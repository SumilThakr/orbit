"""Periodic-orbit adjoint mode — marginal deaths from a deaths-gradient field.

Given a per-cell deaths-gradient field S_(c,a,k) for the
SAS surface, solve the per-month, per-bin, per-species adjoint orbit

    λ_τ = R_τ + P_τ^T λ_{τ+1}                      (mod-N_BINS periodic)

and read off ∂J/∂e_τ = DTAU · (λ_τ − R_τ) per (bin, species, cell).
``J = Σ_τ ⟨S_τ, G_τ c_τ⟩`` is the linearised total marginal deaths, with
G_τ the per-bin PM2.5-growth Jacobian (see :mod:`orbit.core.growth_jacobian`).

Reuses the forward operator LU via :meth:`_UmfpackLU.solve(b, trans='T')`
so no re-factorisation is paid for the adjoint.

**Scope**: full per-species adjoint including K^T chemistry
propagation (pSO4 → SO2, TotalNO3 → NOx, VBS aging cascade
C01 → C1 → C10 → C100 → C1000). Adjoint solves species in
reverse-forward-DAG order so each source's receptor is augmented
by its receivers' λ_r via −DTAU·K[(r,s),τ]^T·(λ_r,τ − R_r,τ).
All five emission-policy species (PM25_primary, NH3 via TotalNH,
SO2, NOx, VOC via VBS bins) are fully correct.
"""
from __future__ import annotations

import os
import time
from typing import Callable

import numpy as np
import scipy.sparse as sp
import xarray as xr

from orbit.core.deposition import N_SPECIES
from orbit.core.grid_data import load_grid
from orbit.core.growth_jacobian import apply_growth_transpose
from orbit.modes.marginal import baseline_vbs_from_c_orbit
from orbit.core.indexing import CellIndexer
from orbit.core.operator import assemble_species_operators
from orbit.core.orbit import DTAU, N_BINS, _SOLVE_ORDER, _factor_one_bin
from orbit.core.orbit_adjoint import (
    solve_orbit_adjoint_one_species,
    gradient_per_emission_bin,
)
from orbit.core.solve import (
    umfpack_symbolic, umfpack_free_symbolic, _HAS_UMFPACK,
)


def _build_adjoint_dag(K_sources_per_bin_bin0: dict) -> tuple[dict, list[int]]:
    """Construct the adjoint DAG and solve order.

    Forward `sources_for_target[r] = [src, ...]` says species r consumes
    species src as a chemistry source. In the adjoint, the dependency
    flips: src's receptor needs r's solved adjoint via K^T propagation.

    Returns
    -------
    receivers_for_source : dict {src: [r, ...]}
        Reversed DAG — for each source species, which receivers feed
        back into its adjoint receptor via K^T.
    adjoint_solve_order : list of species indices
        Reversed `_SOLVE_ORDER` so that for any (r, src) coupling,
        r comes before src in the adjoint order (r's λ available when
        we build src's receptor).
    """
    receivers_for_source: dict[int, list[int]] = {}
    for (target, source) in K_sources_per_bin_bin0.keys():
        receivers_for_source.setdefault(source, []).append(target)
    return receivers_for_source, list(reversed(_SOLVE_ORDER))


from orbit.core.deposition import (
    IDX_PM25, IDX_POA, IDX_TOTAL_NH, IDX_SO2, IDX_NOX,
    IDX_TOTAL_NO3, IDX_VBS_BINS,
)

EMITTED_SPECIES = {
    # NOTE: after the 2026-08-03 POA split, PM25_primary is the *non-POA*
    # remainder (BC, dust, sea salt, biomass-burning primary) — it is NOT
    # total primary PM2.5. POA is a separate emitted species below and
    # carries 84.7% of CEDS anthropogenic primary mass. Multiplying a
    # PM25_primary damage by total primary tonnage understates POA-heavy
    # sectors (residential, transport, industry) relative to power.
    "PM25_primary": IDX_PM25,
    "POA":          IDX_POA,
    "NH3":          IDX_TOTAL_NH,
    "SO2":          IDX_SO2,
    "NOx":          IDX_NOX,
    "VBS_C01":      IDX_VBS_BINS[4],
    "VBS_C1":       IDX_VBS_BINS[3],
    "VBS_C10":      IDX_VBS_BINS[2],
    "VBS_C100":     IDX_VBS_BINS[1],
    "VBS_C1000":    IDX_VBS_BINS[0],
}

# Pseudo-species that combine the 5 VBS bins via per-cell, per-bin
# source-class yields. Each is a separate adjoint-deliverable; the
# driver expands the user request to its underlying VBS_C* set,
# solves those, then combines.
VOC_PSEUDO_SPECIES = ("VOC_anthro", "VOC_bio", "VOC_bb")
# Low-to-high C* (matching VBS_PARENT_YIELDS row order).
VBS_BIN_KEYS = ("VBS_C01", "VBS_C1", "VBS_C10", "VBS_C100", "VBS_C1000")


def _compute_voc_yields_per_bin(voc_class: str, grids, bio_voc_split=None):
    """Per-bin per-cell mass yields (5_vbs, ny, nx) for one VOC source class.

    Returns shape (N_BINS, 5_vbs, ny, nx). Yields are the same as the
    forward solver's VOC→VBS distribution
    (`orbit.emissions.vbs_distribution.distribute_voc_to_vbs_bins`),
    with NO2/OH-dependent NOx-regime interpolation for `VOC_anthro` and
    a constant SAS monoterpene/isoprene split for `VOC_bio`.
    """
    from orbit.emissions.vbs_distribution import (
        compute_nox_regime, BIO_VOC_SPLIT,
    )
    from orbit.emissions.vbs_yields import VBS_PARENT_YIELDS
    if bio_voc_split is None:
        bio_voc_split = BIO_VOC_SPLIT
    g0 = grids[0]
    ny, nx = g0.ny, g0.nx

    out_per_bin = []
    for g in grids:
        if voc_class == "VOC_anthro":
            F = compute_nox_regime(g)   # (ny, nx)
            y_hi = VBS_PARENT_YIELDS["anthro_high_nox"]        # (5,)
            y_lo = VBS_PARENT_YIELDS["anthro_low_nox"]
            yields_eff = (
                F[None, :, :] * y_hi[:, None, None]
                + (1.0 - F[None, :, :]) * y_lo[:, None, None]
            )  # (5, ny, nx)
        elif voc_class == "VOC_bio":
            y_mono = VBS_PARENT_YIELDS["bio_monoterpene"]
            y_iso  = VBS_PARENT_YIELDS["bio_isoprene"]
            f_mono = float(bio_voc_split.get("monoterpene", 0.15))
            f_iso  = float(bio_voc_split.get("isoprene",    0.85))
            y_combined = f_mono * y_mono + f_iso * y_iso       # (5,)
            yields_eff = np.broadcast_to(
                y_combined[:, None, None], (5, ny, nx)
            ).copy()
        elif voc_class == "VOC_bb":
            y_bb = VBS_PARENT_YIELDS["biomass_burning"]
            yields_eff = np.broadcast_to(
                y_bb[:, None, None], (5, ny, nx)
            ).copy()
        else:
            raise ValueError(
                f"Unknown VOC pseudo-species {voc_class!r}. "
                f"Choose from {VOC_PSEUDO_SPECIES}."
            )
        out_per_bin.append(yields_eff)
    return np.stack(out_per_bin, axis=0)  # (N_BINS, 5, ny, nx)


def _load_gradient_nc(gradient_path: str) -> tuple:
    """Read deaths_gradient_<crf>.nc → all draws of S summed over endpoints.

    The summed-over-endpoints field is what the per-bin receptor needs
    (deaths are additive over endpoints). The per-endpoint cube is dropped
    here — downstream cause-specific aggregation would re-load the file
    with per-endpoint indexing.

    Returns
    -------
    S_summed_per_draw : ndarray (n_draws, lat, lon)
        Per-draw deaths-gradient summed over endpoints.
        n_draws includes the draw=0 deterministic and any MC samples.
    causes, ages : list str/int — endpoint metadata
    lat, lon : ndarray — gradient grid centres
    attrs : dict — passthrough of file-level attrs (crf_mode, cause_set, …)
    """
    ds = xr.open_dataset(gradient_path)
    # S shape: (draw, endpoint, lat, lon). Sum over endpoints.
    S_per_draw_summed = ds["S"].sum("endpoint").values.astype(np.float64)
    causes = ds["cause"].values.tolist()
    ages = ds["age"].values.tolist()
    lat = ds["lat"].values
    lon = ds["lon"].values
    attrs = dict(ds.attrs)
    ds.close()
    return S_per_draw_summed, causes, ages, lat, lon, attrs


def _regrid_gradient_to_orbit_grid(
    S_grad: np.ndarray,           # (n_lat_grad, n_lon_grad) at 0.1°
    lat_grad: np.ndarray, lon_grad: np.ndarray,
    lat_orbit: np.ndarray, lon_orbit: np.ndarray,
) -> np.ndarray:
    """Conservative-sum regrid of the deaths-gradient from its 0.1° to ORBIT
    0.5° × 0.625°.

    S has units of "deaths · year⁻¹ · μg⁻¹ · m³ · cell⁻¹" — per-cell
    extensive. Summing fine cells into a coarse cell preserves the
    sensitivity (an aggregate deaths-per-unit-PM in the coarse cell).

    Implementation: build a gradient→orbit cell mapping by nearest-edge,
    bincount-sum into the coarse layer.
    """
    n_orbit_lat, n_orbit_lon = lat_orbit.size, lon_orbit.size
    # Half-widths of the ORBIT cells.
    dlat_o = float(lat_orbit[1] - lat_orbit[0])
    dlon_o = float(lon_orbit[1] - lon_orbit[0])

    out = np.zeros((n_orbit_lat, n_orbit_lon), dtype=np.float64)
    for i, lat in enumerate(lat_grad):
        # Which ORBIT lat row contains this gradient row?
        oi = int(round((lat - lat_orbit[0]) / dlat_o))
        if not (0 <= oi < n_orbit_lat):
            continue
        for j, lon in enumerate(lon_grad):
            oj = int(round((lon - lon_orbit[0]) / dlon_o))
            if not (0 <= oj < n_orbit_lon):
                continue
            out[oi, oj] += S_grad[i, j]
    return out


def run_adjoint_month(
    month: int,
    *,
    gradient_path: str,
    output_dir: str,
    preproc_path_fn: Callable[[int, int], str],
    constants_path: str,
    orbit_npz_path: str | None = None,
    species_keys: list[str] | None = None,
    krylov_tol: float = 1e-6,
    maxiter: int = 200,
    horizontal_fct: bool = False,
    verbose: bool = True,
    output_filename: str | None = None,
) -> dict:
    """Solve the periodic-orbit adjoint for one month → ∂J/∂e per (bin, sp, cell).

    Parameters
    ----------
    month : 1..12
    gradient_path : path to the deaths_gradient_<crf>.nc.
    output_dir : where to write the adjoint NetCDF.
    preproc_path_fn : callable(month, bin_idx_1based) → preproc bin file path.
    constants_path : MERRA2 constants file path.
    species_keys : subset of EMITTED_SPECIES.
        Default: ['PM25_primary', 'NH3', 'SO2', 'NOx'] — the four direct
        policy levers. Add VBS_C* keys to expose VOC sensitivities by
        volatility bin.
    krylov_tol, maxiter : GMRES tolerance / iteration cap.
    output_filename : override for the output file name.
    """
    if not _HAS_UMFPACK:
        raise RuntimeError("UMFPACK is required for the adjoint mode.")
    if species_keys is None:
        species_keys = ["PM25_primary", "NH3", "SO2", "NOx"]
    # Two species groups: direct emissions (1:1 mapping to a ORBIT species
    # index in EMITTED_SPECIES) and VOC pseudo-species (combinations of
    # the 5 VBS bins via per-cell yield tables).
    direct_keys = [k for k in species_keys if k in EMITTED_SPECIES]
    voc_keys = [k for k in species_keys if k in VOC_PSEUDO_SPECIES]
    unknown = [
        k for k in species_keys
        if k not in EMITTED_SPECIES and k not in VOC_PSEUDO_SPECIES
    ]
    if unknown:
        raise ValueError(
            f"Unknown species keys: {unknown}. Choose from "
            f"{sorted(EMITTED_SPECIES)} + {VOC_PSEUDO_SPECIES}."
        )
    # If any VOC pseudo-species is requested, we MUST solve the 5 VBS bins
    # (regardless of whether the user listed them directly). Add them
    # silently to the solve set; the output cube exposes them as
    # first-class species in the NetCDF only when the user requested them.
    if voc_keys:
        for vbs_key in VBS_BIN_KEYS:
            if vbs_key not in direct_keys:
                direct_keys.append(vbs_key)
    # `species_keys` from here on means "direct solver-index species we
    # need to solve and possibly emit"; VOC pseudo-species are handled
    # in a separate post-process step.
    species_keys = direct_keys

    timings: dict = {"month": month}
    out_name = output_filename or f"adjoint_M{month:02d}.nc"
    out_path = os.path.join(output_dir, out_name)
    os.makedirs(output_dir, exist_ok=True)

    # ── 1. Read the deaths gradient → S_summed_grad per draw ───────────────
    if verbose:
        print(f"  Loading deaths gradient: {gradient_path}")
    t0 = time.time()
    (S_summed_per_draw, causes, ages,
     lat_grad, lon_grad, grad_attrs) = _load_gradient_nc(gradient_path)
    n_draws_total = S_summed_per_draw.shape[0]
    if verbose:
        print(f"  Gradient has {n_draws_total} draws "
              f"(draw=0 deterministic + {n_draws_total - 1} MC)")
    timings["t_load_gradient"] = time.time() - t0

    # ── 2. Build per-bin grids + operators + LU factorisations ────────
    if verbose:
        print(f"  Building per-bin operators (8 × {N_SPECIES} species)")
    t0 = time.time()
    bin_paths = [preproc_path_fn(month, tau + 1) for tau in range(N_BINS)]
    grids = [load_grid(bin_paths[tau], constants_path) for tau in range(N_BINS)]
    g0 = grids[0]
    indexer = CellIndexer(g0.nz, g0.ny, g0.nx)
    N = indexer.N

    # 2a. Regrid each draw's deaths-gradient to the ORBIT grid (sum over
    # endpoints was already done in _load_gradient_nc). Replicate across
    # bins — the gradient is annual, so each bin sees the same
    # surface receptor.
    S_summed_orbit_per_draw = np.stack([
        _regrid_gradient_to_orbit_grid(
            S_summed_per_draw[d], lat_grad, lon_grad, g0.lat, g0.lon,
        )
        for d in range(n_draws_total)
    ], axis=0)   # (n_draws, ny, nx)

    # 2b. Pull production-baseline fields from the orbit NPZ:
    #   - ISORROPIA partitioning fractions (per-bin f_nh4_marg_3d /
    #     f_no3_marg_3d — more accurate than bin-mean iso_f_*_mean).
    #   - Per-bin VBS partitioning (F_p_vbs_marg_3d) and OA mass
    #     (M_OA_marg_3d) — without these, the VBS receptor terms
    #     would be zero (g.F_p_vbs from load_grid is empty by default).
    #   - Iso cross-partials (f_nh_dno3, f_no3_dnh, f_nh_dso4,
    #     f_no3_dso4 — the cyclic NH↔NO3 + sulfate-driven blocks the
    #     forward Picard handles via assemble_iso_cross_blocks).
    p_nh4_surface_per_bin = None
    p_no3_surface_per_bin = None
    iso_cross_3d = None
    baseline_vbs_c = None
    if orbit_npz_path is not None and os.path.exists(orbit_npz_path):
        if verbose:
            print(f"  Loading orbit baseline: {orbit_npz_path}")
        orbit_npz = np.load(orbit_npz_path, allow_pickle=False)
        # Partitioning: prefer per-bin marg field if available; fall
        # back to bin-mean iso_f_*_mean broadcast.
        if "f_nh4_marg_3d" in orbit_npz.files:
            arr = orbit_npz["f_nh4_marg_3d"]   # (N_BINS, nz, ny, nx)
            p_nh4_surface_per_bin = arr[:, 0, :, :].astype(np.float64)
        elif "iso_f_nh4_mean" in orbit_npz.files:
            iso_nh4_surf = orbit_npz["iso_f_nh4_mean"][0]
            p_nh4_surface_per_bin = np.broadcast_to(
                iso_nh4_surf, (N_BINS, g0.ny, g0.nx)
            ).copy()
        if "f_no3_marg_3d" in orbit_npz.files:
            arr = orbit_npz["f_no3_marg_3d"]
            p_no3_surface_per_bin = arr[:, 0, :, :].astype(np.float64)
        elif "iso_f_no3_mean" in orbit_npz.files:
            iso_no3_surf = orbit_npz["iso_f_no3_mean"][0]
            p_no3_surface_per_bin = np.broadcast_to(
                iso_no3_surf, (N_BINS, g0.ny, g0.nx)
            ).copy()
        # Baseline per-bin VBS bin masses for the Pankow feedback factor 1/D
        # on the POA and VBS receptors, built exactly as marginal mode builds
        # them. Until 2026-09-26 this stayed None, so D was 1 everywhere: the
        # POA receptor equalled the primary-PM2.5 receptor and the VBS bins
        # carried F_p alone.
        if "c_orbit" in orbit_npz.files:
            baseline_vbs_c = baseline_vbs_from_c_orbit(
                orbit_npz["c_orbit"], g0.nz, g0.ny, g0.nx)
        elif verbose:
            print("  WARNING: baseline NPZ has no c_orbit; the organic "
                  "partitioning feedback (1/D) is off in the receptor.")
        # Per-bin VBS state for the growth Jacobian's F_p × 1/D term.
        # apply_growth_transpose looks at g.F_p_vbs / g.M_OA_3d; we
        # inject the marg fields directly into the grids list below.
        F_p_vbs_marg_3d = None
        M_OA_marg_3d = None
        if "F_p_vbs_marg_3d" in orbit_npz.files:
            F_p_vbs_marg_3d = orbit_npz["F_p_vbs_marg_3d"].astype(np.float64)
        if "M_OA_marg_3d" in orbit_npz.files:
            M_OA_marg_3d = orbit_npz["M_OA_marg_3d"].astype(np.float64)
        for tau in range(N_BINS):
            if F_p_vbs_marg_3d is not None:
                grids[tau].F_p_vbs = F_p_vbs_marg_3d[tau]   # (5, nz, ny, nx)
            if M_OA_marg_3d is not None:
                grids[tau].M_OA_3d = M_OA_marg_3d[tau]      # (nz, ny, nx)
            # Mirror marginal.py:336-344: overwrite NH/NO3 partitioning with
            # the ISORROPIA-marg fields BEFORE assemble_species_operators.
            # The deposition operator uses g.NHPartitioning / NO3Partitioning
            # to split each total-pool species into gas vs particle dep rates;
            # without this overwrite, NO3Partitioning is empty → deposition
            # treats TotalNO3 as 100% HNO3 (fast-dep bound, deposition.py:166)
            # and L_adjoint ≠ L_forward-marginal. That asymmetry collapses
            # the adjoint's λ_TotalNO3 by ~7× at Delhi-like high-pNO3 cells.
            # Equilibrium fields are not consumed by the deposition operator
            # but kept self-consistent in case any post-hoc PM2.5 split runs.
            if (p_nh4_surface_per_bin is not None
                    and "f_nh4_marg_3d" in orbit_npz.files):
                f_nh4_3d_full = np.asarray(orbit_npz["f_nh4_marg_3d"][tau], dtype=np.float64)
                grids[tau].NHPartitioning   = f_nh4_3d_full
                grids[tau].NHPartitioningEq = f_nh4_3d_full
            if (p_no3_surface_per_bin is not None
                    and "f_no3_marg_3d" in orbit_npz.files):
                f_no3_3d_full = np.asarray(orbit_npz["f_no3_marg_3d"][tau], dtype=np.float64)
                grids[tau].NO3Partitioning   = f_no3_3d_full
                grids[tau].NO3PartitioningEq = f_no3_3d_full
        # Iso cross-partials — for the cyclic NH↔NO3 + sulfate-driven
        # iso K-blocks the forward marginal builds via
        # orbit.modes.iso_coupling.assemble_iso_cross_blocks.
        iso_keys = ("f_nh_dno3_3d", "f_no3_dnh_3d",
                    "f_nh_dso4_3d", "f_no3_dso4_3d")
        if all(k in orbit_npz.files for k in iso_keys):
            iso_cross_3d = {
                k.replace("_3d", ""): orbit_npz[k].astype(np.float64)
                for k in iso_keys
            }
            if verbose:
                print(f"  Loaded iso cross-partials: {list(iso_cross_3d)}")
        orbit_npz.close()
    timings["t_setup_receptors"] = time.time() - t0

    # 2c. Assemble per-bin operators (same call as forward marginal). KEEP
    # K_sources_per_bin — we need K^T contributions for the adjoint
    # chemistry chain (pSO4→SO2, TotalNO3→NOx, VBS aging cascade). If
    # iso cross-partials were loaded, merge the iso K-blocks into
    # K_sources_per_bin too (handles the NH↔NO3 cyclic coupling that
    # the forward Picard iterates on).
    t0 = time.time()
    L_species_per_bin: list[list[sp.csc_matrix]] = []
    K_sources_per_bin: list[dict] = []
    endbin_keys: set = set()
    for tau in range(N_BINS):
        L_species, K_sources, _T, _d = assemble_species_operators(
            grids[tau], indexer, verbose=False, scheme="exp",
        )
        if iso_cross_3d is not None:
            from orbit.modes.iso_coupling import assemble_iso_cross_blocks
            iso_blocks = assemble_iso_cross_blocks(
                grids[tau], indexer,
                f_nh_dno3=iso_cross_3d["f_nh_dno3"][tau],
                f_no3_dnh=iso_cross_3d["f_no3_dnh"][tau],
                f_nh_dso4=iso_cross_3d["f_nh_dso4"][tau],
                f_no3_dso4=iso_cross_3d["f_no3_dso4"][tau],
            )
            # Merge: any existing chem-DAG block at the same (target, src)
            # is summed with the iso contribution (rare in practice). All
            # iso blocks use the end-of-bin convention (the source-state
            # they read is c_src[τ+1], i.e. start of bin τ+1).
            for key, K_iso in iso_blocks.items():
                if key in K_sources:
                    K_sources[key] = (K_sources[key] + K_iso).tocsc()
                else:
                    K_sources[key] = K_iso
                endbin_keys.add(key)
        L_species_per_bin.append(L_species)
        K_sources_per_bin.append(K_sources)
    if iso_cross_3d is not None and verbose:
        from orbit.core.orbit import _SPECIES_NAMES as _sn
        print(f"  Merged {len(endbin_keys)} iso K-block keys into K_sources "
              f"({sorted([(_sn[t], _sn[s]) for (t, s) in endbin_keys])})")
    timings["t_assemble"] = time.time() - t0

    # 2d. Determine which species need adjoint solves: the user-requested
    # set, plus its closure over forward-receivers (so K^T propagation
    # has all the downstream λ_r it needs). Adjoint DAG = REVERSED of
    # forward, so we iterate user_set then walk receivers BFS.
    receivers_for_source, _ = _build_adjoint_dag(K_sources_per_bin[0])
    user_set = {EMITTED_SPECIES[k] for k in species_keys}
    to_solve: set[int] = set(user_set)
    queue = list(user_set)
    while queue:
        s = queue.pop()
        for r in receivers_for_source.get(s, []):
            if r not in to_solve:
                to_solve.add(r)
                queue.append(r)
    # Adjoint solve order: pass `receivers_for_source` to the same wave-
    # building helper the forward uses — it interprets the dict as
    # "deps_for_species" (each species in wave max(deps' waves) + 1).
    # Cyclic edges (NH↔NO3 from iso) are broken at the DAG level; the
    # Picard outer loop below repairs them. This gives the correct
    # topological order under iso coupling: NH/NO3 solved BEFORE pSO4
    # (which has K^T from both), and pSO4 BEFORE SO2 (chem-DAG).
    from orbit.core.orbit import _build_solve_waves
    waves = _build_solve_waves(receivers_for_source, _SOLVE_ORDER, set())
    solve_order = [s for wave in waves for s in wave if s in to_solve]
    if verbose:
        from orbit.core.orbit import _SPECIES_NAMES
        print(f"  Adjoint solve plan: {len(solve_order)} species in order "
              f"{[_SPECIES_NAMES[s] for s in solve_order]}")

    # 2d-FCT. Frozen-coefficient anti-diffusive operator L_AD for the FCT
    # tangent-linear, applied by DEFERRED CORRECTION (NOT added to the LHS).
    # Adding L_AD to L_low and factoring (L_low+L_AD)^T is pathological: L_AD is
    # anti-diffusion (non-M-matrix) comparable in magnitude to L_low — it removes
    # ~60% of the transport's effective mixing — so the periodic (I−M^T) GMRES is
    # ~80× slower / effectively non-convergent (diag_fct_marginal_gmres, job
    # 10254401: L_low 18 iters/11s vs L_low+L_AD >16min). Instead keep L_low^T as
    # the factored, well-conditioned propagator and iterate the transposed
    # anti-diffusive term as a deferred RECEPTOR source — the exact transpose of
    # the forward deferred correction (which adds extra_rhs_τ = −L_AD,τ·c[τ+1] to
    # bin τ). Transposing the whole-orbit forward iteration gives the per-draw
    # adjoint update ν_σ += −DTAU·L_AD,(σ−1)^T·μ_(σ−1), with the co-forcing
    # μ_τ = λ_τ − ν_τ (= ∂J/∂e_τ / DTAU) and a one-bin shift mirroring the
    # forward end-of-bin convention. Here we just assemble L_AD^T per
    # (bin, species), frozen at the FCT baseline orbit (end-of-bin c_orbit[s,τ+1],
    # matching the forward/marginal convention); the per-draw FCT outer loop
    # below injects it.
    fct_L_AD_T_per_bin: list[dict] = [dict() for _ in range(N_BINS)]
    if horizontal_fct:
        from orbit.core.fct import assemble_fct_linear_operator
        from orbit.core.orbit import DTAU as _DTAU
        if orbit_npz_path is None or not os.path.exists(orbit_npz_path):
            raise RuntimeError(
                "horizontal_fct adjoint needs the baseline orbit NPZ "
                "(--orbit-npz) pointing at an FCT forward orbit.")
        _onpz = np.load(orbit_npz_path, allow_pickle=False)
        if "c_orbit" not in _onpz.files:
            _onpz.close()
            raise RuntimeError(
                "horizontal_fct adjoint needs the baseline orbit field "
                "(c_orbit) in the orbit NPZ — point --orbit-npz at an FCT "
                "forward orbit.")
        c_orbit_base = np.asarray(_onpz["c_orbit"], dtype=np.float64)
        order = (list(np.asarray(_onpz["orbit_species_order"]).ravel())
                 if "orbit_species_order" in _onpz.files
                 else list(range(c_orbit_base.shape[0])))
        _onpz.close()
        row_for_species = {int(sp): i for i, sp in enumerate(order)}
        for tau in range(N_BINS):
            for s in solve_order:
                if s not in row_for_species:
                    continue
                c_bs = c_orbit_base[row_for_species[s], tau + 1]  # end-of-bin
                L_AD = assemble_fct_linear_operator(grids[tau], indexer, c_bs, _DTAU)
                fct_L_AD_T_per_bin[tau][s] = L_AD.T.tocsc()
        if verbose:
            print(f"  [FCT] frozen L_AD^T assembled for {len(solve_order)} "
                  f"species × {N_BINS} bins; applied by deferred correction (adjoint)")

    # 2e. Factor LU per (species ∈ to_solve, bin). UMFPACK symbolic
    # is reused across all (species × bin) since they share L's
    # sparsity pattern.
    if verbose:
        print(f"  Factoring LUs (UMFPACK; {len(solve_order)} × {N_BINS} factors)")
    t0 = time.time()
    L_sample = L_species_per_bin[0][0]
    umfpack_sym = umfpack_symbolic(L_sample, verbose=False)
    I_N = sp.eye(N, format="csc")
    lu_per_species: dict[int, list] = {}
    for sp_idx in solve_order:
        lu_per_species[sp_idx] = [
            _factor_one_bin(L_species_per_bin[tau][sp_idx], I_N, umfpack_sym, None)
            for tau in range(N_BINS)
        ]
    timings["t_factor"] = time.time() - t0

    # ── 3. MC outer loop × per-species adjoint inner loop ─────────────
    # Outer loop over draws (deterministic + MC) is cheap because the
    # LU factorisations are SHARED — only the RHS (G^T S_d) varies per
    # draw. Per-species K^T propagation happens inside the per-draw
    # loop because λ_r and R_r are draw-specific.
    #
    # Output cube is surface-only (z=0); the aggregator only uses
    # surface and the (n_draws, n_species, n_bins, 15, ny, nx) full-3D
    # cube would balloon storage by 15×.
    from orbit.core.orbit import _SPECIES_NAMES
    t0 = time.time()
    # Output species axis: direct keys first (preserving order), then VOC
    # pseudo-species. Direct keys map to solver indices via EMITTED_SPECIES;
    # VOC pseudos are built from VBS rows post-solve and don't have one.
    species_keys_out = list(direct_keys) + list(voc_keys)
    # Per-species-key row index in the output cube.
    out_row_for_key = {k: i for i, k in enumerate(species_keys_out)}
    grad_surface_cube = np.zeros(
        (n_draws_total, len(species_keys_out), N_BINS, g0.ny, g0.nx),
        dtype=np.float32,
    )
    # Persist solver_info from the deterministic (draw=0) pass only —
    # MC residuals/iters look statistically the same.
    solver_info_d0: dict[int, dict] = {}
    surface_size = g0.ny * g0.nx
    for d in range(n_draws_total):
        if verbose and (d == 0 or n_draws_total <= 5 or d % max(1, n_draws_total // 10) == 0):
            print(f"  --- draw {d}/{n_draws_total - 1} ---")
        # Per-draw receptors: G^T applied to the per-draw S field.
        S_surface_per_bin_d = np.broadcast_to(
            S_summed_orbit_per_draw[d], (N_BINS, g0.ny, g0.nx)
        ).copy()
        receptors_d = apply_growth_transpose(
            S_surface_per_bin_d, grids, indexer, baseline_vbs_c,
            p_nh4_surface_per_bin=p_nh4_surface_per_bin,
            p_no3_surface_per_bin=p_no3_surface_per_bin,
        )

        # ── Per-draw species solve, optionally with FCT deferred correction.
        # _run_species_passes solves every to-solve species' adjoint orbit
        # (reverse-DAG first pass + iso Picard + downstream refresh), with an
        # optional FCT deferred-correction receptor `fct_recep` added to each
        # species' R_list. fct_recep=None reproduces the low-order adjoint
        # byte-for-byte (non-FCT path unchanged). Returns (lam, R) where R is the
        # EFFECTIVE receptor list used (G^T S + K^T + iso + FCT), so the standard
        # gradient identity ∂J/∂e_τ = DTAU·(λ_τ − R_τ) holds for the output.
        def _run_species_passes(fct_recep, pass_verbose):
            lam_per_species_d: dict[int, list] = {}
            R_per_species_d: dict[int, list] = {}

            def _build_R_list_for(sp_idx_):
                """Build R_list for species `sp_idx_`: G^T S receptor + K^T
                contributions from every already-solved receiver (chem-DAG
                start-of-bin τ and iso endbin τ → τ+1 conventions), plus the
                FCT deferred-correction source if present. Cyclic receivers
                not yet solved contribute zero this pass (Picard fixes it).
                """
                R_list_ = [
                    receptors_d[tau, sp_idx_].copy() for tau in range(N_BINS)
                ]
                n_coup = 0
                for r_idx_ in receivers_for_source.get(sp_idx_, []):
                    if r_idx_ not in lam_per_species_d:
                        continue  # cyclic — picked up by Picard later
                    lam_r_ = lam_per_species_d[r_idx_]
                    R_r_ = R_per_species_d[r_idx_]
                    for tau in range(N_BINS):
                        K_block = K_sources_per_bin[tau].get((r_idx_, sp_idx_))
                        if K_block is None:
                            continue
                        # Iso K-blocks use end-of-bin convention: the forward
                        # has c_recv[τ+1] depending on c_src[τ+1] (not c_src[τ]),
                        # so the adjoint contribution to ∂J/∂c_src[τ+1] lands
                        # in R_src[(τ+1) mod N_BINS] rather than R_src[τ].
                        is_endbin = (r_idx_, sp_idx_) in endbin_keys
                        target_bin = (tau + 1) % N_BINS if is_endbin else tau
                        R_list_[target_bin] -= DTAU * (
                            K_block.T @ (lam_r_[tau] - R_r_[tau])
                        )
                        n_coup += 1
                # FCT deferred-correction source (added last, after the K^T/iso
                # couplings, so the returned R = ν includes it and the (λ − R)
                # gradient identity stays exact). See §13.7.
                if fct_recep is not None and sp_idx_ in fct_recep:
                    fr = fct_recep[sp_idx_]
                    for tau in range(N_BINS):
                        R_list_[tau] = R_list_[tau] + fr[tau]
                return R_list_, n_coup

            # ── 3a. First pass: solve every species in reverse-DAG order.
            # Cyclic-iso pairs (NH↔NO3) get zero K^T contribution this pass
            # for their cyclic counterpart; the Picard loop below fixes it.
            for sp_idx in solve_order:
                R_list, n_couplings = _build_R_list_for(sp_idx)
                adjoint, info = solve_orbit_adjoint_one_species(
                    lu_per_species[sp_idx], R_list,
                    tol=krylov_tol, maxiter=maxiter,
                )
                lam_per_species_d[sp_idx] = adjoint
                R_per_species_d[sp_idx] = R_list
                if pass_verbose:
                    solver_info_d0[sp_idx] = info
                    if verbose:
                        name = _SPECIES_NAMES[sp_idx]
                        coup_tag = (
                            f", K^T from {n_couplings // N_BINS} receivers"
                            if n_couplings else ""
                        )
                        print(f"    [{name}]  gmres_iters={info['gmres_iters']}, "
                              f"residual={info['residual']:.2e}, "
                              f"periodicity={info['periodicity']:.2e}, "
                              f"|λ_0|={np.linalg.norm(adjoint[0]):.3e}{coup_tag}")

            # ── 3b. Picard outer loop for cyclic iso species (NH ↔ NO3).
            # Only runs if both TotalNH and TotalNO3 are in `solve_order`.
            # Mirrors the forward marginal's Picard iteration on species
            # [2, 6]. Convergence target: max relative change < 1e-4 (a few
            # iters in practice, like the forward's 3-4 iters).
            cyclic_species = [IDX_TOTAL_NH, IDX_TOTAL_NO3]
            cyclic_active = (
                iso_cross_3d is not None
                and all(s in lam_per_species_d for s in cyclic_species)
            )
            if cyclic_active:
                max_picard_iters = 10
                picard_rel_tol = 1e-4
                for picard_iter in range(max_picard_iters):
                    lam_prev = {
                        s: [a.copy() for a in lam_per_species_d[s]]
                        for s in cyclic_species
                    }
                    for sp_idx in cyclic_species:
                        R_list, _ = _build_R_list_for(sp_idx)
                        adjoint, info = solve_orbit_adjoint_one_species(
                            lu_per_species[sp_idx], R_list,
                            tol=krylov_tol, maxiter=maxiter,
                        )
                        lam_per_species_d[sp_idx] = adjoint
                        R_per_species_d[sp_idx] = R_list
                    # Convergence check: max relative L2 change across the
                    # two cyclic species' adjoint orbits.
                    max_rel = 0.0
                    for s in cyclic_species:
                        new_o = lam_per_species_d[s]
                        old_o = lam_prev[s]
                        for tau in range(N_BINS):
                            delta = float(np.linalg.norm(new_o[tau] - old_o[tau]))
                            base = float(np.linalg.norm(new_o[tau]))
                            rel = delta / (base + 1e-30)
                            if rel > max_rel:
                                max_rel = rel
                    if pass_verbose and verbose:
                        print(f"    Picard iso iter {picard_iter+1}: "
                              f"max rel change = {max_rel:.3e}")
                    if max_rel < picard_rel_tol:
                        break

            # ── 3b'. Refresh species that pull K^T from the (now-converged)
            # cyclic pair. In Phase 3a (first pass) these were solved with
            # stale or zero-valued NH/NO3 adjoints; with the corrected
            # topological order, anything in a wave AFTER the cyclic species
            # may need re-solving. Simplest correct rule: re-solve every
            # species in waves after the cyclic species' wave (wave order is
            # topological).
            if cyclic_active:
                cyclic_wave_idx = None
                for w_idx, wave in enumerate(waves):
                    if any(s in wave for s in cyclic_species):
                        cyclic_wave_idx = w_idx
                        break
                if cyclic_wave_idx is not None:
                    downstream = [
                        s for w_idx in range(cyclic_wave_idx + 1, len(waves))
                        for s in waves[w_idx]
                        if s in to_solve
                    ]
                    if downstream and pass_verbose and verbose:
                        from orbit.core.orbit import _SPECIES_NAMES as _sn2
                        print(f"    Re-solving {len(downstream)} downstream-of-iso "
                              f"species: {[_sn2[s] for s in downstream]}")
                    for sp_idx in downstream:
                        R_list, _ = _build_R_list_for(sp_idx)
                        adjoint, info = solve_orbit_adjoint_one_species(
                            lu_per_species[sp_idx], R_list,
                            tol=krylov_tol, maxiter=maxiter,
                        )
                        lam_per_species_d[sp_idx] = adjoint
                        R_per_species_d[sp_idx] = R_list

            return lam_per_species_d, R_per_species_d

        # ── Invoke: single low-order pass, or FCT deferred-correction solve.
        if horizontal_fct:
            # Deferred correction: keep L_low^T factored (fast solve), treat
            # the transposed anti-diffusive term as a receptor source. The
            # co-forcing μ satisfies the AFFINE fixed point μ = K μ + b where
            # b is the low-order solution and each application of K costs one
            # inner adjoint solve.
            #
            # K is NOT a contraction: on the January 2022 production baseline
            # it has a real eigenvalue ≈ +1.33 (measured 2026-08-03 by plain
            # Picard — residual grows at ratio 1.33/iter and saturates at
            # (λ−1)/λ = 0.2475), so Picard diverges and Anderson(5) can only
            # hold the residual at ~1e-2. That ~1% error shows up as small
            # NEGATIVE ∂J/∂e for primary aerosol (physically impossible —
            # inert primary mass cannot reduce deaths). The fixed point still
            # exists and is unique (1 is not an eigenvalue), so the default
            # solver is outer GMRES on (I−K)μ = b, which handles isolated
            # eigenvalues outside the unit disk without difficulty.
            # ORBIT_ADJOINT_FCT_SOLVER selects gmres|anderson|picard (the
            # last two retained as diagnostics; Picard's residual ratios read
            # off K's dominant eigenvalue directly).
            from orbit.core.dcomp_iter import AndersonAccelerator
            FCT_SOLVER = os.environ.get("ORBIT_ADJOINT_FCT_SOLVER", "gmres")
            FCT_MAX_ITERS = int(os.environ.get(
                "ORBIT_ADJOINT_FCT_MAX_ITERS",
                "40" if FCT_SOLVER == "gmres" else "12"))
            FCT_REL_TOL = float(os.environ.get("ORBIT_ADJOINT_FCT_REL_TOL", "2.0e-3"))
            flat_keys = sorted(solve_order)

            def _mu_from(lam_d, R_d):
                # co-forcing μ_s[τ] = λ_s[τ] − R_s[τ] (= ∂J/∂e_s,τ / DTAU)
                return {s: [lam_d[s][tau] - R_d[s][tau] for tau in range(N_BINS)]
                        for s in flat_keys}

            def _recep_from_mu(mu_d):
                # ν_σ += −DTAU·L_AD,(σ−1)^T·μ_(σ−1)  (one-bin shift = transpose
                # of the forward end-of-bin source; L_AD frozen at the baseline).
                recep: dict[int, list] = {}
                for s in flat_keys:
                    lst = []
                    for tau in range(N_BINS):
                        src = (tau - 1) % N_BINS
                        LADT = fct_L_AD_T_per_bin[src].get(s)
                        if LADT is not None:
                            lst.append(-DTAU * (LADT @ mu_d[s][src]))
                        else:
                            lst.append(np.zeros(N, dtype=np.float64))
                    recep[s] = lst
                return recep

            def _flat_mu(mu_d):
                return np.concatenate(
                    [np.concatenate(mu_d[s]) for s in flat_keys])

            def _unflat_mu(vec):
                out, i = {}, 0
                for s in flat_keys:
                    lst = []
                    for _ in range(N_BINS):
                        lst.append(vec[i:i + N]); i += N
                    out[s] = lst
                return out

            lam_per_species_d, R_per_species_d = _run_species_passes(
                None, d == 0)                                  # μ_0 = b: low-order
            b_flat = _flat_mu(_mu_from(lam_per_species_d, R_per_species_d))

            def _apply_g(vec):
                """One application of the affine map g(x) = K x + b: build the
                FCT receptor from x, run one inner adjoint solve, read off μ.
                Returns (g_flat, lam, R) so the accepted solution's λ/R can be
                reused for output without an extra pass."""
                lam_v, R_v = _run_species_passes(
                    _recep_from_mu(_unflat_mu(vec)), False)
                return _flat_mu(_mu_from(lam_v, R_v)), lam_v, R_v

            def _rel_inf(g_flat, x_flat):
                denom = float(np.abs(g_flat).max())
                return (float(np.abs(g_flat - x_flat).max())
                        / (denom if denom > 1e-30 else 1.0))

            n_fct_solves = [1]                    # the b pass above
            if FCT_SOLVER == "gmres":
                import scipy.sparse.linalg as _spla

                def _outer_matvec(vec):
                    # (I − K) v, with K v = g(v) − b (g affine).
                    g_v, _, _ = _apply_g(vec)
                    n_fct_solves[0] += 1
                    return vec - (g_v - b_flat)

                op = _spla.LinearOperator(
                    (b_flat.size, b_flat.size), matvec=_outer_matvec,
                    dtype=np.float64)
                b_norm = float(np.linalg.norm(b_flat))
                pr_hist: list = []

                # ── Optional checkpointing ──────────────────────────
                # The NetCDF is only written after ALL draws finish, so a
                # walltime kill loses the entire run. Set
                # ORBIT_ADJOINT_FCT_CHECKPOINT_DIR to have the outer
                # solve dump its current μ every CHECKPOINT_EVERY
                # iterations. μ IS the deliverable up to a constant
                # (∂J/∂e_τ = Δτ·μ_τ), so a checkpoint is directly usable
                # — no re-solve needed to recover the science.
                #
                # Mechanism note: scipy calls back with the *solution*
                # only once per restart cycle (`callback_type="x"`), so
                # enabling checkpoints pins restart = CHECKPOINT_EVERY.
                # That makes the run restarted-GMRES with a shorter
                # cycle, which converges slightly slower per iteration
                # than one long cycle — a real trade, taken only when
                # checkpointing is explicitly asked for. Per-iteration
                # residual logging is unavailable in this mode (scipy
                # gives either the iterate or the residual, not both);
                # the honest ∞-norm is still computed at the end.
                ckpt_dir = os.environ.get("ORBIT_ADJOINT_FCT_CHECKPOINT_DIR")
                ckpt_every = int(os.environ.get(
                    "ORBIT_ADJOINT_FCT_CHECKPOINT_EVERY", "10"))

                if ckpt_dir:
                    os.makedirs(ckpt_dir, exist_ok=True)
                    n_ck = [0]

                    def _cb_x(xk):
                        n_ck[0] += 1
                        it = n_ck[0] * ckpt_every
                        path = os.path.join(
                            ckpt_dir,
                            f"fct_ckpt_M{month:02d}_d{d}_it{it:03d}.npz")
                        np.savez_compressed(
                            path, mu_flat=np.asarray(xk, dtype=np.float64),
                            iters=it, month=month, draw=d,
                            species=np.array(
                                [str(k) for k in flat_keys]),
                            n_cells=N, n_bins=N_BINS, dtau=DTAU,
                            rel_inf_of_b=_rel_inf(b_flat, xk))
                        if d == 0 and verbose:
                            print(f"    [FCT] checkpoint at iter {it} "
                                  f"-> {os.path.basename(path)}", flush=True)

                    restart = max(1, ckpt_every)
                    cb, cb_type = _cb_x, "x"
                else:
                    def _cb(pr_norm):
                        pr_hist.append(float(pr_norm))
                        if d == 0 and verbose:
                            print(f"    [FCT] outer GMRES iter {len(pr_hist)}: "
                                  f"rel resid = {pr_norm:.3e}", flush=True)

                    restart = min(FCT_MAX_ITERS, 30)
                    cb, cb_type = _cb, "pr_norm"

                x_sol, info_code = _spla.gmres(
                    op, b_flat, x0=b_flat,
                    rtol=1e-30, atol=0.3 * FCT_REL_TOL * b_norm,
                    restart=restart,
                    maxiter=max(1, -(-FCT_MAX_ITERS // restart)),
                    callback=cb, callback_type=cb_type,
                )
                # pr_hist is empty in checkpoint mode (scipy gives the
                # iterate OR the residual, not both), so count matvecs:
                # n_fct_solves is 1 (the b pass) + one per outer iteration.
                n_outer = len(pr_hist) if pr_hist else max(0, n_fct_solves[0] - 1)
                if info_code != 0 and d == 0:
                    print(f"    WARNING: [FCT] outer GMRES not converged "
                          f"(info={info_code}, iters={n_outer})")
                # Final pass at the solution: λ/R consistent with x_sol for
                # the output, and the honest ∞-norm verdict for the NetCDF.
                g_sol, lam_per_species_d, R_per_species_d = _apply_g(x_sol)
                n_fct_solves[0] += 1
                rel_final = _rel_inf(g_sol, x_sol)
                if d == 0:
                    if verbose:
                        print(f"    [FCT] outer GMRES done: {n_outer} iters, "
                              f"max rel μ residual = {rel_final:.3e}")
                    timings["fct_deferred_iters"] = n_fct_solves[0]
                    timings["fct_converged"] = bool(rel_final < FCT_REL_TOL)
            else:
                # Picard / Anderson diagnostics (Picard DIVERGES on operators
                # with ρ(K) > 1 — keep only for measuring K's spectrum).
                accel = AndersonAccelerator(m=5)
                x = b_flat
                fct_changes: list = []
                for fct_it in range(1, FCT_MAX_ITERS + 1):
                    g, lam_per_species_d, R_per_species_d = _apply_g(x)
                    n_fct_solves[0] += 1
                    rel = _rel_inf(g, x)
                    fct_changes.append(rel)
                    if d == 0 and verbose:
                        print(f"    [FCT] deferred iter {fct_it}: max rel μ "
                              f"change = {rel:.3e}")
                    if rel < FCT_REL_TOL:
                        break
                    if FCT_SOLVER == "anderson":
                        x, _accepted = accel.apply_with_safeguard(x, g)
                    else:
                        x = g
                if d == 0:
                    timings["fct_deferred_iters"] = n_fct_solves[0]
                    timings["fct_converged"] = bool(
                        fct_changes and fct_changes[-1] < FCT_REL_TOL)
        else:
            lam_per_species_d, R_per_species_d = _run_species_passes(
                None, d == 0)

        # ── 3c. Per-draw output cube fill (after Picard convergence) ──
        # Write surface slab of ∂J/∂e for direct user-requested species
        # (including VBS_C* if VOC pseudos are requested). Non-user
        # species in `to_solve` (chemistry-closure only) are skipped.
        for sp_idx_w in lam_per_species_d:
            sp_key_for_idx = next(
                (k for k, v in EMITTED_SPECIES.items() if v == sp_idx_w),
                None,
            )
            if sp_key_for_idx is None or sp_key_for_idx not in out_row_for_key:
                continue
            s_i = out_row_for_key[sp_key_for_idx]
            grad_list = gradient_per_emission_bin(
                lam_per_species_d[sp_idx_w], R_per_species_d[sp_idx_w]
            )
            for tau in range(N_BINS):
                grad_surface = grad_list[tau][:surface_size].reshape(
                    g0.ny, g0.nx
                )
                grad_surface_cube[d, s_i, tau] = grad_surface
    timings["t_adjoint"] = time.time() - t0

    # ── 3a. VOC pseudo-species combination (post per-species solves) ──
    # For each requested VOC class, compute per-cell per-bin yields and
    # combine the 5 VBS surface ∂J/∂e rows. Yields are static (no MC
    # uncertainty propagation on chamber yields in v1), so they're
    # computed once and applied per draw.
    if voc_keys:
        if verbose:
            print(f"  Combining VOC pseudo-species: {voc_keys}")
        # Indices of the 5 VBS bins in species_keys_out (low-to-high C*).
        vbs_row_in_cube = []
        for vbs_key in VBS_BIN_KEYS:
            if vbs_key not in out_row_for_key:
                raise RuntimeError(
                    f"{vbs_key} not in output cube but VOC pseudos requested — "
                    f"check direct_keys expansion."
                )
            vbs_row_in_cube.append(out_row_for_key[vbs_key])
        for voc_key in voc_keys:
            yields = _compute_voc_yields_per_bin(voc_key, grids)  # (N_BINS, 5, ny, nx)
            out_row = out_row_for_key[voc_key]
            for d in range(n_draws_total):
                # voc_dJde[tau, y, x] = Σ_i yields[tau, i, y, x] · vbs_cube[d, i, tau, y, x]
                vbs_cube = np.stack(
                    [grad_surface_cube[d, vbs_row_in_cube[i]] for i in range(5)],
                    axis=1,
                )   # (N_BINS, 5, ny, nx)
                voc_dJde = (yields * vbs_cube).sum(axis=1)   # (N_BINS, ny, nx)
                grad_surface_cube[d, out_row] = voc_dJde.astype(np.float32)
            if verbose:
                # Report draw-0 magnitude at the global-max cell.
                vmax = float(np.abs(grad_surface_cube[0, out_row]).max())
                print(f"    [{voc_key}] max |∂J/∂e| = {vmax:.3e}")

    # ── 4. Write output NetCDF ────────────────────────────────────────
    t0 = time.time()
    # Save surface-layer cell volume in m³ — the aggregator needs it for
    # the µg/m³/s → kg/s conversion. The solver's emissions are in
    # µg/m³/s (per-cell-volume normalised, see orbit/emissions/netcdf.py:247
    # `emis_4d += value / vol`), so ∂J/∂e_τ from the adjoint is in
    # (deaths/yr) per (µg/m³/s). To translate to "deaths per 1000 kg/yr",
    # the aggregator must divide by cell_volume_m3.
    surface_volume_m3 = g0.volume[0].astype(np.float32)  # (ny, nx)

    ds_out = xr.Dataset(
        data_vars=dict(
            dJ_de=(("draw", "species", "bin", "y", "x"), grad_surface_cube),
            S_orbit=(("draw", "y", "x"),
                     S_summed_orbit_per_draw.astype(np.float32)),
            surface_volume_m3=(("y", "x"), surface_volume_m3),
        ),
        coords=dict(
            draw=np.arange(n_draws_total),
            species=np.array(species_keys_out),
            bin=np.arange(N_BINS),
            lat=("y", g0.lat),
            lon=("x", g0.lon),
        ),
        attrs=dict(
            month=month,
            gradient_source=os.path.abspath(gradient_path),
            crf_mode=grad_attrs.get("crf_mode", "unknown"),
            crf_name=grad_attrs.get("crf_name", "unknown"),
            cause_set=grad_attrs.get("cause_set", "unknown"),
            n_draws_total=int(n_draws_total),
            # Whether the FCT deferred correction actually converged. This
            # was previously computed and printed but never persisted, so a
            # consumer of the NetCDF could not tell an unconverged solve
            # from a converged one — and an unconverged one leaves small
            # negative ∂J/∂e for primary aerosol.
            horizontal_fct=int(bool(horizontal_fct)),
            fct_deferred_iters=int(timings.get("fct_deferred_iters", 0)),
            fct_converged=int(bool(timings.get("fct_converged", True))),
            description=(
                "∂J/∂e_(draw, τ, species, cell): marginal change in J for "
                "unit emission of `species` at (bin τ, surface cell), one "
                "per MC draw (draw=0 = deterministic mean; 1.. = MC). "
                "J = annual deaths linearised around the production ORBIT "
                "baseline. Surface-only — the aggregator only uses z=0. "
                "Full K^T chemistry coupling propagates pSO4/TotalNO3/"
                "VBS-aging receptors to SO2/NOx/upstream-VBS sources. "
                "All five emission-policy species are fully correct."
            ),
        ),
    )
    ds_out.dJ_de.attrs.update(
        units="deaths · year-1 / (kg s-1 cell-1)",
        long_name="adjoint emission gradient",
    )
    encoding = {"dJ_de": {"zlib": True, "complevel": 4, "_FillValue": None}}
    ds_out.to_netcdf(out_path, encoding=encoding)
    timings["t_write"] = time.time() - t0

    umfpack_free_symbolic(umfpack_sym)
    if verbose:
        total = sum(v for k, v in timings.items() if k.startswith("t_"))
        print(f"  Adjoint month {month}: {total:.1f}s total → {out_path}")
    return {
        "out_path": out_path,
        "timings": timings,
        "solver_info": {
            key: solver_info_d0[EMITTED_SPECIES[key]]
            for key in species_keys if key in EMITTED_SPECIES
        },
        "n_draws_total": int(n_draws_total),
    }
