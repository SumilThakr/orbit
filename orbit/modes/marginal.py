"""Marginal simulation mode.

Linearised concentration response to an emission perturbation, computed
around a saved forward baseline. Solves the tangent-linear periodic-orbit
system for ``δc`` -- the same monodromy composition as the forward solve,
driven by ``-δe`` -- with the partitioning fraction ``f_marg`` held fixed
at the baseline state. Output is δc, not c.

See ``MODES.md`` (top level) for: when to use marginal vs. zero-out,
the three-tier Jacobian picture, sign conventions, NPZ schema, and
the perturbation YAML format.
"""

from __future__ import annotations

import os
import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from orbit.core.grid_data import load_grid
from orbit.core.indexing import CellIndexer
from orbit.core.operator import assemble_species_operators
from orbit.core.soa_photolysis import load_photolysis_lut, attach_jno2_to_grid, get_a_photo
from orbit.core.orbit import solve_orbit_all_species, N_BINS
from orbit.core.deposition import (
    N_SPECIES,
    IDX_SOA, IDX_PM25, IDX_POA, IDX_TOTAL_NH, IDX_SO2, IDX_NOX,
    IDX_PSO4, IDX_TOTAL_NO3, IDX_VBS_BINS, VBS_AGING_PAIRS, C_STAR_VALS,
)
from orbit.core.solve import (
    umfpack_symbolic, umfpack_free_symbolic, _HAS_UMFPACK,
    compute_metis_ordering, _HAS_METIS,
)
from orbit.emissions.loader import DiurnalConfig
from orbit.emissions.sources import EmissionSource

# Re-export the perturbation spec from its module so existing imports
# (``from orbit.modes.marginal import Perturbation, parse_cli_perturbation``)
# keep working unchanged. The redundant aliases mark these as deliberate
# re-exports (else a linter strips the ones unused within this module).
from orbit.modes.perturbation import (
    Perturbation as Perturbation,
    parse_cli_perturbation as parse_cli_perturbation,
    build_delta_emissions as build_delta_emissions,
    build_perturbed_emissions as build_perturbed_emissions,
)


# ── Solve-skip DAG walker ──────────────────────────────────────────────────


# Mirror of the solver's source DAG: target → list of source species whose
# orbits the target depends on. Must match what K_sources encodes at runtime
# (orbit/core/operator: pSO4←SO2, TotalNO3←NOx, plus the 1D-VBS aging cascade
# C1000→C100→C10→C1→C01). The VBS pairs are derived from the authoritative
# VBS_AGING_PAIRS = ((src, dst), ...) so this stays in sync if the chain
# changes; fragmentation is pure mass-loss to gas (no inter-bin coupling), so
# the aging pairs are the complete VBS dependency set.
_SOURCES_FOR_TARGET: Dict[int, List[int]] = {
    IDX_PSO4: [IDX_SO2],
    IDX_TOTAL_NO3: [IDX_NOX],
}
for _src_bin, _dst_bin in VBS_AGING_PAIRS:        # (src, dst): dst gains from src
    _SOURCES_FOR_TARGET.setdefault(_dst_bin, []).append(_src_bin)


def _species_with_nonzero_delta(delta_solver: np.ndarray, N: int) -> set:
    """Indices of species whose direct δe is non-zero."""
    out = set()
    for s in range(N_SPECIES):
        if delta_solver.ndim == 1:
            slab = delta_solver[s * N:(s + 1) * N]
        else:
            slab = delta_solver[:, s * N:(s + 1) * N]
        if np.any(np.abs(slab) > 0.0):
            out.add(s)
    return out


def _cyclic_iso_species_in_solve_set(solve_species: set) -> set:
    """Return the iso-coupling cyclic species that are also in the solve set.

    The iso DAG ``ISO_SOURCES_FOR_TARGET`` declares one cycle: NH ↔ NO3
    (each lists the other as a cross-partial source). pSO4 also feeds NH
    and NO3 but pSO4 doesn't depend on them, so it's not in the cycle.

    Picard outer iteration is needed only when both members of the cycle
    are in the solve set. For perturbations that touch only one of the
    cyclic species (rare — NH3-only would propagate into NO3 anyway via
    the cross partial, putting both in the solve set), we still seed both
    so the K @ orbit lookup never trips ``KeyError``.
    """
    from orbit.modes.iso_coupling import ISO_SOURCES_FOR_TARGET
    cyclic = set()
    for tgt, srcs in ISO_SOURCES_FOR_TARGET.items():
        for src in srcs:
            if (src in ISO_SOURCES_FOR_TARGET and
                    tgt in ISO_SOURCES_FOR_TARGET[src]):
                cyclic.add(tgt)
                cyclic.add(src)
    return cyclic & solve_species


def _solve_set(direct_nonzero: set, iso_coupling: bool = False) -> set:
    """Closure of direct-perturbed species under the source DAG.

    A species s must be solved iff it has direct δe != 0 OR any of its
    sources is in the solve set (that source's δc != 0 propagates through
    K_sources). Walked iteratively until stable.

    With ``iso_coupling=True`` (Phase 6c), the iso-extended DAG is used
    so an NH3 perturbation propagates into TotalNO3 (via ∂p_NO3/∂c_NH)
    and a SO4/SO2 perturbation propagates into both TotalNH and
    TotalNO3.
    """
    from orbit.modes.iso_coupling import merged_sources_for_target
    dag = merged_sources_for_target(_SOURCES_FOR_TARGET, iso=iso_coupling)
    solve = set(direct_nonzero)
    changed = True
    while changed:
        changed = False
        for tgt, srcs in dag.items():
            if tgt in solve:
                continue
            if any(src in solve for src in srcs):
                solve.add(tgt)
                changed = True
    return solve


# ── Main driver ────────────────────────────────────────────────────────────


def run_marginal_month(
    month: int,
    *,
    baseline_npz_path: str,
    perturbation: Perturbation,
    output_dir: str,
    preproc_path_fn: Callable[[int, int], str],
    constants_path: str,
    baseline_sources: List[EmissionSource],
    diurnal_cfg: Optional[DiurnalConfig] = None,
    krylov_tol: float = 1e-6,
    maxiter: int = 200,
    skip_zero_delta_species: bool = True,
    output_filename: Optional[str] = None,
    iso_coupling: bool = True,
    picard_max_iters: int = 8,
    picard_rel_tol: float = 1e-4,
    horizontal_fct: bool = False,
    verbose: bool = True,
) -> dict:
    """Solve ``L · δc = -δe`` for one month around a saved baseline.

    Parameters
    ----------
    month : int (1-12).
    baseline_npz_path : path to the per-month forward NPZ. Must carry
        f_nh4_marg_3d and f_no3_marg_3d.
    perturbation : Perturbation. If empty, δe = 0 and δc ≈ 0 (used as a
        sanity check by tests).
    output_dir : where to write the marginal NPZ.
    preproc_path_fn : callable(month, bin_idx_1based) → preproc path.
    constants_path : MERRA2 constants file path.
    baseline_sources : the same EmissionSource list the baseline used.
        Required so that --scale-source factors can be applied.
    diurnal_cfg : optional. Same diurnal config as the baseline.
    krylov_tol : GMRES tolerance for the back-solve.
    maxiter : GMRES maxiter.
    skip_zero_delta_species : use the source-DAG to skip species whose
        δe and δc-source-couplings are both zero. Saves ~70% wall on
        single-precursor perturbations.
    output_filename : override for the output file name within output_dir.

    Returns a dict of timings and key diagnostics, or None on failure.
    """
    out_name = output_filename or f"marginal_M{month:02d}.npz"
    out_path = os.path.join(output_dir, out_name)
    os.makedirs(output_dir, exist_ok=True)

    timings: Dict[str, float] = {"month": month}

    # 1. Load baseline NPZ — extract f_marg fields and grid shape.
    if not os.path.exists(baseline_npz_path):
        if verbose:
            print(f"  Missing baseline: {baseline_npz_path}")
        return None
    t0 = time.time()
    baseline = np.load(baseline_npz_path, allow_pickle=False)
    have_nh4_3d = "f_nh4_marg_3d" in baseline.files
    have_no3_3d = "f_no3_marg_3d" in baseline.files

    # Phase 6c: iso-coupled marginal needs the cross-partials.
    iso_cross_keys = ("f_nh_dno3_3d", "f_no3_dnh_3d", "f_nh_dso4_3d", "f_no3_dso4_3d")
    have_iso_cross = all(k in baseline.files for k in iso_cross_keys)
    if iso_coupling and not have_iso_cross:
        missing = [k for k in iso_cross_keys if k not in baseline.files]
        raise ValueError(
            f"iso_coupling=True but baseline {baseline_npz_path} lacks "
            f"the cross-partial fields: {missing}. Either re-run forward "
            f"with --iso-cross-partials (default ON post-Phase-6a-rev), or "
            f"call run_marginal_month with iso_coupling=False (--no-iso-"
            f"coupling) for the diagonal-only diagnostic."
        )

    # 2026-05-01 fix (revised): refuse pre-clipped baselines for iso-
    # coupled marg.
    #   "eq"  : pre-2026-05-01-AM. Wrong formula (∂f_eq/∂c_other).
    #   "marg": 2026-05-01-AM. Right formula but on UNCLIPPED f_marg —
    #           K-block over-predicts cross-coupling at regime-edge
    #           cells (Delhi sulfate-saturated regime: ~5/8 bins).
    #           Drives the empirical 50% marg/zo gap.
    #   "marg_clipped" (current): clipped, symmetric±δ-averaged f_marg
    #           on each c_other ± δ slice. Cross-derivative consistent
    #           with the operator's clipped f_marg.
    if iso_coupling:
        defn_arr = baseline.get("iso_cross_partial_definition")
        defn = str(defn_arr) if defn_arr is not None else "eq"
        if defn != "marg_clipped":
            raise ValueError(
                f"iso_coupling=True but baseline {baseline_npz_path} has "
                f"iso_cross_partial_definition='{defn}' (expected "
                f"'marg_clipped'). Pre-clipped baselines (defn='eq' or "
                f"'marg') used cross-partials inconsistent with the "
                f"operator's clipped f_marg, driving a structural ~50%-"
                f"underprediction at sulfate-saturated populated cells. "
                f"Re-run forward to regenerate the baseline with the "
                f"clipped cross-derivative extraction (forward solver "
                f"and dcomp_isorropia updated 2026-05-01). Or call with "
                f"iso_coupling=False to disable iso-coupling entirely "
                f"(chemistry-DAG only)."
            )
    if have_nh4_3d and have_no3_3d:
        f_nh4_marg = np.asarray(baseline["f_nh4_marg_3d"], dtype=np.float64)
        f_no3_marg = np.asarray(baseline["f_no3_marg_3d"], dtype=np.float64)
    elif "p_nh" in baseline.files and "p_no" in baseline.files:
        # Backward-compat fallback for baselines saved before the
        # per-bin f_*_marg_3d schema fix. Broadcasts the bin-0
        # post-closure marginal partitioning to all 8 bins. Less
        # accurate than per-bin (T/RH varies diurnally) — for paper
        # runs, refresh the baseline by re-running forward mode.
        if verbose:
            print("  WARNING: baseline lacks f_nh4_marg_3d / f_no3_marg_3d; "
                  "broadcasting bin-0 p_nh / p_no across 8 bins. "
                  "Refresh baseline with the updated forward driver "
                  "to get per-bin marginal partitioning.")
        p_nh = np.asarray(baseline["p_nh"], dtype=np.float64)
        p_no = np.asarray(baseline["p_no"], dtype=np.float64)
        f_nh4_marg = np.broadcast_to(p_nh, (N_BINS,) + p_nh.shape).copy()
        f_no3_marg = np.broadcast_to(p_no, (N_BINS,) + p_no.shape).copy()
    else:
        raise ValueError(
            f"Baseline NPZ {baseline_npz_path} lacks f_nh4_marg_3d / "
            f"f_no3_marg_3d AND p_nh / p_no fallbacks. Was it produced "
            f"by a forward run with ISORROPIA closure?"
        )
    if "grid_shape" in baseline.files:
        nz, ny, nx = (int(x) for x in baseline["grid_shape"])
    else:
        nz, ny, nx = f_nh4_marg.shape[-3:]
    if f_nh4_marg.shape != (N_BINS, nz, ny, nx):
        raise ValueError(
            f"f_nh4_marg_3d shape {f_nh4_marg.shape} != ({N_BINS}, {nz}, {ny}, {nx})"
        )

    # Per-bin VBS partitioning (linearisation point for the VBS deposition
    # operator). The forward driver saves F_p_vbs_marg_3d (N_BINS,5,nz,ny,nx)
    # and M_OA_marg_3d (N_BINS,nz,ny,nx) — the exact per-bin baseline state.
    have_vbs_3d = ("F_p_vbs_marg_3d" in baseline.files
                   and "M_OA_marg_3d" in baseline.files)
    F_p_vbs_marg = M_OA_marg = None
    if have_vbs_3d:
        F_p_vbs_marg = np.asarray(baseline["F_p_vbs_marg_3d"], dtype=np.float64)
        M_OA_marg = np.asarray(baseline["M_OA_marg_3d"], dtype=np.float64)
        n_vbs = len(IDX_VBS_BINS)
        if F_p_vbs_marg.shape != (N_BINS, n_vbs, nz, ny, nx):
            raise ValueError(
                f"F_p_vbs_marg_3d shape {F_p_vbs_marg.shape} != "
                f"({N_BINS}, {n_vbs}, {nz}, {ny}, {nx})"
            )

    # Phase 6c: cross-partial fields (already validated above when
    # iso_coupling=True). Always load if present so the validation
    # path can compare iso-coupled vs. diagonal results.
    iso_cross_3d = None
    if have_iso_cross:
        iso_cross_3d = {
            "f_nh_dno3":  np.asarray(baseline["f_nh_dno3_3d"],  dtype=np.float64),
            "f_no3_dnh":  np.asarray(baseline["f_no3_dnh_3d"],  dtype=np.float64),
            "f_nh_dso4":  np.asarray(baseline["f_nh_dso4_3d"],  dtype=np.float64),
            "f_no3_dso4": np.asarray(baseline["f_no3_dso4_3d"], dtype=np.float64),
        }
        for k, arr in iso_cross_3d.items():
            if arr.shape != (N_BINS, nz, ny, nx):
                raise ValueError(
                    f"{k}_3d shape {arr.shape} != ({N_BINS}, {nz}, {ny}, {nx})"
                )
    timings["baseline_load"] = time.time() - t0

    # 2. Load 8 grids, verify shape.
    t0 = time.time()
    bin_paths = [preproc_path_fn(month, b + 1) for b in range(N_BINS)]
    for bp in bin_paths:
        if not os.path.exists(bp):
            if verbose:
                print(f"  Missing: {bp}")
            return None
    grids = [load_grid(bin_paths[tau], constants_path) for tau in range(N_BINS)]
    # Clear-sky j(NO2) for the VBS SoA photolytic sink — must match the forward
    # baseline's operator (deposition.assemble_deposition consumes it), else the
    # linearisation is inconsistent. The sink is OFF in production
    # (ORBIT_VBS_A_PHOTO=0), so the LUT is loaded ONLY when explicitly enabled
    # (A_PHOTO > 0); otherwise neither the file nor the TUV LUT is required.
    _photo_lut = load_photolysis_lut() if get_a_photo() > 0 else None
    try:
        _photo_year = int(os.environ.get("ORBIT_PREPROC_YEAR_TAG", "2022"))
    except ValueError:
        _photo_year = 2022
    if _photo_lut is not None:
        for tau in range(N_BINS):
            attach_jno2_to_grid(grids[tau], _photo_lut, _photo_year, month, tau)
    g0_shape = (grids[0].nz, grids[0].ny, grids[0].nx)
    if g0_shape != (nz, ny, nx):
        raise ValueError(
            f"Preproc grid shape {g0_shape} != baseline grid_shape ({nz}, {ny}, {nx})"
        )
    indexer = CellIndexer(nz, ny, nx)
    N = indexer.N
    timings["grid_load"] = time.time() - t0
    if verbose:
        print(f"  Grid: {nz}x{ny}x{nx} = {N:,} cells, 8 bins ({timings['grid_load']:.1f}s)")

    # 3. Override marginal partitioning per bin from baseline.
    for tau in range(N_BINS):
        grids[tau].NHPartitioning = f_nh4_marg[tau].astype(np.float64)
        grids[tau].NO3Partitioning = f_no3_marg[tau].astype(np.float64)
        # Equilibrium fields are not consumed by the deposition operator,
        # but PM2.5 mass extraction uses them. Set them to f_marg's values
        # too so any post-hoc PM2.5 split (which we don't compute here)
        # is at least self-consistent.
        grids[tau].NHPartitioningEq = f_nh4_marg[tau].astype(np.float64)
        grids[tau].NO3PartitioningEq = f_no3_marg[tau].astype(np.float64)
        # Per-bin VBS partitioning so the linearised VBS deposition operator
        # matches the baseline exactly (the forward updates F_p_vbs per bin).
        if have_vbs_3d:
            grids[tau].F_p_vbs = F_p_vbs_marg[tau]              # (5, nz, ny, nx)
            grids[tau].F_p_vbs_surface = F_p_vbs_marg[tau][:, 0]
            grids[tau].M_OA_3d = M_OA_marg[tau]
            grids[tau].M_OA_surface = M_OA_marg[tau][0]

    if not have_vbs_3d:
        # Fallback for baselines saved before per-bin VBS partitioning: rebuild
        # F_p_vbs from the orbit-mean c_mean (same field for all bins). Less
        # accurate than per-bin (M_OA varies ~10% diurnally) — refresh the
        # baseline with the updated forward driver for an exact VBS marginal.
        if "c_mean" in baseline.files:
            from orbit.core.dcomp_vbs import update_vbs_partitioning
            if verbose:
                print("  WARNING: baseline lacks F_p_vbs_marg_3d; rebuilding "
                      "VBS partitioning from orbit-mean c_mean (broadcast to "
                      "all bins). Refresh baseline for an exact VBS marginal.")
            c_mean_base = np.asarray(baseline["c_mean"], dtype=np.float64)
            for tau in range(N_BINS):
                update_vbs_partitioning(grids[tau], c_mean_base, indexer)
        elif verbose:
            print("  WARNING: baseline lacks F_p_vbs_marg_3d and c_mean; VBS "
                  "deposition will use operator fallback partitioning.")

    # 4. UMFPACK / METIS symbolic on a sample operator.
    t0 = time.time()
    from orbit.core.operator import assemble_transport_block
    from orbit.core.deposition import assemble_deposition as _assdep
    T_sample = assemble_transport_block(grids[0], indexer, scheme="exp")
    L_sample = T_sample + _assdep(grids[0], indexer, 0)
    umfpack_sym = None
    perm = None
    if _HAS_UMFPACK:
        umfpack_sym = umfpack_symbolic(L_sample, verbose=False)
    elif _HAS_METIS:
        perm = compute_metis_ordering(L_sample, verbose=False)
    timings["symbolic"] = time.time() - t0

    # 5. Build δe.
    t0 = time.time()
    delta_e_SN = build_delta_emissions(
        perturbation, baseline_sources, grids[0], indexer,
        month=month, n_bins=N_BINS, diurnal_cfg=diurnal_cfg,
        verbose=verbose,
    )
    timings["build_delta_e"] = time.time() - t0
    if verbose:
        delta_total = float(np.abs(delta_e_SN).sum())
        print(f"  δe: max|δe|={float(np.abs(delta_e_SN).max()):.3e}, "
              f"|δe|_1={delta_total:.3e}, shape={delta_e_SN.shape} "
              f"({timings['build_delta_e']:.1f}s)")

    # 6. Identify species we actually need to solve. Skip species with
    # zero δe and no upstream-perturbed source.
    direct_nonzero = _species_with_nonzero_delta(delta_e_SN, N)
    solve_species = (
        _solve_set(direct_nonzero, iso_coupling=iso_coupling)
        if skip_zero_delta_species
        else set(range(N_SPECIES))
    )
    skip_species = set(range(N_SPECIES)) - solve_species

    # Build zero cached_orbits for skipped species.
    if skip_species:
        zero_orbit = [np.zeros(N, dtype=np.float64) for _ in range(N_BINS + 1)]
        cached_orbits = {s: [c.copy() for c in zero_orbit] for s in skip_species}
    else:
        cached_orbits = None

    if verbose:
        from orbit.core.orbit import _SPECIES_NAMES
        solve_names = sorted(solve_species)
        print(f"  Solving {len(solve_species)}/{N_SPECIES} species: "
              f"{[_SPECIES_NAMES[s] for s in solve_names]}")

    # 7. Assemble per-bin operators (baseline prescribed-chemistry path —
    # archive oxidants drive K_sources unchanged from the forward run).
    t0 = time.time()
    L_species_per_bin = []
    K_sources_per_bin = []
    for tau in range(N_BINS):
        L_species, K_sources, _T, _d = assemble_species_operators(
            grids[tau], indexer, verbose=False, scheme="exp",
        )
        L_species_per_bin.append(L_species)
        K_sources_per_bin.append(K_sources)

    # Phase 4 (FCT): add the frozen-coefficient anti-diffusive operator L_AD to
    # each transported species' operator, so the marginal δc reflects the same
    # ~2nd-order horizontal transport as the FCT forward baseline. The limiter
    # (van Leer φ + Zalesak C) is FROZEN at the baseline orbit field for that
    # species/bin, making L_AD a fixed linear 5-point operator (same sparsity as
    # L_low). baseline_npz_path must be the FCT forward orbit. See
    # orbit.core.fct.assemble_fct_linear_operator.
    # FCT tangent-linear is applied by DEFERRED CORRECTION, not by adding L_AD to
    # the LHS operator: the frozen L_low+L_AD is non-M-matrix and its periodic
    # GMRES is ~80× slower / effectively non-convergent (diag_fct_marginal_gmres,
    # job 10254401: L_low 18 iters/11s vs L_low+L_AD >16min). Instead keep L_low
    # as the factored LHS and iterate the anti-diffusive term −L_AD·δc as a RHS
    # source (same structure as the forward, which converges in ~6 iters). Here
    # we just assemble L_AD per (bin, species), frozen at the FCT baseline orbit;
    # the deferred loop below injects it.
    fct_L_AD_per_bin: list[dict] = [dict() for _ in range(N_BINS)]
    if horizontal_fct:
        from orbit.core.fct import assemble_fct_linear_operator
        from orbit.core.orbit import DTAU as _DTAU
        if "c_orbit" not in baseline.files:
            raise RuntimeError(
                "horizontal_fct marginal needs the baseline orbit field "
                "(c_orbit) in the baseline NPZ; point --baseline at an FCT "
                "forward orbit.")
        c_orbit_base = np.asarray(baseline["c_orbit"], dtype=np.float64)  # (S,9,N)
        order = (list(np.asarray(baseline["orbit_species_order"]).ravel())
                 if "orbit_species_order" in baseline.files
                 else list(range(c_orbit_base.shape[0])))
        row_for_species = {int(sp): i for i, sp in enumerate(order)}
        for tau in range(N_BINS):
            for s in solve_species:
                if s not in row_for_species:
                    continue
                c_bs = c_orbit_base[row_for_species[s], tau + 1]  # end-of-bin (N,)
                fct_L_AD_per_bin[tau][s] = assemble_fct_linear_operator(
                    grids[tau], indexer, c_bs, _DTAU)
        if verbose:
            print(f"  [FCT] frozen L_AD assembled for {len(solve_species)} species "
                  f"× {N_BINS} bins; applied by deferred correction (marginal)")

    # Phase 6c: iso-coupled cross-blocks slot into the same
    # K_sources_per_bin dict that the orbit code already iterates over
    # (``e_target -= sum K[(target, source)] @ c_source``). The 4 new
    # blocks per bin add the ISORROPIA NH4↔NO3↔SO4 cross-couplings
    # linearly. Off by default for the diagnostic --no-iso-coupling flow.
    # Iso K-block keys flagged for the end-of-bin time-level convention
    # in the orbit solver. Iso K-blocks are the linearisation of an
    # implicit operator coefficient; their c_baseline factor is built at
    # end-of-bin τ (see dcomp_isorropia._concentration_for_bin), so the
    # corresponding δc_other forcing must also read end-of-bin (τ+1) to
    # keep time-levels matched within the linearised term. Chemistry-DAG
    # K-blocks (SO2 → pSO4, NOx → TotalNO3) use start-of-bin per the
    # forward solver's convention and are not added here.
    endbin_keys: set = set()
    if iso_coupling:
        from orbit.modes.iso_coupling import assemble_iso_cross_blocks
        for tau in range(N_BINS):
            iso_blocks = assemble_iso_cross_blocks(
                grids[tau], indexer,
                f_nh_dno3=iso_cross_3d["f_nh_dno3"][tau],
                f_no3_dnh=iso_cross_3d["f_no3_dnh"][tau],
                f_nh_dso4=iso_cross_3d["f_nh_dso4"][tau],
                f_no3_dso4=iso_cross_3d["f_no3_dso4"][tau],
            )
            for key, K in iso_blocks.items():
                # Sum into existing block if (target, source) is already
                # present (e.g. nominal NOx → TotalNO3 chemistry coupling
                # combined with the iso-mediated NH3 → TotalNO3 path
                # would target the same (target, source) pair only if
                # the source happened to overlap; in practice keys are
                # disjoint between the two layers, so this is just
                # defensive).
                if key in K_sources_per_bin[tau]:
                    K_sources_per_bin[tau][key] = K_sources_per_bin[tau][key] + K
                else:
                    K_sources_per_bin[tau][key] = K
                endbin_keys.add(key)
    timings["assemble"] = time.time() - t0

    # Skip-solve DAG drift assertion (deferred-backlog, fixed for
    # tuple-keyed K_sources). _SOURCES_FOR_TARGET (and its iso extension)
    # together must cover every TARGET in the operator's K_sources keys.
    # A target appearing in K_sources but missing from the DAG would
    # silently fall outside the solve set when its source perturbs.
    from orbit.modes.iso_coupling import merged_sources_for_target
    actual_targets = {t for (t, _src) in K_sources_per_bin[0].keys()}
    declared_targets = set(merged_sources_for_target(
        _SOURCES_FOR_TARGET, iso=iso_coupling,
    ).keys())
    extra_in_operator = actual_targets - declared_targets
    if extra_in_operator:
        raise RuntimeError(
            f"K_sources contains target species {sorted(extra_in_operator)} "
            f"that aren't in the marginal-mode source DAG (iso_coupling="
            f"{iso_coupling}). The DAG must be extended whenever a new "
            f"chemistry / iso coupling lands in the operator, or marginal "
            f"will silently miss the new downstream response."
        )
    if verbose:
        print(f"  Operators assembled ({timings['assemble']:.1f}s)")

    # 8. Solve. extra_rhs_per_bin_per_species MUST be None — δBC = 0.
    #
    # Picard outer iteration for the iso-coupled cyclic DAG (NH↔NO3 via
    # ∂p_NH/∂c_NO3 + ∂p_NO3/∂c_NH cross-partials). The one-pass orbit
    # solver puts both NH and NO3 in the same wave because they
    # mutually depend, and `_solve_one_species` raises KeyError if
    # `orbits[src]` is missing for a cyclic source. The Picard driver
    # seeds iter 0 with zeros for the cyclic species, then re-solves
    # with iter k's orbits as the seed for iter k+1, until the max
    # relative change in cyclic species drops below `picard_rel_tol`.
    #
    # Convergence: for linear-regime perturbations (NH3 −1%, etc.) the
    # cyclic NH↔NO3 coupling is small relative to the diagonal
    # response, so Picard converges in 2-3 iters. For larger
    # perturbations (NH3 −50% scaled from the linear marginal) the
    # operator is unchanged (linearised at baseline) so convergence
    # behaviour matches the −1% case — the Picard iteration always
    # operates on the linearised system, not the perturbation strength.
    #
    # Diagonal-only (iso_coupling=False): one solve, no iteration. The
    # K_sources dict has no iso entries, so there are no cyclic
    # dependencies and seed_orbits is unused.
    t0 = time.time()
    if iso_coupling:
        cyclic_species = _cyclic_iso_species_in_solve_set(solve_species)
    else:
        cyclic_species = set()

    # Shared LU cache across all solve passes (L_low is fixed → factor once).
    _lu_holder: Dict[str, Optional[Dict[int, list]]] = {"lu": None}

    def _solve_pass(extra_rhs_fct):
        """One marginal solve (iso-cyclic Picard handled internally) with an
        optional FCT deferred-correction RHS. Reuses the shared LU cache."""
        if cyclic_species:
            seed_orbits = {
                s: [np.zeros(N, dtype=np.float64) for _ in range(N_BINS + 1)]
                for s in cyclic_species
            }
            res = None
            for picard_it in range(picard_max_iters):
                res = solve_orbit_all_species(
                    L_species_per_bin, K_sources_per_bin, delta_e_SN, N,
                    umfpack_sym=umfpack_sym, perm=perm,
                    tol=krylov_tol, maxiter=maxiter,
                    verbose=(verbose and picard_it == 0),
                    c_warm_SN=None,
                    skip_solve_species=skip_species if skip_species else None,
                    cached_orbits=cached_orbits,
                    extra_rhs_per_bin_per_species=extra_rhs_fct,
                    seed_orbits=seed_orbits,
                    lu_cache=_lu_holder["lu"], return_lu_cache=True,
                    endbin_keys=endbin_keys if endbin_keys else None,
                )
                _lu_holder["lu"] = res.get("lu_cache")
                new_orbits = res["orbits"]
                max_rel = 0.0
                for s in cyclic_species:
                    prev = np.stack(seed_orbits[s][:N_BINS], axis=0)
                    curr = np.stack(new_orbits[s][:N_BINS], axis=0)
                    denom = float(np.abs(curr).max())
                    rel = (float(np.abs(curr - prev).max()) if denom < 1e-30
                           else float(np.abs(curr - prev).max() / denom))
                    max_rel = max(max_rel, rel)
                if picard_it > 0 and max_rel < picard_rel_tol:
                    break
                seed_orbits = {
                    s: [c.copy() for c in new_orbits[s]] for s in cyclic_species
                }
            return res
        res = solve_orbit_all_species(
            L_species_per_bin, K_sources_per_bin, delta_e_SN, N,
            umfpack_sym=umfpack_sym, perm=perm,
            tol=krylov_tol, maxiter=maxiter, verbose=verbose, c_warm_SN=None,
            skip_solve_species=skip_species if skip_species else None,
            cached_orbits=cached_orbits,
            extra_rhs_per_bin_per_species=extra_rhs_fct,
            lu_cache=_lu_holder["lu"], return_lu_cache=True,
            endbin_keys=endbin_keys if endbin_keys else None,
        )
        _lu_holder["lu"] = res.get("lu_cache")
        return res

    def _fct_rhs(prev_orbits):
        """extra_rhs[s][τ] = −L_AD[τ][s] · δc_prev[s][τ+1] (end-of-bin)."""
        er: Dict[int, list] = {}
        for s in solve_species:
            er[s] = [(-(fct_L_AD_per_bin[tau][s] @ prev_orbits[s][tau + 1])
                      if s in fct_L_AD_per_bin[tau]
                      else np.zeros(N, dtype=np.float64))
                     for tau in range(N_BINS)]
        return er

    if horizontal_fct:
        # Deferred correction: keep L_low factored (fast solve), iterate
        # −L_AD·δc as an extra RHS. δc satisfies the AFFINE fixed point
        # δc = K δc + b where b is the low-order solution and each
        # application of K costs one inner solve.
        #
        # K is NOT a contraction: on the January 2022 production baseline
        # the adjoint's K^T has a measured real eigenvalue ≈ +1.33 (Picard
        # residual grows at ratio 1.33/iter and saturates at (λ−1)/λ =
        # 0.2475, run C 2026-08-03), and this forward K is its transpose so
        # it shares the spectrum. Plain Picard diverges and Anderson(5)
        # only holds the residual at ~1e-2. The fixed point still exists
        # and is unique (1 is not an eigenvalue), so the default solver is
        # outer GMRES on (I−K)δc = b, mirroring adjoint.py.
        # ORBIT_MARGINAL_FCT_SOLVER selects gmres|anderson|picard (the
        # last two retained as diagnostics; Picard's residual ratios read
        # off K's dominant eigenvalue directly).
        from orbit.core.dcomp_iter import AndersonAccelerator
        FCT_SOLVER = os.environ.get("ORBIT_MARGINAL_FCT_SOLVER", "gmres")
        FCT_MAX_ITERS = int(os.environ.get(
            "ORBIT_MARGINAL_FCT_MAX_ITERS",
            "40" if FCT_SOLVER == "gmres" else "12"))
        FCT_REL_TOL = float(os.environ.get("ORBIT_MARGINAL_FCT_REL_TOL", "2.0e-3"))
        flat_keys = sorted(solve_species)

        def _flat(orbits):
            return np.concatenate([np.concatenate(orbits[s]) for s in flat_keys])

        def _unflat(vec):
            out, i = {}, 0
            for s in flat_keys:
                lst = []
                for _ in range(N_BINS + 1):
                    lst.append(vec[i:i + N]); i += N
                out[s] = lst
            return out

        def _rel_inf(g_flat, x_flat):
            denom = float(np.abs(g_flat).max())
            return (float(np.abs(g_flat - x_flat).max())
                    / (denom if denom > 1e-30 else 1.0))

        result = _solve_pass(None)                       # b: low-order δc
        b_flat = _flat(result["orbits"])
        n_fct_solves = [1]                               # the b pass above

        def _apply_g(vec):
            """One application of the affine map g(x) = K x + b: build the
            FCT RHS from x, run one inner solve. Returns (g_flat, result)
            so the accepted solution's orbits/diagnostics can be reused."""
            g_result = _solve_pass(_fct_rhs(_unflat(vec)))
            n_fct_solves[0] += 1
            return _flat(g_result["orbits"]), g_result

        if FCT_SOLVER == "gmres":
            import scipy.sparse.linalg as _spla

            def _outer_matvec(vec):
                # (I − K) v, with K v = g(v) − b (g affine).
                g_v, _ = _apply_g(vec)
                return vec - (g_v - b_flat)

            op = _spla.LinearOperator(
                (b_flat.size, b_flat.size), matvec=_outer_matvec,
                dtype=np.float64)
            b_norm = float(np.linalg.norm(b_flat))
            pr_hist: List[float] = []

            def _cb(pr_norm):
                pr_hist.append(float(pr_norm))
                if verbose:
                    print(f"  [FCT] outer GMRES iter {len(pr_hist)}: "
                          f"rel resid = {pr_norm:.3e}")

            restart = min(FCT_MAX_ITERS, 30)
            x_sol, info_code = _spla.gmres(
                op, b_flat, x0=b_flat,
                rtol=1e-30, atol=0.3 * FCT_REL_TOL * b_norm,
                restart=restart,
                maxiter=max(1, -(-FCT_MAX_ITERS // restart)),
                callback=_cb, callback_type="pr_norm",
            )
            if info_code != 0:
                print(f"  WARNING: [FCT] outer GMRES not converged "
                      f"(info={info_code}, iters={len(pr_hist)})")
            # Final pass at the solution: orbits consistent with x_sol for
            # the output, and the honest ∞-norm verdict for the NPZ.
            g_sol, result = _apply_g(x_sol)
            rel_final = _rel_inf(g_sol, x_sol)
            if verbose:
                print(f"  [FCT] outer GMRES done: {len(pr_hist)} iters, "
                      f"max rel δc residual = {rel_final:.3e}")
            timings["fct_deferred_iters"] = n_fct_solves[0]
            timings["fct_converged"] = bool(rel_final < FCT_REL_TOL)
        else:
            # Picard / Anderson diagnostics (Picard DIVERGES on operators
            # with ρ(K) > 1 — keep only for measuring K's spectrum).
            accel = AndersonAccelerator(m=5)
            x = b_flat
            fct_changes: List[float] = []
            for fct_it in range(1, FCT_MAX_ITERS + 1):
                g, result = _apply_g(x)
                rel = _rel_inf(g, x)
                fct_changes.append(rel)
                if verbose:
                    print(f"  [FCT] deferred iter {fct_it}: max rel δc change "
                          f"= {rel:.3e}")
                if rel < FCT_REL_TOL:
                    break
                if FCT_SOLVER == "anderson":
                    x, _accepted = accel.apply_with_safeguard(x, g)
                else:
                    x = g
            timings["fct_deferred_iters"] = n_fct_solves[0]
            timings["fct_converged"] = bool(
                fct_changes and fct_changes[-1] < FCT_REL_TOL)
    else:
        result = _solve_pass(None)
    timings["solve"] = time.time() - t0

    # 9. Stack δc orbits and compute orbit-mean δc.
    orbits = result["orbits"]
    delta_c_orbit = np.stack(
        [np.stack(orbits[s], axis=0) for s in sorted(orbits.keys())], axis=0,
    )  # (N_SPECIES, 9, N)
    delta_c_mean = delta_c_orbit[:, :N_BINS, :].mean(axis=1)  # (N_SPECIES, N)

    # 10. Compute δPM2.5 with baseline f_marg fields (linearised). Pass the
    # baseline per-bin VBS bin masses (from c_orbit) so the δSoA carries the
    # M_OA→F_p partitioning-feedback factor 1/D.
    baseline_vbs_c = None
    if "c_orbit" in baseline.files:
        baseline_vbs_c = baseline_vbs_from_c_orbit(baseline["c_orbit"], nz, ny, nx)
    delta_pm25_orbit, delta_pm25_mean = _compute_delta_pm25(
        delta_c_orbit, grids, nz, ny, nx, N, baseline_vbs_c=baseline_vbs_c,
    )

    # 11. Save δNPZ.
    t0 = time.time()
    save_dict = {
        "mode": np.array("marginal"),
        "month": np.array(month),
        "baseline_npz_path": np.array(os.path.abspath(baseline_npz_path)),
        "perturbation_name": np.array(perturbation.name),
        "perturbation_description": np.array(perturbation.description),
        # Phase 4c: sign convention. marginal solves L·δc = -δe, so a
        # downward perturbation (factor=0.99) yields δc < 0. zero-out
        # reports c_full - c_perturbed, which is +ve for the same case.
        # See orbit.modes.perturbation.unify_sign for cross-mode flips.
        "sign_convention": np.array("perturbation_response"),
        "iso_coupling": np.array(bool(iso_coupling)),
        "picard_iters": np.array(int(timings.get("picard_iters", 1))),
        "picard_converged": np.array(bool(timings.get("picard_converged", True))),
        "picard_final_change": np.array(float(timings.get("picard_final_change", 0.0))),
        "picard_rel_tol": np.array(float(picard_rel_tol)),
        "cyclic_species": np.array(sorted(cyclic_species), dtype=np.int64),
        "delta_c_orbit": delta_c_orbit.astype(np.float64),
        "delta_c_mean": delta_c_mean.astype(np.float64),
        "delta_pm25_orbit": delta_pm25_orbit.astype(np.float64),
        "delta_pm25_mean": delta_pm25_mean.astype(np.float64),
        "orbit_species_order": np.arange(N_SPECIES),
        "solved_species": np.array(sorted(solve_species), dtype=np.int64),
        "gmres_iters": result["gmres_iters"],
        "gmres_resid": result["gmres_resid"],
        "periodicity": result["periodicity"],
        "lon": grids[0].lon,
        "lat": grids[0].lat,
        "grid_shape": np.array([nz, ny, nx]),
    }
    if perturbation.factors:
        save_dict["perturbation_factors_keys"] = np.array(list(perturbation.factors.keys()))
        save_dict["perturbation_factors_values"] = np.array(list(perturbation.factors.values()))
    if perturbation.add_sources:
        save_dict["perturbation_add_paths"] = np.array(
            [os.path.abspath(s.path) for s in perturbation.add_sources]
        )
    np.savez_compressed(out_path, **save_dict)
    timings["save"] = time.time() - t0
    if verbose:
        out_mb = os.path.getsize(out_path) / 1024 / 1024
        print(f"  Saved: {out_path} ({out_mb:.1f} MB, {timings['save']:.1f}s)")

    # Free UMFPACK symbolic.
    if umfpack_sym is not None:
        umfpack_free_symbolic(umfpack_sym)

    timings["delta_c_max_abs"] = float(np.abs(delta_c_orbit).max())
    timings["delta_pm25_max_abs"] = float(np.abs(delta_pm25_orbit).max())
    return timings


def baseline_vbs_from_c_orbit(c_orbit, nz, ny, nx):
    """Per-bin baseline VBS bin masses for the Pankow feedback factor 1/D.

    ``c_orbit`` is the forward output's concentration along the periodic
    orbit, shape (N_SPECIES, N_BINS + 1, N); entry ``tau + 1`` is the state at
    the end of bin ``tau``, which is what the marginal solve pairs with the
    bin-``tau`` operator. Returns (N_BINS, 5, nz, ny, nx) in ``IDX_VBS_BINS``
    order, clipped at zero. Marginal mode (``_compute_delta_pm25``) and
    adjoint mode (``growth_jacobian.apply_growth_transpose``) must build D
    from the same field, or the adjoint-versus-marginal duality breaks.
    """
    c = np.asarray(c_orbit, dtype=np.float64)
    out = np.empty((N_BINS, len(IDX_VBS_BINS), nz, ny, nx), dtype=np.float64)
    for tau in range(N_BINS):
        for k, s in enumerate(IDX_VBS_BINS):
            out[tau, k] = np.maximum(c[s, tau + 1].reshape(nz, ny, nx), 0.0)
    return out


def _compute_delta_pm25(
    delta_c_orbit: np.ndarray,  # (N_SPECIES, 9, N)
    grids,
    nz: int, ny: int, nx: int, N: int,
    baseline_vbs_c: np.ndarray = None,  # (N_BINS, 5, nz, ny, nx), IDX_VBS_BINS order
) -> Tuple[np.ndarray, np.ndarray]:
    """Linearised δPM2.5 from δc using the baseline f_marg fields.

    Uses *_marg (not *_eq) to match the deposition operator's linearisation.
    SoA is the 1D-VBS particle mass: total SoA = Σ_i F_p,i × C_i over the 5
    volatility bins. Linearising INCLUDING the M_OA→F_p partitioning feedback
    (Pankow) gives the closed form

        δSoA = (Σ_i F_p,i · δC_i) / D ,   D = 1 − Σ_k C_k·C*_k/(M_OA+C*_k)² ,

    where F_p, M_OA and the baseline bin masses C_k are the per-bin baseline
    state. The 1/D factor is the partitioning-feedback amplification (adding
    SoA mass raises M_OA, which raises F_p); omitting it leaves a ~14%
    first-order error (validated: 1/D closes marg vs exact to ~0.2%). Falls
    back to F_p,i·δC_i (no feedback) when M_OA / baseline C_k are unavailable.
    p_nh and p_no3 come from the baseline marginal partitioning fields.
    """
    from orbit.core.constants import N_TO_NH4, N_TO_NO3, S_TO_SO4

    delta_pm25_orbit = np.zeros((N_BINS, nz, ny, nx), dtype=np.float64)
    C_star = np.asarray(C_STAR_VALS, dtype=np.float64)

    for tau in range(N_BINS):
        g = grids[tau]
        p_nh = g.NHPartitioning.ravel()
        p_no3 = g.NO3Partitioning.ravel()

        # δSoA = Σ_i F_p,i · δC_i over the 5 VBS bins (IDX_VBS_BINS order),
        # divided by the partitioning-feedback factor D.
        F_p = getattr(g, "F_p_vbs", None)            # (5, nz, ny, nx)
        M_OA = getattr(g, "M_OA_3d", None)           # (nz, ny, nx)
        # 1/D is the Pankow partitioning-feedback amplification. It applies
        # to POA as well as to the VBS bins (see the d_poa comment below);
        # 1.0 means "no baseline VBS state available, no feedback".
        inv_D = 1.0
        if F_p is not None and F_p.shape[0] == len(IDX_VBS_BINS):
            d_soa = sum(
                F_p[i].ravel() * delta_c_orbit[IDX_VBS_BINS[i], tau + 1]
                for i in range(len(IDX_VBS_BINS))
            )
            if M_OA is not None and baseline_vbs_c is not None:
                M_safe = np.maximum(M_OA.ravel(), 1.0e-6)
                D = np.ones_like(M_safe)
                for k in range(len(IDX_VBS_BINS)):
                    C_k = baseline_vbs_c[tau, k].ravel()
                    D -= C_k * C_star[k] / (M_safe + C_star[k]) ** 2
                # D ∈ (0, 1]; clip to keep 1/D bounded in pathological cells.
                inv_D = 1.0 / np.clip(D, 0.1, 1.0)
                d_soa = d_soa * inv_D
        else:
            # Fallback for baselines without per-bin F_p_vbs: C100 bin only
            # (legacy single-tracer SoA). Under-counts the other 4 bins.
            d_soa = delta_c_orbit[IDX_SOA, tau + 1]
        d_pm = delta_c_orbit[IDX_PM25, tau + 1]
        d_nh = delta_c_orbit[IDX_TOTAL_NH, tau + 1]
        d_pso4 = delta_c_orbit[IDX_PSO4, tau + 1]
        d_no3 = delta_c_orbit[IDX_TOTAL_NO3, tau + 1]

        # POA is primary and non-volatile in TRANSPORT, but it is not inert
        # THERMODYNAMICALLY: it sits in M_OA = C_POA + Σ F_p,i C_i, so adding
        # POA raises the absorbing mass, raises F_p, and pulls semi-volatiles
        # into the particle phase. Differentiating the M_OA fixed point,
        #
        #     dM_OA/dC_POA · (1 − Σ_k C_k C*_k/(M_OA+C*_k)²) = 1
        #     ⇒ dM_OA/dC_POA = 1/D
        #
        # and since the organic part of δPM2.5 is δ(POA + SoA) = δM_OA, POA
        # carries the SAME 1/D amplification the VBS bins do. Writing
        # d_poa·(1/D) + d_soa gives the total organic response
        # (δC_POA + Σ F_p,i δC_i)/D, which is symmetric and closes exactly.
        #
        # Measured on the January 2022 baseline, omitting 1/D understated POA
        # damages by ~4.5% at POA top-decile cells (~4.8% POA-mass-weighted,
        # ~15% at p90). Before the 2026-08-03 split POA was inside PrimaryPM25
        # and was counted automatically; the split first dropped it entirely
        # (fixed in 0f25708), and this restores its partitioning feedback.
        #
        # growth_jacobian.apply_growth_transpose MUST apply the same 1/D to
        # its POA receptor — an asymmetry there breaks adjoint/marginal
        # duality rather than merely losing accuracy. Guarded so pre-split
        # 13-species baselines still load.
        d_poa = (delta_c_orbit[IDX_POA, tau + 1] * inv_D
                 if delta_c_orbit.shape[0] > IDX_POA else 0.0)

        d_pm25 = (
            d_pm
            + d_poa
            + d_soa
            + p_nh * d_nh * N_TO_NH4
            + d_pso4 * S_TO_SO4
            + p_no3 * d_no3 * N_TO_NO3
        )
        delta_pm25_orbit[tau] = d_pm25.reshape(nz, ny, nx)

    delta_pm25_mean = delta_pm25_orbit.mean(axis=0)
    return delta_pm25_orbit, delta_pm25_mean
