"""DCOMP chemistry closure: per-bin oxidants + operator-assembly rates.

Bridges the orbit solve to the cell-local oxidants module.  For each of the
8 UTC 3h bins, this module:
  1. Gathers NOx, SO2, PM2.5 (element/compound mass ug/m3) from the latest
     orbit solution, plus HEMCO background O3/CO/CH4/HNO2, plus speciated
     VOCs and meteorology from the bin grid.
  2. Calls `diagnose_oxidants_per_bin` to get OH, HO2, RO2, PSS NO/NO2,
     NO3, N2O5, and N2O5 hydrolysis rate.
  3. Calls `build_chemistry_rates` to convert oxidants into SO2->pSO4 gas
     and NOx->TotalNO3 first-order rates.

The resulting per-bin (k_so2_rate, k_nox_rate) pairs are fed to
`assemble_species_operators` which bakes them into the per-bin transport-
and-chemistry matrices.  The orbit solver sees only those matrices; no
changes to the GMRES core are needed.

This module is the *workhorse* for Phase 3c (single-pass chemistry) and
Phase 3d (outer iteration with ISORROPIA and S_a closure).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional, List

import numpy as np

from orbit.core.grid_data import GridData
from orbit.core.deposition import (
    IDX_SOA, IDX_PM25, IDX_TOTAL_NH, IDX_SO2, IDX_NOX,
    IDX_PSO4, IDX_TOTAL_NO3, IDX_O3, IDX_CO,
)
from orbit.core.oxidants import (
    OxidantFields, diagnose_oxidants_per_bin, build_chemistry_rates,
    build_o3_rates, build_co_loss_rate,
)
from orbit.core.photolysis import PhotolysisLUT
from orbit.core.constants import N_TO_NH4, N_TO_NO3, S_TO_SO4


# ORBIT bin centres (UTC minutes past midnight)
_BIN_CENTER_MIN = (90, 270, 450, 630, 810, 990, 1170, 1350)


@dataclass
class ChemistryPerBin:
    """Per-bin chemistry closure fields returned for downstream use."""
    oxidants: List[OxidantFields]            # length N_BINS
    k_so2_rate: List[np.ndarray]             # (nz, ny, nx) s^-1 per bin
    k_nox_to_no3_rate: List[np.ndarray]      # (nz, ny, nx) s^-1 per bin
    jNO2: np.ndarray                         # (N_BINS, ny, nx) s^-1
    jO1D: np.ndarray                         # (N_BINS, ny, nx)
    jNO3: np.ndarray                         # (N_BINS, ny, nx)
    jHONO: np.ndarray                        # (N_BINS, ny, nx)
    jHCHO: Optional[np.ndarray] = None       # (N_BINS, ny, nx)  Phase 3f
    jH2O2: Optional[np.ndarray] = None       # (N_BINS, ny, nx)
    # Phase 3e: O3 chemistry rates for operator assembly + RHS.
    k_o3_loss_rate: Optional[List[np.ndarray]] = None    # (nz, ny, nx) s^-1 per bin
    s_o3_source_rate: Optional[List[np.ndarray]] = None  # (nz, ny, nx) ug/m3/s per bin
    # CO chemistry rate (diagonal loss only; no source species).
    k_co_loss_rate: Optional[List[np.ndarray]] = None    # (nz, ny, nx) s^-1 per bin
    # Context retained for Opt 1's `rebuild_rates_from_chem` so that
    # Anderson acceleration can modify `oxidants[tau].OH` and then
    # re-derive the operator rates without redoing the expensive
    # inner OH fixed-point via diagnose_oxidants_per_bin.
    c_o3_per_bin: Optional[List[np.ndarray]] = None
    o3_bc_target_3d: Optional[np.ndarray] = None
    o3_bc_rate_3d: Optional[np.ndarray] = None
    jNO3_for_branch: bool = True


def _voc_dict_from_grid(grid: GridData) -> Dict[str, np.ndarray]:
    """Collect speciated VOC fields off a GridData into the name->array dict
    that `diagnose_oxidants_per_bin` expects.  Missing species are skipped.
    """
    out = {}
    for name in ("ISOP", "MTPA", "MTPO", "LIMO", "BENZ", "TOLU", "XYLE", "NAP"):
        arr = getattr(grid, name, np.array([]))
        if arr.size > 0:
            out[name] = arr
    return out


def _broadcast_to_3d(field_2d_or_3d, nz):
    """If already 3D, return as-is; if 2D (ny, nx), broadcast to (nz, ny, nx)."""
    if field_2d_or_3d.ndim == 3:
        return field_2d_or_3d
    return np.broadcast_to(field_2d_or_3d[None, :, :], (nz, *field_2d_or_3d.shape)).copy()


def _hemco_to_grid(hemco_species_ugm3: Dict[str, np.ndarray],
                    species: str, grid: GridData) -> np.ndarray:
    """Fetch a HEMCO species on the ORBIT horizontal grid, broadcast vertically.

    HEMCO arrives as (nz_hemco, ny, nx) on the target horizontal grid but the
    native 72-layer GC vertical. For Phase 3c we take the surface level only
    and broadcast it across all 15 ORBIT layers — surface-level chemistry
    dominates the OH/HO2 budget, and the error aloft is small relative to
    the overall OH/HO2 uncertainty. A proper pressure remap is an obvious
    later refinement.

    Missing species -> returns a zero array.
    """
    if species not in hemco_species_ugm3:
        return np.zeros((grid.nz, grid.ny, grid.nx), dtype=np.float64)
    arr = hemco_species_ugm3[species]
    if arr.ndim == 3:
        # (nz_hemco, ny, nx) -> take surface, broadcast
        surface = arr[0]
    else:
        surface = arr
    return np.broadcast_to(surface[None, :, :],
                           (grid.nz, grid.ny, grid.nx)).copy()


def _bin_center_utc(year, month, day, bin_idx):
    """UTC datetime at the centre of bin `bin_idx` (0..7) of a given day."""
    base = datetime(year, month, day, tzinfo=timezone.utc)
    return base + timedelta(minutes=int(_BIN_CENTER_MIN[bin_idx]))


def compute_chemistry_per_bin(
    grids: List[GridData],
    orbits: Dict[int, list],
    photolysis_lut: PhotolysisLUT,
    hemco_species_ugm3: Dict[str, np.ndarray],
    year: int, month: int, day: int = 15,
    o3_column_du: float = 300.0,
    gamma_N2O5: float = 0.02,
    use_orbit_o3: bool = False,
    use_orbit_co: bool = False,
    jNO3_for_branch: bool = True,
    o3_bc_target_3d: Optional[np.ndarray] = None,
    o3_bc_rate_3d: Optional[np.ndarray] = None,
    prescribed_oh: bool = False,
) -> ChemistryPerBin:
    """Compute per-bin chemistry rates from current orbit state.

    Parameters
    ----------
    grids : list of 8 GridData
        One per UTC 3h bin.
    orbits : dict {species_idx: list of N_BINS+1 ndarray}
        Orbit solution from `solve_orbit_all_species`.  Uses orbits[s][tau+1]
        (end-of-bin concentration) as the representative c for that bin,
        which is consistent with the backward-Euler P_tau(c) = c_{tau+1}.
    photolysis_lut : PhotolysisLUT
        TUV clear-sky LUT (jNO2, jO1D, jNO3, jHONO vs SZA x O3 column).
    hemco_species_ugm3 : dict
        HEMCO climatology species -> 2D/3D ug/m3 on the ORBIT horizontal grid.
        Used here for O3, CO, CH4 only. HNO3 is never loaded (it is the gas
        phase of ORBIT's transported TotalNO3, split post-solve by ISORROPIA).
    year, month, day : int
        Date for solar geometry (day defaults to mid-month).
    o3_column_du : float
        Overhead O3 column (Dobson Units) used in the photolysis LUT
        interpolation.  300 DU is a reasonable tropical/subtropical default;
        for SAS winter 250-275 is typical but the LUT sensitivity is modest.
    gamma_N2O5 : float
        N2O5 heterogeneous uptake coefficient.  0.02 mid-range default.
    use_orbit_o3 : bool
        If True, use the orbit's O3 species (index IDX_O3) for oxidants.
        Phase 3c default is False — O3 isn't yet closed self-consistently
        and HEMCO provides a stable monthly-mean boundary condition.
    use_orbit_co : bool
        If True, feed the orbit's CO species (index IDX_CO) into the
        OH calculator's CO loss term, closing the CO <-> OH feedback.
        When False, uses HEMCO CO climatology (bin-flat) — safe default
        before CO is transported.  Iteration pattern mirrors use_orbit_o3:
        iter 1 uses HEMCO CO, iter 2+ uses orbit CO.
    prescribed_oh : bool
        If True, override each bin's ``ox.OH`` field with the HEMCO
        monthly-mean OH (same value for every bin — HEMCO is a monthly
        climatology, not diurnal).  This is the "prescribed-OH" ablation
        from the PM-first scope doc: isolates the contribution of
        diagnostic/responsive OH vs archive-style prescribed OH.  HO2 is
        NOT prescribed (HEMCO lacks HO2), so HO2 stays as computed by
        the PSS + fixed-point solve — slightly inconsistent with the
        overridden OH but close enough for the ablation since HO2
        affects PM2.5 only indirectly via O3 production.
    jNO3_for_branch : bool
        Include jNO3 in the N2O5 hetero branching (reduces the branch
        fraction in daylight, correct physics).  Default True.

    Returns
    -------
    ChemistryPerBin
    """
    n_bins = len(grids)
    nz, ny, nx = grids[0].nz, grids[0].ny, grids[0].nx

    # --- Photolysis for all bins (bin-averaged, twilight-safe) ---
    # Cloud fraction per bin: stacked (n_bins, ny, nx)
    cf_stack = np.stack([g.CLDTOT if g.CLDTOT.size > 0
                         else np.zeros((ny, nx)) for g in grids], axis=0)
    from orbit.core.photolysis import j_all_bins
    jvals = j_all_bins(
        photolysis_lut,
        grids[0].lat, grids[0].lon,
        year, month, day,
        cloud_fraction_per_bin=cf_stack,
        o3_column_du=o3_column_du,
    )
    jNO2_all = jvals["jNO2"]         # (N_BINS, ny, nx)
    jO1D_all = jvals["jO1D"]
    jNO3_all = jvals["jNO3"]
    jHONO_all = jvals["jHONO"]
    jHCHO_all = jvals.get("jHCHO")    # (N_BINS, ny, nx) or None
    jH2O2_all = jvals.get("jH2O2")

    # --- HEMCO backgrounds, shared across bins (monthly-mean) ---
    O3_hemco   = _hemco_to_grid(hemco_species_ugm3, "O3",   grids[0])
    CO_hemco   = _hemco_to_grid(hemco_species_ugm3, "CO",   grids[0])
    CH4_hemco  = _hemco_to_grid(hemco_species_ugm3, "CH4",  grids[0])
    # HCHO: 3h GC archive lacks IJ_AVG_S__CH2O, so we use the monthly mean.
    # jHCHO (TUV) is still bin-resolved, so 2·jHCHO·[HCHO] keeps its diurnal.
    HCHO_hemco = _hemco_to_grid(hemco_species_ugm3, "CH2O", grids[0])
    # HNO2: grid-level HNO2 (from preprocessor) if present; otherwise HEMCO.
    # ORBIT preprocessor HNO2 is spatially resolved; prefer it.

    # Prescribed-OH ablation: convert HEMCO OH (ug/m3) -> molec/cm3 once
    # (bin-flat; HEMCO is a monthly climatology without diurnal variation).
    if prescribed_oh:
        from orbit.core.oxidants import _compound_to_molcm3, MW_OH
        OH_hemco_ugm3 = _hemco_to_grid(hemco_species_ugm3, "OH", grids[0])
        OH_hemco_n = _compound_to_molcm3(OH_hemco_ugm3, MW_OH)
    else:
        OH_hemco_n = None

    oxidants_list: List[OxidantFields] = []
    k_so2_list:    List[np.ndarray]    = []
    k_nox_list:    List[np.ndarray]    = []
    k_o3_loss_list:   List[np.ndarray] = []
    s_o3_source_list: List[np.ndarray] = []
    k_co_loss_list:   List[np.ndarray] = []
    c_o3_list:        List[np.ndarray] = []  # per-bin c_o3 (HEMCO or orbit) for rate rebuild

    for tau in range(n_bins):
        grid = grids[tau]

        # Current concentrations from the orbit (end-of-bin state; equivalent
        # to the backward-Euler propagator's output at that bin).
        def _c(idx):
            return np.maximum(orbits[idx][tau + 1], 0.0).reshape((nz, ny, nx))
        c_NOx_ugN = _c(IDX_NOX)
        c_SO2_ugS = _c(IDX_SO2)
        c_pm      = _c(IDX_PM25)
        c_soa     = _c(IDX_SOA)
        c_nh      = _c(IDX_TOTAL_NH)
        c_pso4    = _c(IDX_PSO4)
        c_no3     = _c(IDX_TOTAL_NO3)

        # Prescribed partitioning for the PM2.5 estimate used in S_a(PM).
        # Use equilibrium partitioning when available; otherwise marginal.
        # SoA (species 0) is now a pure-particle tracer; no p_org blending.
        p_nh = grid.NHPartitioningEq if grid.NHPartitioningEq.size > 0 \
            else grid.NHPartitioning
        if grid.NO3PartitioningEq.size > 0:
            p_no3 = grid.NO3PartitioningEq
        elif grid.NO3Partitioning.size > 0:
            p_no3 = grid.NO3Partitioning
        else:
            p_no3 = np.ones((nz, ny, nx))
        pm25_approx = (c_pm + c_soa
                       + p_nh * c_nh * N_TO_NH4
                       + c_pso4 * S_TO_SO4
                       + p_no3 * c_no3 * N_TO_NO3)

        # O3: HEMCO background (default) or orbit self-consistent O3
        if use_orbit_o3:
            c_o3 = _c(IDX_O3)
        else:
            c_o3 = O3_hemco

        # CO: HEMCO background (default) or orbit self-consistent CO.
        # Per the PM-first scope doc, orbit CO closing feedback into OH
        # via the OH denominator is the point of transporting CO.
        if use_orbit_co:
            c_co = _c(IDX_CO)
        else:
            c_co = CO_hemco

        # Broadcast photolysis (ny, nx) to (nz, ny, nx)
        jNO2_3d = np.broadcast_to(jNO2_all[tau][None, :, :], (nz, ny, nx))
        jO1D_3d = np.broadcast_to(jO1D_all[tau][None, :, :], (nz, ny, nx))
        jNO3_3d = np.broadcast_to(jNO3_all[tau][None, :, :], (nz, ny, nx))
        jHONO_3d = np.broadcast_to(jHONO_all[tau][None, :, :], (nz, ny, nx))
        jHCHO_3d = (np.broadcast_to(jHCHO_all[tau][None, :, :], (nz, ny, nx))
                    if jHCHO_all is not None else None)
        jH2O2_3d = (np.broadcast_to(jH2O2_all[tau][None, :, :], (nz, ny, nx))
                    if jH2O2_all is not None else None)

        # HNO2 compound mass: from preprocessor (bin-resolved) if present.
        HONO_ugm3 = grid.HNO2 if grid.HNO2.size > 0 else None
        # HCHO: HEMCO monthly mean, flat across bins. Diurnal comes from jHCHO.
        HCHO_ugm3 = HCHO_hemco
        # H2O2 not yet wired through preprocessor; pass None for now.
        H2O2_ugm3 = None

        vocs = _voc_dict_from_grid(grid)

        ox = diagnose_oxidants_per_bin(
            grid,
            c_NOx_ugN=c_NOx_ugN, c_SO2_ugS=c_SO2_ugS, pm25_ugm3=pm25_approx,
            O3_ugm3=c_o3, CO_ugm3=c_co, CH4_ugm3=CH4_hemco,
            VOC_speciated_ugm3=vocs,
            jNO2=jNO2_3d, jO1D=jO1D_3d, jNO3=jNO3_3d,
            HONO_ugm3=HONO_ugm3, jHONO=jHONO_3d,
            HCHO_ugm3=HCHO_ugm3, jHCHO=jHCHO_3d,
            H2O2_ugm3=H2O2_ugm3, jH2O2=jH2O2_3d,
            gamma_N2O5=gamma_N2O5,
        )

        # Prescribed-OH ablation: overwrite diagnosed OH with HEMCO
        # monthly-mean.  Downstream rate builders (build_chemistry_rates,
        # build_co_loss_rate) use ox.OH directly, so they pick up the
        # prescribed value.  HO2 stays as computed — HEMCO lacks HO2,
        # so this is prescribed-OH only, not prescribed-(OH, HO2).
        if OH_hemco_n is not None:
            ox.OH = OH_hemco_n.copy()

        # Chemistry rates for operator assembly
        jNO3_for_rates = jNO3_3d if jNO3_for_branch else None
        k_so2_gas, k_nox_to_no3 = build_chemistry_rates(
            ox, c_o3, grid, jNO3=jNO3_for_rates,
        )

        # Combine with preprocessor aqueous SO2 oxidation
        k_so2_total = grid.SO2oxidation + k_so2_gas

        # Phase 3e: O3 chemistry (loss diagonal + production term) +
        # Newtonian BC nudging via the per-cell rate field ``o3_bc_rate_3d``
        # toward HEMCO 3D climatology ``o3_bc_target_3d``.  The caller
        # builds the rate field to combine top-of-model nudging (rate
        # concentrated at top N layers) and lateral-edge nudging (rate
        # concentrated at outermost M cells of each horizontal boundary),
        # typically via cell-wise max.  Multi-layer top nudging alone
        # cannot fix surface O3 in a regional domain because UT air is
        # advected out through the lateral boundaries faster than top-BC
        # can replenish — lateral nudging is required for a realistic
        # interior O3 field.
        k_o3_loss, s_o3_source = build_o3_rates(
            ox, grid, jO1D=jO1D_3d,
            bc_target_ugm3_3d=o3_bc_target_3d,
            bc_rate_3d=o3_bc_rate_3d,
        )

        oxidants_list.append(ox)
        k_so2_list.append(k_so2_total)
        k_nox_list.append(k_nox_to_no3)
        k_o3_loss_list.append(k_o3_loss)
        s_o3_source_list.append(s_o3_source)
        k_co_loss_list.append(build_co_loss_rate(ox))
        c_o3_list.append(c_o3)

    return ChemistryPerBin(
        oxidants=oxidants_list,
        k_so2_rate=k_so2_list,
        k_nox_to_no3_rate=k_nox_list,
        jNO2=jNO2_all, jO1D=jO1D_all, jNO3=jNO3_all, jHONO=jHONO_all,
        jHCHO=jHCHO_all, jH2O2=jH2O2_all,
        k_o3_loss_rate=k_o3_loss_list,
        s_o3_source_rate=s_o3_source_list,
        k_co_loss_rate=k_co_loss_list,
        c_o3_per_bin=c_o3_list,
        o3_bc_target_3d=o3_bc_target_3d,
        o3_bc_rate_3d=o3_bc_rate_3d,
        jNO3_for_branch=jNO3_for_branch,
    )


def splice_oh_into_chem(chem: ChemistryPerBin, oh_flat: np.ndarray,
                         nz: int, ny: int, nx: int) -> None:
    """Write a flat (N_BINS * nz * ny * nx) OH vector back into
    ``chem.oxidants[tau].OH`` for each bin, in place.

    Used by the Anderson closure (Opt 1) to propagate the accelerated
    OH field into the chemistry struct before calling
    ``rebuild_rates_from_chem``.
    """
    N = nz * ny * nx
    n_bins = len(chem.oxidants)
    oh_flat = np.asarray(oh_flat, dtype=np.float64).ravel()
    expected = N * n_bins
    if oh_flat.size != expected:
        raise ValueError(
            f"splice_oh_into_chem: oh_flat has {oh_flat.size} entries, "
            f"expected {expected} ({n_bins} bins * {N} cells)"
        )
    for tau in range(n_bins):
        chem.oxidants[tau].OH = oh_flat[tau * N:(tau + 1) * N].reshape((nz, ny, nx))


def rebuild_rates_from_chem(chem: ChemistryPerBin,
                             grids: List[GridData]) -> None:
    """Re-derive operator rates from (possibly updated) oxidants.

    After Anderson acceleration rewrites ``chem.oxidants[tau].OH``
    (via :func:`splice_oh_into_chem`), the downstream rates
    (k_so2, k_nox_to_no3, k_o3_loss, s_o3_source) must be re-derived
    to stay consistent with the new OH field.  This mirrors the
    cheap "turn oxidants into operator rates" pass inside
    :func:`compute_chemistry_per_bin` but skips the expensive
    per-cell OH fixed-point iteration (``diagnose_oxidants_per_bin``).

    Requires that ``chem`` was produced by
    :func:`compute_chemistry_per_bin`, which populates the context
    fields (``c_o3_per_bin``, ``o3_bc_target_3d``, ``o3_bc_rate_3d``,
    ``jNO3_for_branch``).

    Mutates ``chem.k_so2_rate``, ``chem.k_nox_to_no3_rate``,
    ``chem.k_o3_loss_rate``, ``chem.s_o3_source_rate`` in place.
    """
    if chem.c_o3_per_bin is None:
        raise ValueError(
            "rebuild_rates_from_chem: chem.c_o3_per_bin not populated. "
            "Was chem produced by compute_chemistry_per_bin?"
        )
    n_bins = len(chem.oxidants)
    nz, ny, nx = chem.oxidants[0].OH.shape
    for tau in range(n_bins):
        ox = chem.oxidants[tau]
        grid = grids[tau]
        # Broadcast photolysis from (ny, nx) to (nz, ny, nx); chem.jO1D
        # has shape (N_BINS, ny, nx).
        jNO3_3d = (np.broadcast_to(chem.jNO3[tau][None, :, :], (nz, ny, nx))
                   if chem.jNO3_for_branch else None)
        jO1D_3d = np.broadcast_to(chem.jO1D[tau][None, :, :], (nz, ny, nx))
        c_o3 = chem.c_o3_per_bin[tau]

        k_so2_gas, k_nox_to_no3 = build_chemistry_rates(
            ox, c_o3, grid, jNO3=jNO3_3d,
        )
        k_so2_total = grid.SO2oxidation + k_so2_gas
        k_o3_loss, s_o3_source = build_o3_rates(
            ox, grid, jO1D=jO1D_3d,
            bc_target_ugm3_3d=chem.o3_bc_target_3d,
            bc_rate_3d=chem.o3_bc_rate_3d,
        )

        chem.k_so2_rate[tau] = k_so2_total
        chem.k_nox_to_no3_rate[tau] = k_nox_to_no3
        chem.k_o3_loss_rate[tau] = k_o3_loss
        chem.s_o3_source_rate[tau] = s_o3_source
        if chem.k_co_loss_rate is not None:
            chem.k_co_loss_rate[tau] = build_co_loss_rate(ox)


def chemistry_diagnostics(chem: ChemistryPerBin, grids: List[GridData]):
    """Compact per-bin surface-mean summary for validation."""
    out = []
    for tau, (ox, k_s, k_n) in enumerate(zip(
        chem.oxidants, chem.k_so2_rate, chem.k_nox_to_no3_rate,
    )):
        out.append({
            "bin": tau + 1,
            "OH_surf_mean": float(ox.OH[0].mean()),
            "OH_surf_max":  float(ox.OH[0].max()),
            "HO2_surf_mean": float(ox.HO2[0].mean()),
            "f_NO2_surf_mean": float(ox.f_NO2[0].mean()),
            "NO3_surf_mean": float(ox.NO3[0].mean()),
            "N2O5_surf_mean": float(ox.N2O5[0].mean()),
            "k_so2_tot_surf_mean": float(k_s[0].mean()),
            "k_so2_aq_surf_mean":  float(grids[tau].SO2oxidation[0].mean()),
            "k_nox_to_no3_surf_mean": float(k_n[0].mean()),
            "jNO2_mean": float(chem.jNO2[tau].mean()),
            "jO1D_mean": float(chem.jO1D[tau].mean()),
            "jHONO_mean": float(chem.jHONO[tau].mean()),
        })
    return out
