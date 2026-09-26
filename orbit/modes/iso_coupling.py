"""ISORROPIA cross-coupling blocks for iso-coupled marginal mode.

The marginal back-solve treats the deposition operator's diagonal
``L[NH, NH] = T + (1-p_NH) k_gas + p_NH k_aer`` (and similarly for NO3)
with ``p`` frozen at baseline. Iso-coupled marginal extends this
linearisation to include the off-diagonal cross-blocks captured by the
ISORROPIA equilibrium:

  ∂(D_NH · c_NH) / ∂c_NO3  |_{c_NH fixed}
    = c_NH · (k_aer - k_gas) · ∂p_NH/∂c_NO3
    = (k_aer - k_gas) · f_nh_dno3              (since f_nh_dno3 = c_NH · ∂p_NH/∂c_NO3)

So the cross-block ``L[NH, NO3]`` (per cell) is just
``Δk_NH · f_nh_dno3`` where ``Δk_NH`` = particle dep rate − gas dep rate.

Same form for the other 3 cross-couplings (Δk_NO3 · f_no3_dnh,
Δk_NH · f_nh_dso4, Δk_NO3 · f_no3_dso4). The pSO4 row of the 3×3
ISORROPIA Jacobian block is not represented here — the LUT does not
carry f_so4 (only f_nh4 + f_no3 are stored). pSO4 is treated as ~100%
particle in ORBIT; the ∂f_SO4/∂c_NH and ∂f_SO4/∂c_NO3 partials are
small (per the iso-coupled-marginal addendum, "approximately 1.0
doesn't mean its derivatives are zero" but they're small) and their
inclusion would require a full LUT regeneration.

These blocks slot into the existing ``K_sources_per_bin`` mechanism
in ``orbit.core.orbit.solve_orbit_all_species`` — the orbit code
treats each ``K[(target, source)]`` generically as
``e_target -= K @ c_source``.
"""

from __future__ import annotations

import os
from typing import Dict, Tuple

import numpy as np
import scipy.sparse as sp

from orbit.core.deposition import (
    IDX_TOTAL_NH, IDX_TOTAL_NO3,
    _get_hno3_vd,
)
from orbit.core.grid_data import GridData
from orbit.core.indexing import CellIndexer


# Diagnostic env-var: disable specific iso K-blocks at runtime by setting
# ORBIT_ISO_DISABLE_BLOCKS to a comma-separated list of block names from
# {NH_NO3, NH_SO4, NO3_NH, NO3_SO4}. Used by the iso-coupling magnitude
# investigation to localize which block contributes the over-amplification.
def _disabled_blocks() -> set:
    raw = os.environ.get("ORBIT_ISO_DISABLE_BLOCKS", "").strip()
    if not raw:
        return set()
    return {tok.strip() for tok in raw.split(",") if tok.strip()}


def _delta_k_nh(grid: GridData) -> np.ndarray:
    """Δk for TotalNH = (particle-rate − gas-rate). 3D, shape (nz, ny, nx).

    Wet:  particle_wet_dep − other_gas_wet_dep
    Dry:  (particle_dry_dep − NH3_dry_dep) / Dz at k=0 only

    Returns a 3D array combining both contributions.
    """
    nz, ny, nx = grid.nz, grid.ny, grid.nx
    dk_wet = grid.particle_wet_dep - grid.other_gas_wet_dep  # (nz, ny, nx)
    dk_dry_surf = grid.particle_dry_dep[0] - grid.NH3_dry_dep[0]  # (ny, nx)
    dz_surf = grid.Dz[0]
    safe_dz = np.where(dz_surf > 0, dz_surf, 1.0)
    dk_dry_3d = np.zeros((nz, ny, nx), dtype=np.float64)
    dk_dry_3d[0] = np.where(dz_surf > 0, dk_dry_surf / safe_dz, 0.0)
    return dk_wet + dk_dry_3d


def _delta_k_no3(grid: GridData) -> np.ndarray:
    """Δk for TotalNO3 = (particle-rate − HNO3-rate). 3D, shape (nz, ny, nx).

    Wet:  particle_wet_dep − other_gas_wet_dep. The deposition operator
          scavenges the HNO3 branch of TotalNO3 at the soluble-gas rate and
          the particle branch at the particle rate (deposition.py,
          ``(1-p)·other_gas_wet_dep + p·particle_wet_dep``, since
          2026-08-02), so the contrast is the same as for TotalNH. Until
          2026-09-26 this term was zero, left over from the earlier
          all-particle placeholder.
    Dry:  (particle_dry_dep − HNO3_dry_dep) / Dz at k=0 only.

    Returns a 3D array combining both contributions.
    """
    nz, ny, nx = grid.nz, grid.ny, grid.nx
    dk_wet = grid.particle_wet_dep - grid.other_gas_wet_dep  # (nz, ny, nx)
    vd_hno3 = _get_hno3_vd(grid)
    dk_dry_surf = grid.particle_dry_dep[0] - vd_hno3[0]
    dz_surf = grid.Dz[0]
    safe_dz = np.where(dz_surf > 0, dz_surf, 1.0)
    dk_dry_3d = np.zeros((nz, ny, nx), dtype=np.float64)
    dk_dry_3d[0] = np.where(dz_surf > 0, dk_dry_surf / safe_dz, 0.0)
    return dk_wet + dk_dry_3d


def assemble_iso_cross_blocks(
    grid: GridData,
    indexer: CellIndexer,
    f_nh_dno3: np.ndarray,    # (nz, ny, nx) — ∂(p_NH·NH)/∂NO3
    f_no3_dnh: np.ndarray,    # (nz, ny, nx) — ∂(p_NO3·NO3)/∂NH
    f_nh_dso4: np.ndarray,    # (nz, ny, nx) — ∂(p_NH·NH)/∂SO4
    f_no3_dso4: np.ndarray,   # (nz, ny, nx) — ∂(p_NO3·NO3)/∂SO4
) -> Dict[Tuple[int, int], "sp.csc_matrix"]:
    """Build the 4 iso-coupling cross-blocks for one bin.

    Returns ``{(target, source): K_diag_csc}`` keyed by:
      (IDX_TOTAL_NH,  IDX_TOTAL_NO3)
      (IDX_TOTAL_NH,  IDX_PSO4)
      (IDX_TOTAL_NO3, IDX_TOTAL_NH)
      (IDX_TOTAL_NO3, IDX_PSO4)

    Each K is a (N, N) diagonal sparse matrix; the orbit code does
    ``src -= K @ c_source`` so a positive K entry means c_source acts
    as additional loss for c_target. The signs work out: more c_NO3
    increases p_NH which (typically) deposits faster than NH3 gas, so
    c_NH loses extra mass when c_NO3 rises.
    """
    N = indexer.N

    dk_nh = _delta_k_nh(grid).ravel()   # (N,)
    dk_no3 = _delta_k_no3(grid).ravel()  # (N,)

    idx = np.arange(N, dtype=np.int64)
    blocks: Dict[Tuple[int, int], sp.csc_matrix] = {}

    disabled = _disabled_blocks()
    from orbit.core.deposition import IDX_PSO4

    diag_nh_dno3 = dk_nh * f_nh_dno3.ravel()
    if "NH_NO3" in disabled:
        diag_nh_dno3 = np.zeros_like(diag_nh_dno3)
    blocks[(IDX_TOTAL_NH, IDX_TOTAL_NO3)] = sp.csc_matrix(
        (diag_nh_dno3, (idx, idx)), shape=(N, N),
    )

    diag_nh_dso4 = dk_nh * f_nh_dso4.ravel()
    if "NH_SO4" in disabled:
        diag_nh_dso4 = np.zeros_like(diag_nh_dso4)
    blocks[(IDX_TOTAL_NH, IDX_PSO4)] = sp.csc_matrix(
        (diag_nh_dso4, (idx, idx)), shape=(N, N),
    )

    diag_no3_dnh = dk_no3 * f_no3_dnh.ravel()
    if "NO3_NH" in disabled:
        diag_no3_dnh = np.zeros_like(diag_no3_dnh)
    blocks[(IDX_TOTAL_NO3, IDX_TOTAL_NH)] = sp.csc_matrix(
        (diag_no3_dnh, (idx, idx)), shape=(N, N),
    )

    diag_no3_dso4 = dk_no3 * f_no3_dso4.ravel()
    if "NO3_SO4" in disabled:
        diag_no3_dso4 = np.zeros_like(diag_no3_dso4)
    blocks[(IDX_TOTAL_NO3, IDX_PSO4)] = sp.csc_matrix(
        (diag_no3_dso4, (idx, idx)), shape=(N, N),
    )

    return blocks


# Iso-coupled extension of the marginal-mode skip-solve DAG. With cross-
# coupling on, an NH3 perturbation propagates into TotalNO3 (via
# ∂p_NO3/∂c_NH) and a SO4/SO2 perturbation propagates into both
# TotalNH and TotalNO3.
ISO_SOURCES_FOR_TARGET: Dict[int, list] = {
    IDX_TOTAL_NH:  [IDX_TOTAL_NO3],   # NO3 → NH via ∂p_NH/∂c_NO3
    IDX_TOTAL_NO3: [IDX_TOTAL_NH],    # NH  → NO3 via ∂p_NO3/∂c_NH
}


def merged_sources_for_target(
    base: Dict[int, list],
    iso: bool,
) -> Dict[int, list]:
    """Combine the chemistry-DAG with iso couplings (if iso=True).

    Note: ISO contributions through pSO4 (IDX_PSO4 → IDX_TOTAL_NH and
    IDX_PSO4 → IDX_TOTAL_NO3) are also added when iso=True. pSO4 itself
    is downstream of SO2 in the chemistry DAG, so a SO2 perturbation
    indirectly perturbs both TotalNH and TotalNO3 through this path.
    """
    if not iso:
        return dict(base)
    from orbit.core.deposition import IDX_PSO4
    out = {k: list(v) for k, v in base.items()}
    for target, sources in ISO_SOURCES_FOR_TARGET.items():
        cur = out.setdefault(target, [])
        for s in sources:
            if s not in cur:
                cur.append(s)
        # pSO4 also drives both NH and NO3 cross-couplings
        if IDX_PSO4 not in cur:
            cur.append(IDX_PSO4)
    return out
