"""Deposition operator assembly: dry + wet deposition as diagonal matrices.

For the steady-state matrix:
  Dry dep rate = vd / Dz  (linearized dry-deposition rate)
  Wet dep rate = wd_rate   (linearized wet-deposition rate)

For merged species (Total NH, Total NO3), effective deposition rates use
partition-weighted combination of gas and particle rates.

14-species inventory (indices below; POA at 13 and the four further VBS
bins at 9 to 12 are listed after the constants):

  0  SoA           pure-particle SOA tracer (yield-at-emission scheme; was
                   previously a lumped TotalOrg with p_org partitioning).
                   Particle-phase dry/wet dep, no gas channel, no chemistry
                   coupling. Emissions = Σ_source (yield_source × VOC_emis).
  1  Primary PM2.5 primary PM
  2  Total NH      NH3 + NH4, split by ISORROPIA post-solve
  3  SO2           gas, oxidized to pSO4
  4  NOx           NO + NO2, oxidized to TotalNO3
  5  pSO4          aerosol sulfate
  6  Total NO3     HNO3 + pNO3, split by ISORROPIA post-solve
  7  O3            ozone, chemistry-driven
  8  CO            carbon monoxide, major OH sink, chemistry-driven loss

Kinetic off-diagonal couplings:
  SO2 -> pSO4   (existing)
  NOx -> TotalNO3 (new in Phase 2/DCOMP)

O3 has no transported precursor; its chemistry comes from diagnostic
HO2/RO2 produced by the oxidants module.

CO has no transported precursor; its chemistry is a simple diagonal
loss term k_OH_CO * [OH] assembled into its own operator. Negligible
dry/wet deposition at regional scales. Enters the OH budget via the
denominator, closing the CO <-> OH feedback within the outer iteration.
"""

import numpy as np
import scipy.sparse as sp
from orbit.core.indexing import CellIndexer
from orbit.core.grid_data import GridData
from orbit.core.soa_photolysis import get_a_photo, soa_photolysis_rate

# 13-species indices (DCOMP + CO + 1D VBS for SoA).
# Solve order groups sources before sinks:
#   SO2 before pSO4, NOx before TotalNO3,
#   VBS_C1000 before C100 before C10 before C1 before C01 (aging cascade).
# CO is a diagonal-only transported species (no source couplings).
#
# IDX_SOA = 0 is preserved as an alias for IDX_VBS_C100 so legacy code
# paths that read "the SoA tracer" still compile; the canonical SoA mass
# that contributes to PM2.5 is the F_p-weighted sum across all 5 VBS
# bins (computed by the closure helper, see orbit/core/dcomp_vbs.py).
IDX_VBS_C100  = 0
IDX_SOA       = IDX_VBS_C100   # alias; total SoA = F_p-weighted sum over bins
IDX_PM25      = 1
IDX_TOTAL_NH  = 2
IDX_SO2       = 3
IDX_NOX       = 4
IDX_PSO4      = 5
IDX_TOTAL_NO3 = 6
IDX_O3        = 7
IDX_CO        = 8
IDX_VBS_C10   = 9
IDX_VBS_C1    = 10
IDX_VBS_C01   = 11
IDX_VBS_C1000 = 12
# POA (primary organic aerosol), carved out of PrimaryPM25 so that it can
# act as absorbing mass in the VBS Pankow partitioning (orbit/core/dcomp_vbs)
# and as the IVOC source term. Chemically inert and depositionally identical
# to PrimaryPM25, so its operator is bit-identical and the LU factorisation
# is shared. Appended last so every pre-existing index is unchanged.
IDX_POA       = 13
N_SPECIES = 14

# VBS conveniences. Bin order is low-to-high C* (smallest volatility
# first), matching the natural deposition / particle-fraction ordering.
C_STAR_VALS = np.array([0.1, 1.0, 10.0, 100.0, 1000.0])  # µg/m³
IDX_VBS_BINS = (IDX_VBS_C01, IDX_VBS_C1, IDX_VBS_C10,
                IDX_VBS_C100, IDX_VBS_C1000)
N_VBS_BINS = len(IDX_VBS_BINS)

# Aging cascade: source bin → destination bin (one step downward in C*).
# IDX_VBS_C01 has no destination (lowest bin, no further aging).
VBS_AGING_PAIRS = (
    (IDX_VBS_C1000, IDX_VBS_C100),
    (IDX_VBS_C100,  IDX_VBS_C10),
    (IDX_VBS_C10,   IDX_VBS_C1),
    (IDX_VBS_C1,    IDX_VBS_C01),
)

# Back-compat alias: old Total NO lumped NOx + pNO3 as index 5. Phase 2
# splits this into NOx (4) and TotalNO3 (6). Any remaining reference to
# IDX_TOTAL_NO is a bug and should be replaced. Left as None to fail loudly.
IDX_TOTAL_NO = None


def assemble_deposition(grid: GridData, indexer: CellIndexer, species_idx: int) -> sp.csc_matrix:
    """Assemble deposition diagonal matrix for one merged species.

    Parameters
    ----------
    grid : GridData
    indexer : CellIndexer
    species_idx : int
        One of IDX_SOA..IDX_TOTAL_NO

    Returns
    -------
    D : csc_matrix, shape (N, N)
        Diagonal matrix of deposition rates [1/s]
    """
    N = indexer.N

    # Wet deposition rate (all levels) - vectorized
    wd = _get_wet_dep_rate_3d(grid, species_idx)  # (nz, ny, nx)
    diag_3d = wd.copy()

    # Dry deposition (k=0 only)
    dz_surface = grid.Dz[0]  # (ny, nx)
    vd = _get_dry_dep_velocity_2d(grid, species_idx)  # (ny, nx)
    safe_dz = np.where(dz_surface > 0, dz_surface, 1.0)
    diag_3d[0] += np.where(dz_surface > 0, vd / safe_dz, 0.0)

    # VBS SoA photolytic / heterogeneous loss (Lambda = A_PHOTO * j(NO2) * F_p).
    # A partition-dependent, pure-loss particle-phase sink, structurally like the
    # F_p-weighted deposition above (NOT the aging cascade, which routes mass
    # down-bin and is partition-independent). Gated on grid.j_no2_soa being
    # attached at solve time (orbit/core/soa_photolysis); absent => no sink, so
    # baselines/tests without the LUT are unaffected.
    diag_3d += _get_soa_photolysis_rate_3d(grid, species_idx)

    diag = diag_3d.ravel()
    indices = np.arange(N, dtype=np.int64)
    return sp.csc_matrix((diag, (indices, indices)), shape=(N, N))


def _get_soa_photolysis_rate_3d(grid: GridData, species_idx: int) -> np.ndarray:
    """SOA photolytic-loss rate [1/s], (nz, ny, nx), for a VBS bin.

    Returns A_PHOTO * j(NO2) * F_p,i for VBS bins, else 0. Zero when the LUT
    field grid.j_no2_soa is absent or A_PHOTO == 0. F_p,i uses the closure-
    converged grid.F_p_vbs; on the first solver pass (before the VBS closure
    has run) it falls back to pure-particle (F_p = 1), mirroring the wet/dry
    dep VBS branches.
    """
    j = getattr(grid, "j_no2_soa", None)
    if (species_idx not in IDX_VBS_BINS or j is None
            or np.asarray(j).size == 0 or get_a_photo() <= 0.0):
        return 0.0
    bin_pos = IDX_VBS_BINS.index(species_idx)
    F_p_3d = getattr(grid, "F_p_vbs", None)
    if F_p_3d is None or F_p_3d.shape[0] != N_VBS_BINS:
        F_p_i = 1.0  # pure particle until the closure populates F_p_vbs
    else:
        F_p_i = F_p_3d[bin_pos]
    return soa_photolysis_rate(j, F_p_i)


def _get_no3_partitioning(grid: GridData) -> np.ndarray:
    """Return the partitioning field used for Total NO3 deposition blending.

    Prefers the marginal NO3Partitioning (response-coefficient, correct for
    operator assembly) when present, falling back to the equilibrium
    NO3PartitioningEq. These are populated at runtime by the Phase 3d
    outer iteration from ISORROPIA LUT queries on ORBIT's converged
    TotalNO3 (not from the preprocessor — GC/HEMCO don't provide HNO3).

    Until the outer iteration lands, both are empty and this returns zeros.
    That corresponds to treating the TotalNO3 pool as all HNO3 for
    deposition — the fast-dep bound (worst case for loss), which is a
    conservative starting point but wrong for the final answer.
    """
    if hasattr(grid, "NO3Partitioning") and grid.NO3Partitioning.size > 0:
        return grid.NO3Partitioning
    if hasattr(grid, "NO3PartitioningEq") and grid.NO3PartitioningEq.size > 0:
        return grid.NO3PartitioningEq
    return np.zeros((grid.nz, grid.ny, grid.nx))


def _get_hno3_vd(grid: GridData) -> np.ndarray:
    """HNO3 dry deposition velocity (m/s), 3D -> (nz, ny, nx).

    If grid.HNO3_dry_dep is populated, use it. Otherwise fall back to a
    constant 3 cm/s at surface (0.03 m/s), zero aloft — a reasonable
    mid-range value for vegetated/polluted SAS continental conditions
    (Nemitz et al. 2000, Zhang et al. 2003: 1-5 cm/s over most land).
    Full land-use-dependent resistance formulation is a later refinement.
    """
    if hasattr(grid, "HNO3_dry_dep") and grid.HNO3_dry_dep.size > 0:
        return grid.HNO3_dry_dep
    vd = np.zeros((grid.nz, grid.ny, grid.nx))
    vd[0] = 0.03  # 3 cm/s at surface
    return vd


def _get_o3_vd(grid: GridData) -> np.ndarray:
    """O3 dry deposition velocity (m/s), 3D -> (nz, ny, nx).

    Standard values: ~0.5 cm/s over vegetation, ~0.05 cm/s over water.
    Falls back to 0.4 cm/s over land, 0.05 over ocean using grid.is_land.
    """
    if hasattr(grid, "O3_dry_dep") and grid.O3_dry_dep.size > 0:
        return grid.O3_dry_dep
    vd = np.zeros((grid.nz, grid.ny, grid.nx))
    land = grid.is_land if grid.is_land.size > 0 else np.ones((grid.ny, grid.nx))
    vd[0] = np.where(land > 0, 0.004, 0.0005)
    return vd


def _get_dry_dep_velocity_2d(grid: GridData, species_idx: int) -> np.ndarray:
    """Get effective dry deposition velocity array for surface cells. Returns (ny, nx)."""
    if species_idx in IDX_VBS_BINS:
        # VBS bin gas/particle partitioning per Pankow (1994). Uses the
        # closure-converged F_p,i fields written by
        # orbit.core.dcomp_vbs.update_vbs_partitioning(). On the very
        # first solver pass (before the closure has run) F_p_vbs_surface
        # may be absent; fall back to pure particle in that case (the
        # closure overwrites this on iter 1+).
        bin_pos = IDX_VBS_BINS.index(species_idx)
        F_p_surf = getattr(grid, "F_p_vbs_surface", None)
        if F_p_surf is None or F_p_surf.shape[0] != N_VBS_BINS:
            return grid.particle_dry_dep[0]
        F_p_i = F_p_surf[bin_pos]
        # gas-channel for organic gas. VOC_dry_dep loaded from preproc;
        # falls back to other_gas dry-dep proxy if absent.
        if grid.VOC_dry_dep.size > 0:
            voc_dep = grid.VOC_dry_dep[0]
        else:
            # Conservative fallback: 0.1 cm/s for semi-volatile organic gas
            # (Karl 2010, Niinemets 2014). Avoids zero gas-channel which
            # would make the C*=1000 bin behave like a pure aerosol.
            voc_dep = np.full_like(grid.particle_dry_dep[0], 1.0e-3)
        return F_p_i * grid.particle_dry_dep[0] + (1.0 - F_p_i) * voc_dep
    elif species_idx in (IDX_PM25, IDX_POA):
        return grid.particle_dry_dep[0]
    elif species_idx == IDX_TOTAL_NH:
        p = grid.NHPartitioning[0]
        return (1.0 - p) * grid.NH3_dry_dep[0] + p * grid.particle_dry_dep[0]
    elif species_idx == IDX_SO2:
        return grid.SO2_dry_dep[0]
    elif species_idx == IDX_NOX:
        return grid.NOx_dry_dep[0]
    elif species_idx == IDX_PSO4:
        return grid.particle_dry_dep[0]
    elif species_idx == IDX_TOTAL_NO3:
        p = _get_no3_partitioning(grid)[0]
        vd_hno3 = _get_hno3_vd(grid)[0]
        return (1.0 - p) * vd_hno3 + p * grid.particle_dry_dep[0]
    elif species_idx == IDX_O3:
        return _get_o3_vd(grid)[0]
    elif species_idx == IDX_CO:
        # CO has negligible dry deposition at regional scales.
        return np.zeros((grid.ny, grid.nx))
    return np.zeros((grid.ny, grid.nx))


def _get_wet_dep_rate_3d(grid: GridData, species_idx: int) -> np.ndarray:
    """Get effective wet deposition rate array. Returns (nz, ny, nx)."""
    if species_idx in IDX_VBS_BINS:
        # VBS bin gas/particle partitioning per Pankow. Mirrors the dry-dep
        # branch above; F_p_vbs is full 3D (5, nz, ny, nx) when populated.
        bin_pos = IDX_VBS_BINS.index(species_idx)
        F_p_3d = getattr(grid, "F_p_vbs", None)
        if F_p_3d is None or F_p_3d.shape[0] != N_VBS_BINS:
            return grid.particle_wet_dep.copy()
        F_p_i = F_p_3d[bin_pos]
        return F_p_i * grid.particle_wet_dep + (1.0 - F_p_i) * grid.other_gas_wet_dep
    elif species_idx in (IDX_PM25, IDX_POA):
        return grid.particle_wet_dep.copy()
    elif species_idx == IDX_TOTAL_NH:
        p = grid.NHPartitioning
        return (1.0 - p) * grid.other_gas_wet_dep + p * grid.particle_wet_dep
    elif species_idx == IDX_SO2:
        return grid.SO2_wet_dep.copy()
    elif species_idx == IDX_NOX:
        # NOx: gas-phase deposition (NOx is not appreciably wet-deposited;
        # the dominant wet-dep pathway is through HNO3 after oxidation).
        return grid.other_gas_wet_dep.copy()
    elif species_idx == IDX_PSO4:
        return grid.particle_wet_dep.copy()
    elif species_idx == IDX_TOTAL_NO3:
        p = _get_no3_partitioning(grid)
        # Gas fraction (HNO3) scavenges as a soluble gas, symmetric with
        # the TotalNH branch above. HNO3's effective solubility exceeds
        # NH3's, so the soluble-gas rate is still a lower bound; a
        # dedicated HNO3 wet-scavenging field from the preprocessor is the
        # full fix (2026-08-02, replacing the all-particle placeholder
        # that reduced to particle_wet_dep for any partitioning).
        return (1.0 - p) * grid.other_gas_wet_dep + p * grid.particle_wet_dep
    elif species_idx == IDX_O3:
        # O3 wet deposition is negligible (low solubility).
        return np.zeros_like(grid.other_gas_wet_dep)
    elif species_idx == IDX_CO:
        # CO wet deposition is negligible (low solubility).
        return np.zeros_like(grid.other_gas_wet_dep)
    return np.zeros((grid.nz, grid.ny, grid.nx))


# --- Loop version for equivalence testing ---

def _assemble_deposition_loop(grid: GridData, indexer: CellIndexer, species_idx: int) -> sp.csc_matrix:
    nz, ny, nx = grid.nz, grid.ny, grid.nx
    N = indexer.N
    diag = np.zeros(N, dtype=np.float64)

    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                n = indexer.to_flat(k, j, i)
                rate = 0.0
                if k == 0:
                    dz = grid.Dz[0, j, i]
                    if dz > 0:
                        vd = _get_dry_dep_velocity_scalar(grid, species_idx, j, i)
                        rate += vd / dz
                wd = _get_wet_dep_rate_scalar(grid, species_idx, k, j, i)
                rate += wd
                diag[n] = rate

    indices = np.arange(N, dtype=np.int64)
    return sp.csc_matrix((diag, (indices, indices)), shape=(N, N))


def _get_dry_dep_velocity_scalar(grid, species_idx, j, i):
    # Loop form, used only for equivalence testing. Delegates to the
    # vectorized dispatch for new species to avoid duplicating logic.
    if species_idx in IDX_VBS_BINS:
        return _get_dry_dep_velocity_2d(grid, species_idx)[j, i]
    elif species_idx in (IDX_PM25, IDX_POA):
        return grid.particle_dry_dep[0, j, i]
    elif species_idx == IDX_TOTAL_NH:
        p = grid.NHPartitioning[0, j, i]
        return (1.0 - p) * grid.NH3_dry_dep[0, j, i] + p * grid.particle_dry_dep[0, j, i]
    elif species_idx == IDX_SO2:
        return grid.SO2_dry_dep[0, j, i]
    elif species_idx == IDX_NOX:
        return grid.NOx_dry_dep[0, j, i]
    elif species_idx == IDX_PSO4:
        return grid.particle_dry_dep[0, j, i]
    elif species_idx == IDX_TOTAL_NO3:
        return _get_dry_dep_velocity_2d(grid, IDX_TOTAL_NO3)[j, i]
    elif species_idx == IDX_O3:
        return _get_dry_dep_velocity_2d(grid, IDX_O3)[j, i]
    elif species_idx == IDX_CO:
        return 0.0
    return 0.0


def _get_wet_dep_rate_scalar(grid, species_idx, k, j, i):
    if species_idx in IDX_VBS_BINS:
        return _get_wet_dep_rate_3d(grid, species_idx)[k, j, i]
    elif species_idx in (IDX_PM25, IDX_POA):
        return grid.particle_wet_dep[k, j, i]
    elif species_idx == IDX_TOTAL_NH:
        p = grid.NHPartitioning[k, j, i]
        return (1.0 - p) * grid.other_gas_wet_dep[k, j, i] + p * grid.particle_wet_dep[k, j, i]
    elif species_idx == IDX_SO2:
        return grid.SO2_wet_dep[k, j, i]
    elif species_idx == IDX_NOX:
        return grid.other_gas_wet_dep[k, j, i]
    elif species_idx == IDX_PSO4:
        return grid.particle_wet_dep[k, j, i]
    elif species_idx == IDX_TOTAL_NO3:
        return _get_wet_dep_rate_3d(grid, IDX_TOTAL_NO3)[k, j, i]
    elif species_idx == IDX_O3:
        return 0.0
    elif species_idx == IDX_CO:
        return 0.0
    return 0.0
