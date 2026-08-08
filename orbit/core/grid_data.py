"""GridData: dataclass holding all arrays from preprocessor + computed geometry."""

import warnings

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from orbit.io.reader import load_preprocessor, truncate_u, truncate_v, load_frland

# Earth radius (WGS84 semi-major axis)
EARTH_RADIUS = 6378137.0
DEG_TO_RAD = np.pi / 180.0


@dataclass
class GridData:
    """All arrays needed for operator assembly.

    Loaded from preprocessor NetCDF + computed geometry.
    All 3D arrays are (nz, ny, nx). Wind arrays are truncated from staggered.
    """

    # Dimensions
    nz: int = 0
    ny: int = 0
    nx: int = 0

    # Coordinates
    lon: np.ndarray = field(default_factory=lambda: np.array([]))
    lat: np.ndarray = field(default_factory=lambda: np.array([]))
    dlon: float = 0.625
    dlat: float = 0.5

    # Sigma coordinates
    Ap: np.ndarray = field(default_factory=lambda: np.array([]))  # (nz+1,) Pa
    Bp: np.ndarray = field(default_factory=lambda: np.array([]))  # (nz+1,)
    Psurf: np.ndarray = field(default_factory=lambda: np.array([]))  # (ny, nx) Pa

    # Computed from sigma coords
    dP: np.ndarray = field(default_factory=lambda: np.array([]))  # (nz, ny, nx) Pa
    Dz: np.ndarray = field(default_factory=lambda: np.array([]))  # (nz, ny, nx) m

    # Wind (truncated from staggered)
    UAvg: np.ndarray = field(default_factory=lambda: np.array([]))  # (nz, ny, nx) m/s
    VAvg: np.ndarray = field(default_factory=lambda: np.array([]))  # (nz, ny, nx) m/s
    omega: np.ndarray = field(default_factory=lambda: np.array([]))  # (nz, ny, nx) Pa/s
    omega_plus: np.ndarray = field(default_factory=lambda: np.array([]))   # (nz, ny, nx) max(omega,0) avg
    omega_minus: np.ndarray = field(default_factory=lambda: np.array([]))  # (nz, ny, nx) max(-omega,0) avg
    has_split_omega: bool = False

    # Split-flux wind averages (from preprocessor, or computed from UAvg/VAvg)
    UAvg_plus: np.ndarray = field(default_factory=lambda: np.array([]))   # (nz, ny, nx) max(U,0) avg
    UAvg_minus: np.ndarray = field(default_factory=lambda: np.array([]))  # (nz, ny, nx) max(-U,0) avg
    VAvg_plus: np.ndarray = field(default_factory=lambda: np.array([]))   # (nz, ny, nx) max(V,0) avg
    VAvg_minus: np.ndarray = field(default_factory=lambda: np.array([]))  # (nz, ny, nx) max(-V,0) avg
    has_split_fluxes: bool = False

    # PROTOTYPE flag (off by default): when True, assemble_transport_block uses
    # a single unified vertical Patankar operator (convdiff.assemble_vertical_
    # convdiff) in place of separate omega advection + Kzz diffusion. Research
    # only.
    unified_vertical_patankar: bool = False

    # Periodic longitude boundary
    periodic_lon: bool = False
    UAvg_wrap: np.ndarray = field(default_factory=lambda: np.array([]))       # (nz, ny) wrap face
    UAvg_plus_wrap: np.ndarray = field(default_factory=lambda: np.array([]))  # (nz, ny)
    UAvg_minus_wrap: np.ndarray = field(default_factory=lambda: np.array([]))  # (nz, ny)
    K_meander_u_wrap: np.ndarray = field(default_factory=lambda: np.array([]))  # (nz, ny)

    # Diffusivities
    Kzz: np.ndarray = field(default_factory=lambda: np.array([]))  # (nz, ny, nx) m^2/s
    Kxxyy: np.ndarray = field(default_factory=lambda: np.array([]))  # (nz, ny, nx)
    K_meander_u: np.ndarray = field(default_factory=lambda: np.array([]))  # (nz, ny, nx)
    K_meander_v: np.ndarray = field(default_factory=lambda: np.array([]))  # (nz, ny, nx)

    # Deposition velocities
    particle_dry_dep: np.ndarray = field(default_factory=lambda: np.array([]))  # m/s
    SO2_dry_dep: np.ndarray = field(default_factory=lambda: np.array([]))
    NOx_dry_dep: np.ndarray = field(default_factory=lambda: np.array([]))
    NH3_dry_dep: np.ndarray = field(default_factory=lambda: np.array([]))
    VOC_dry_dep: np.ndarray = field(default_factory=lambda: np.array([]))

    # Wet deposition rates
    particle_wet_dep: np.ndarray = field(default_factory=lambda: np.array([]))  # 1/s
    SO2_wet_dep: np.ndarray = field(default_factory=lambda: np.array([]))
    other_gas_wet_dep: np.ndarray = field(default_factory=lambda: np.array([]))

    # Chemistry
    SO2oxidation: np.ndarray = field(default_factory=lambda: np.array([]))  # 1/s
    NHPartitioning: np.ndarray = field(default_factory=lambda: np.array([]))   # MARGINAL <Δp/(Δp+Δg)>
    NOPartitioning: np.ndarray = field(default_factory=lambda: np.array([]))   # MARGINAL (legacy lumped NOx+pNO3)
    AOrgPartitioning: np.ndarray = field(default_factory=lambda: np.array([]))  # MARGINAL
    # DCOMP NO3 partitioning of the HNO3+pNO3 pool, populated by the Phase 3d
    # outer iteration from ISORROPIA LUT queries at ORBIT's converged
    # TotalNO3 (not a preprocessor input). Marginal goes into the deposition
    # operator, equilibrium into PM2.5 mass extraction.
    NO3Partitioning: np.ndarray = field(default_factory=lambda: np.array([]))
    NO3PartitioningEq: np.ndarray = field(default_factory=lambda: np.array([]))
    # Equilibrium partitioning = <p>/(<p>+<g>) from averaged gas/particle fields.
    # Use this for PM2.5 mass extraction (state-variable split); marginal is for
    # operator-side effective deposition.
    NHPartitioningEq: np.ndarray = field(default_factory=lambda: np.array([]))
    NOPartitioningEq: np.ndarray = field(default_factory=lambda: np.array([]))
    SPartitioningEq: np.ndarray = field(default_factory=lambda: np.array([]))
    # Time-averaged gas/particle concentrations (element mass, ug/m3) from bin files
    gNH: np.ndarray = field(default_factory=lambda: np.array([]))
    pNH: np.ndarray = field(default_factory=lambda: np.array([]))
    gNO: np.ndarray = field(default_factory=lambda: np.array([]))
    pNO: np.ndarray = field(default_factory=lambda: np.array([]))
    # gNO3 is the HNO3-gas slot populated post-solve by the ISORROPIA split
    # of TotalNO3 (never read from an archive; GC/HEMCO don't provide it).
    gNO3: np.ndarray = field(default_factory=lambda: np.array([]))
    gS: np.ndarray = field(default_factory=lambda: np.array([]))
    pS: np.ndarray = field(default_factory=lambda: np.array([]))
    # HNO3 and O3 dry dep fields (3D). Populated by deposition module
    # with defaults if absent from bin files.
    HNO3_dry_dep: np.ndarray = field(default_factory=lambda: np.array([]))
    O3_dry_dep: np.ndarray = field(default_factory=lambda: np.array([]))

    # Land mask
    is_land: np.ndarray = field(default_factory=lambda: np.array([]))  # (ny, nx) uint8

    # Computed geometry
    dx: np.ndarray = field(default_factory=lambda: np.array([]))  # (ny,) m
    dy: float = 0.0  # m
    volume: np.ndarray = field(default_factory=lambda: np.array([]))  # (nz, ny, nx) m^3

    # Terrain ratios for sigma-native transport
    dP_ratio_west: np.ndarray = field(default_factory=lambda: np.array([]))  # (nz, ny, nx)
    dP_ratio_south: np.ndarray = field(default_factory=lambda: np.array([]))  # (nz, ny, nx)
    dP_ratio_east: np.ndarray = field(default_factory=lambda: np.array([]))   # (nz, ny, nx)
    dP_ratio_north: np.ndarray = field(default_factory=lambda: np.array([]))  # (nz, ny, nx)

    # Meteorological fields for plume rise (optional, loaded if present)
    LayerHeights: np.ndarray = field(default_factory=lambda: np.array([]))  # (nz+1, ny, nx) m
    Temperature: np.ndarray = field(default_factory=lambda: np.array([]))  # (nz, ny, nx) K
    WindSpeed: np.ndarray = field(default_factory=lambda: np.array([]))  # (nz, ny, nx) m/s
    S1: np.ndarray = field(default_factory=lambda: np.array([]))  # (nz, ny, nx) stability param
    Sclass: np.ndarray = field(default_factory=lambda: np.array([]))  # (nz, ny, nx) stability class
    WindSpeedInverse: np.ndarray = field(default_factory=lambda: np.array([]))  # (nz, ny, nx) 1/ws
    WindSpeedMinusThird: np.ndarray = field(default_factory=lambda: np.array([]))  # (nz, ny, nx) ws^(-1/3)
    WindSpeedMinusOnePointFour: np.ndarray = field(default_factory=lambda: np.array([]))  # (nz, ny, nx) ws^(-1.4)

    # ISORROPIA inputs (optional — only in reprocessed bin files)
    RH: np.ndarray = field(default_factory=lambda: np.array([]))            # (nz, ny, nx) percent
    dust_fine: np.ndarray = field(default_factory=lambda: np.array([]))     # (nz, ny, nx) ug/m3
    sea_salt_fine: np.ndarray = field(default_factory=lambda: np.array([]))  # (nz, ny, nx) ug/m3

    # Bin-resolved archive oxidants (feat/orbit-isorropia-archive).
    # archive_OH in molec/cm3 (native), archive_NO / archive_NO2 /
    # archive_NO3rad in ppbv. Consumed at operator-assembly time to build
    # first-order NOx -> TotalNO3 rates (k_OH+NO2 day + N2O5 hydrolysis
    # night via thermal equilibrium from NO2+NO3rad).
    archive_OH: np.ndarray = field(default_factory=lambda: np.array([]))          # molec/cm3
    archive_NO: np.ndarray = field(default_factory=lambda: np.array([]))          # ppbv
    archive_NO2: np.ndarray = field(default_factory=lambda: np.array([]))         # ppbv
    archive_NO3rad: np.ndarray = field(default_factory=lambda: np.array([]))      # ppbv

    # DCOMP oxidant-diagnosis inputs (optional — ISOP..NAP from Phase 2a rerun,
    # HNO2 from Phase 3 rerun). Compound mass ug/m3 except CLDTOT (0-1) and Pblh (m).
    ISOP: np.ndarray = field(default_factory=lambda: np.array([]))
    MTPA: np.ndarray = field(default_factory=lambda: np.array([]))
    MTPO: np.ndarray = field(default_factory=lambda: np.array([]))
    LIMO: np.ndarray = field(default_factory=lambda: np.array([]))
    BENZ: np.ndarray = field(default_factory=lambda: np.array([]))
    TOLU: np.ndarray = field(default_factory=lambda: np.array([]))
    XYLE: np.ndarray = field(default_factory=lambda: np.array([]))
    NAP:  np.ndarray = field(default_factory=lambda: np.array([]))
    HNO2: np.ndarray = field(default_factory=lambda: np.array([]))
    CLDTOT: np.ndarray = field(default_factory=lambda: np.array([]))        # (ny, nx) 0-1
    Pblh:   np.ndarray = field(default_factory=lambda: np.array([]))        # (ny, nx) m
    Pressure: np.ndarray = field(default_factory=lambda: np.array([]))      # (nz, ny, nx) Pa layer-center

    # Convective mass flux (optional, interface-level) — from feat/cmfmc-convective-transport
    CMFMC: np.ndarray = field(default_factory=lambda: np.array([]))  # (nz+1, ny, nx) kg/m²/s

    # Clear-sky j(NO2) [1/s] per orbit bin, attached at solve time by
    # orbit/core/soa_photolysis.attach_jno2_to_grid. Drives the VBS SoA
    # photolytic-loss sink in deposition.assemble_deposition. Empty/absent
    # => no photolytic sink. (nz, ny, nx)
    j_no2_soa: np.ndarray = field(default_factory=lambda: np.array([]))


# Every raw variable load_grid consumes, verbatim from the code below.
# tests/test_grid_read_restriction.py proves a restricted load produces a
# GridData identical to a full one; extend this set when adding a field.
_GRID_INPUT_VARS = frozenset([
    "lon", "lat", "Ap", "Bp", "Psurf", "Dz", "LayerHeights", "CMFMC",
    "omega", "omega_plus", "omega_minus",
    "UAvg", "VAvg", "UAvg_plus", "UAvg_minus", "VAvg_plus", "VAvg_minus",
    "K_meander_u", "K_meander_v",
    "Kzz", "Kxxyy",
    "ParticleDryDep", "SO2DryDep", "NOxDryDep", "NH3DryDep", "VOCDryDep",
    "ParticleWetDep", "SO2WetDep", "OtherGasWetDep",
    "SO2oxidation", "NHPartitioning", "NOPartitioning",
    "aOrgPartitioning", "AOrgPartitioning",
    "Temperature", "WindSpeed", "S1", "Sclass",
    "WindSpeedInverse", "WindSpeedMinusThird", "WindSpeedMinusOnePointFour",
    "RH", "dust_fine", "sea_salt_fine",
    "gNH", "pNH", "gNO", "pNO", "gS", "pS",
    "ISOP", "MTPA", "MTPO", "LIMO", "BENZ", "TOLU", "XYLE", "NAP", "HNO2",
    "archive_OH", "archive_NO", "archive_NO2", "archive_NO3rad",
    "CLDTOT", "Pblh",
])


def load_grid(preprocessor_path: str, constants_path: Optional[str] = None,
              read_all: bool = False) -> GridData:
    """Load preprocessor NetCDF and compute all derived geometry.

    Parameters
    ----------
    preprocessor_path : str
        Path to preprocessed-input preprocessor NetCDF
    constants_path : str, optional
        Path to MERRA-2 constants file for land mask.
        If None, all cells treated as land.
    read_all : bool
        Read every variable in the file instead of only _GRID_INPUT_VARS.
        The result is identical (the extra variables are never consumed);
        exists for the equivalence test.

    Returns
    -------
    GridData
        Fully populated grid data
    """
    raw = load_preprocessor(preprocessor_path,
                            variables=None if read_all else _GRID_INPUT_VARS)
    attrs = raw.get("_attrs", {})

    g = GridData()

    # Coordinates
    g.lon = np.asarray(raw["lon"], dtype=np.float64)
    g.lat = np.asarray(raw["lat"], dtype=np.float64)
    g.nx = len(g.lon)
    g.ny = len(g.lat)
    g.dlon = float(attrs.get("dx", 0.625))
    g.dlat = float(attrs.get("dy", 0.5))

    # Get nz from a 3D variable
    g.nz = raw["Kzz"].shape[0]

    # Sigma coordinates
    g.Ap = np.asarray(raw["Ap"], dtype=np.float64)
    # Auto-convert Ap from hPa to Pa if needed
    if np.max(g.Ap) < 1100:
        g.Ap = g.Ap * 100.0
    g.Bp = np.asarray(raw["Bp"], dtype=np.float64)
    g.Psurf = np.asarray(raw["Psurf"], dtype=np.float64)

    # Compute dP from sigma coordinates
    g.dP = np.zeros((g.nz, g.ny, g.nx), dtype=np.float64)
    g.Pressure = np.zeros((g.nz, g.ny, g.nx), dtype=np.float64)
    for k in range(g.nz):
        P_bot = g.Ap[k] + g.Bp[k] * g.Psurf
        P_top = g.Ap[k + 1] + g.Bp[k + 1] * g.Psurf
        g.dP[k] = P_bot - P_top
        g.Pressure[k] = 0.5 * (P_bot + P_top)

    # Layer thickness
    g.Dz = np.asarray(raw["Dz"], dtype=np.float64)

    # Wind (truncate staggered arrays)
    g.UAvg = truncate_u(np.asarray(raw["UAvg"], dtype=np.float64), g.nx)
    g.VAvg = truncate_v(np.asarray(raw["VAvg"], dtype=np.float64), g.ny)
    # omega here is the HYBRID CROSS-LEVEL velocity (terrain-corrected in the
    # preprocessor: omega - V.grad_eta(P); see meteorology.hybrid_cross_omega),
    # NOT raw pressure velocity. Used directly as the cross-model-level flux.
    g.omega = np.asarray(raw["omega"], dtype=np.float64)

    # Split-flux omega averages: if preprocessor has pre-split fields, use them.
    # Otherwise, fall back to splitting the time-averaged omega.
    if "omega_plus" in raw:
        g.omega_plus = np.asarray(raw["omega_plus"], dtype=np.float64)
        g.omega_minus = np.asarray(raw["omega_minus"], dtype=np.float64)
        g.has_split_omega = True
    else:
        warnings.warn(
            "Preprocessor output lacks omega_plus/omega_minus; "
            "falling back to splitting time-averaged omega. "
            "Re-run the preprocessor to get proper upwind-split vertical fluxes.",
            stacklevel=2,
        )
        g.omega_plus = np.maximum(g.omega, 0.0)
        g.omega_minus = np.maximum(-g.omega, 0.0)
        g.has_split_omega = False

    # Meander diffusivities (also staggered)
    g.K_meander_u = truncate_u(np.asarray(raw["K_meander_u"], dtype=np.float64), g.nx)
    g.K_meander_v = truncate_v(np.asarray(raw["K_meander_v"], dtype=np.float64), g.ny)

    # Split-flux wind averages: if preprocessor has pre-split fields, use them
    # (avoids double-counting K_meander in convdiff). Otherwise, compute from UAvg/VAvg.
    if "UAvg_plus" in raw:
        g.UAvg_plus = truncate_u(np.asarray(raw["UAvg_plus"], dtype=np.float64), g.nx)
        g.UAvg_minus = truncate_u(np.asarray(raw["UAvg_minus"], dtype=np.float64), g.nx)
        g.VAvg_plus = truncate_v(np.asarray(raw["VAvg_plus"], dtype=np.float64), g.ny)
        g.VAvg_minus = truncate_v(np.asarray(raw["VAvg_minus"], dtype=np.float64), g.ny)
        g.has_split_fluxes = True
    else:
        g.UAvg_plus = np.maximum(g.UAvg, 0.0)
        g.UAvg_minus = np.maximum(-g.UAvg, 0.0)
        g.VAvg_plus = np.maximum(g.VAvg, 0.0)
        g.VAvg_minus = np.maximum(-g.VAvg, 0.0)
        g.has_split_fluxes = False

    # Periodic longitude detection and wrap-face extraction
    g.periodic_lon = (g.nx * g.dlon >= 359.5)
    if g.periodic_lon:
        # Extract wrap-face data from raw staggered arrays (face nx, between
        # cell nx-1 and cell 0) before truncation dropped it.
        # Raw U-staggered has shape (nz, ny, nx+1); face 0 is west of cell 0,
        # face nx is east of cell nx-1 (= west of cell 0 in periodic grid).
        raw_U_full = np.asarray(raw["UAvg"], dtype=np.float64)  # (nz, ny, nx+1)
        # Wrap face = average of face 0 and face nx (they bound the same gap)
        g.UAvg_wrap = 0.5 * (raw_U_full[:, :, 0] + raw_U_full[:, :, g.nx])  # (nz, ny)
        if "UAvg_plus" in raw:
            raw_Up = np.asarray(raw["UAvg_plus"], dtype=np.float64)
            raw_Um = np.asarray(raw["UAvg_minus"], dtype=np.float64)
            g.UAvg_plus_wrap = 0.5 * (raw_Up[:, :, 0] + raw_Up[:, :, g.nx])
            g.UAvg_minus_wrap = 0.5 * (raw_Um[:, :, 0] + raw_Um[:, :, g.nx])
        else:
            g.UAvg_plus_wrap = np.maximum(g.UAvg_wrap, 0.0)
            g.UAvg_minus_wrap = np.maximum(-g.UAvg_wrap, 0.0)
        raw_Km = np.asarray(raw["K_meander_u"], dtype=np.float64)
        g.K_meander_u_wrap = 0.5 * (raw_Km[:, :, 0] + raw_Km[:, :, g.nx])

    # Simple 3D fields
    for raw_name, attr_name in [
        ("Kzz", "Kzz"), ("Kxxyy", "Kxxyy"),
        ("ParticleDryDep", "particle_dry_dep"),
        ("SO2DryDep", "SO2_dry_dep"),
        ("NOxDryDep", "NOx_dry_dep"),
        ("NH3DryDep", "NH3_dry_dep"),
        ("VOCDryDep", "VOC_dry_dep"),
        ("ParticleWetDep", "particle_wet_dep"),
        ("SO2WetDep", "SO2_wet_dep"),
        ("OtherGasWetDep", "other_gas_wet_dep"),
        ("SO2oxidation", "SO2oxidation"),
        ("NHPartitioning", "NHPartitioning"),
        ("NOPartitioning", "NOPartitioning"),
        ("aOrgPartitioning", "AOrgPartitioning"),
        ("Temperature", "Temperature"),
        ("WindSpeed", "WindSpeed"),
        ("S1", "S1"),
        ("Sclass", "Sclass"),
        ("WindSpeedInverse", "WindSpeedInverse"),
        ("WindSpeedMinusThird", "WindSpeedMinusThird"),
        ("WindSpeedMinusOnePointFour", "WindSpeedMinusOnePointFour"),
        ("RH", "RH"),
        ("dust_fine", "dust_fine"),
        ("sea_salt_fine", "sea_salt_fine"),
        ("gNH", "gNH"), ("pNH", "pNH"),
        ("gNO", "gNO"), ("pNO", "pNO"),
        ("gS", "gS"), ("pS", "pS"),
        # gNO3 / NO3Partitioning / NO3PartitioningEq are NOT bin-file inputs;
        # populated at runtime in Phase 3d from the ISORROPIA LUT.
        ("ISOP", "ISOP"), ("MTPA", "MTPA"), ("MTPO", "MTPO"), ("LIMO", "LIMO"),
        ("BENZ", "BENZ"), ("TOLU", "TOLU"), ("XYLE", "XYLE"), ("NAP", "NAP"),
        ("HNO2", "HNO2"),
        # Bin-resolved archive oxidants (feat/orbit-isorropia-archive)
        ("archive_OH", "archive_OH"),
        ("archive_NO", "archive_NO"),
        ("archive_NO2", "archive_NO2"),
        ("archive_NO3rad", "archive_NO3rad"),
        ("CLDTOT", "CLDTOT"), ("Pblh", "Pblh"),
    ]:
        if raw_name in raw:
            setattr(g, attr_name, np.asarray(raw[raw_name], dtype=np.float64))
    # Fallback for AOrgPartitioning key
    if "AOrgPartitioning" in raw and g.AOrgPartitioning.size == 0:
        g.AOrgPartitioning = np.asarray(raw["AOrgPartitioning"], dtype=np.float64)

    # Compute equilibrium partitioning from averaged gas/particle fields.
    # NOTE: this is <p>/(<p>+<g>) — the ratio of time-averaged concentrations.
    # Observationally this matches what monthly-mean composition measurements see.
    # For thermodynamic interpretation this is physically distinct from (and for
    # our purposes preferable to) <p/(p+g)> (Jensen inequality; they coincide
    # only when p and g are perfectly correlated).
    def _safe_ratio(p, g):
        if p.size == 0 or g.size == 0:
            return np.array([])
        total = p + g
        return np.where(total > 0, p / np.maximum(total, 1e-30), 0.0)
    g.NHPartitioningEq = _safe_ratio(g.pNH, g.gNH)
    g.NOPartitioningEq = _safe_ratio(g.pNO, g.gNO)
    g.SPartitioningEq = _safe_ratio(g.pS, g.gS)

    # LayerHeights: staggered (nz+1, ny, nx), loaded separately
    if "LayerHeights" in raw:
        g.LayerHeights = np.asarray(raw["LayerHeights"], dtype=np.float64)

    # Convective mass flux: staggered (nz+1, ny, nx), optional
    if "CMFMC" in raw:
        g.CMFMC = np.asarray(raw["CMFMC"], dtype=np.float64)

    # Land mask
    if constants_path is not None:
        g.is_land = load_frland(constants_path, g.lon, g.lat)
    else:
        g.is_land = np.ones((g.ny, g.nx), dtype=np.uint8)

    # Compute geometry
    _compute_geometry(g)
    _compute_terrain_ratios(g)

    return g


def _compute_geometry(g: GridData) -> None:
    """Compute dx (varies with latitude), dy, and volume."""
    g.dy = EARTH_RADIUS * g.dlat * DEG_TO_RAD
    g.dx = np.zeros(g.ny, dtype=np.float64)
    for j in range(g.ny):
        g.dx[j] = EARTH_RADIUS * np.cos(g.lat[j] * DEG_TO_RAD) * g.dlon * DEG_TO_RAD

    g.volume = np.zeros((g.nz, g.ny, g.nx), dtype=np.float64)
    for k in range(g.nz):
        for j in range(g.ny):
            g.volume[k, j, :] = g.dx[j] * g.dy * g.Dz[k, j, :]


def _compute_terrain_ratios(g: GridData) -> None:
    """Compute terrain ratios for sigma-native transport.

    dP_ratio_west[k,j,i]  = dP[k,j,i-1] / dP[k,j,i]  (west neighbor / cell)
    dP_ratio_south[k,j,i] = dP[k,j-1,i] / dP[k,j,i]  (south neighbor / cell)
    dP_ratio_east[k,j,i]  = dP[k,j,i+1] / dP[k,j,i]  (east neighbor / cell)
    dP_ratio_north[k,j,i] = dP[k,j+1,i] / dP[k,j,i]  (north neighbor / cell)

    At boundaries or where either cell has dP <= 0, ratio = 1.0.
    """
    g.dP_ratio_west = np.ones((g.nz, g.ny, g.nx), dtype=np.float64)
    g.dP_ratio_south = np.ones((g.nz, g.ny, g.nx), dtype=np.float64)
    g.dP_ratio_east = np.ones((g.nz, g.ny, g.nx), dtype=np.float64)
    g.dP_ratio_north = np.ones((g.nz, g.ny, g.nx), dtype=np.float64)

    dP = g.dP

    # Use safe denominators to avoid divide-by-zero warnings in np.where
    # (np.where evaluates both branches before selecting)
    safe_dP = np.where(dP > 0, dP, 1.0)

    # West: ratio[k,j,i] = dP[k,j,i-1] / dP[k,j,i] for i > 0
    valid_w = (dP[:, :, 1:] > 0) & (dP[:, :, :-1] > 0)
    g.dP_ratio_west[:, :, 1:] = np.where(valid_w, dP[:, :, :-1] / safe_dP[:, :, 1:], 1.0)

    # East: ratio[k,j,i] = dP[k,j,i+1] / dP[k,j,i] for i < nx-1
    valid_e = (dP[:, :, :-1] > 0) & (dP[:, :, 1:] > 0)
    g.dP_ratio_east[:, :, :-1] = np.where(valid_e, dP[:, :, 1:] / safe_dP[:, :, :-1], 1.0)

    # South: ratio[k,j,i] = dP[k,j-1,i] / dP[k,j,i] for j > 0
    valid_s = (dP[:, 1:, :] > 0) & (dP[:, :-1, :] > 0)
    g.dP_ratio_south[:, 1:, :] = np.where(valid_s, dP[:, :-1, :] / safe_dP[:, 1:, :], 1.0)

    # North: ratio[k,j,i] = dP[k,j+1,i] / dP[k,j,i] for j < ny-1
    valid_n = (dP[:, :-1, :] > 0) & (dP[:, 1:, :] > 0)
    g.dP_ratio_north[:, :-1, :] = np.where(valid_n, dP[:, 1:, :] / safe_dP[:, :-1, :], 1.0)

    # Periodic longitude: wrap ratios at dateline
    if g.periodic_lon:
        # West neighbor of cell 0 is cell nx-1
        valid_w0 = (dP[:, :, 0] > 0) & (dP[:, :, -1] > 0)
        g.dP_ratio_west[:, :, 0] = np.where(valid_w0, dP[:, :, -1] / safe_dP[:, :, 0], 1.0)
        # East neighbor of cell nx-1 is cell 0
        valid_e_last = (dP[:, :, -1] > 0) & (dP[:, :, 0] > 0)
        g.dP_ratio_east[:, :, -1] = np.where(valid_e_last, dP[:, :, 0] / safe_dP[:, :, -1], 1.0)
