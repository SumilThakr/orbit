"""Emission source configuration and constants.

Emission source configuration for the loader's 7-slot emission layout
(SPECIES_MAP below); the solver's 14 transported species are filled from
these slots and the VOC-to-VBS distribution.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


# =============================================================================
# EmissionSource dataclass
# =============================================================================

@dataclass
class EmissionSource:
    """Configuration for a single emission source file.

    Supported units:
        Rate units (mass per time, converted to ug/s internally):
            - "kg/s"         Kilograms per second
            - "ug/s"         Micrograms per second (native internal rate)
            - "kg/year"      Kilograms per year (annualized rate, 365.25-day year)
            - "tons/year"    US short tons per year (1 ton = 907.185 kg)
            - "tonnes/year"  Metric tonnes per year (1 tonne = 1000 kg)

        Flux units (mass per area per time, NetCDF only):
            - "kg/m2/s"      Flux rate; multiplied by cell area to get ug/s
            - "kg/m2"        Flux total over a period; requires averaging_period (days).

    For shapefiles/geopackages:
        - Species columns are auto-detected (PM25, NOx, SOx, NH3, VOC, etc.)
        - If elevated=True, reads height/diam/temp/velocity columns for plume rise

    For NetCDF:
        - variable_mapping: Explicit {species: varname} mapping, or auto-detect
        - height_variable: Variable name for 2D injection heights [m]
        - layer_index: Fixed layer index for all emissions
        - time_index: Select specific timestep from multi-time NetCDF
    """
    path: str = ""
    format: str = "netcdf"      # shapefile, netcdf, geopackage
    units: str = "kg/m2/s"      # kg/year, tons/year, tonnes/year, ug/s, kg/s, kg/m2/s, kg/m2
    elevated: bool = False
    averaging_period: Optional[float] = None  # Days; required when units='kg/m2'
    # NetCDF-specific options
    variable_mapping: Optional[Dict[str, str]] = None
    height_variable: Optional[str] = None
    layer_index: Optional[int] = None
    time_index: Optional[int] = None
    # Scalar stack params for ASME (1973) plume-rise. When all four are
    # > 0 and the gridded loader sees no per-cell height_variable, these
    # are applied uniformly across the source so plumerise.find_injection_layer
    # computes the per-cell effective injection layer from local met
    # (stability, wind, temperature). Defaults to 0 = use simple stack
    # height lookup with no rise.
    stack_height: float = 0.0   # m above ground
    stack_diam: float = 0.0     # m
    stack_temp: float = 0.0     # K (stack-exit gas temperature)
    stack_vel: float = 0.0      # m/s (stack-exit gas velocity)
    # Diurnal-resolved input: file carries N_BINS slabs per month rather
    # than a single monthly mean.  When True, the loader reads
    # (n_bins, lat, lon) slabs at month=time_index and the YAML diurnal
    # profile (if any) is ignored for this source.
    bin_axis: bool = False
    bin_axis_name: str = "bin"
    # SoA mass yield for the yield-at-emission scheme (Variant A).
    # DEPRECATED on the VBS branch — kept only for interface compatibility
    # with eval scripts that still import EmissionSource. The active VBS
    # branch uses voc_parent_class instead.
    soa_yield: float = 0.0
    # VBS parent class for 5-bin SoA distribution. Maps to a yields tuple
    # in orbit/emissions/vbs_yields.VBS_PARENT_YIELDS:
    #   "anthro_high_nox", "anthro_low_nox",
    #   "bio_monoterpene", "bio_isoprene",
    #   "biomass_burning", "ivoc"
    # Or "anthro" — triggers cell-dependent high/low NOx-regime switching
    # via archive [NO2]/[OH] (see run_orbit.py:_compute_nox_regime).
    # None means this source does not contribute to VBS at all (its slot 0
    # mass is dropped — important so that legacy VOC sources without a
    # parent class don't contaminate the C*=100 bin).
    voc_parent_class: Optional[str] = None


# =============================================================================
# Physical constants
# =============================================================================

# Julian year, as in deposition_maps.UG_M2_S_TO_KG_HA_YR and the health
# post-processing (subdistrict_aggregation.SECONDS_PER_YEAR). Until 2026-09-26
# the emissions path used 365 days, a 0.07% offset from the rest of the model.
SECONDS_PER_YEAR = 365.25 * 24.0 * 3600.0
R_EARTH = 6378137.0  # Earth radius [m] (WGS84, matches grid_data.py)

# Molar masses [g/mol]
MW_N = 14.0067
MW_S = 32.0655
MW_NOx = 46.0055
MW_NH3 = 17.03056
MW_SO2 = 64.0644

# Element conversion factors
NOx_TO_N = MW_N / MW_NOx    # 0.30449
SOx_TO_S = MW_S / MW_SO2    # 0.49652
NH3_TO_N = MW_N / MW_NH3    # 0.82245


# =============================================================================
# Unit conversion factors: source units -> ug/s
# =============================================================================

UNIT_CONVERSIONS = {
    'kg/year': 1e9 / SECONDS_PER_YEAR,
    'tons/year': 1e9 * 907.185 / SECONDS_PER_YEAR,   # US short tons
    'tonnes/year': 1e9 * 1000.0 / SECONDS_PER_YEAR,  # metric tonnes
    'ug/s': 1.0,
    'kg/s': 1e9,
    'kg/m2/s': None,  # Sentinel: flux rate, handled via cell area
    'kg/m2': None,    # Sentinel: flux total, requires averaging_period
}


# =============================================================================
# Species column mappings (case-insensitive matching)
# Maps column/variable names to legacy 9-species indices
# =============================================================================

SPECIES_MAP = {
    # Index 0: VOC/gOrg
    'voc': 0, 'vocs': 0, 'gorg': 0,
    # Index 1: SOA/pOrg
    'soa': 1, 'porg': 1,
    # Index 2: Primary PM2.5
    'pm25': 2, 'pm2_5': 2, 'pm2.5': 2, 'primarypm25': 2, 'primarypm2_5': 2,
    # Index 3: NH3/gNH (convert to N mass)
    'nh3': 3, 'gnh': 3,
    # Index 4: pNH4 (already as N)
    'nh4': 4, 'pnh': 4, 'pnh4': 4,
    # Index 5: SO2/SOx/gS (convert to S mass)
    'so2': 5, 'sox': 5, 'gs': 5,
    # Index 6: pSO4 (already as S)
    'so4': 6, 'ps': 6, 'pso4': 6,
    # Index 7: NOx/gNO (convert to N mass)
    'nox': 7, 'no2': 7, 'gno': 7,
    # Index 8: pNO3 (already as N)
    'no3': 8, 'pno': 8, 'pno3': 8,
    # Index 9: POA / POM (primary organic aerosol), split out of PM2.5
    'poa': 9, 'pom': 9,
}

# NetCDF variable name patterns for species auto-detection
NETCDF_SPECIES_PATTERNS = {
    'voc': 0, 'gorg': 0,
    'soa': 1, 'porg': 1,
    'pm25': 2, 'pm2_5': 2,
    'nh3': 3, 'gnh': 3, 'ammonia': 3,
    'nh4': 4, 'pnh': 4,
    'so2': 5, 'sox': 5, 'gs': 5, 'sulfur': 5,
    'so4': 6, 'ps': 6, 'sulfate': 6,
    'nox': 7, 'no2': 7, 'gno': 7, 'nitrogen': 7,
    'no3': 8, 'pno': 8, 'nitrate': 8,
    'poa': 9, 'pom': 9,
}

# Element conversion for precursor species (legacy index -> factor)
ELEMENT_CONVERSION = {
    3: NH3_TO_N,   # NH3 -> N
    5: SOx_TO_S,   # SO2 -> S
    7: NOx_TO_N,   # NOx -> N
}

# Column names that trigger element conversion (case-normalized)
CONVERT_COLS = {'nh3', 'sox', 'so2', 'nox', 'no2'}


# =============================================================================
# emitted-species names -> loader slot
# =============================================================================

# the reference model idx -> ORBIT idx
LEGACY9_TO_ORBIT = {
    0: 0,  # gOrg (VOC)  -> SoA
    1: 0,  # pOrg (SOA)  -> SoA
    2: 1,  # PM2.5       -> PM2.5
    3: 2,  # gNH (NH3)   -> Total NH
    4: 2,  # pNH (NH4)   -> Total NH
    5: 3,  # gS (SO2)    -> SO2
    6: 4,  # pS (SO4)    -> pSO4
    7: 5,  # gNO (NOx)   -> Total NO
    8: 5,  # pNO (NO3)   -> Total NO
    9: 6,  # POA         -> POA
}

N_ORBIT_SPECIES = 7


# =============================================================================
# Helpers
# =============================================================================

def compute_cell_areas(lat_arr: np.ndarray, dlon_deg: float, dlat_deg: float) -> np.ndarray:
    """Compute cell areas for each latitude band.

    area(lat) = R_earth^2 * cos(lat_rad) * dlon_rad * dlat_rad  [m^2]

    Parameters
    ----------
    lat_arr : ndarray, shape (ny,)
        Cell-center latitudes [degrees]
    dlon_deg : float
        Cell width in degrees longitude
    dlat_deg : float
        Cell height in degrees latitude

    Returns
    -------
    ndarray, shape (ny,)
        Cell area in m^2 per latitude band
    """
    lat_rad = np.deg2rad(lat_arr)
    dlon_rad = np.deg2rad(dlon_deg)
    dlat_rad = np.deg2rad(dlat_deg)
    return R_EARTH**2 * np.cos(lat_rad) * dlon_rad * dlat_rad


def parse_emission_sources(
    yaml_list: list, config_dir: str
) -> List[EmissionSource]:
    """Parse YAML emission entries into EmissionSource objects.

    Parameters
    ----------
    yaml_list : list
        List of dicts from YAML config ``input.emissions``
    config_dir : str
        Directory of the config file (for resolving relative paths)

    Returns
    -------
    list of EmissionSource
    """
    sources = []
    for entry in yaml_list:
        if isinstance(entry, str):
            source = EmissionSource(path=entry)
        else:
            source = EmissionSource(
                path=entry.get('path', ''),
                format=entry.get('format', 'netcdf'),
                units=entry.get('units', 'kg/m2/s'),
                elevated=entry.get('elevated', False),
                averaging_period=entry.get('averaging_period'),
                variable_mapping=entry.get('variable_mapping'),
                height_variable=entry.get('height_variable'),
                layer_index=entry.get('layer_index'),
                time_index=entry.get('time_index'),
            )
        # Resolve relative paths
        if source.path and not Path(source.path).is_absolute():
            source.path = str(Path(config_dir) / source.path)
        sources.append(source)
    return sources
