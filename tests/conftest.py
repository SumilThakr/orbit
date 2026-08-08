"""pytest fixtures for ORBIT tests."""

import numpy as np
import pytest
import os

# Paths to real input data, used only by tests marked ``slow``. They are not
# part of the release; point ORBIT_TEST_DATA_DIR at a directory holding the
# published input archive to run those tests.
_TEST_DATA_DIR = os.environ.get("ORBIT_TEST_DATA_DIR", "/path/to/data")

SAS_PREPROCESSOR = os.path.join(_TEST_DATA_DIR, "COMBINED", "sas_2022_01.nc")
CONSTANTS_PATH = os.path.join(_TEST_DATA_DIR, "MERRA2.20150101.CN.05x0625.nc4")
SOLAR_MAP_PATH = os.path.join(_TEST_DATA_DIR, "EMISSIONS", "SAS", "solar_map_2016.npz")
WEEKLY_SUBSET_BASE = os.path.join(_TEST_DATA_DIR, "SUBSET", "SAS", "2016")


@pytest.fixture
def small_grid_params():
    """Parameters for a synthetic 4x4x3 grid with known values."""
    nz, ny, nx = 3, 4, 4
    N = nz * ny * nx  # 48 cells

    # Coordinates
    lon = np.array([60.0, 61.0, 62.0, 63.0])
    lat = np.array([10.0, 11.0, 12.0, 13.0])
    dlon = 1.0
    dlat = 1.0

    # Sigma coordinates (3 layers + 1 edge = 4 edges)
    Ap = np.array([0.0, 5000.0, 20000.0, 60000.0])  # Pa
    Bp = np.array([1.0, 0.90, 0.70, 0.30])

    # Surface pressure: uniform 101325 Pa (flat terrain)
    Psurf = np.full((ny, nx), 101325.0)

    # Compute dP
    dP = np.zeros((nz, ny, nx))
    for k in range(nz):
        P_bot = Ap[k] + Bp[k] * Psurf
        P_top = Ap[k + 1] + Bp[k + 1] * Psurf
        dP[k] = P_bot - P_top

    # Layer thickness (uniform 500m)
    Dz = np.full((nz, ny, nx), 500.0)

    # Uniform eastward wind (2 m/s)
    UAvg = np.full((nz, ny, nx), 2.0)  # west-face velocity
    VAvg = np.full((nz, ny, nx), 0.0)  # south-face velocity
    omega = np.zeros((nz, ny, nx))  # no vertical motion

    # Diffusivities
    Kzz = np.full((nz, ny, nx), 10.0)  # m^2/s
    Kxxyy = np.full((nz, ny, nx), 100.0)  # m^2/s
    K_meander_u = np.full((nz, ny, nx), 50.0)
    K_meander_v = np.full((nz, ny, nx), 50.0)

    # Deposition velocities (m/s for dry, 1/s for wet)
    particle_dry_dep = np.full((nz, ny, nx), 0.001)
    SO2_dry_dep = np.full((nz, ny, nx), 0.005)
    NOx_dry_dep = np.full((nz, ny, nx), 0.004)
    NH3_dry_dep = np.full((nz, ny, nx), 0.003)
    VOC_dry_dep = np.full((nz, ny, nx), 0.002)

    particle_wet_dep = np.full((nz, ny, nx), 1e-5)
    SO2_wet_dep = np.full((nz, ny, nx), 2e-5)
    other_gas_wet_dep = np.full((nz, ny, nx), 5e-6)

    # Chemistry
    SO2oxidation = np.full((nz, ny, nx), 1e-5)  # 1/s
    NHPartitioning = np.full((nz, ny, nx), 0.4)  # 40% particle
    NOPartitioning = np.full((nz, ny, nx), 0.3)  # 30% particle
    AOrgPartitioning = np.full((nz, ny, nx), 0.5)  # 50% particle

    # Land mask (all land)
    is_land = np.ones((ny, nx), dtype=np.uint8)

    # Meteorological fields for plume rise
    # LayerHeights: staggered (nz+1, ny, nx), built from cumulative Dz
    LayerHeights = np.zeros((nz + 1, ny, nx))
    for k in range(nz):
        LayerHeights[k + 1] = LayerHeights[k] + Dz[k]

    # Temperature: ~290K at surface with -6.5 K/km lapse rate
    Temperature = np.zeros((nz, ny, nx))
    for k in range(nz):
        mid_height = (LayerHeights[k] + LayerHeights[k + 1]) / 2.0
        Temperature[k] = 290.0 - 6.5e-3 * mid_height

    WindSpeed = np.full((nz, ny, nx), 5.0)  # 5 m/s
    Sclass = np.full((nz, ny, nx), 0.3)  # unstable
    S1 = np.full((nz, ny, nx), 0.01)  # stability parameter

    # Precomputed wind speed powers
    WindSpeedInverse = 1.0 / WindSpeed
    WindSpeedMinusThird = np.power(WindSpeed, -1.0 / 3.0)
    WindSpeedMinusOnePointFour = np.power(WindSpeed, -1.4)

    return {
        "nz": nz, "ny": ny, "nx": nx, "N": N,
        "lon": lon, "lat": lat, "dlon": dlon, "dlat": dlat,
        "Ap": Ap, "Bp": Bp, "Psurf": Psurf, "dP": dP, "Dz": Dz,
        "UAvg": UAvg, "VAvg": VAvg, "omega": omega,
        "Kzz": Kzz, "Kxxyy": Kxxyy,
        "K_meander_u": K_meander_u, "K_meander_v": K_meander_v,
        "particle_dry_dep": particle_dry_dep,
        "SO2_dry_dep": SO2_dry_dep,
        "NOx_dry_dep": NOx_dry_dep,
        "NH3_dry_dep": NH3_dry_dep,
        "VOC_dry_dep": VOC_dry_dep,
        "particle_wet_dep": particle_wet_dep,
        "SO2_wet_dep": SO2_wet_dep,
        "other_gas_wet_dep": other_gas_wet_dep,
        "SO2oxidation": SO2oxidation,
        "NHPartitioning": NHPartitioning,
        "NOPartitioning": NOPartitioning,
        "AOrgPartitioning": AOrgPartitioning,
        "is_land": is_land,
        "LayerHeights": LayerHeights,
        "Temperature": Temperature,
        "WindSpeed": WindSpeed,
        "Sclass": Sclass,
        "S1": S1,
        "WindSpeedInverse": WindSpeedInverse,
        "WindSpeedMinusThird": WindSpeedMinusThird,
        "WindSpeedMinusOnePointFour": WindSpeedMinusOnePointFour,
    }


@pytest.fixture
def january_preprocessor_path():
    """Path to real January preprocessor file (for slow tests)."""
    return os.path.abspath(SAS_PREPROCESSOR)


@pytest.fixture
def constants_path():
    """Path to MERRA-2 constants file."""
    return os.path.abspath(CONSTANTS_PATH)
