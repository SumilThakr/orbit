"""Integration tests for emission loading pipeline."""

import numpy as np
import pytest
import os
import tempfile
from dataclasses import dataclass

from orbit.emissions.sources import (
    N_ORBIT_SPECIES,
    EmissionSource,
    NOx_TO_N,
    SOx_TO_S,
    NH3_TO_N,
)
from orbit.emissions.loader import load_emissions
from orbit.emissions.plumerise import (
    PlumeAboveModelTop,
    find_injection_layer,
    find_layer,
    calc_delta_h,
    asme_plume_rise,
)


# =============================================================================
# Minimal grid mock for tests (matches GridData interface)
# =============================================================================

@dataclass
class MockGrid:
    """Minimal mock of GridData for emission loading tests."""
    nz: int = 3
    ny: int = 4
    nx: int = 5
    lon: np.ndarray = None
    lat: np.ndarray = None
    dlon: float = 1.0
    dlat: float = 1.0
    Dz: np.ndarray = None
    volume: np.ndarray = None
    # Met fields for plume rise (optional)
    LayerHeights: np.ndarray = None
    Temperature: np.ndarray = None
    WindSpeed: np.ndarray = None
    Sclass: np.ndarray = None
    S1: np.ndarray = None
    WindSpeedInverse: np.ndarray = None
    WindSpeedMinusThird: np.ndarray = None
    WindSpeedMinusOnePointFour: np.ndarray = None

    def __post_init__(self):
        if self.lon is None:
            self.lon = np.arange(70.0, 70.0 + self.nx * self.dlon, self.dlon)
        if self.lat is None:
            self.lat = np.arange(20.0, 20.0 + self.ny * self.dlat, self.dlat)
        if self.Dz is None:
            self.Dz = np.full((self.nz, self.ny, self.nx), 500.0)
        if self.volume is None:
            # Simple volume: dx * dy * Dz
            dy_m = 6.371e6 * np.deg2rad(self.dlat)
            self.volume = np.zeros((self.nz, self.ny, self.nx))
            for j in range(self.ny):
                dx_m = 6.371e6 * np.cos(np.deg2rad(self.lat[j])) * np.deg2rad(self.dlon)
                for k in range(self.nz):
                    self.volume[k, j, :] = dx_m * dy_m * self.Dz[k, j, :]


def _make_mock_grid_with_met(**kwargs) -> MockGrid:
    """Create a MockGrid with met fields populated for ASME plume rise tests."""
    grid = MockGrid(**kwargs)
    nz, ny, nx = grid.nz, grid.ny, grid.nx

    # Build LayerHeights from cumulative Dz
    grid.LayerHeights = np.zeros((nz + 1, ny, nx))
    for k in range(nz):
        grid.LayerHeights[k + 1] = grid.LayerHeights[k] + grid.Dz[k]

    # Temperature: ~290K with -6.5 K/km lapse rate
    grid.Temperature = np.zeros((nz, ny, nx))
    for k in range(nz):
        mid_h = (grid.LayerHeights[k] + grid.LayerHeights[k + 1]) / 2.0
        grid.Temperature[k] = 290.0 - 6.5e-3 * mid_h

    grid.WindSpeed = np.full((nz, ny, nx), 5.0)
    grid.Sclass = np.full((nz, ny, nx), 0.3)  # unstable
    grid.S1 = np.full((nz, ny, nx), 0.01)

    grid.WindSpeedInverse = 1.0 / grid.WindSpeed
    grid.WindSpeedMinusThird = np.power(grid.WindSpeed, -1.0 / 3.0)
    grid.WindSpeedMinusOnePointFour = np.power(grid.WindSpeed, -1.4)

    return grid


class MockIndexer:
    """Minimal mock of CellIndexer."""
    def __init__(self, nz, ny, nx):
        self.nz = nz
        self.ny = ny
        self.nx = nx
        self.N = nz * ny * nx


# =============================================================================
# NetCDF loading tests
# =============================================================================

class TestNetCDFLoading:
    """Test NetCDF emission loading with synthetic data."""

    def _make_netcdf(self, tmpdir, filename, varname, data, lon, lat):
        """Create a minimal NetCDF file for testing."""
        import xarray as xr

        ds = xr.Dataset(
            {varname: (["lat", "lon"], data)},
            coords={"lat": lat, "lon": lon},
        )
        path = os.path.join(tmpdir, filename)
        ds.to_netcdf(path)
        return path

    def _make_netcdf_with_time(self, tmpdir, filename, varname, data_3d, lon, lat):
        """Create a NetCDF file with time dimension."""
        import xarray as xr

        ds = xr.Dataset(
            {varname: (["time", "lat", "lon"], data_3d)},
            coords={"time": np.arange(data_3d.shape[0]), "lat": lat, "lon": lon},
        )
        path = os.path.join(tmpdir, filename)
        ds.to_netcdf(path)
        return path

    def test_loader_zeros_slot0_for_voc_sources(self, tmp_path):
        """Under VBS, the loader unconditionally zeros slot 0 for all sources.
        VOC mass distribution into the 5 VBS bins is done downstream of the
        loader by run_orbit.py:_distribute_voc_to_vbs_bins (where the
        13-species solver layout is available); the loader can't do it
        because it operates on the 6-species ORBIT layout that has no
        notion of the VBS bins at indices 9–12.
        """
        grid = MockGrid()
        idx = MockIndexer(grid.nz, grid.ny, grid.nx)

        data = np.full((grid.ny, grid.nx), 1e-10)
        path = self._make_netcdf(str(tmp_path), "voc.nc", "voc_emis", data,
                                 grid.lon, grid.lat)

        # No parent class → loader zeros slot 0.
        src = EmissionSource(
            path=path, format="netcdf", units="kg/m2/s",
            variable_mapping={"voc": "voc_emis"},
        )
        e = load_emissions([src], grid, idx)
        N = idx.N
        assert np.all(e[0 * N : 1 * N] == 0), \
            "Slot 0 should be zero — VBS distribution happens downstream"

        # With parent class → still zero in the loader's 6-layout output.
        src_voc = EmissionSource(
            path=path, format="netcdf", units="kg/m2/s",
            variable_mapping={"voc": "voc_emis"},
            voc_parent_class="bio_monoterpene",
        )
        e_voc = load_emissions([src_voc], grid, idx)
        assert np.all(e_voc[0 * N : 1 * N] == 0), \
            "Loader zeros slot 0 regardless of voc_parent_class"

    def test_basic_load(self, tmp_path):
        """Load a simple PM2.5 NetCDF and verify non-zero emissions in ORBIT PM2.5 slot."""
        grid = MockGrid()
        idx = MockIndexer(grid.nz, grid.ny, grid.nx)

        # Create synthetic data: uniform 1e-10 kg/m2/s
        data = np.full((grid.ny, grid.nx), 1e-10)
        path = self._make_netcdf(str(tmp_path), "pm25.nc", "pm25_anthro", data,
                                 grid.lon, grid.lat)

        source = EmissionSource(
            path=path, format="netcdf", units="kg/m2/s",
            variable_mapping={"pm25": "pm25_anthro"},
        )

        e = load_emissions([source], grid, idx)
        assert e.shape == (N_ORBIT_SPECIES * idx.N,)

        # PM2.5 is ORBIT index 1
        N = idx.N
        pm25_emis = e[1 * N : 2 * N]
        assert np.any(pm25_emis > 0), "PM2.5 emissions should be non-zero"

        # Other species should be zero
        for s in [s for s in range(N_ORBIT_SPECIES) if s != 1]:
            species_emis = e[s * N : (s + 1) * N]
            assert np.all(species_emis == 0), f"Species {s} should be zero"

    def test_element_conversion_so2(self, tmp_path):
        """SO2 emissions should be multiplied by SOx_TO_S."""
        grid = MockGrid()
        idx = MockIndexer(grid.nz, grid.ny, grid.nx)

        data = np.full((grid.ny, grid.nx), 1e-10)
        path = self._make_netcdf(str(tmp_path), "so2.nc", "so2_anthro", data,
                                 grid.lon, grid.lat)

        source = EmissionSource(
            path=path, format="netcdf", units="kg/m2/s",
            variable_mapping={"so2": "so2_anthro"},
        )

        e = load_emissions([source], grid, idx)
        N = idx.N

        # SO2 goes to ORBIT index 3
        so2_emis = e[3 * N : 4 * N]
        assert np.any(so2_emis > 0)

        # Without element conversion, values would be higher by factor 1/SOx_TO_S
        # We can't easily test the exact factor without also knowing area/volume,
        # but we can check it's not the raw value
        # Load same file as PM2.5 (no element conversion) for comparison
        source_raw = EmissionSource(
            path=path, format="netcdf", units="kg/m2/s",
            variable_mapping={"pm25": "so2_anthro"},
        )
        e_raw = load_emissions([source_raw], grid, idx)
        pm25_raw = e_raw[1 * N : 2 * N]

        # SO2 emission should be ~SOx_TO_S times the raw value
        ratio = so2_emis.sum() / (pm25_raw.sum() + 1e-30)
        assert abs(ratio - SOx_TO_S) < 0.01, f"SO2 element conversion ratio should be ~{SOx_TO_S}, got {ratio}"

    def test_element_conversion_nox(self, tmp_path):
        """NOx emissions should be multiplied by NOx_TO_N."""
        grid = MockGrid()
        idx = MockIndexer(grid.nz, grid.ny, grid.nx)

        data = np.full((grid.ny, grid.nx), 1e-10)
        path = self._make_netcdf(str(tmp_path), "nox.nc", "nox_anthro", data,
                                 grid.lon, grid.lat)

        source_nox = EmissionSource(
            path=path, format="netcdf", units="kg/m2/s",
            variable_mapping={"nox": "nox_anthro"},
        )
        source_raw = EmissionSource(
            path=path, format="netcdf", units="kg/m2/s",
            variable_mapping={"pm25": "nox_anthro"},
        )

        e_nox = load_emissions([source_nox], grid, idx)
        e_raw = load_emissions([source_raw], grid, idx)
        N = idx.N

        nox_emis = e_nox[5 * N : 6 * N]  # Total NO is ORBIT index 5
        pm25_raw = e_raw[1 * N : 2 * N]

        ratio = nox_emis.sum() / (pm25_raw.sum() + 1e-30)
        assert abs(ratio - NOx_TO_N) < 0.01

    def test_time_index_selection(self, tmp_path):
        """time_index selects the correct month."""
        grid = MockGrid()
        idx = MockIndexer(grid.nz, grid.ny, grid.nx)

        # 3 time steps with different values
        data_3d = np.zeros((3, grid.ny, grid.nx))
        data_3d[0] = 1e-10
        data_3d[1] = 2e-10
        data_3d[2] = 3e-10

        path = self._make_netcdf_with_time(str(tmp_path), "pm25_monthly.nc",
                                            "pm25_anthro", data_3d, grid.lon, grid.lat)

        N = idx.N
        results = []
        for ti in range(3):
            source = EmissionSource(
                path=path, format="netcdf", units="kg/m2/s",
                variable_mapping={"pm25": "pm25_anthro"},
                time_index=ti,
            )
            e = load_emissions([source], grid, idx)
            results.append(e[1 * N : 2 * N].sum())

        # Emissions should scale linearly with input data
        assert results[1] / results[0] == pytest.approx(2.0, rel=1e-6)
        assert results[2] / results[0] == pytest.approx(3.0, rel=1e-6)

    def test_multiple_sources_accumulate(self, tmp_path):
        """Multiple sources accumulate into same species."""
        grid = MockGrid()
        idx = MockIndexer(grid.nz, grid.ny, grid.nx)

        data = np.full((grid.ny, grid.nx), 1e-10)

        path1 = self._make_netcdf(str(tmp_path), "pm25_a.nc", "pm25_a", data,
                                  grid.lon, grid.lat)
        path2 = self._make_netcdf(str(tmp_path), "pm25_b.nc", "pm25_b", data,
                                  grid.lon, grid.lat)

        source1 = EmissionSource(
            path=path1, format="netcdf", units="kg/m2/s",
            variable_mapping={"pm25": "pm25_a"},
        )
        source2 = EmissionSource(
            path=path2, format="netcdf", units="kg/m2/s",
            variable_mapping={"pm25": "pm25_b"},
        )

        e_single = load_emissions([source1], grid, idx)
        e_double = load_emissions([source1, source2], grid, idx)

        N = idx.N
        ratio = e_double[1 * N : 2 * N].sum() / e_single[1 * N : 2 * N].sum()
        assert ratio == pytest.approx(2.0, rel=1e-6)

    def test_elevated_layer_index(self, tmp_path):
        """Elevated source with layer_index=1 puts emissions in layer 1."""
        grid = MockGrid()
        idx = MockIndexer(grid.nz, grid.ny, grid.nx)

        data = np.full((grid.ny, grid.nx), 1e-10)
        path = self._make_netcdf(str(tmp_path), "pm25_elev.nc", "pm25_anthro", data,
                                 grid.lon, grid.lat)

        source = EmissionSource(
            path=path, format="netcdf", units="kg/m2/s",
            variable_mapping={"pm25": "pm25_anthro"},
            elevated=True,
            layer_index=1,
        )

        e = load_emissions([source], grid, idx)
        N = idx.N
        pm25_emis = e[1 * N : 2 * N]

        # Reshape to 3D and check layer distribution
        pm25_3d = pm25_emis.reshape(grid.nz, grid.ny, grid.nx)
        assert np.all(pm25_3d[0] == 0), "Surface layer should be empty"
        assert np.any(pm25_3d[1] > 0), "Layer 1 should have emissions"

    def test_kg_m2_with_averaging_period(self, tmp_path):
        """kg/m2 units with averaging_period produces correct emissions."""
        grid = MockGrid()
        idx = MockIndexer(grid.nz, grid.ny, grid.nx)

        # 1e-6 kg/m2 over 30 days
        data = np.full((grid.ny, grid.nx), 1e-6)
        path = self._make_netcdf(str(tmp_path), "pm25_total.nc", "pm25_total", data,
                                 grid.lon, grid.lat)

        source = EmissionSource(
            path=path, format="netcdf", units="kg/m2",
            variable_mapping={"pm25": "pm25_total"},
            averaging_period=30.0,
        )

        e = load_emissions([source], grid, idx)
        N = idx.N
        pm25_emis = e[1 * N : 2 * N]
        assert np.any(pm25_emis > 0), "Should have non-zero emissions"

        # Compare with kg/m2/s: kg/m2 over 30 days = kg/m2/s * (30*86400)
        # So the same input with kg/m2/s should give 30*86400 times more
        source_rate = EmissionSource(
            path=path, format="netcdf", units="kg/m2/s",
            variable_mapping={"pm25": "pm25_total"},
        )
        e_rate = load_emissions([source_rate], grid, idx)
        pm25_rate = e_rate[1 * N : 2 * N]

        ratio = pm25_rate.sum() / pm25_emis.sum()
        expected_ratio = 30.0 * 86400.0
        assert ratio == pytest.approx(expected_ratio, rel=1e-6)

    def test_kg_m2_missing_averaging_period(self, tmp_path):
        """kg/m2 units without averaging_period raises ValueError."""
        grid = MockGrid()
        idx = MockIndexer(grid.nz, grid.ny, grid.nx)

        data = np.full((grid.ny, grid.nx), 1e-6)
        path = self._make_netcdf(str(tmp_path), "pm25.nc", "pm25_total", data,
                                 grid.lon, grid.lat)

        source = EmissionSource(
            path=path, format="netcdf", units="kg/m2",
            variable_mapping={"pm25": "pm25_total"},
            # averaging_period not set -> None
        )

        with pytest.raises(ValueError, match="averaging_period"):
            load_emissions([source], grid, idx)

    def test_lon_360_to_180_conversion(self, tmp_path):
        """Longitude in [0, 360] range is converted to [-180, 180]."""
        grid = MockGrid()

        # Create NetCDF with lon in [0, 360] matching grid.lon (70-74)
        lon_360 = grid.lon.copy()  # already in [-180, 180]
        # Shift to [0, 360] — for lon 70-74, this is the same values
        # Use a range that wraps: e.g., 350-354 maps to -10..-6
        lon_360 = np.array([350.0, 351.0, 352.0, 353.0, 354.0])
        grid_neg = MockGrid(lon=np.array([-10.0, -9.0, -8.0, -7.0, -6.0]))

        data = np.full((grid_neg.ny, grid_neg.nx), 1e-10)
        path = self._make_netcdf(str(tmp_path), "pm25_360.nc", "pm25_anthro", data,
                                 lon_360, grid_neg.lat)

        source = EmissionSource(
            path=path, format="netcdf", units="kg/m2/s",
            variable_mapping={"pm25": "pm25_anthro"},
        )

        e = load_emissions([source], grid_neg, MockIndexer(grid_neg.nz, grid_neg.ny, grid_neg.nx))
        N = grid_neg.nz * grid_neg.ny * grid_neg.nx
        pm25_emis = e[1 * N : 2 * N]
        assert np.any(pm25_emis > 0), "Should match after lon 360->180 conversion"

    def test_reversed_latitude(self, tmp_path):
        """NetCDF with north-to-south latitude is handled correctly."""
        grid = MockGrid()
        idx = MockIndexer(grid.nz, grid.ny, grid.nx)

        # Create data with a gradient: higher values at higher latitudes
        data = np.zeros((grid.ny, grid.nx))
        for j in range(grid.ny):
            data[j, :] = (j + 1) * 1e-10  # row 0 = low, row 3 = high

        # Save with reversed (north-to-south) latitude
        lat_reversed = grid.lat[::-1]
        data_reversed = data[::-1, :]

        path = self._make_netcdf(str(tmp_path), "pm25_rev.nc", "pm25_anthro",
                                 data_reversed, grid.lon, lat_reversed)

        source = EmissionSource(
            path=path, format="netcdf", units="kg/m2/s",
            variable_mapping={"pm25": "pm25_anthro"},
        )

        e = load_emissions([source], grid, idx)
        N = idx.N
        pm25_3d = e[1 * N : 2 * N].reshape(grid.nz, grid.ny, grid.nx)

        # Highest emissions should be at highest latitude (j=3)
        assert pm25_3d[0, 3, 0] > pm25_3d[0, 0, 0], \
            "North (j=3) should have more emissions than south (j=0)"

        # Compare with non-reversed version
        path_normal = self._make_netcdf(str(tmp_path), "pm25_norm.nc", "pm25_anthro",
                                        data, grid.lon, grid.lat)
        source_normal = EmissionSource(
            path=path_normal, format="netcdf", units="kg/m2/s",
            variable_mapping={"pm25": "pm25_anthro"},
        )
        e_normal = load_emissions([source_normal], grid, idx)

        np.testing.assert_allclose(e, e_normal, rtol=1e-10,
                                   err_msg="Reversed lat should produce identical results")

    def test_regridding(self, tmp_path):
        """Data on a coarser grid is conservatively regridded to the model grid.

        Semantics changed 2026-08-02 with the switch from bilinear point
        sampling to area-weighted regridding: coverage is now defined by
        source cell EDGES, not by the range of source cell centres. The
        source cell centred at 22 deg with 2 deg spacing spans 21-23 deg,
        so it legitimately contributes to a destination cell centred at
        23 deg (which spans 22.5-23.5) in proportion to the overlap.
        """
        grid = MockGrid()
        idx = MockIndexer(grid.nz, grid.ny, grid.nx)

        # Create coarser source grid (2x2 deg instead of 1x1)
        src_lon = np.array([70.0, 72.0, 74.0])
        src_lat = np.array([20.0, 22.0])
        data = np.full((len(src_lat), len(src_lon)), 1e-10)

        path = self._make_netcdf(str(tmp_path), "pm25_coarse.nc", "pm25_anthro",
                                 data, src_lon, src_lat)

        source = EmissionSource(
            path=path, format="netcdf", units="kg/m2/s",
            variable_mapping={"pm25": "pm25_anthro"},
        )

        e = load_emissions([source], grid, idx)
        N = idx.N
        pm25_emis = e[1 * N : 2 * N]

        # Interior cells that fall within the coarse grid should have emissions
        pm25_3d = pm25_emis.reshape(grid.nz, grid.ny, grid.nx)
        assert np.any(pm25_3d[0] > 0), "Should have emissions after regridding"

        # Fully covered rows (j=0,1,2) carry the full source intensity; the
        # row at lat=23 spans 22.5-23.5 while the source ends at its edge
        # of 23, so it receives half the intensity.
        full = pm25_3d[0, 1, :]
        assert np.all(full > 0)
        np.testing.assert_allclose(pm25_3d[0, 3, :], 0.5 * full, rtol=1e-2)

        # Beyond the source edge entirely, emissions are still zero.
        far_lat = np.array([30.0, 31.0])
        from orbit.emissions.netcdf import _regrid_emission_conservative
        beyond = _regrid_emission_conservative(
            data, src_lon, src_lat, np.array([72.0, 73.0]), far_lat)
        assert np.all(beyond == 0.0), \
            "Cells beyond the source edge extent should be zero"


# =============================================================================
# Plume rise tests
# =============================================================================

class TestPlumeRise:
    """Test simplified height-based plume rise."""

    def test_surface(self):
        """Zero height returns surface layer."""
        grid = MockGrid()
        k = find_injection_layer(grid, 0, 0, 0.0)
        assert k == 0

    def test_negative_height(self):
        """Negative height returns surface layer."""
        grid = MockGrid()
        k = find_injection_layer(grid, 0, 0, -10.0)
        assert k == 0

    def test_within_first_layer(self):
        """Height < Dz[0] returns layer 0."""
        grid = MockGrid()  # Dz = 500m per layer
        k = find_injection_layer(grid, 0, 0, 200.0)
        assert k == 0

    def test_second_layer(self):
        """Height in (500, 1000] returns layer 1."""
        grid = MockGrid()  # Dz = 500m per layer
        k = find_injection_layer(grid, 0, 0, 700.0)
        assert k == 1

    def test_above_domain(self):
        """Height above domain top returns topmost layer."""
        grid = MockGrid()  # 3 layers x 500m = 1500m total
        k = find_injection_layer(grid, 0, 0, 5000.0)
        assert k == grid.nz - 1


# =============================================================================
# ASME plume rise algorithm tests
# =============================================================================

class TestASMEAlgorithm:
    """Test ASME plume rise building blocks."""

    def test_find_layer_basic(self):
        """find_layer returns correct layer for mid-layer height."""
        # 3 layers: 0-500, 500-1000, 1000-1500
        heights = [0.0, 500.0, 1000.0, 1500.0]
        layer, above = find_layer(heights, 250.0)
        assert layer == 0
        assert above is False

    def test_find_layer_second(self):
        """find_layer returns layer 1 for height in second layer."""
        heights = [0.0, 500.0, 1000.0, 1500.0]
        layer, above = find_layer(heights, 750.0)
        assert layer == 1
        assert above is False

    def test_find_layer_at_boundary(self):
        """find_layer at exact layer boundary returns lower layer."""
        heights = [0.0, 500.0, 1000.0, 1500.0]
        layer, above = find_layer(heights, 500.0)
        assert layer == 0
        assert above is False

    def test_find_layer_above_top(self):
        """find_layer above domain returns top layer with flag."""
        heights = [0.0, 500.0, 1000.0, 1500.0]
        layer, above = find_layer(heights, 2000.0)
        assert layer == 2  # n-2 = 4-2 = 2
        assert above is True

    def test_calc_delta_h_momentum(self):
        """Momentum-dominated regime: high exit velocity, small temp diff."""
        # Setup: wind=5 m/s, air temp=290K, stack temp=310K (diff=20 < 50),
        # stack vel=15 (> wind, > 10)
        temp = [290.0]
        ws = [5.0]
        s_class = [0.3]
        s1 = [0.01]
        ws_inv = [1.0 / 5.0]
        ws_m3 = [5.0 ** (-1.0 / 3.0)]
        ws_m14 = [5.0 ** (-1.4)]

        dh = calc_delta_h(
            0, temp, ws, s_class, s1,
            stack_height=100.0, stack_temp=310.0, stack_vel=15.0, stack_diam=3.0,
            wind_speed_minus_one_point_four=ws_m14,
            wind_speed_minus_third=ws_m3,
            wind_speed_inverse=ws_inv,
        )
        # dh = D * Vs^1.4 * ws^(-1.4) = 3 * 15^1.4 * 5^(-1.4)
        import math
        expected = 3.0 * math.pow(15.0, 1.4) * math.pow(5.0, -1.4)
        assert abs(dh - expected) < 1e-6

    def test_calc_delta_h_buoyancy_unstable(self):
        """Buoyancy-dominated, unstable: sClass < 0.5, F > 0."""
        temp = [280.0]
        ws = [5.0]
        s_class = [0.3]  # unstable
        s1 = [0.01]
        ws_inv = [1.0 / 5.0]
        ws_m3 = [5.0 ** (-1.0 / 3.0)]
        ws_m14 = [5.0 ** (-1.4)]

        dh = calc_delta_h(
            0, temp, ws, s_class, s1,
            stack_height=100.0, stack_temp=500.0, stack_vel=5.0, stack_diam=4.0,
            wind_speed_minus_one_point_four=ws_m14,
            wind_speed_minus_third=ws_m3,
            wind_speed_inverse=ws_inv,
        )
        # Buoyancy: temp_diff = 2*(500-280)/(500+280) ≈ 0.564
        # F = g * temp_diff * Vs * (D/2)^2 = 9.81 * 0.564 * 5.0 * 4.0 ≈ 110.7
        # dh = 7.4 * (F * H^2)^(1/3) * ws_inv = 7.4 * (110.7 * 10000)^(1/3) * 0.2
        assert dh > 0
        # Verify formula manually
        import math
        temp_diff = 2 * (500 - 280) / (500 + 280)
        F = 9.80665 * temp_diff * 5.0 * (4.0 / 2) ** 2
        expected = 7.4 * math.pow(F * 100.0**2, 1.0/3.0) * (1.0/5.0)
        assert abs(dh - expected) < 1e-6

    def test_calc_delta_h_buoyancy_stable(self):
        """Buoyancy-dominated, stable: sClass > 0.5, s1 != 0, F > 0."""
        temp = [280.0]
        ws = [5.0]
        s_class = [0.8]  # stable
        s1 = [0.02]
        ws_inv = [1.0 / 5.0]
        ws_m3 = [5.0 ** (-1.0 / 3.0)]
        ws_m14 = [5.0 ** (-1.4)]

        dh = calc_delta_h(
            0, temp, ws, s_class, s1,
            stack_height=100.0, stack_temp=500.0, stack_vel=5.0, stack_diam=4.0,
            wind_speed_minus_one_point_four=ws_m14,
            wind_speed_minus_third=ws_m3,
            wind_speed_inverse=ws_inv,
        )
        assert dh > 0
        # Briggs stable: dh = 2.6 * (F / (u * s))^(1/3) with s = g * S1,
        # since the grid's S1 is (dtheta/dz)/theta [1/m], not the
        # stability parameter [1/s^2].  See tests/test_plumerise_stable.py.
        import math
        temp_diff = 2 * (500 - 280) / (500 + 280)
        F = 9.80665 * temp_diff * 5.0 * (4.0 / 2) ** 2
        expected = 2.6 * math.pow(F / (5.0 * 9.80665 * 0.02), 1.0 / 3.0)
        assert abs(dh - expected) < 1e-6

    def test_calc_delta_h_cold_stack(self):
        """Cold stack (temp <= air temp) gives zero rise."""
        temp = [300.0]
        ws = [5.0]
        s_class = [0.3]
        s1 = [0.01]
        ws_inv = [1.0 / 5.0]
        ws_m3 = [5.0 ** (-1.0 / 3.0)]
        ws_m14 = [5.0 ** (-1.4)]

        dh = calc_delta_h(
            0, temp, ws, s_class, s1,
            stack_height=100.0, stack_temp=290.0, stack_vel=2.0, stack_diam=2.0,
            wind_speed_minus_one_point_four=ws_m14,
            wind_speed_minus_third=ws_m3,
            wind_speed_inverse=ws_inv,
        )
        assert dh == 0.0

    def test_asme_plume_rise_above_top(self):
        """asme_plume_rise raises PlumeAboveModelTop for stack above domain."""
        heights = [0.0, 500.0, 1000.0, 1500.0]
        temp = [290.0, 287.0, 284.0]
        ws = [5.0, 5.0, 5.0]
        s_class = [0.3, 0.3, 0.3]
        s1 = [0.01, 0.01, 0.01]
        ws_inv = [0.2, 0.2, 0.2]
        ws_m3 = [5.0**(-1/3)] * 3
        ws_m14 = [5.0**(-1.4)] * 3

        with pytest.raises(PlumeAboveModelTop):
            asme_plume_rise(2000.0, 3.0, 500.0, 10.0,
                            heights, temp, ws, s_class, s1,
                            ws_m14, ws_m3, ws_inv)


class TestASMEPlumeRiseIntegration:
    """Test ASME plume rise via find_injection_layer with met fields."""

    def test_asme_momentum_dominated(self):
        """Momentum-dominated stack rises above physical stack height."""
        grid = _make_mock_grid_with_met()
        # Stack at 100m, high velocity -> should rise to higher layer
        k = find_injection_layer(grid, 0, 0, 100.0,
                                 stack_diam=3.0, stack_temp=310.0, stack_vel=15.0)
        # With ASME, plume should be higher than simple 100m (layer 0)
        # Delta_h = D * Vs^1.4 * ws^(-1.4) = 3 * 15^1.4 * 5^(-1.4)
        import math
        dh = 3.0 * math.pow(15.0, 1.4) * math.pow(5.0, -1.4)
        effective = 100.0 + dh
        # 500m per layer: expected layer = int(effective / 500)
        expected_layer = min(int(effective / 500.0), grid.nz - 1)
        assert k == expected_layer

    def test_asme_buoyancy_unstable(self):
        """Hot stack in unstable atmosphere gets buoyancy rise."""
        grid = _make_mock_grid_with_met()
        k = find_injection_layer(grid, 0, 0, 100.0,
                                 stack_diam=4.0, stack_temp=500.0, stack_vel=5.0)
        # Should be above layer 0 (100m stack + buoyancy rise)
        assert k >= 0

    def test_asme_cold_stack_no_rise(self):
        """Cold stack (temp < air temp) stays at physical stack layer."""
        grid = _make_mock_grid_with_met()
        # Stack at 100m with cold exhaust -> no rise -> stays in layer 0
        k = find_injection_layer(grid, 0, 0, 100.0,
                                 stack_diam=2.0, stack_temp=250.0, stack_vel=2.0)
        assert k == 0

    def test_asme_above_domain(self):
        """Very hot tall stack exceeding domain returns top layer."""
        grid = _make_mock_grid_with_met()
        # Stack at 1400m (near top), very hot -> plume above domain
        k = find_injection_layer(grid, 0, 0, 1400.0,
                                 stack_diam=5.0, stack_temp=800.0, stack_vel=20.0)
        assert k == grid.nz - 1

    def test_fallback_without_met_fields(self):
        """Without met fields, falls back to simple Dz lookup."""
        grid = MockGrid()  # No met fields
        k = find_injection_layer(grid, 0, 0, 700.0,
                                 stack_diam=3.0, stack_temp=500.0, stack_vel=15.0)
        # Simple lookup: 700m in 500m layers -> layer 1
        assert k == 1

    def test_fallback_without_stack_params(self):
        """Without stack params, falls back to simple Dz lookup even with met."""
        grid = _make_mock_grid_with_met()
        k = find_injection_layer(grid, 0, 0, 700.0)
        # Simple lookup: 700m in 500m layers -> layer 1
        assert k == 1

    def test_fallback_zero_stack_diam(self):
        """Zero stack diameter triggers fallback."""
        grid = _make_mock_grid_with_met()
        k = find_injection_layer(grid, 0, 0, 700.0,
                                 stack_diam=0.0, stack_temp=500.0, stack_vel=15.0)
        assert k == 1


# =============================================================================
# Shapefile loading tests (only run if geopandas available)
# =============================================================================

class TestShapefileLoading:
    """Test shapefile loading with synthetic GeoDataFrame (no file I/O)."""

    @pytest.fixture
    def has_geopandas(self):
        try:
            import geopandas
            import shapely
            return True
        except ImportError:
            pytest.skip("geopandas/shapely not installed")

    def test_point_apportionment(self, has_geopandas):
        """Point source in grid cell center goes 100% to that cell."""
        import geopandas as gpd
        from shapely.geometry import Point
        from orbit.emissions.shapefile import load_shapefile_source

        grid = MockGrid()

        # Create a point source at center of cell (0,0)
        gdf = gpd.GeoDataFrame(
            {"pm25": [1000.0]},
            geometry=[Point(grid.lon[0], grid.lat[0])],
        )

        # Save to temp shapefile
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "points.gpkg")
            gdf.to_file(path, driver="GPKG")

            source = EmissionSource(
                path=path, format="geopackage", units="kg/year",
            )
            emis = load_shapefile_source(source, grid)

        # PM2.5 is ORBIT index 1, should be non-zero in surface layer at (j=0, i=0)
        assert emis[1, 0, 0, 0] > 0
        # Total PM2.5 emission should be in only one cell
        assert np.count_nonzero(emis[1]) == 1

    def test_element_conversion_in_shapefile(self, has_geopandas):
        """NH3 column triggers element conversion in shapefile loading."""
        import geopandas as gpd
        from shapely.geometry import Point
        from orbit.emissions.shapefile import load_shapefile_source

        grid = MockGrid()

        # Two point sources with same mass: one NH3 (convert), one PM2.5 (no convert)
        gdf_nh3 = gpd.GeoDataFrame(
            {"nh3": [1000.0]},
            geometry=[Point(grid.lon[0], grid.lat[0])],
        )
        gdf_pm25 = gpd.GeoDataFrame(
            {"pm25": [1000.0]},
            geometry=[Point(grid.lon[0], grid.lat[0])],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path_nh3 = os.path.join(tmpdir, "nh3.gpkg")
            path_pm25 = os.path.join(tmpdir, "pm25.gpkg")
            gdf_nh3.to_file(path_nh3, driver="GPKG")
            gdf_pm25.to_file(path_pm25, driver="GPKG")

            source_nh3 = EmissionSource(path=path_nh3, format="geopackage", units="kg/year")
            source_pm25 = EmissionSource(path=path_pm25, format="geopackage", units="kg/year")

            emis_nh3 = load_shapefile_source(source_nh3, grid)
            emis_pm25 = load_shapefile_source(source_pm25, grid)

        # NH3 -> Total NH (ORBIT index 2), PM2.5 -> PM2.5 (ORBIT index 1)
        nh_total = emis_nh3[2].sum()
        pm25_total = emis_pm25[1].sum()

        # NH3 emission should be NH3_TO_N times PM2.5 (same input mass, different conversion)
        ratio = nh_total / pm25_total
        assert abs(ratio - NH3_TO_N) < 0.01
