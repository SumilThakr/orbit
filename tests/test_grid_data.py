"""Tests for GridData loading and geometry computation."""

import numpy as np
import pytest
import os


class TestGridDataSynthetic:
    def test_geometry_dx_varies_with_lat(self, small_grid_params):
        from orbit.core.grid_data import GridData, _compute_geometry, EARTH_RADIUS, DEG_TO_RAD

        g = GridData()
        g.nz = small_grid_params["nz"]
        g.ny = small_grid_params["ny"]
        g.nx = small_grid_params["nx"]
        g.lon = small_grid_params["lon"]
        g.lat = small_grid_params["lat"]
        g.dlon = small_grid_params["dlon"]
        g.dlat = small_grid_params["dlat"]
        g.Dz = small_grid_params["Dz"]
        _compute_geometry(g)

        # dx should decrease with latitude (cos decreases)
        assert g.dx[0] > g.dx[-1] or np.isclose(g.dx[0], g.dx[-1], rtol=0.01)
        # dy should be constant
        assert g.dy > 0
        # Check dx formula
        expected_dx_0 = EARTH_RADIUS * np.cos(g.lat[0] * DEG_TO_RAD) * g.dlon * DEG_TO_RAD
        np.testing.assert_allclose(g.dx[0], expected_dx_0)

    def test_terrain_ratios_flat(self, small_grid_params):
        """Flat terrain -> all ratios = 1.0."""
        from orbit.core.grid_data import GridData, _compute_terrain_ratios

        g = GridData()
        g.nz = small_grid_params["nz"]
        g.ny = small_grid_params["ny"]
        g.nx = small_grid_params["nx"]
        g.dP = small_grid_params["dP"]
        _compute_terrain_ratios(g)

        # Flat terrain: Psurf uniform -> dP uniform -> ratios = 1.0
        np.testing.assert_allclose(g.dP_ratio_west, 1.0, atol=1e-12)
        np.testing.assert_allclose(g.dP_ratio_south, 1.0, atol=1e-12)
        np.testing.assert_allclose(g.dP_ratio_east, 1.0, atol=1e-12)
        np.testing.assert_allclose(g.dP_ratio_north, 1.0, atol=1e-12)

    def test_terrain_ratios_reciprocal(self, small_grid_params):
        """East/north ratios are reciprocals of west/south at interior faces."""
        from orbit.core.grid_data import GridData, _compute_terrain_ratios

        g = GridData()
        g.nz = small_grid_params["nz"]
        g.ny = small_grid_params["ny"]
        g.nx = small_grid_params["nx"]

        # Non-uniform dP to test reciprocal property
        dP = small_grid_params["dP"].copy()
        # Vary Psurf to create terrain variation
        Psurf = small_grid_params["Psurf"].copy()
        Psurf[0, 0] = 90000.0  # lower pressure = higher terrain
        Psurf[1, 1] = 95000.0
        Ap = small_grid_params["Ap"]
        Bp = small_grid_params["Bp"]
        nz = small_grid_params["nz"]
        ny = small_grid_params["ny"]
        nx = small_grid_params["nx"]
        dP = np.zeros((nz, ny, nx))
        for k in range(nz):
            P_bot = Ap[k] + Bp[k] * Psurf
            P_top = Ap[k + 1] + Bp[k + 1] * Psurf
            dP[k] = P_bot - P_top
        g.dP = dP
        _compute_terrain_ratios(g)

        assert g.dP_ratio_east.shape == (nz, ny, nx)
        assert g.dP_ratio_north.shape == (nz, ny, nx)

        # Check reciprocal: dP_ratio_east[k,j,i] = 1/dP_ratio_west[k,j,i+1]
        for k in range(nz):
            for j in range(ny):
                for i in range(nx - 1):
                    east = g.dP_ratio_east[k, j, i]
                    west_next = g.dP_ratio_west[k, j, i + 1]
                    np.testing.assert_allclose(east, 1.0 / west_next, rtol=1e-12)
        # Check reciprocal: dP_ratio_north[k,j,i] = 1/dP_ratio_south[k,j+1,i]
        for k in range(nz):
            for j in range(ny - 1):
                for i in range(nx):
                    north = g.dP_ratio_north[k, j, i]
                    south_next = g.dP_ratio_south[k, j + 1, i]
                    np.testing.assert_allclose(north, 1.0 / south_next, rtol=1e-12)


class TestLoadGridFallbacks:
    """Test load_grid with missing optional fields (Issue #8)."""

    def test_no_constants_path(self, january_preprocessor_path):
        """load_grid without constants_path should treat all cells as land."""
        if not os.path.exists(january_preprocessor_path):
            pytest.skip("Preprocessor file not found")

        from orbit.core.grid_data import load_grid
        g = load_grid(january_preprocessor_path, constants_path=None)

        assert g.is_land.shape == (g.ny, g.nx)
        assert np.all(g.is_land == 1)

    def test_split_flux_fallback(self, january_preprocessor_path, constants_path):
        """When preprocessor lacks UAvg_plus/minus, should compute from UAvg/VAvg."""
        if not os.path.exists(january_preprocessor_path):
            pytest.skip("Preprocessor file not found")

        from orbit.core.grid_data import load_grid
        g = load_grid(january_preprocessor_path, constants_path)

        # Current preprocessor files lack split fields, so has_split_fluxes=False
        # and UAvg_plus/minus should be derived from UAvg/VAvg
        if not g.has_split_fluxes:
            np.testing.assert_array_equal(g.UAvg_plus, np.maximum(g.UAvg, 0.0))
            np.testing.assert_array_equal(g.UAvg_minus, np.maximum(-g.UAvg, 0.0))
            np.testing.assert_array_equal(g.VAvg_plus, np.maximum(g.VAvg, 0.0))
            np.testing.assert_array_equal(g.VAvg_minus, np.maximum(-g.VAvg, 0.0))

    def test_degenerate_dP_terrain_ratios(self):
        """Terrain ratios default to 1.0 when dP <= 0 (degenerate layers)."""
        from orbit.core.grid_data import GridData, _compute_terrain_ratios

        g = GridData()
        g.nz, g.ny, g.nx = 2, 3, 3
        g.dP = np.ones((2, 3, 3), dtype=np.float64) * 1000.0
        # Set one cell to zero dP (degenerate)
        g.dP[0, 1, 1] = 0.0
        _compute_terrain_ratios(g)

        # All ratios involving the degenerate cell should be 1.0
        assert g.dP_ratio_west[0, 1, 1] == 1.0
        assert g.dP_ratio_east[0, 1, 1] == 1.0
        assert g.dP_ratio_south[0, 1, 1] == 1.0
        assert g.dP_ratio_north[0, 1, 1] == 1.0
        # Neighbors looking at the degenerate cell should also get 1.0
        assert g.dP_ratio_east[0, 1, 0] == 1.0   # cell (0,1,0) looking east at (0,1,1)
        assert g.dP_ratio_west[0, 1, 2] == 1.0   # cell (0,1,2) looking west at (0,1,1)


@pytest.mark.slow
class TestGridDataReal:
    def test_load_january(self, january_preprocessor_path, constants_path):
        if not os.path.exists(january_preprocessor_path):
            pytest.skip("Preprocessor file not found")

        from orbit.core.grid_data import load_grid
        g = load_grid(january_preprocessor_path, constants_path)

        # Check dimensions (47z x 71y x 68x)
        assert g.nz == 47
        assert g.ny == 71
        assert g.nx == 68

        # Check coordinate arrays
        assert len(g.lon) == 68
        assert len(g.lat) == 71

        # Check Ap/Bp
        assert len(g.Ap) == 48
        assert len(g.Bp) == 48

        # Check dP is positive
        assert np.all(g.dP > 0)

        # Check geometry
        assert g.dy > 0
        assert np.all(g.dx > 0)
        assert np.all(g.volume > 0)

        # Check terrain ratios exist
        assert g.dP_ratio_west.shape == (47, 71, 68)
        assert g.dP_ratio_east.shape == (47, 71, 68)
        assert g.dP_ratio_north.shape == (47, 71, 68)
