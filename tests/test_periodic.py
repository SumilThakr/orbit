"""Tests for periodic longitude boundary conditions."""

import numpy as np
import pytest
from orbit.core.indexing import CellIndexer
from orbit.core.grid_data import GridData, _compute_geometry, _compute_terrain_ratios
from orbit.core.convdiff import (
    assemble_horizontal_convdiff,
    _assemble_horizontal_convdiff_loop,
)
from orbit.core.mixing import (
    assemble_horizontal_diffusion,
    _assemble_horizontal_diffusion_loop,
)


def _make_periodic_grid(params):
    """Build a GridData from params dict, like test_advection._make_grid."""
    g = GridData()
    for key in ["nz", "ny", "nx", "lon", "lat", "dlon", "dlat",
                 "Ap", "Bp", "Psurf", "dP", "Dz",
                 "UAvg", "VAvg", "omega",
                 "omega_plus", "omega_minus", "has_split_omega",
                 "Kzz", "Kxxyy", "K_meander_u", "K_meander_v",
                 "is_land",
                 "UAvg_plus", "UAvg_minus", "VAvg_plus", "VAvg_minus",
                 "has_split_fluxes",
                 "periodic_lon",
                 "UAvg_wrap", "UAvg_plus_wrap", "UAvg_minus_wrap",
                 "K_meander_u_wrap"]:
        if key in params:
            setattr(g, key, params[key])
    if g.omega_plus.size == 0 and g.omega.size > 0:
        g.omega_plus = np.maximum(g.omega, 0.0)
        g.omega_minus = np.maximum(-g.omega, 0.0)
        g.has_split_omega = False
    _compute_geometry(g)
    _compute_terrain_ratios(g)
    return g


@pytest.fixture
def periodic_grid_params():
    """Synthetic periodic grid: nx=8, dlon=45.0 (8*45=360), ny=4, nz=2."""
    nz, ny, nx = 2, 4, 8
    dlon = 45.0
    dlat = 10.0

    lon = np.arange(nx) * dlon
    lat = np.array([-15.0, -5.0, 5.0, 15.0])

    Ap = np.array([0.0, 0.0, 0.0])
    Bp = np.array([1.0, 0.85, 0.50])
    Psurf = np.full((ny, nx), 101325.0)

    dP = np.zeros((nz, ny, nx))
    for k in range(nz):
        P_bot = Ap[k] + Bp[k] * Psurf
        P_top = Ap[k + 1] + Bp[k + 1] * Psurf
        dP[k] = P_bot - P_top

    Dz = np.full((nz, ny, nx), 1000.0)

    rng = np.random.RandomState(42)
    UAvg = rng.randn(nz, ny, nx) * 3.0
    VAvg = rng.randn(nz, ny, nx) * 2.0
    omega = np.zeros((nz, ny, nx))

    Kxxyy = rng.uniform(50, 200, (nz, ny, nx))
    K_meander_u = rng.uniform(10, 100, (nz, ny, nx))
    K_meander_v = rng.uniform(10, 100, (nz, ny, nx))
    Kzz = np.full((nz, ny, nx), 10.0)

    is_land = np.ones((ny, nx), dtype=np.uint8)

    # Wrap-face data: for synthetic grids, use average of cell 0 and cell nx-1
    UAvg_wrap = 0.5 * (UAvg[:, :, 0] + UAvg[:, :, -1])
    UAvg_plus_wrap = np.maximum(UAvg_wrap, 0.0)
    UAvg_minus_wrap = np.maximum(-UAvg_wrap, 0.0)
    K_meander_u_wrap = 0.5 * (K_meander_u[:, :, 0] + K_meander_u[:, :, -1])

    return {
        "nz": nz, "ny": ny, "nx": nx,
        "lon": lon, "lat": lat, "dlon": dlon, "dlat": dlat,
        "Ap": Ap, "Bp": Bp, "Psurf": Psurf, "dP": dP, "Dz": Dz,
        "UAvg": UAvg, "VAvg": VAvg, "omega": omega,
        "Kzz": Kzz, "Kxxyy": Kxxyy,
        "K_meander_u": K_meander_u, "K_meander_v": K_meander_v,
        "is_land": is_land,
        "periodic_lon": True,
        "UAvg_wrap": UAvg_wrap,
        "UAvg_plus_wrap": UAvg_plus_wrap,
        "UAvg_minus_wrap": UAvg_minus_wrap,
        "K_meander_u_wrap": K_meander_u_wrap,
    }


class TestDetection:
    """Test periodic_lon detection logic."""

    def test_periodic_detected(self, periodic_grid_params):
        g = _make_periodic_grid(periodic_grid_params)
        assert g.periodic_lon is True

    def test_regional_not_periodic(self, periodic_grid_params):
        params = dict(periodic_grid_params)
        params["dlon"] = 1.0  # 8*1 = 8 degrees << 360
        params["periodic_lon"] = False
        g = _make_periodic_grid(params)
        assert g.periodic_lon is False


class TestWrapTerrainRatios:
    """Test terrain ratios at the wrap boundary."""

    def test_wrap_terrain_ratios_flat(self, periodic_grid_params):
        """Flat terrain: all ratios should be 1.0."""
        g = _make_periodic_grid(periodic_grid_params)
        np.testing.assert_allclose(g.dP_ratio_west[:, :, 0], 1.0)
        np.testing.assert_allclose(g.dP_ratio_east[:, :, -1], 1.0)

    def test_wrap_terrain_ratios_varying(self, periodic_grid_params):
        """Non-uniform Psurf: wrap ratios should match dP neighbor/cell."""
        params = dict(periodic_grid_params)
        nz, ny, nx = params["nz"], params["ny"], params["nx"]
        Psurf = np.full((ny, nx), 101325.0)
        Psurf[:, 0] = 90000.0   # lower Psurf at i=0
        Psurf[:, -1] = 80000.0  # even lower at i=nx-1
        params["Psurf"] = Psurf

        Ap, Bp = params["Ap"], params["Bp"]
        dP = np.zeros((nz, ny, nx))
        for k in range(nz):
            P_bot = Ap[k] + Bp[k] * Psurf
            P_top = Ap[k + 1] + Bp[k + 1] * Psurf
            dP[k] = P_bot - P_top
        params["dP"] = dP

        g = _make_periodic_grid(params)

        # West of cell 0 is cell nx-1
        expected_w = g.dP[:, :, -1] / g.dP[:, :, 0]
        np.testing.assert_allclose(g.dP_ratio_west[:, :, 0], expected_w, rtol=1e-12)

        # East of cell nx-1 is cell 0
        expected_e = g.dP[:, :, 0] / g.dP[:, :, -1]
        np.testing.assert_allclose(g.dP_ratio_east[:, :, -1], expected_e, rtol=1e-12)


class TestConvdiffVectorizedMatchesLoop:
    """Verify vectorized periodic convdiff matches loop reference."""

    def test_periodic_vec_eq_loop(self, periodic_grid_params):
        g = _make_periodic_grid(periodic_grid_params)
        idx = CellIndexer(g.nz, g.ny, g.nx)

        vec = assemble_horizontal_convdiff(g, idx)
        loop = _assemble_horizontal_convdiff_loop(g, idx)

        diff = vec - loop
        if diff.nnz > 0:
            max_diff = np.max(np.abs(diff.data))
        else:
            max_diff = 0.0
        assert max_diff < 1e-10, f"max abs diff = {max_diff}"

    def test_periodic_split_flux_vec_eq_loop(self, periodic_grid_params):
        """Split-flux mode with periodic boundary."""
        params = dict(periodic_grid_params)
        nz, ny, nx = params["nz"], params["ny"], params["nx"]
        rng = np.random.RandomState(88)
        params["UAvg_plus"] = rng.uniform(0.5, 3.0, (nz, ny, nx))
        params["UAvg_minus"] = rng.uniform(0.1, 1.5, (nz, ny, nx))
        params["VAvg_plus"] = rng.uniform(0.3, 2.0, (nz, ny, nx))
        params["VAvg_minus"] = rng.uniform(0.1, 1.0, (nz, ny, nx))
        params["UAvg"] = params["UAvg_plus"] - params["UAvg_minus"]
        params["VAvg"] = params["VAvg_plus"] - params["VAvg_minus"]
        params["has_split_fluxes"] = True

        # Wrap face split fluxes
        params["UAvg_plus_wrap"] = rng.uniform(0.5, 2.0, (nz, ny))
        params["UAvg_minus_wrap"] = rng.uniform(0.1, 1.0, (nz, ny))
        params["UAvg_wrap"] = params["UAvg_plus_wrap"] - params["UAvg_minus_wrap"]

        g = _make_periodic_grid(params)
        idx = CellIndexer(g.nz, g.ny, g.nx)

        vec = assemble_horizontal_convdiff(g, idx)
        loop = _assemble_horizontal_convdiff_loop(g, idx)

        diff = vec - loop
        max_diff = np.max(np.abs(diff.data)) if diff.nnz > 0 else 0.0
        assert max_diff < 1e-10, f"max abs diff = {max_diff}"


class TestMMatrix:
    """M-matrix property: non-negative diagonal, non-positive off-diagonal."""

    def test_periodic_m_matrix(self, periodic_grid_params):
        g = _make_periodic_grid(periodic_grid_params)
        idx = CellIndexer(g.nz, g.ny, g.nx)

        T = assemble_horizontal_convdiff(g, idx)
        _check_m_matrix(T)


class TestNoBoundaryLoss:
    """With periodic + flat terrain, T @ ones == 0 for ALL cells."""

    def test_uniform_concentration_zero_residual(self, periodic_grid_params):
        """Pure diffusion + periodic: no mass leak anywhere."""
        params = dict(periodic_grid_params)
        nz, ny, nx = params["nz"], params["ny"], params["nx"]
        params["UAvg"] = np.zeros((nz, ny, nx))
        params["VAvg"] = np.zeros((nz, ny, nx))
        params["UAvg_wrap"] = np.zeros((nz, ny))
        params["UAvg_plus_wrap"] = np.zeros((nz, ny))
        params["UAvg_minus_wrap"] = np.zeros((nz, ny))

        g = _make_periodic_grid(params)
        idx = CellIndexer(g.nz, g.ny, g.nx)

        T = assemble_horizontal_convdiff(g, idx)
        c = np.ones(idx.N)
        residual = T @ c

        # All cells (not just interior) should have zero residual
        # Y-boundaries still absorb, but x-boundaries should not
        for k in range(nz):
            for j in range(1, ny - 1):  # avoid y-boundaries
                for i in range(nx):  # ALL x, including 0 and nx-1
                    n = idx.to_flat(k, j, i)
                    assert abs(residual[n]) < 1e-10, (
                        f"Non-zero residual at ({k},{j},{i}): {residual[n]}"
                    )

    def test_uniform_wind_zero_residual_x_wrap(self, periodic_grid_params):
        """Uniform wind + periodic: zero residual for y-interior cells.

        Uses uniform wind so wrap face velocity is consistent with interior
        face velocities (divergence-free). Non-uniform synthetic wind would
        create artificial divergence at the wrap face; that case is covered
        by the mass-weighted conservation test.
        """
        params = dict(periodic_grid_params)
        nz, ny, nx = params["nz"], params["ny"], params["nx"]
        params["UAvg"] = np.full((nz, ny, nx), 3.0)
        params["VAvg"] = np.zeros((nz, ny, nx))
        params["UAvg_wrap"] = np.full((nz, ny), 3.0)
        params["UAvg_plus_wrap"] = np.full((nz, ny), 3.0)
        params["UAvg_minus_wrap"] = np.zeros((nz, ny))

        g = _make_periodic_grid(params)
        idx = CellIndexer(g.nz, g.ny, g.nx)

        T = assemble_horizontal_convdiff(g, idx)
        c = np.ones(idx.N)
        residual = T @ c

        for k in range(nz):
            for j in range(1, ny - 1):
                for i in range(nx):
                    n = idx.to_flat(k, j, i)
                    assert abs(residual[n]) < 1e-10, (
                        f"Non-zero residual at ({k},{j},{i}): {residual[n]}"
                    )


class TestMixingWrapConnection:
    """Mixing operator should connect cell 0 and cell nx-1 when periodic."""

    def test_wrap_coupling_exists(self, periodic_grid_params):
        g = _make_periodic_grid(periodic_grid_params)
        idx = CellIndexer(g.nz, g.ny, g.nx)

        T = assemble_horizontal_diffusion(g, idx)
        T_dense = T.toarray()

        # Check that cell (0,1,0) is coupled to cell (0,1,nx-1)
        n0 = idx.to_flat(0, 1, 0)
        n_last = idx.to_flat(0, 1, g.nx - 1)
        assert T_dense[n0, n_last] < 0, (
            f"No wrap coupling: T[{n0},{n_last}] = {T_dense[n0, n_last]}"
        )
        assert T_dense[n_last, n0] < 0, (
            f"No wrap coupling: T[{n_last},{n0}] = {T_dense[n_last, n0]}"
        )

    def test_mixing_vec_eq_loop(self, periodic_grid_params):
        """Vectorized mixing matches loop reference for periodic grid."""
        g = _make_periodic_grid(periodic_grid_params)
        idx = CellIndexer(g.nz, g.ny, g.nx)

        vec = assemble_horizontal_diffusion(g, idx)
        loop = _assemble_horizontal_diffusion_loop(g, idx)

        diff = vec - loop
        max_diff = np.max(np.abs(diff.data)) if diff.nnz > 0 else 0.0
        assert max_diff < 1e-10, f"max abs diff = {max_diff}"


class TestMixingSymmetric:
    """Pure diffusion with flat terrain: T == T.T."""

    def test_symmetric(self, periodic_grid_params):
        params = dict(periodic_grid_params)
        nz, ny, nx = params["nz"], params["ny"], params["nx"]
        # Uniform K for exact symmetry
        params["Kxxyy"] = np.full((nz, ny, nx), 100.0)
        params["K_meander_u"] = np.full((nz, ny, nx), 50.0)
        params["K_meander_v"] = np.full((nz, ny, nx), 50.0)
        params["K_meander_u_wrap"] = np.full((nz, ny), 50.0)

        g = _make_periodic_grid(params)
        idx = CellIndexer(g.nz, g.ny, g.nx)

        T = assemble_horizontal_diffusion(g, idx)
        diff = T - T.T
        max_diff = np.max(np.abs(diff.data)) if diff.nnz > 0 else 0.0
        assert max_diff < 1e-12, f"Mixing not symmetric: max diff = {max_diff}"


class TestRegionalUnaffected:
    """Same grid with periodic_lon=False should give original absorbing boundaries."""

    def test_regional_has_boundary_loss(self, periodic_grid_params):
        """Non-periodic grid: boundary cells should lose mass (T@ones != 0)."""
        params = dict(periodic_grid_params)
        params["periodic_lon"] = False
        nz, ny, nx = params["nz"], params["ny"], params["nx"]

        g = _make_periodic_grid(params)
        idx = CellIndexer(g.nz, g.ny, g.nx)

        T = assemble_horizontal_convdiff(g, idx)
        c = np.ones(idx.N)
        residual = T @ c

        # Boundary cells (i=0 or i=nx-1) should generally have nonzero residual
        # (mass lost to boundaries)
        has_boundary_loss = False
        for k in range(nz):
            for j in range(1, ny - 1):
                for i_bnd in [0, nx - 1]:
                    n = idx.to_flat(k, j, i_bnd)
                    if abs(residual[n]) > 1e-12:
                        has_boundary_loss = True
                        break
        assert has_boundary_loss, "Regional grid should have x-boundary loss"

    def test_regional_no_wrap_in_mixing(self, periodic_grid_params):
        """Non-periodic grid: cell 0 and cell nx-1 should not be coupled."""
        params = dict(periodic_grid_params)
        params["periodic_lon"] = False

        g = _make_periodic_grid(params)
        idx = CellIndexer(g.nz, g.ny, g.nx)

        T = assemble_horizontal_diffusion(g, idx)
        T_dense = T.toarray()

        n0 = idx.to_flat(0, 1, 0)
        n_last = idx.to_flat(0, 1, g.nx - 1)
        assert T_dense[n0, n_last] == 0.0, (
            f"Unexpected wrap coupling: T[{n0},{n_last}] = {T_dense[n0, n_last]}"
        )


class TestMassWeightedConservation:
    """Mass-weighted column sums = 0 for y-interior cells (periodic x)."""

    def test_periodic_mass_conservation(self, periodic_grid_params):
        """Mass-weighted column sum = 0 for all y-interior cells."""
        params = dict(periodic_grid_params)
        nz, ny, nx = params["nz"], params["ny"], params["nx"]

        # Non-uniform terrain
        Psurf = np.full((ny, nx), 101325.0)
        Psurf[:, 0] = 90000.0
        Psurf[:, -1] = 80000.0
        Psurf[2, 3] = 70000.0
        params["Psurf"] = Psurf

        Ap, Bp = params["Ap"], params["Bp"]
        dP = np.zeros((nz, ny, nx))
        for k in range(nz):
            P_bot = Ap[k] + Bp[k] * Psurf
            P_top = Ap[k + 1] + Bp[k + 1] * Psurf
            dP[k] = P_bot - P_top
        params["dP"] = dP

        g = _make_periodic_grid(params)
        idx = CellIndexer(g.nz, g.ny, g.nx)

        T = assemble_horizontal_convdiff(g, idx)

        dx_3d = np.broadcast_to(g.dx[np.newaxis, :, np.newaxis], (nz, ny, nx))
        W = (dx_3d * g.dy * g.dP).ravel()
        mass_col_sum = W @ T.toarray()

        scale = np.max(np.abs(mass_col_sum)) if np.max(np.abs(mass_col_sum)) > 0 else 1.0
        for k in range(nz):
            for j in range(1, ny - 1):
                for i in range(nx):  # ALL x including boundaries
                    n = idx.to_flat(k, j, i)
                    assert abs(mass_col_sum[n]) < 1e-10 * scale, (
                        f"Non-zero mass-weighted col sum at ({k},{j},{i}): "
                        f"{mass_col_sum[n]}"
                    )


def _check_m_matrix(T):
    """Assert M-matrix properties on a sparse matrix."""
    N = T.shape[0]
    diag = T.diagonal()
    assert np.all(diag >= -1e-15), f"Negative diagonal: min = {diag.min()}"

    T_csr = T.tocsr()
    for i in range(N):
        row_start = T_csr.indptr[i]
        row_end = T_csr.indptr[i + 1]
        for jj in range(row_start, row_end):
            col = T_csr.indices[jj]
            val = T_csr.data[jj]
            if col != i:
                assert val <= 1e-15, (
                    f"Positive off-diagonal at ({i},{col}): {val}"
                )
