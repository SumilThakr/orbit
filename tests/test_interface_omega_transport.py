"""The vertical block on an interface mass flux, and the convection block's
column balance.

Grids made on or after 2026-09-26 carry omega_edge, the flux through every
layer interface diagnosed in the preprocessor from the same face fluxes the
horizontal block uses, with the ground closed. The vertical block must read
it at the interfaces, treat the domain top as outflow when upward and as
tracer-free inflow when downward, and, together with the horizontal block,
leave a uniform mixing ratio alone wherever the flux was diagnosed from the
horizontal divergence. The convection block must move as much air down as
it moves up in every column.
"""
import numpy as np
import scipy.sparse as sp

from orbit.core.advection import (
    assemble_vertical_advection, _assemble_vertical_advection_loop, interface_omega,
)
from orbit.core.convdiff import assemble_horizontal_convdiff
from orbit.core.convection import assemble_cmfmc_transport, _assemble_cmfmc_transport_loop
from orbit.core.grid_data import GridData, _compute_geometry, _compute_terrain_ratios
from orbit.core.indexing import CellIndexer

from tests.test_advection import _make_grid


def _row_sums(T, shape):
    return np.asarray(T.sum(axis=1)).ravel().reshape(shape)


def _with_edge(params, w_edge):
    g = _make_grid(params)
    g.omega_edge = w_edge
    g.omega_edge_plus = np.maximum(w_edge, 0.0)
    g.omega_edge_minus = np.maximum(-w_edge, 0.0)
    g.has_interface_omega = True
    return g


class TestInterfaceConvention:
    def test_fallback_reproduces_the_cell_centred_reading(self, small_grid_params):
        """Without omega_edge the bottom face of layer k takes the cell-k value,
        the ground and the top are closed: the operator of the 2026-09-25 fix."""
        params = dict(small_grid_params)
        nz, ny, nx = params["nz"], params["ny"], params["nx"]
        rng = np.random.RandomState(1)
        params["omega_plus"] = rng.uniform(0.0, 0.02, (nz, ny, nx))
        params["omega_minus"] = rng.uniform(0.0, 0.01, (nz, ny, nx))
        params["omega"] = params["omega_plus"] - params["omega_minus"]
        params["has_split_omega"] = True
        g = _make_grid(params)
        wp, wm = interface_omega(g)
        assert wp.shape == (nz + 1, ny, nx)
        assert np.all(wp[0] == 0) and np.all(wm[0] == 0)
        assert np.all(wp[nz] == 0) and np.all(wm[nz] == 0)
        np.testing.assert_array_equal(wp[1:nz], params["omega_plus"][1:])
        np.testing.assert_array_equal(wm[1:nz], params["omega_minus"][1:])

    def test_edge_values_are_read_at_the_interfaces(self, small_grid_params):
        """Layer k loses w_plus[k]/dP_k downward and w_minus[k+1]/dP_k upward,
        gains w_minus[k]/dP_k from below and w_plus[k+1]/dP_k from above."""
        params = dict(small_grid_params)
        nz, ny, nx = params["nz"], params["ny"], params["nx"]
        rng = np.random.RandomState(2)
        w = rng.uniform(-0.02, 0.02, (nz + 1, ny, nx))
        w[0] = 0.0
        g = _with_edge(params, w)
        idx = CellIndexer(nz, ny, nx)
        T = assemble_vertical_advection(g, idx).toarray()
        wp, wm = np.maximum(w, 0), np.maximum(-w, 0)
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    n = idx.to_flat(k, j, i)
                    dp = g.dP[k, j, i]
                    assert np.isclose(T[n, n], (wp[k, j, i] + wm[k + 1, j, i]) / dp)
                    if k > 0:
                        assert np.isclose(T[n, idx.to_flat(k - 1, j, i)], -wm[k, j, i] / dp)
                    if k < nz - 1:
                        assert np.isclose(T[n, idx.to_flat(k + 1, j, i)], -wp[k + 1, j, i] / dp)

    def test_top_face_is_outflow_up_and_tracer_free_inflow_down(self, small_grid_params):
        params = dict(small_grid_params)
        nz, ny, nx = params["nz"], params["ny"], params["nx"]
        w = np.zeros((nz + 1, ny, nx))
        w[nz, :, :2] = -0.01      # upward through the top in two columns
        w[nz, :, 2:] = +0.01      # downward through the top elsewhere
        g = _with_edge(params, w)
        idx = CellIndexer(nz, ny, nx)
        T = assemble_vertical_advection(g, idx)
        diag_top = T.diagonal().reshape(nz, ny, nx)[nz - 1]
        np.testing.assert_allclose(diag_top[:, :2], 0.01 / g.dP[nz - 1, :, :2])
        assert np.all(diag_top[:, 2:] == 0.0)
        assert T.nnz == diag_top[:, :2].size   # no other entries at all

    def test_loop_form_matches(self, small_grid_params):
        params = dict(small_grid_params)
        nz, ny, nx = params["nz"], params["ny"], params["nx"]
        rng = np.random.RandomState(3)
        w = rng.uniform(-0.02, 0.02, (nz + 1, ny, nx)); w[0] = 0
        g = _with_edge(params, w)
        idx = CellIndexer(nz, ny, nx)
        A = assemble_vertical_advection(g, idx)
        B = _assemble_vertical_advection_loop(g, idx)
        assert abs(A - B).max() < 1e-14

    def test_ground_is_closed_even_if_the_file_says_otherwise(self, small_grid_params):
        params = dict(small_grid_params)
        nz, ny, nx = params["nz"], params["ny"], params["nx"]
        w = np.zeros((nz + 1, ny, nx)); w[0] = 0.05
        g = _with_edge(params, w)
        T = assemble_vertical_advection(g, CellIndexer(nz, ny, nx))
        assert T.nnz == 0


class TestClosureWithTheHorizontalBlock:
    def test_diagnosed_flux_closes_the_advective_rows(self, small_grid_params):
        """Build a horizontally divergent wind over terrain, diagnose the
        interface flux from the horizontal block's own row sums (which is what
        the preprocessor computes from the face fluxes), and check that the
        horizontal plus vertical blocks leave a uniform field alone in the
        interior, with the top flux carrying the column residual."""
        params = dict(small_grid_params)
        nz, ny, nx = params["nz"], params["ny"], params["nx"]
        rng = np.random.RandomState(4)
        # Terrain: varying surface pressure, so dP varies horizontally (kept
        # within the range where the fixture's thin top layer stays positive).
        params["Psurf"] = 101325.0 - 1000.0 * rng.random((ny, nx))
        dP = np.zeros((nz, ny, nx))
        for k in range(nz):
            dP[k] = (params["Ap"][k] + params["Bp"][k] * params["Psurf"]) - \
                    (params["Ap"][k + 1] + params["Bp"][k + 1] * params["Psurf"])
        params["dP"] = dP
        U = 2.0 + 3.0 * rng.random((nz, ny, nx)); V = -1.0 + 2.0 * rng.random((nz, ny, nx))
        params["UAvg"] = U; params["VAvg"] = V
        params["UAvg_plus"] = np.maximum(U, 0); params["UAvg_minus"] = np.maximum(-U, 0)
        params["VAvg_plus"] = np.maximum(V, 0); params["VAvg_minus"] = np.maximum(-V, 0)
        params["has_split_fluxes"] = True
        params["Kxxyy"] = np.zeros((nz, ny, nx)); params["K_meander_u"] = np.zeros((nz, ny, nx))
        params["K_meander_v"] = np.zeros((nz, ny, nx))
        g = _make_grid(params)
        idx = CellIndexer(nz, ny, nx)
        T_h = assemble_horizontal_convdiff(g, idx)
        R_h = _row_sums(T_h, (nz, ny, nx))          # 1/s, > 0 = net outflow / dP
        # The preprocessor's rule: w[k+1] = w[k] + div_k, div_k = R_h[k] dP_k.
        w = np.zeros((nz + 1, ny, nx))
        w[1:] = np.cumsum(R_h * g.dP, axis=0)
        g.omega_edge = w; g.omega_edge_plus = np.maximum(w, 0); g.omega_edge_minus = np.maximum(-w, 0)
        T_v = assemble_vertical_advection(g, idx)
        R = _row_sums(T_h + T_v, (nz, ny, nx))
        # Every layer but the top closes exactly. At the top, an upward
        # residual carries tracer out at the same mixing ratio that the
        # horizontal convergence brings in, so the row closes too; a downward
        # residual is tracer-free air replacing tracer that left sideways,
        # a dilution of max(w[nz], 0) / dP.
        assert np.abs(R[:-1]).max() < 1e-14
        expected_top = np.maximum(w[nz], 0.0) / g.dP[nz - 1]
        np.testing.assert_allclose(R[-1], expected_top, atol=1e-14)


class TestConvectionColumnBalance:
    def _grid(self, small_grid_params, seed=5):
        params = dict(small_grid_params)
        nz, ny, nx = params["nz"], params["ny"], params["nx"]
        g = _make_grid(params)
        rng = np.random.RandomState(seed)
        cm = np.zeros((nz + 1, ny, nx))
        cm[1:nz] = rng.uniform(0.0, 0.05, (nz - 1, ny, nx))     # kg/m2/s at interior interfaces
        cm[nz] = rng.uniform(0.0, 0.01, (ny, nx))                # some flux out of the top
        g.CMFMC = cm
        return g, CellIndexer(nz, ny, nx)

    def test_interior_rows_sum_to_zero_and_top_row_is_the_detrained_flux(self, small_grid_params):
        g, idx = self._grid(small_grid_params)
        nz = g.nz
        T = assemble_cmfmc_transport(g, idx)
        R = _row_sums(T, (nz, g.ny, g.nx))
        assert np.abs(R[:-1]).max() < 1e-14
        from orbit.core.convection import GRAVITY
        np.testing.assert_allclose(R[-1], g.CMFMC[nz] * GRAVITY / g.dP[nz - 1])

    def test_mass_measure_is_conserved_in_the_interior(self, small_grid_params):
        """W^T T with W = dP area vanishes for every source cell below the top:
        what leaves a cell arrives in its neighbours."""
        g, idx = self._grid(small_grid_params)
        g.CMFMC[g.nz] = 0.0
        T = assemble_cmfmc_transport(g, idx)
        W = (np.broadcast_to(g.dx[None, :, None], g.dP.shape) * g.dP).ravel()
        col = (W @ T.toarray()).reshape(g.dP.shape)
        assert np.abs(col).max() < 1e-9 * np.abs(W).max()

    def test_up_and_down_branches_are_symmetric(self, small_grid_params):
        g, idx = self._grid(small_grid_params)
        T = assemble_cmfmc_transport(g, idx).toarray()
        from orbit.core.convection import GRAVITY
        k = 1; j = 0; i = 0
        n, m = idx.to_flat(k, j, i), idx.to_flat(k + 1, j, i)
        F = g.CMFMC[k + 1, j, i]
        assert np.isclose(T[m, n], -F * GRAVITY / g.dP[k + 1, j, i])   # updraft gain at k+1
        assert np.isclose(T[n, m], -F * GRAVITY / g.dP[k, j, i])       # subsidence gain at k

    def test_loop_form_matches(self, small_grid_params):
        g, idx = self._grid(small_grid_params, seed=6)
        A = assemble_cmfmc_transport(g, idx)
        B = _assemble_cmfmc_transport_loop(g, idx)
        assert abs(A - B).max() < 1e-12

    def test_m_matrix_structure(self, small_grid_params):
        g, idx = self._grid(small_grid_params)
        T = assemble_cmfmc_transport(g, idx)
        d = T.diagonal()
        off = T - sp.diags(d)
        assert np.all(d >= 0) and off.min() <= 0 and off.max() <= 1e-300


class TestMassBalanceDiagnostics:
    def test_closed_column_reports_zero_and_a_leak_is_seen(self, small_grid_params):
        from orbit.core.operator import mass_balance_diagnostics
        params = dict(small_grid_params)
        nz, ny, nx = params["nz"], params["ny"], params["nx"]
        # Still air, no diffusion gradients: the transport block conserves.
        params["UAvg"] = np.zeros((nz, ny, nx)); params["VAvg"] = np.zeros((nz, ny, nx))
        g = _make_grid(params)
        idx = CellIndexer(nz, ny, nx)
        mb = mass_balance_diagnostics([g, g], idx)
        assert mb["interior_max"] < 1e-12
        assert len(mb["layer_p90"]) == nz
        # Reopen the ground the way the leak did: a downward flux out of layer 0.
        w = np.zeros((nz + 1, ny, nx)); w[1] = 0.0   # nothing between layers
        g.omega_edge = w; g.omega_edge_plus = np.maximum(w, 0); g.omega_edge_minus = np.maximum(-w, 0)
        g.has_interface_omega = True
        T0 = mass_balance_diagnostics([g], idx)["interior_max"]
        assert T0 < 1e-12
        # A vertical flux that does not close: upward through the top only,
        # with nothing feeding the column. The top layer loses 50 per day.
        w[nz] = -50.0 * g.dP[nz - 1] / 86400.0
        g.omega_edge_minus = np.maximum(-w, 0)
        mb = mass_balance_diagnostics([g], idx)
        assert mb["interior_max"] < 1e-12          # interior layers untouched
        np.testing.assert_allclose(mb["top_median"], 50.0, rtol=1e-9)
