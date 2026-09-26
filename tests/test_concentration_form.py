"""The transport block in concentration form.

The four transport blocks conserve the pressure measure dP*area and leave a
uniform mixing ratio alone. The state the model reads is a concentration at
local density, so the production block is diag(rho) T diag(1/rho) with
rho = dP/(g Dz): it conserves the cell volume and leaves c = rho * const
alone. The FCT source and its frozen linear operator are formed on c/rho
and returned for c, so that L_AD @ c_baseline still equals the negative of
the source at the baseline.
"""
import numpy as np

from orbit.core.convdiff import assemble_horizontal_convdiff
from orbit.core.fct import compute_horizontal_fct_source, assemble_fct_linear_operator
from orbit.core.grid_data import density_weights
from orbit.core.indexing import CellIndexer
from orbit.core.operator import assemble_transport_block, to_concentration_form
from orbit.core.orbit import DTAU

from tests.test_advection import _make_grid


def _terrain_grid(small_grid_params, seed=11, with_omega=True):
    params = dict(small_grid_params)
    nz, ny, nx = params["nz"], params["ny"], params["nx"]
    rng = np.random.RandomState(seed)
    params["Psurf"] = 101325.0 - 1000.0 * rng.random((ny, nx))
    dP = np.zeros((nz, ny, nx))
    for k in range(nz):
        dP[k] = (params["Ap"][k] + params["Bp"][k] * params["Psurf"]) - \
                (params["Ap"][k + 1] + params["Bp"][k + 1] * params["Psurf"])
    params["dP"] = dP
    params["Dz"] = 400.0 + 300.0 * rng.random((nz, ny, nx))      # non-uniform density
    U = 2.0 + 3.0 * rng.random((nz, ny, nx)); V = -1.0 + 2.0 * rng.random((nz, ny, nx))
    params["UAvg"] = U; params["VAvg"] = V
    params["UAvg_plus"] = np.maximum(U, 0); params["UAvg_minus"] = np.maximum(-U, 0)
    params["VAvg_plus"] = np.maximum(V, 0); params["VAvg_minus"] = np.maximum(-V, 0)
    params["has_split_fluxes"] = True
    g = _make_grid(params)
    idx = CellIndexer(nz, ny, nx)
    if with_omega:
        g.concentration_state = False
        R_h = np.asarray(assemble_horizontal_convdiff(g, idx).sum(axis=1)).ravel().reshape(nz, ny, nx)
        g.concentration_state = True
        w = np.zeros((nz + 1, ny, nx)); w[1:] = np.cumsum(R_h * g.dP, axis=0)
        g.omega_edge = w; g.omega_edge_plus = np.maximum(w, 0); g.omega_edge_minus = np.maximum(-w, 0)
        g.has_interface_omega = True
    return g, idx


def test_density_weights_are_dp_over_g_dz_and_one_where_degenerate(small_grid_params):
    g, _ = _terrain_grid(small_grid_params)
    rho = density_weights(g).reshape(g.dP.shape)
    np.testing.assert_allclose(rho, g.dP / (9.80665 * g.Dz))
    g.Dz[0, 0, 0] = 0.0
    assert density_weights(g).reshape(g.dP.shape)[0, 0, 0] == 1.0


def test_transformed_block_is_the_similarity_transform(small_grid_params):
    g, idx = _terrain_grid(small_grid_params)
    g.concentration_state = False
    T = assemble_transport_block(g, idx)
    g.concentration_state = True
    Tc = assemble_transport_block(g, idx)
    assert abs(Tc - to_concentration_form(T, g)).max() < 1e-15


def test_uniform_mixing_ratio_is_the_equilibrium_of_the_transformed_block(small_grid_params):
    """T x = 0 for x = 1 in the interior (mass-consistent flux), so
    T_c c = 0 for c = rho: a column of air in hydrostatic balance with a
    uniform mixing ratio is not disturbed, while a uniform concentration is."""
    g, idx = _terrain_grid(small_grid_params)
    nz, ny, nx = g.nz, g.ny, g.nx
    Tc = assemble_transport_block(g, idx)
    rho = density_weights(g)
    r = (Tc @ rho).reshape(nz, ny, nx)[:-1, 1:-1, 1:-1] / rho.reshape(nz, ny, nx)[:-1, 1:-1, 1:-1]
    assert np.abs(r).max() < 1e-13
    ones = (Tc @ np.ones(idx.N)).reshape(nz, ny, nx)[:-1, 1:-1, 1:-1]
    assert np.abs(ones).max() > 1e-7          # a uniform concentration is not conserved


def test_volume_measure_is_conserved(small_grid_params):
    g, idx = _terrain_grid(small_grid_params)
    g.omega_edge[-1] = 0.0; g.omega_edge_plus[-1] = 0.0; g.omega_edge_minus[-1] = 0.0   # close the top
    Tc = assemble_transport_block(g, idx)
    vol = (np.broadcast_to(g.dx[None, :, None], g.dP.shape) * g.Dz).ravel()
    col = (vol @ Tc.toarray()).reshape(g.dP.shape)[:, 1:-1, 1:-1]
    assert np.abs(col).max() < 1e-9 * np.abs(vol).max()


def test_fct_source_and_linear_operator_are_consistent_in_concentration_form(small_grid_params):
    g, idx = _terrain_grid(small_grid_params, seed=12)
    rng = np.random.RandomState(13)
    c = (1.0 + rng.random(idx.N)) * density_weights(g)
    c[idx.N // 3: idx.N // 2] *= 4.0                       # a front, so the limiter acts
    d_c = compute_horizontal_fct_source(g, idx, c, DTAU)
    L = assemble_fct_linear_operator(g, idx, c, DTAU)
    np.testing.assert_allclose(L @ c, -d_c, rtol=1e-9, atol=1e-12 * np.abs(d_c).max())
    # and the source for c is rho times the source for x = c / rho
    rho = density_weights(g)
    g.concentration_state = False
    d_x = compute_horizontal_fct_source(g, idx, c / rho, DTAU)
    g.concentration_state = True
    np.testing.assert_allclose(d_c, rho * d_x, rtol=1e-12, atol=1e-15)
