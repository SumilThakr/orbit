"""Tests for the lateral boundary inflow source (orbit/core/lateral_bc.py).

1. Mirror: the inflow rate of a grid equals the outflow (boundary loss) rate of
   the same grid with the wind reversed, face by face (split and non-split).
2. Steady state: with a uniform zonal wind and a uniform boundary
   concentration, no emissions, chemistry or deposition, the steady solution of
   the horizontal convection-diffusion operator is c = c_bc in every cell.
   Without the inflow it would be zero.
3. Loader: grid mismatch and negative values fail loudly; species map by name.
"""

import numpy as np
import pytest
import scipy.sparse.linalg as spla

from orbit.core.convdiff import assemble_horizontal_convdiff
from orbit.core.indexing import CellIndexer
from orbit.core.lateral_bc import compute_lateral_boundary_inflow, load_lateral_bc, merge_into
from orbit.core.operator import compute_lateral_boundary_loss
from tests.test_detection_operator import _make_grid, _regional_params

SPECIES = ["VBS_C100", "PM2.5", "NH", "SO2", "NOx", "pSO4", "TotalNO3",
           "O3", "CO", "VBS_C10", "VBS_C1", "VBS_C01", "VBS_C1000", "POA"]


def _grid(u, v, split=False, seed=None):
    p = _regional_params()
    shape = (p["nz"], p["ny"], p["nx"])
    if seed is not None:  # spatially varying winds of both signs
        rng = np.random.default_rng(seed)
        p["UAvg"] = rng.normal(u, 3.0, shape)
        p["VAvg"] = rng.normal(v, 3.0, shape)
    else:
        p["UAvg"] = np.full(shape, float(u))
        p["VAvg"] = np.full(shape, float(v))
    if split:
        rng = np.random.default_rng(7)
        p["UAvg_plus"] = np.abs(rng.normal(1.0, 1.0, shape))
        p["UAvg_minus"] = np.abs(rng.normal(1.0, 1.0, shape))
        p["VAvg_plus"] = np.abs(rng.normal(1.0, 1.0, shape))
        p["VAvg_minus"] = np.abs(rng.normal(1.0, 1.0, shape))
        p["has_split_fluxes"] = True
    return _make_grid(p)


def _reverse(g):
    g.UAvg = -g.UAvg
    g.VAvg = -g.VAvg
    if g.has_split_fluxes:
        g.UAvg_plus, g.UAvg_minus = g.UAvg_minus.copy(), g.UAvg_plus.copy()
        g.VAvg_plus, g.VAvg_minus = g.VAvg_minus.copy(), g.VAvg_plus.copy()
    return g


@pytest.mark.parametrize("split", [False, True])
@pytest.mark.parametrize("seed", [None, 1, 2])
def test_inflow_mirrors_outflow(split, seed):
    g = _grid(2.0, 1.0, split=split, seed=seed)
    r_in = compute_lateral_boundary_inflow(g)
    out_rev = compute_lateral_boundary_loss(_reverse(_grid(2.0, 1.0, split=split, seed=seed)),
                                            CellIndexer(g.nz, g.ny, g.nx))
    if seed is None and not split:
        # Non-split, uniform wind: exact zero-velocity ties are absent, so the
        # mirror holds cell by cell.
        np.testing.assert_allclose(r_in, out_rev, rtol=1e-14, atol=0)
    else:
        np.testing.assert_allclose(r_in, out_rev, rtol=1e-14, atol=1e-30)
    assert np.all(r_in >= 0)
    interior = np.zeros((g.nz, g.ny, g.nx), bool)
    interior[:, 1:-1, 1:-1] = True
    assert np.all(r_in.reshape(g.nz, g.ny, g.nx)[interior] == 0)


@pytest.mark.parametrize("u", [2.0, -3.0])
def test_uniform_inflow_gives_uniform_interior(u):
    g = _grid(u, 0.0)
    T = assemble_horizontal_convdiff(g, CellIndexer(g.nz, g.ny, g.nx)).tocsc()
    c_bc = 7.5
    s = compute_lateral_boundary_inflow(g) * c_bc
    c = spla.spsolve(T, s)
    np.testing.assert_allclose(c, c_bc, rtol=1e-10)
    # Control: without inflow the steady state is zero.
    np.testing.assert_allclose(spla.spsolve(T, np.zeros_like(s)), 0.0)


def _write_bc(path, g, names, value=1.0, lat=None):
    c = np.full((len(names), 8, g.nz, g.ny, g.nx), value, np.float32)
    np.savez(path, lat=g.lat if lat is None else lat, lon=g.lon, nz=g.nz,
             species=np.array(names), c_bc=c)


def test_loader(tmp_path):
    g = _grid(2.0, 1.0)
    grids = [g] * 8
    f = tmp_path / "bc.npz"
    _write_bc(f, g, ["PM2.5", "SO2"], 2.0)
    src = load_lateral_bc(f, grids, SPECIES)
    assert set(src) == {1, 3}
    np.testing.assert_allclose(src[1][0], 2.0 * compute_lateral_boundary_inflow(g))
    merged = merge_into({1: [np.ones(src[1][0].size)] * 8}, src)
    np.testing.assert_allclose(merged[1][3], 1.0 + src[1][3])
    _write_bc(f, g, ["PM2.5"], 1.0, lat=g.lat + 0.5)
    with pytest.raises(ValueError, match="grid mismatch"):
        load_lateral_bc(f, grids, SPECIES)
    _write_bc(f, g, ["PM2.5"], -1.0)
    with pytest.raises(ValueError, match="negative"):
        load_lateral_bc(f, grids, SPECIES)
    _write_bc(f, g, ["PM25"], 1.0)
    with pytest.raises(ValueError, match="unknown species"):
        load_lateral_bc(f, grids, SPECIES)


def _with_true_faces(g, seed=3):
    """Attach distinct east/north outer-face winds (as load_grid does from a staggered input)."""
    rng = np.random.default_rng(seed)
    g.U_face_e = rng.normal(1.0, 2.0, (g.nz, g.ny))
    g.V_face_n = rng.normal(-1.0, 2.0, (g.nz, g.nx))
    g.U_plus_face_e, g.U_minus_face_e = np.maximum(g.U_face_e, 0), np.maximum(-g.U_face_e, 0)
    g.V_plus_face_n, g.V_minus_face_n = np.maximum(g.V_face_n, 0), np.maximum(-g.V_face_n, 0)
    return g


def test_edge_face_fallback_and_true():
    from orbit.core.grid_data import edge_face
    g = _grid(2.0, 1.0, split=True, seed=1)
    np.testing.assert_array_equal(edge_face(g, "U"), g.UAvg[:, :, -1])
    np.testing.assert_array_equal(edge_face(g, "V_minus"), g.VAvg_minus[:, -1, :])
    _with_true_faces(g)
    np.testing.assert_array_equal(edge_face(g, "U"), g.U_face_e)
    np.testing.assert_array_equal(edge_face(g, "V_minus"), g.V_minus_face_n)


@pytest.mark.parametrize("split", [False, True])
def test_mirror_with_true_faces(split):
    """Inflow and outflow use the same east/north faces when the true faces are stored."""
    g = _with_true_faces(_grid(2.0, 1.0, split=split, seed=2))
    r_in = compute_lateral_boundary_inflow(g)
    gr = _reverse(_with_true_faces(_grid(2.0, 1.0, split=split, seed=2)))
    gr.U_face_e, gr.V_face_n = -gr.U_face_e, -gr.V_face_n
    gr.U_plus_face_e, gr.U_minus_face_e = gr.U_minus_face_e.copy(), gr.U_plus_face_e.copy()
    gr.V_plus_face_n, gr.V_minus_face_n = gr.V_minus_face_n.copy(), gr.V_plus_face_n.copy()
    out_rev = compute_lateral_boundary_loss(gr, CellIndexer(g.nz, g.ny, g.nx))
    np.testing.assert_allclose(r_in, out_rev, rtol=1e-14, atol=1e-30)
    # and the true faces change the east/north rates relative to the stand-in
    g0 = _grid(2.0, 1.0, split=split, seed=2)
    assert not np.allclose(r_in, compute_lateral_boundary_inflow(g0))


def test_convdiff_change_is_boundary_diagonal_only():
    """Storing the true faces changes convdiff only on the boundary diagonal, by exactly the change in
    operator.compute_lateral_boundary_loss (the two boundary-loss implementations stay consistent)."""
    for split in (False, True):
        g0 = _grid(2.0, 1.0, split=split, seed=4)
        g1 = _with_true_faces(_grid(2.0, 1.0, split=split, seed=4))
        idx = CellIndexer(g0.nz, g0.ny, g0.nx)
        dT = (assemble_horizontal_convdiff(g1, idx) - assemble_horizontal_convdiff(g0, idx)).tocoo()
        assert np.all(dT.row[np.abs(dT.data) > 0] == dT.col[np.abs(dT.data) > 0]), "off-diagonal change"
        d = np.zeros(idx.N); np.add.at(d, dT.row, dT.data)
        dl = compute_lateral_boundary_loss(g1, idx) - compute_lateral_boundary_loss(g0, idx)
        np.testing.assert_allclose(d, dl, rtol=1e-12, atol=1e-18)
        assert np.abs(dl).max() > 0
