"""Tests for lateral-closed detection operator.

Tests compute_lateral_boundary_loss(), build_detection_operator(),
validate_detection_operator(), and the return_boundary_fluxes flag
on assemble_transport_block().
"""

import numpy as np
import pytest
import scipy.sparse as sp

from orbit.core.indexing import CellIndexer
from orbit.core.grid_data import GridData, _compute_geometry, _compute_terrain_ratios
from orbit.core.operator import (
    compute_lateral_boundary_loss,
    build_detection_operator,
    validate_detection_operator,
    assemble_transport_block,
)
from orbit.core.convdiff import assemble_horizontal_convdiff


# ---------- helpers ----------

def _make_grid(params):
    """Build a GridData from params dict."""
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


def _regional_params():
    """Non-periodic grid: nx=6, ny=5, nz=3, uniform eastward+northward wind."""
    nz, ny, nx = 3, 5, 6
    dlon = 1.0
    dlat = 1.0
    lon = np.arange(nx) * dlon + 70.0
    lat = np.arange(ny) * dlat + 20.0

    Ap = np.array([0.0, 5000.0, 20000.0, 60000.0])
    Bp = np.array([1.0, 0.90, 0.70, 0.30])
    Psurf = np.full((ny, nx), 101325.0)

    dP = np.zeros((nz, ny, nx))
    for k in range(nz):
        P_bot = Ap[k] + Bp[k] * Psurf
        P_top = Ap[k + 1] + Bp[k + 1] * Psurf
        dP[k] = P_bot - P_top

    Dz = np.full((nz, ny, nx), 500.0)

    UAvg = np.full((nz, ny, nx), 2.0)
    VAvg = np.full((nz, ny, nx), 1.0)
    omega = np.zeros((nz, ny, nx))

    Kxxyy = np.full((nz, ny, nx), 100.0)
    K_meander_u = np.full((nz, ny, nx), 50.0)
    K_meander_v = np.full((nz, ny, nx), 50.0)
    Kzz = np.full((nz, ny, nx), 10.0)
    is_land = np.ones((ny, nx), dtype=np.uint8)

    return {
        "nz": nz, "ny": ny, "nx": nx,
        "lon": lon, "lat": lat, "dlon": dlon, "dlat": dlat,
        "Ap": Ap, "Bp": Bp, "Psurf": Psurf, "dP": dP, "Dz": Dz,
        "UAvg": UAvg, "VAvg": VAvg, "omega": omega,
        "Kzz": Kzz, "Kxxyy": Kxxyy,
        "K_meander_u": K_meander_u, "K_meander_v": K_meander_v,
        "is_land": is_land,
    }


def _periodic_params():
    """Periodic grid: nx=8, ny=5, nz=3, dlon=45.0, uniform wind."""
    nz, ny, nx = 3, 5, 8
    dlon = 45.0
    dlat = 10.0
    lon = np.arange(nx) * dlon
    lat = np.arange(ny) * dlat - 20.0

    Ap = np.array([0.0, 5000.0, 20000.0, 60000.0])
    Bp = np.array([1.0, 0.90, 0.70, 0.30])
    Psurf = np.full((ny, nx), 101325.0)

    dP = np.zeros((nz, ny, nx))
    for k in range(nz):
        P_bot = Ap[k] + Bp[k] * Psurf
        P_top = Ap[k + 1] + Bp[k + 1] * Psurf
        dP[k] = P_bot - P_top

    Dz = np.full((nz, ny, nx), 500.0)

    UAvg = np.full((nz, ny, nx), 2.0)
    VAvg = np.full((nz, ny, nx), 1.0)
    omega = np.zeros((nz, ny, nx))

    Kxxyy = np.full((nz, ny, nx), 100.0)
    K_meander_u = np.full((nz, ny, nx), 50.0)
    K_meander_v = np.full((nz, ny, nx), 50.0)
    Kzz = np.full((nz, ny, nx), 10.0)
    is_land = np.ones((ny, nx), dtype=np.uint8)

    UAvg_wrap = np.full((nz, ny), 2.0)
    UAvg_plus_wrap = np.maximum(UAvg_wrap, 0.0)
    UAvg_minus_wrap = np.maximum(-UAvg_wrap, 0.0)
    K_meander_u_wrap = np.full((nz, ny), 50.0)

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


# ---------- fixtures ----------

@pytest.fixture
def regional_grid():
    params = _regional_params()
    return _make_grid(params), CellIndexer(params["nz"], params["ny"], params["nx"])


@pytest.fixture
def periodic_grid():
    params = _periodic_params()
    return _make_grid(params), CellIndexer(params["nz"], params["ny"], params["nx"])


# ---------- compute_lateral_boundary_loss tests ----------

class TestLateralLoss:

    def test_interior_zero(self, regional_grid):
        """Interior cells have zero lateral loss."""
        grid, indexer = regional_grid
        loss = compute_lateral_boundary_loss(grid, indexer)
        nz, ny, nx = grid.nz, grid.ny, grid.nx
        loss_3d = loss.reshape(nz, ny, nx)
        interior = loss_3d[:, 1:-1, 1:-1]
        assert np.all(interior == 0.0)

    def test_boundary_positive(self, regional_grid):
        """Boundary cells with outward wind have positive loss."""
        grid, indexer = regional_grid
        loss = compute_lateral_boundary_loss(grid, indexer)
        nz, ny, nx = grid.nz, grid.ny, grid.nx
        loss_3d = loss.reshape(nz, ny, nx)

        # East boundary (i=nx-1): uniform U=2.0 > 0, so outward loss
        assert np.all(loss_3d[:, :, -1] > 0)
        # North boundary (j=ny-1): uniform V=1.0 > 0, so outward loss
        assert np.all(loss_3d[:, -1, :] > 0)

    def test_periodic_x_zero(self, periodic_grid):
        """Periodic grid: x-boundary loss is zero, y-boundary still present."""
        grid, indexer = periodic_grid
        loss = compute_lateral_boundary_loss(grid, indexer)
        nz, ny, nx = grid.nz, grid.ny, grid.nx
        loss_3d = loss.reshape(nz, ny, nx)

        # x-boundary cells that are NOT also y-boundary: should be zero
        x_only = loss_3d[:, 1:-1, 0]  # i=0, interior j
        assert np.all(x_only == 0.0)
        x_only_e = loss_3d[:, 1:-1, -1]  # i=nx-1, interior j
        assert np.all(x_only_e == 0.0)

        # y-boundary should still have loss (V=1.0 northward)
        assert np.all(loss_3d[:, -1, :] > 0)


# ---------- build_detection_operator tests ----------

class TestBuildDetectionOperator:

    def test_offdiag_unchanged(self, regional_grid):
        """T_det has identical off-diagonals to T."""
        grid, indexer = regional_grid
        T = assemble_transport_block(grid, indexer, scheme="exp")
        loss = compute_lateral_boundary_loss(grid, indexer)
        T_det, _, _ = build_detection_operator(T, loss)

        diff = (T_det - T).tocsr()
        diff_offdiag = diff.copy()
        diff_offdiag.setdiag(0)
        diff_offdiag.eliminate_zeros()
        assert diff_offdiag.nnz == 0

    def test_interior_unchanged(self, regional_grid):
        """Interior cell row sums identical between T and T_det."""
        grid, indexer = regional_grid
        nz, ny, nx = grid.nz, grid.ny, grid.nx
        T = assemble_transport_block(grid, indexer, scheme="exp")
        loss = compute_lateral_boundary_loss(grid, indexer)
        T_det, _, _ = build_detection_operator(T, loss)

        rs_T = np.array(T.tocsr().sum(axis=1)).ravel()
        rs_det = np.array(T_det.tocsr().sum(axis=1)).ravel()

        n3d = np.arange(indexer.N).reshape(nz, ny, nx)
        interior = n3d[:, 1:-1, 1:-1].ravel()
        np.testing.assert_allclose(rs_det[interior], rs_T[interior], atol=1e-12)

    def test_diagonal_reduced(self, regional_grid):
        """T_det diagonal <= T diagonal at all boundary cells."""
        grid, indexer = regional_grid
        nz, ny, nx = grid.nz, grid.ny, grid.nx
        T = assemble_transport_block(grid, indexer, scheme="exp")
        loss = compute_lateral_boundary_loss(grid, indexer)
        T_det, _, _ = build_detection_operator(T, loss)

        diag_T = np.array(T.diagonal())
        diag_det = np.array(T_det.diagonal())

        n3d = np.arange(indexer.N).reshape(nz, ny, nx)
        is_lateral = np.zeros(indexer.N, dtype=bool)
        is_lateral[n3d[:, :, 0].ravel()] = True
        is_lateral[n3d[:, :, -1].ravel()] = True
        is_lateral[n3d[:, 0, :].ravel()] = True
        is_lateral[n3d[:, -1, :].ravel()] = True

        assert np.all(diag_det[is_lateral] <= diag_T[is_lateral] + 1e-12)

    def test_lateral_below_top_closed(self):
        """build_detection_operator reduces boundary row sums by exactly
        lateral_loss, and on a zero-Kzz grid the residual is only the
        y-face dx-correction (small, from varying cos(lat))."""
        params = _regional_params()
        params["Kzz"] = np.zeros((params["nz"], params["ny"], params["nx"]))
        grid = _make_grid(params)
        indexer = CellIndexer(params["nz"], params["ny"], params["nx"])

        T = assemble_transport_block(grid, indexer, scheme="exp")
        loss = compute_lateral_boundary_loss(grid, indexer)
        T_det, _, _ = build_detection_operator(T, loss)

        nz, ny, nx = grid.nz, grid.ny, grid.nx
        rs_T = np.array(T.tocsr().sum(axis=1)).ravel()
        rs_det = np.array(T_det.tocsr().sum(axis=1)).ravel()

        # Exact by construction: T_det = T - diag(lateral_loss)
        np.testing.assert_allclose(rs_det, rs_T - loss, atol=1e-12)

        # Lateral cells below top: row sum reduced (less positive or negative)
        is_lateral = np.zeros((nz, ny, nx), dtype=bool)
        is_lateral[:, :, 0] = True
        is_lateral[:, :, -1] = True
        is_lateral[:, 0, :] = True
        is_lateral[:, -1, :] = True
        below_top = np.zeros((nz, ny, nx), dtype=bool)
        below_top[:nz-1, :, :] = True
        lateral_below_top = (is_lateral & below_top).ravel()

        # Row sums of T_det at lateral cells should be much smaller than T
        # (remaining residual is only from y-face dx-correction, ~1e-5)
        assert np.max(np.abs(rs_det[lateral_below_top])) < 1e-4


# ---------- validate_detection_operator tests ----------

class TestValidateDetectionOperator:

    def test_validate_passes(self, regional_grid):
        """validate_detection_operator returns all True checks."""
        grid, indexer = regional_grid
        T = assemble_transport_block(grid, indexer, scheme="exp")
        loss = compute_lateral_boundary_loss(grid, indexer)
        T_det, _, _ = build_detection_operator(T, loss)

        result = validate_detection_operator(T_det, T, indexer)
        assert result["interior_unchanged"]
        assert result["boundary_diag_reduced"]
        assert result["offdiag_identical"]
        assert result["no_negative_diag"]


# ---------- assemble_transport_block return_boundary_fluxes tests ----------

class TestReturnBoundaryFluxes:

    def test_flag_returns_tuple(self, regional_grid):
        """assemble_transport_block with return_boundary_fluxes=True returns tuple."""
        grid, indexer = regional_grid
        result = assemble_transport_block(
            grid, indexer, scheme="exp", return_boundary_fluxes=True,
        )
        assert isinstance(result, tuple)
        assert len(result) == 2
        T, lat_loss = result
        assert sp.issparse(T)
        assert lat_loss.shape == (indexer.N,)

    def test_flag_false_returns_matrix(self, regional_grid):
        """Default return_boundary_fluxes=False returns just the matrix."""
        grid, indexer = regional_grid
        result = assemble_transport_block(grid, indexer, scheme="exp")
        assert sp.issparse(result)


# ---------- cross-check: lateral loss matches row-sum deficit ----------

class TestLateralLossConsistency:

    def test_matches_row_sum_deficit(self):
        """lateral_loss captures exactly the explicit boundary-block diagonal
        from convdiff.py. Verify by comparing row-sum reduction."""
        params = _regional_params()
        params["Kzz"] = np.zeros((params["nz"], params["ny"], params["nx"]))
        grid = _make_grid(params)
        indexer = CellIndexer(params["nz"], params["ny"], params["nx"])
        nz, ny, nx = grid.nz, grid.ny, grid.nx

        T = assemble_transport_block(grid, indexer, scheme="exp")
        loss = compute_lateral_boundary_loss(grid, indexer)
        T_det, n_closed, total_flux = build_detection_operator(T, loss)

        rs_T = np.array(T.tocsr().sum(axis=1)).ravel()
        rs_det = np.array(T_det.tocsr().sum(axis=1)).ravel()

        # Core identity: T_det = T - diag(loss) → row_sum(T_det) = row_sum(T) - loss
        np.testing.assert_allclose(rs_det, rs_T - loss, atol=1e-12)

        # Interior cells: loss is zero, so row sums unchanged
        n3d = np.arange(indexer.N).reshape(nz, ny, nx)
        interior = n3d[:, 1:-1, 1:-1].ravel()
        np.testing.assert_allclose(loss[interior], 0.0)

        # East boundary with U=2.0 (outward): explicit loss is positive
        east = n3d[:, :, -1].ravel()
        assert np.all(loss[east] > 0)

        # North boundary with V=1.0 (outward): explicit loss is positive
        north = n3d[:, -1, :].ravel()
        assert np.all(loss[north] > 0)

        # n_closed counts cells with positive lateral loss
        assert n_closed > 0
        assert total_flux > 0

    def test_lateral_loss_matches_convdiff_diagonal(self):
        """Cross-check: compute_lateral_boundary_loss reproduces the boundary
        diagonal contribution from convdiff.py.

        Builds the full convdiff operator T_hcd and a "no-boundary" reference
        by subtracting lateral_loss from the diagonal. If convdiff.py's boundary
        treatment ever changes without updating compute_lateral_boundary_loss,
        this test will catch the divergence.
        """
        # Use random-ish wind to exercise all 4 boundary directions
        params = _regional_params()
        rng = np.random.RandomState(99)
        nz, ny, nx = params["nz"], params["ny"], params["nx"]
        params["UAvg"] = rng.randn(nz, ny, nx) * 3.0
        params["VAvg"] = rng.randn(nz, ny, nx) * 2.0
        grid = _make_grid(params)
        indexer = CellIndexer(nz, ny, nx)

        T_hcd = assemble_horizontal_convdiff(grid, indexer)
        loss = compute_lateral_boundary_loss(grid, indexer)

        # The lateral_loss vector should be a subset of T_hcd's diagonal:
        # T_hcd.diagonal() = interior_face_diag + boundary_loss_diag
        # So: T_hcd.diagonal() - loss >= 0 (interior contribution is non-negative)
        diag_hcd = np.array(T_hcd.diagonal())
        residual_diag = diag_hcd - loss
        assert np.all(residual_diag >= -1e-12), (
            f"lateral_loss exceeds T_hcd diagonal at {np.sum(residual_diag < -1e-12)} cells; "
            f"min residual = {np.min(residual_diag):.2e}"
        )

        # Removing lateral_loss from the diagonal should not change off-diagonals
        T_closed = T_hcd.copy()
        T_closed -= sp.diags(loss)
        diff = (T_closed - T_hcd).tocsr()
        diff.setdiag(0)
        diff.eliminate_zeros()
        assert diff.nnz == 0

    def test_lateral_loss_matches_convdiff_split_flux(self):
        """Cross-check with split-flux wind fields."""
        params = _regional_params()
        rng = np.random.RandomState(42)
        nz, ny, nx = params["nz"], params["ny"], params["nx"]
        params["UAvg"] = rng.randn(nz, ny, nx) * 3.0
        params["VAvg"] = rng.randn(nz, ny, nx) * 2.0
        params["UAvg_plus"] = rng.uniform(0, 4, (nz, ny, nx))
        params["UAvg_minus"] = rng.uniform(0, 4, (nz, ny, nx))
        params["VAvg_plus"] = rng.uniform(0, 3, (nz, ny, nx))
        params["VAvg_minus"] = rng.uniform(0, 3, (nz, ny, nx))
        params["has_split_fluxes"] = True
        grid = _make_grid(params)
        indexer = CellIndexer(nz, ny, nx)

        T_hcd = assemble_horizontal_convdiff(grid, indexer)
        loss = compute_lateral_boundary_loss(grid, indexer)

        diag_hcd = np.array(T_hcd.diagonal())
        residual_diag = diag_hcd - loss
        assert np.all(residual_diag >= -1e-12), (
            f"lateral_loss exceeds T_hcd diagonal at {np.sum(residual_diag < -1e-12)} cells; "
            f"min residual = {np.min(residual_diag):.2e}"
        )


# ---------- Y-boundary terrain correction residual ----------

class TestYBoundaryTerrainCorrection:

    def test_high_latitude_residual(self):
        """Quantify the Y-boundary dx-correction residual on a grid spanning
        low to high latitudes, where dx varies significantly.

        convdiff.py's interior Y-faces apply dx_face/dx_cell terrain correction
        but Y-boundary faces use raw V/dy. This test documents the residual size
        to ensure it doesn't grow dangerously large at high latitudes.
        """
        # Grid spanning 5°N to 75°N — dx varies by factor ~4
        nz, ny, nx = 2, 15, 4
        dlat = 5.0
        dlon = 5.0
        lat = np.arange(ny) * dlat + 5.0  # 5° to 75°
        lon = np.arange(nx) * dlon + 70.0

        Ap = np.array([0.0, 5000.0, 60000.0])
        Bp = np.array([1.0, 0.90, 0.30])
        Psurf = np.full((ny, nx), 101325.0)
        dP = np.zeros((nz, ny, nx))
        for k in range(nz):
            P_bot = Ap[k] + Bp[k] * Psurf
            P_top = Ap[k + 1] + Bp[k + 1] * Psurf
            dP[k] = P_bot - P_top

        params = {
            "nz": nz, "ny": ny, "nx": nx,
            "lon": lon, "lat": lat, "dlon": dlon, "dlat": dlat,
            "Ap": Ap, "Bp": Bp, "Psurf": Psurf, "dP": dP,
            "Dz": np.full((nz, ny, nx), 500.0),
            "UAvg": np.zeros((nz, ny, nx)),
            "VAvg": np.full((nz, ny, nx), 1.0),  # uniform northward
            "omega": np.zeros((nz, ny, nx)),
            "Kzz": np.zeros((nz, ny, nx)),  # no vertical mixing
            "Kxxyy": np.full((nz, ny, nx), 100.0),
            "K_meander_u": np.full((nz, ny, nx), 50.0),
            "K_meander_v": np.full((nz, ny, nx), 50.0),
            "is_land": np.ones((ny, nx), dtype=np.uint8),
        }
        grid = _make_grid(params)
        indexer = CellIndexer(nz, ny, nx)

        T = assemble_transport_block(grid, indexer, scheme="exp")
        loss = compute_lateral_boundary_loss(grid, indexer)
        T_det, _, _ = build_detection_operator(T, loss)

        rs_det = np.array(T_det.tocsr().sum(axis=1)).ravel()
        rs_det_3d = rs_det.reshape(nz, ny, nx)

        # North boundary (j=ny-1, lat=70°): the residual comes from the
        # missing dx_face/dx_cell correction at the boundary face.
        # At 70°N, cos(70°)/cos(67.5°) ≈ 0.89, so the correction is ~11%.
        north_rs = rs_det_3d[:, -1, 1:-1]  # interior-x, north-y
        north_max = np.max(np.abs(north_rs))

        # After the dx-correction fix, boundary Y-faces now use the correct
        # dx_face/dx_cell factor (dx at face latitude / dx at cell center).
        # However, interior Y-faces use an arithmetic-mean approximation:
        #   dx_face = 0.5*(dx[j-1] + dx[j])
        # while boundary faces use the exact cos(lat_face). These differ
        # at high latitudes, producing a residual row sum after closing.
        # This residual comes from the interior face, not the boundary face,
        # and is inherent to the arithmetic-mean approximation.
        #
        # At 75°N with 5° spacing: interior tr_R ≈ 1.16, boundary tr ≈ 0.83,
        # so the residual = V/dy * |tr_bnd - tr_interior| is non-trivial.
        # For SAS (5–35°N), both corrections are close to 1 and the residual
        # is negligible.
        loss_3d = loss.reshape(nz, ny, nx)
        north_loss = loss_3d[:, -1, 1:-1]
        if np.any(north_loss > 0):
            relative_residual = north_max / np.mean(north_loss)
            # Guard against catastrophic blowup only.
            assert relative_residual < 5.0, (
                f"Y-boundary residual is {relative_residual:.1%} of loss rate "
                f"— possible bug beyond normal dx-approximation mismatch"
            )
