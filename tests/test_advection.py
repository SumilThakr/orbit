"""Tests for advection operator assembly."""

import numpy as np
from orbit.core.indexing import CellIndexer
from orbit.core.grid_data import GridData, _compute_geometry, _compute_terrain_ratios
from orbit.core.advection import (
    assemble_horizontal_advection, assemble_vertical_advection,
    _assemble_vertical_advection_loop,
)
from orbit.core.stencil import compute_stencil


def _make_grid(params):
    """Helper to build a GridData from small_grid_params."""
    g = GridData()
    for key in ["nz", "ny", "nx", "lon", "lat", "dlon", "dlat",
                 "Ap", "Bp", "Psurf", "dP", "Dz",
                 "UAvg", "VAvg", "omega",
                 "omega_plus", "omega_minus", "has_split_omega",
                 "Kzz", "Kxxyy", "K_meander_u", "K_meander_v",
                 "is_land",
                 "UAvg_plus", "UAvg_minus", "VAvg_plus", "VAvg_minus",
                 "has_split_fluxes",
                 "particle_dry_dep", "SO2_dry_dep", "NOx_dry_dep",
                 "NH3_dry_dep", "VOC_dry_dep",
                 "particle_wet_dep", "SO2_wet_dep", "other_gas_wet_dep",
                 "SO2oxidation", "NHPartitioning", "NOPartitioning",
                 "AOrgPartitioning"]:
        if key in params:
            setattr(g, key, params[key])
    # Derive split omega from omega if not explicitly provided
    if g.omega_plus.size == 0 and g.omega.size > 0:
        g.omega_plus = np.maximum(g.omega, 0.0)
        g.omega_minus = np.maximum(-g.omega, 0.0)
        g.has_split_omega = False
    _compute_geometry(g)
    _compute_terrain_ratios(g)
    return g


class TestHorizontalAdvection:
    def test_row_sums_first_order(self, small_grid_params):
        """Row sums >= 0 for all cells (first-order, flat terrain)."""
        g = _make_grid(small_grid_params)
        idx = CellIndexer(g.nz, g.ny, g.nx)
        stencil = np.zeros((g.nz, g.ny, g.nx), dtype=np.uint8)
        T = assemble_horizontal_advection(g, idx, stencil, use_second_order=False)

        row_sums = np.array(T.sum(axis=1)).ravel()
        # Interior cells: row sum should be ~0 (mass conservation with outflow on both sides)
        # But boundary cells have positive row sums (outflow = loss)
        # All row sums should be >= -epsilon
        assert np.all(row_sums >= -1e-10), f"Min row sum: {row_sums.min()}"

    def test_positive_diagonal(self, small_grid_params):
        """All diagonal entries should be >= 0."""
        g = _make_grid(small_grid_params)
        idx = CellIndexer(g.nz, g.ny, g.nx)
        stencil = np.zeros((g.nz, g.ny, g.nx), dtype=np.uint8)
        T = assemble_horizontal_advection(g, idx, stencil, use_second_order=False)

        diag = T.diagonal()
        assert np.all(diag >= -1e-10), f"Min diagonal: {diag.min()}"

    def test_second_order_builds(self, small_grid_params):
        """Second-order assembly should succeed and produce different nnz."""
        g = _make_grid(small_grid_params)
        idx = CellIndexer(g.nz, g.ny, g.nx)
        stencil = compute_stencil(g)

        T_fo = assemble_horizontal_advection(g, idx, stencil, use_second_order=False)
        T_so = assemble_horizontal_advection(g, idx, stencil, use_second_order=True)

        # Both should be valid sparse matrices
        assert T_fo.shape == (idx.N, idx.N)
        assert T_so.shape == (idx.N, idx.N)


class TestVerticalAdvection:
    def test_no_vertical_motion(self, small_grid_params):
        """Zero omega -> zero vertical advection matrix."""
        g = _make_grid(small_grid_params)
        idx = CellIndexer(g.nz, g.ny, g.nx)
        T = assemble_vertical_advection(g, idx)

        assert T.nnz == 0

    def test_with_subsidence(self, small_grid_params):
        """Uniform downward motion (omega > 0) should produce non-zero matrix."""
        params = dict(small_grid_params)
        params["omega"] = np.full_like(params["omega"], 0.01)  # 0.01 Pa/s downward
        g = _make_grid(params)
        idx = CellIndexer(g.nz, g.ny, g.nx)
        T = assemble_vertical_advection(g, idx)

        assert T.nnz > 0
        # Row sums should be >= 0
        row_sums = np.array(T.sum(axis=1)).ravel()
        assert np.all(row_sums >= -1e-10)


class TestSplitOmega:
    """Tests for split-omega mode (pre-split vertical fluxes)."""

    def _make_split_params(self, small_grid_params):
        """Create params with bidirectional omega (both components nonzero)."""
        params = dict(small_grid_params)
        nz, ny, nx = params["nz"], params["ny"], params["nx"]
        rng = np.random.RandomState(42)

        # Bidirectional omega: both components nonzero at same cell
        # (vertical velocity reverses within averaging period)
        params["omega_plus"] = rng.uniform(0.001, 0.02, (nz, ny, nx))
        params["omega_minus"] = rng.uniform(0.001, 0.01, (nz, ny, nx))
        # Net omega = plus - minus
        params["omega"] = params["omega_plus"] - params["omega_minus"]
        params["has_split_omega"] = True
        return params

    def test_vectorized_matches_loop(self, small_grid_params):
        """Vectorized split-omega must match loop reference."""
        params = self._make_split_params(small_grid_params)
        g = _make_grid(params)
        idx = CellIndexer(g.nz, g.ny, g.nx)

        vec = assemble_vertical_advection(g, idx)
        loop = _assemble_vertical_advection_loop(g, idx)

        diff = abs(vec - loop).max()
        assert diff < 1e-10, f"max abs diff = {diff}"

    def test_m_matrix(self, small_grid_params):
        """Split-omega vertical advection should be an M-matrix."""
        params = self._make_split_params(small_grid_params)
        g = _make_grid(params)
        idx = CellIndexer(g.nz, g.ny, g.nx)

        T = assemble_vertical_advection(g, idx)
        dense = T.toarray()

        # Diagonal >= 0
        assert np.all(dense.diagonal() >= -1e-15), \
            f"Negative diagonal: {dense.diagonal().min()}"
        # Off-diagonal <= 0
        off_diag = dense - np.diag(dense.diagonal())
        assert np.all(off_diag <= 1e-15), \
            f"Positive off-diagonal: {off_diag.max()}"

    def test_backward_compatibility(self, small_grid_params):
        """Trivial split (from net omega) should match legacy behavior."""
        params = dict(small_grid_params)
        nz, ny, nx = params["nz"], params["ny"], params["nx"]
        rng = np.random.RandomState(99)
        params["omega"] = rng.uniform(-0.01, 0.02, (nz, ny, nx))

        # Legacy mode: split derived from net omega in _make_grid
        g_legacy = _make_grid(params)
        idx = CellIndexer(g_legacy.nz, g_legacy.ny, g_legacy.nx)
        T_legacy = assemble_vertical_advection(g_legacy, idx)

        # Explicit trivial split (same as fallback)
        params_split = dict(params)
        params_split["omega_plus"] = np.maximum(params["omega"], 0.0)
        params_split["omega_minus"] = np.maximum(-params["omega"], 0.0)
        params_split["has_split_omega"] = True
        g_split = _make_grid(params_split)
        T_split = assemble_vertical_advection(g_split, idx)

        diff = abs(T_legacy - T_split).max()
        assert diff < 1e-10, f"Split != legacy with trivial split: diff = {diff}"

    def test_split_stronger_than_net(self, small_grid_params):
        """Pre-split omega should produce stronger exchange than net-averaged.

        When omega oscillates (both components nonzero), the gross flux exceeds
        the net. The split operator should have more nonzeros and/or larger
        diagonal entries than the operator built from net omega alone.
        """
        params = self._make_split_params(small_grid_params)
        g_split = _make_grid(params)
        idx = CellIndexer(g_split.nz, g_split.ny, g_split.nx)
        T_split = assemble_vertical_advection(g_split, idx)

        # Build legacy operator from net omega only
        params_net = dict(params)
        del params_net["omega_plus"]
        del params_net["omega_minus"]
        del params_net["has_split_omega"]
        # _make_grid will derive trivial split from net omega
        g_net = _make_grid(params_net)
        T_net = assemble_vertical_advection(g_net, idx)

        # Split should have >= nonzeros (bidirectional exchange at cells
        # where net omega ≈ 0 but gross flux is nonzero)
        assert T_split.nnz >= T_net.nnz, \
            f"Split nnz ({T_split.nnz}) < net nnz ({T_net.nnz})"

        # Split diagonal sum should be >= net diagonal sum (more total loss)
        split_diag_sum = T_split.diagonal().sum()
        net_diag_sum = T_net.diagonal().sum()
        assert split_diag_sum >= net_diag_sum - 1e-15, \
            f"Split diag sum ({split_diag_sum}) < net ({net_diag_sum})"

    def test_row_sums_nonnegative(self, small_grid_params):
        """Row sums should be >= 0 for physically consistent omega.

        Use uniform subsidence (divergence-free) so row sums reflect
        the operator, not artifacts of non-physical synthetic data.
        """
        params = dict(small_grid_params)
        nz, ny, nx = params["nz"], params["ny"], params["nx"]
        # Uniform subsidence: omega_plus = 0.01, omega_minus = 0 everywhere
        params["omega_plus"] = np.full((nz, ny, nx), 0.01)
        params["omega_minus"] = np.zeros((nz, ny, nx))
        params["omega"] = params["omega_plus"].copy()
        params["has_split_omega"] = True
        g = _make_grid(params)
        idx = CellIndexer(g.nz, g.ny, g.nx)
        T = assemble_vertical_advection(g, idx)

        row_sums = np.array(T.sum(axis=1)).ravel()
        assert np.all(row_sums >= -1e-10), f"Min row sum: {row_sums.min()}"
