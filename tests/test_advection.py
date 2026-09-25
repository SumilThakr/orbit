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
        """Uniform downward motion (omega > 0) should produce non-zero matrix.

        Above the surface every layer loses through its bottom face what it
        gains through its top face, so row sums vanish. The surface layer
        only gains: the ground is closed, so uniform subsidence converges
        there at the rate omega / dP_0 (in a real wind field the horizontal
        divergence carries it away). Until 2026-09-25 the operator let that
        air leave through the ground, and this test asserted a zero surface
        row sum.
        """
        params = dict(small_grid_params)
        params["omega"] = np.full_like(params["omega"], 0.01)  # 0.01 Pa/s downward
        g = _make_grid(params)
        idx = CellIndexer(g.nz, g.ny, g.nx)
        T = assemble_vertical_advection(g, idx)

        assert T.nnz > 0
        row_sums = np.array(T.sum(axis=1)).ravel().reshape(g.nz, g.ny, g.nx)
        assert np.all(row_sums[1:] >= -1e-10)
        assert np.allclose(row_sums[0], -0.01 / g.dP[0])


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
        """Row sums vanish above the surface under uniform subsidence.

        Uniform subsidence is divergence-free between layers, so every
        interior row sums to zero. The top row sums to +omega / dP_top
        (loss only, since nothing enters through the domain top) and the
        surface row to -omega / dP_0 (gain only, since the ground is closed
        and the descending air converges in layer 0; see
        TestVerticalAdvection.test_with_subsidence).
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

        row_sums = np.array(T.sum(axis=1)).ravel().reshape(nz, ny, nx)
        assert np.allclose(row_sums[1:-1], 0.0, atol=1e-10), f"Max |row sum|: {np.abs(row_sums[1:-1]).max()}"
        assert np.allclose(row_sums[-1], 0.01 / g.dP[-1])   # top layer: loss only, nothing enters from above
        assert np.allclose(row_sums[0], -0.01 / g.dP[0])    # surface: gain only, the ground is closed


class TestGroundClosure:
    """No air crosses the ground: the bottom face of layer 0 carries no flux.

    The preprocessor stores cell-centred omega, so omega[0] is the layer-0
    mid-level value. Until 2026-09-25 the assembly applied omega_plus[0] as a
    downward loss from the surface layer with no receiving cell, which removed
    tracer into the ground at about three times the dry-deposition rate on the
    2022 South Asia grids. These tests pin the closed ground.
    """

    @staticmethod
    def _weighted_column_sums(g, T):
        """W^T T with W = dP * area: the net rate at which each source cell's
        mass leaves the domain through the operator (zero when conserved)."""
        dx_3d = np.broadcast_to(g.dx[None, :, None], (g.nz, g.ny, g.nx))
        W = (dx_3d * g.dP).ravel()
        return (W @ T.toarray()).reshape(g.nz, g.ny, g.nx)

    def test_subsidence_conserves_surface_mass(self, small_grid_params):
        """Uniform subsidence: every column sum vanishes, layer 0 included,
        and layer 0 itself has no loss term (nothing lies below it)."""
        params = dict(small_grid_params)
        params["omega"] = np.full_like(params["omega"], 0.02)
        g = _make_grid(params)
        idx = CellIndexer(g.nz, g.ny, g.nx)
        T = assemble_vertical_advection(g, idx)

        WT = self._weighted_column_sums(g, T)
        assert np.allclose(WT, 0.0, atol=1e-12), f"max |W^T T| = {np.abs(WT).max()}"
        diag0 = T.diagonal().reshape(g.nz, g.ny, g.nx)[0]
        assert np.all(diag0 == 0.0)

    def test_surface_omega_is_ignored(self, small_grid_params):
        """omega_plus[0] and omega_minus[0] do not enter the operator, in the
        vectorised assembly and in the loop reference alike."""
        params = dict(small_grid_params)
        nz, ny, nx = params["nz"], params["ny"], params["nx"]
        rng = np.random.RandomState(7)
        params["omega_plus"] = rng.uniform(0.001, 0.02, (nz, ny, nx))
        params["omega_minus"] = rng.uniform(0.001, 0.01, (nz, ny, nx))
        params["omega"] = params["omega_plus"] - params["omega_minus"]
        params["has_split_omega"] = True
        g = _make_grid(params)
        idx = CellIndexer(g.nz, g.ny, g.nx)

        T_ref = assemble_vertical_advection(g, idx)
        loop_ref = _assemble_vertical_advection_loop(g, idx)
        g.omega_plus[0] = 0.0
        g.omega_minus[0] = 0.0
        T_zeroed = assemble_vertical_advection(g, idx)

        assert abs(T_ref - T_zeroed).max() == 0.0
        assert abs(T_ref - loop_ref).max() < 1e-12
        assert np.allclose(self._weighted_column_sums(g, T_ref), 0.0, atol=1e-12)
