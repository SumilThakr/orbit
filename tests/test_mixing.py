"""Tests for mixing operator assembly."""

import numpy as np
import scipy.sparse as sp
from orbit.core.indexing import CellIndexer
from orbit.core.mixing import (
    assemble_vertical_diffusion, assemble_horizontal_diffusion,
    _assemble_vertical_diffusion_loop,
)
from tests.test_advection import _make_grid


class TestVerticalDiffusion:
    def test_dp_weighted_symmetric(self, small_grid_params):
        """Vertical diffusion is self-adjoint in the dP inner product.

        After reconciling the block to the Σ dP·c mass measure, T is no longer
        plain-symmetric when dP varies (it was only symmetric in the old
        Dz-measure form because the fixture has uniform Dz). The correct
        invariant is that diag(dP)·T is symmetric — equivalently, dP·area is the
        conserved weight.
        """
        g = _make_grid(small_grid_params)
        idx = CellIndexer(g.nz, g.ny, g.nx)
        T = assemble_vertical_diffusion(g, idx)

        D = sp.diags(g.dP.ravel())
        M = (D @ T).toarray()
        assert np.abs(M - M.T).max() < 1e-10

    def test_vectorized_matches_loop(self, small_grid_params):
        """Vectorized and loop assemblies are identical."""
        g = _make_grid(small_grid_params)
        idx = CellIndexer(g.nz, g.ny, g.nx)
        T_vec = assemble_vertical_diffusion(g, idx)
        T_loop = _assemble_vertical_diffusion_loop(g, idx)
        diff = (T_vec - T_loop).toarray()
        assert np.allclose(diff, 0, atol=1e-12), (
            f"Max |vec - loop|: {np.max(np.abs(diff))}"
        )

    def test_dp_weighted_conservation_interior(self, small_grid_params):
        """Interior column sums vanish under the dP weight (mass conservation)."""
        g = _make_grid(small_grid_params)
        idx = CellIndexer(g.nz, g.ny, g.nx)
        T = assemble_vertical_diffusion(g, idx)

        dP_flat = g.dP.ravel()
        VtL = dP_flat @ T.toarray()
        # Pure diffusion (no boundary flux through top/bottom): every column
        # conserves exactly.
        assert np.allclose(VtL, 0.0, atol=1e-10), (
            f"Max |dP^T L|: {np.max(np.abs(VtL))}"
        )

    def test_row_sums_zero_interior(self, small_grid_params):
        """Interior cells (not top/bottom) should have row sum = 0."""
        g = _make_grid(small_grid_params)
        idx = CellIndexer(g.nz, g.ny, g.nx)
        T = assemble_vertical_diffusion(g, idx)

        row_sums = np.array(T.sum(axis=1)).ravel()
        # Check interior layer (k=1)
        for j in range(g.ny):
            for i in range(g.nx):
                n = idx.to_flat(1, j, i)
                assert abs(row_sums[n]) < 1e-10


class TestHorizontalDiffusion:
    def test_row_sums_nonneg(self, small_grid_params):
        """All row sums should be >= 0 (diffusion = loss from cell)."""
        g = _make_grid(small_grid_params)
        idx = CellIndexer(g.nz, g.ny, g.nx)
        T = assemble_horizontal_diffusion(g, idx)

        row_sums = np.array(T.sum(axis=1)).ravel()
        assert np.all(row_sums >= -1e-10)

    def test_uniform_k_laplacian(self, small_grid_params):
        """For uniform K, diffusion should be a 5-point Laplacian."""
        g = _make_grid(small_grid_params)
        idx = CellIndexer(g.nz, g.ny, g.nx)
        T = assemble_horizontal_diffusion(g, idx)

        # Interior cell: should have 4 off-diagonal entries (W, E, S, N) + diagonal
        n = idx.to_flat(1, 2, 2)  # Interior cell
        row = T.getrow(n)
        # At least 4 off-diag + 1 diag = 5 nonzeros
        assert row.nnz >= 5
