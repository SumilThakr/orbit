"""Tests for chemistry coupling (SO2 oxidation)."""

import numpy as np
from orbit.core.indexing import CellIndexer
from orbit.core.chemistry import assemble_so2_oxidation_loss, assemble_so2_to_pso4_source
from tests.test_advection import _make_grid


class TestChemistry:
    def test_mass_conservation(self, small_grid_params):
        """SO2 loss = pSO4 gain at each cell (chemistry conserves mass)."""
        g = _make_grid(small_grid_params)
        idx = CellIndexer(g.nz, g.ny, g.nx)

        K_loss = assemble_so2_oxidation_loss(g, idx)
        K_source = assemble_so2_to_pso4_source(g, idx)

        # K_loss diagonal = SO2oxidation rate (positive = loss from SO2)
        # K_source diagonal = -SO2oxidation rate (negative = gain to pSO4)
        # Sum should be zero at each cell
        total = K_loss.diagonal() + K_source.diagonal()
        np.testing.assert_allclose(total, 0.0, atol=1e-15)

    def test_loss_positive(self, small_grid_params):
        """SO2 loss rates should be positive."""
        g = _make_grid(small_grid_params)
        idx = CellIndexer(g.nz, g.ny, g.nx)

        K_loss = assemble_so2_oxidation_loss(g, idx)
        assert np.all(K_loss.diagonal() >= 0)

    def test_source_negative(self, small_grid_params):
        """pSO4 source rates should be negative (mass transfer in)."""
        g = _make_grid(small_grid_params)
        idx = CellIndexer(g.nz, g.ny, g.nx)

        K_source = assemble_so2_to_pso4_source(g, idx)
        assert np.all(K_source.diagonal() <= 0)
