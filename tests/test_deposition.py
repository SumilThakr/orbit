"""Tests for deposition operator assembly."""

import numpy as np
from orbit.core.indexing import CellIndexer
from orbit.core.deposition import assemble_deposition, N_SPECIES, IDX_SOA, IDX_PM25
from tests.test_advection import _make_grid


class TestDeposition:
    def test_diagonal(self, small_grid_params):
        """Deposition matrix should be diagonal."""
        g = _make_grid(small_grid_params)
        idx = CellIndexer(g.nz, g.ny, g.nx)

        for s in range(N_SPECIES):
            D = assemble_deposition(g, idx, s)
            # Off-diagonal should be zero
            offdiag = D - sp_diag(D)
            assert offdiag.nnz == 0

    def test_positive(self, small_grid_params):
        """All deposition rates should be >= 0."""
        g = _make_grid(small_grid_params)
        idx = CellIndexer(g.nz, g.ny, g.nx)

        for s in range(N_SPECIES):
            D = assemble_deposition(g, idx, s)
            assert np.all(D.diagonal() >= 0)

    def test_surface_has_dry(self, small_grid_params):
        """Surface cells should have higher deposition (dry + wet) than upper cells."""
        g = _make_grid(small_grid_params)
        idx = CellIndexer(g.nz, g.ny, g.nx)

        D = assemble_deposition(g, idx, 1)  # PM2.5
        diag = D.diagonal()

        n_surface = idx.to_flat(0, 1, 1)
        n_upper = idx.to_flat(1, 1, 1)
        # Surface should have higher rate (dry + wet) vs upper (wet only)
        assert diag[n_surface] > diag[n_upper]

    def test_merged_species_effective_rate(self, small_grid_params):
        """Merged species should have partition-weighted effective rates."""
        g = _make_grid(small_grid_params)
        idx = CellIndexer(g.nz, g.ny, g.nx)

        # Total NH (species 2) with NHPartitioning = 0.4
        D = assemble_deposition(g, idx, 2)
        diag = D.diagonal()

        n = idx.to_flat(0, 1, 1)
        p = g.NHPartitioning[0, 1, 1]
        dz = g.Dz[0, 1, 1]
        expected_dry = (1 - p) * g.NH3_dry_dep[0, 1, 1] + p * g.particle_dry_dep[0, 1, 1]
        expected_wet = (1 - p) * g.other_gas_wet_dep[0, 1, 1] + p * g.particle_wet_dep[0, 1, 1]
        expected = expected_dry / dz + expected_wet
        np.testing.assert_allclose(diag[n], expected, rtol=1e-10)

    def test_soa_is_pure_particle(self, small_grid_params):
        """SoA (species 0, yield-at-emission scheme) deposits as pure
        particle, identical to Primary PM2.5 — no AOrgPartitioning blend,
        no gas-channel (VOC_dry_dep / other_gas_wet_dep) contribution."""
        g = _make_grid(small_grid_params)
        idx = CellIndexer(g.nz, g.ny, g.nx)

        D_soa = assemble_deposition(g, idx, IDX_SOA)
        D_pm = assemble_deposition(g, idx, IDX_PM25)

        np.testing.assert_allclose(
            D_soa.diagonal(), D_pm.diagonal(), rtol=1e-12,
            err_msg="SoA deposition rate should match PrimaryPM25 exactly",
        )


def sp_diag(M):
    """Extract diagonal as a sparse matrix."""
    import scipy.sparse as sp
    d = M.diagonal()
    n = len(d)
    return sp.diags(d, shape=(n, n), format="csc")
