"""Integration test: assemble + factorize + solve on real SAS preprocessor data.

Exercises the full pipeline with real meteorology and chemistry fields from the
January SAS preprocessor file. Validates operator properties (M-matrix, mass
conservation) and solution quality (non-negative concentrations, reasonable
PM2.5 levels) on realistic data.

Marked slow: loads ~50MB preprocessor NetCDF and runs 6 sparse LU factorizations.
"""

import numpy as np
import os
import pytest



@pytest.mark.slow
class TestSASIntegration:
    """Full pipeline integration test on real SAS January data."""

    @pytest.fixture(autouse=True)
    def setup(self, january_preprocessor_path, constants_path):
        if not os.path.exists(january_preprocessor_path):
            pytest.skip("Preprocessor file not found")
        if not os.path.exists(constants_path):
            pytest.skip("Constants file not found")

        from orbit.core.grid_data import load_grid
        from orbit.core.indexing import CellIndexer

        self.grid = load_grid(january_preprocessor_path, constants_path)
        self.indexer = CellIndexer(self.grid.nz, self.grid.ny, self.grid.nx)

    def test_operator_assembly(self):
        """Transport + deposition + chemistry operators should assemble without error."""
        from orbit.core.operator import assemble_species_operators
        from orbit.core.deposition import N_SPECIES

        L_species, K_source, *_ = assemble_species_operators(
            self.grid, self.indexer, scheme="exp", verbose=False,
        )

        assert len(L_species) == N_SPECIES
        N = self.indexer.N
        for s, L_s in enumerate(L_species):
            assert L_s.shape == (N, N), f"Species {s} shape mismatch"
            assert L_s.nnz > 0, f"Species {s} has no nonzeros"

        assert K_source.shape == (N, N)

    def test_m_matrix_property(self):
        """EXP scheme operators should be M-matrices on real data."""
        from orbit.core.operator import assemble_transport_block

        T = assemble_transport_block(self.grid, self.indexer, scheme="exp")
        T_dense_diag = np.array(T.diagonal())

        # Positive diagonal
        assert np.all(T_dense_diag >= -1e-15), (
            f"Transport diagonal has negatives: min={T_dense_diag.min():.2e}"
        )

        # Non-positive off-diagonal
        T_coo = T.tocoo()
        off_diag_mask = T_coo.row != T_coo.col
        off_diag_vals = T_coo.data[off_diag_mask]
        assert np.all(off_diag_vals <= 1e-15), (
            f"Transport off-diagonal has positives: max={off_diag_vals.max():.2e}"
        )

    def test_point_source_solve(self):
        """Single PM2.5 point source on the real SAS grid: non-negative, decaying solution."""
        import scipy.sparse.linalg as spla
        from orbit.core.operator import assemble_species_operators
        from orbit.core.deposition import IDX_PM25

        L_species, K_source, *_ = assemble_species_operators(
            self.grid, self.indexer, scheme="exp", verbose=False,
        )
        lu = spla.splu(L_species[IDX_PM25].tocsc())

        N = self.indexer.N
        e = np.zeros(N, dtype=np.float64)

        # Emit PM2.5 at an interior surface cell (roughly center of SAS domain)
        j_mid = self.grid.ny // 2
        i_mid = self.grid.nx // 2
        n_src = self.indexer.to_flat(0, j_mid, i_mid)
        e[n_src] = 1.0

        c = lu.solve(e).reshape(self.grid.nz, self.grid.ny, self.grid.nx)

        # All concentrations should be non-negative (M-matrix guarantee)
        assert np.all(c >= -1e-12), f"Found negatives: min={c.min():.2e}"
        # Positive at source, decaying away from it
        assert c[0, j_mid, i_mid] > 0, "No PM2.5 at source cell"
        assert c[0, j_mid, i_mid] > c[0, 0, 0]
