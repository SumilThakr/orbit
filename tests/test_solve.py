"""Exp-scheme operator non-negativity (M-matrix property).

The legacy single-shot decoupled solve path (assemble_coupled_operator / factorize /
solve / solve_decoupled / extract_pm25) was retired in the production-branch cleanup
(2026-05-22); the periodic-orbit solver is the only solver. The property still worth
guarding here is structural: the Patankar exponential operator each species is built
on is an M-matrix, so a direct LU back-solve of a non-negative emission yields
non-negative concentrations (no spurious negatives from the discretization).
"""

import numpy as np
import scipy.sparse.linalg as spla
from orbit.core.indexing import CellIndexer
from orbit.core.operator import assemble_species_operators
from orbit.core.deposition import N_SPECIES, IDX_PM25
from tests.test_advection import _make_grid


class TestExpSchemeNonNegativity:
    """EXP scheme guarantees M-matrix -> non-negative concentrations (Issue #9)."""

    def test_point_source_pm25_nonneg(self, small_grid_params):
        """Single PM2.5 point source, direct LU solve: all concentrations >= 0."""
        g = _make_grid(small_grid_params)
        idx = CellIndexer(g.nz, g.ny, g.nx)
        N = idx.N

        L_species, *_ = assemble_species_operators(g, idx, verbose=False, scheme="exp")
        lu = spla.splu(L_species[IDX_PM25].tocsc())

        e = np.zeros(N)
        e[idx.to_flat(0, 2, 2)] = 1.0
        c = lu.solve(e)

        assert np.all(c >= -1e-15), f"Found negatives: min={c.min():.2e}"

    def test_all_species_nonneg(self, small_grid_params):
        """Each per-species exp operator is an M-matrix: non-neg RHS -> non-neg solution."""
        g = _make_grid(small_grid_params)
        idx = CellIndexer(g.nz, g.ny, g.nx)
        N = idx.N

        L_species, *_ = assemble_species_operators(g, idx, verbose=False, scheme="exp")
        rng = np.random.default_rng(123)
        for s in range(N_SPECIES):
            lu = spla.splu(L_species[s].tocsc())
            e = np.zeros(N)
            for _ in range(3):
                j, i = int(rng.integers(0, g.ny)), int(rng.integers(0, g.nx))
                e[idx.to_flat(0, j, i)] = rng.uniform(0.1, 10.0)
            c = lu.solve(e)
            assert np.all(c >= -1e-15), f"species {s}: min={c.min():.2e}"
