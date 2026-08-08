"""Tests for vectorized vs loop equivalence.

Each test builds both vectorized and loop versions on the synthetic grid
and asserts they produce identical sparse matrices (within machine epsilon).
"""

import numpy as np
from orbit.core.indexing import CellIndexer
from orbit.core.advection import (
    assemble_vertical_advection, _assemble_vertical_advection_loop,
)
from orbit.core.mixing import (
    assemble_vertical_diffusion, _assemble_vertical_diffusion_loop,
    assemble_horizontal_diffusion, _assemble_horizontal_diffusion_loop,
)
from orbit.core.deposition import (
    assemble_deposition, _assemble_deposition_loop, N_SPECIES,
)
from orbit.core.chemistry import (
    assemble_so2_oxidation_loss, _assemble_so2_oxidation_loss_loop,
    assemble_so2_to_pso4_source, _assemble_so2_to_pso4_source_loop,
)
from tests.test_advection import _make_grid


def _max_abs_diff(A, B):
    """Max absolute difference between two sparse matrices."""
    diff = A - B
    if diff.nnz == 0:
        return 0.0
    return np.max(np.abs(diff.data))


class TestChemistryEquivalence:
    def test_so2_loss(self, small_grid_params):
        g = _make_grid(small_grid_params)
        idx = CellIndexer(g.nz, g.ny, g.nx)
        vec = assemble_so2_oxidation_loss(g, idx)
        loop = _assemble_so2_oxidation_loss_loop(g, idx)
        assert _max_abs_diff(vec, loop) < 1e-15

    def test_so2_source(self, small_grid_params):
        g = _make_grid(small_grid_params)
        idx = CellIndexer(g.nz, g.ny, g.nx)
        vec = assemble_so2_to_pso4_source(g, idx)
        loop = _assemble_so2_to_pso4_source_loop(g, idx)
        assert _max_abs_diff(vec, loop) < 1e-15


class TestDepositionEquivalence:
    def test_all_species(self, small_grid_params):
        g = _make_grid(small_grid_params)
        idx = CellIndexer(g.nz, g.ny, g.nx)
        for s in range(N_SPECIES):
            vec = assemble_deposition(g, idx, s)
            loop = _assemble_deposition_loop(g, idx, s)
            diff = _max_abs_diff(vec, loop)
            assert diff < 1e-15, f"Species {s}: max diff = {diff}"


class TestVerticalAdvectionEquivalence:
    def test_zero_omega(self, small_grid_params):
        g = _make_grid(small_grid_params)
        idx = CellIndexer(g.nz, g.ny, g.nx)
        vec = assemble_vertical_advection(g, idx)
        loop = _assemble_vertical_advection_loop(g, idx)
        assert vec.nnz == 0 and loop.nnz == 0

    def test_with_omega(self, small_grid_params):
        params = dict(small_grid_params)
        rng = np.random.RandomState(123)
        params["omega"] = rng.randn(*params["omega"].shape) * 0.01
        g = _make_grid(params)
        idx = CellIndexer(g.nz, g.ny, g.nx)
        vec = assemble_vertical_advection(g, idx)
        loop = _assemble_vertical_advection_loop(g, idx)
        assert _max_abs_diff(vec, loop) < 1e-15


class TestVerticalDiffusionEquivalence:
    def test_uniform_kzz(self, small_grid_params):
        g = _make_grid(small_grid_params)
        idx = CellIndexer(g.nz, g.ny, g.nx)
        vec = assemble_vertical_diffusion(g, idx)
        loop = _assemble_vertical_diffusion_loop(g, idx)
        assert _max_abs_diff(vec, loop) < 1e-15

    def test_varying_kzz(self, small_grid_params):
        params = dict(small_grid_params)
        rng = np.random.RandomState(456)
        params["Kzz"] = np.abs(rng.randn(*params["Kzz"].shape)) * 20 + 1.0
        g = _make_grid(params)
        idx = CellIndexer(g.nz, g.ny, g.nx)
        vec = assemble_vertical_diffusion(g, idx)
        loop = _assemble_vertical_diffusion_loop(g, idx)
        assert _max_abs_diff(vec, loop) < 1e-15


class TestHorizontalDiffusionEquivalence:
    def test_uniform(self, small_grid_params):
        g = _make_grid(small_grid_params)
        idx = CellIndexer(g.nz, g.ny, g.nx)
        vec = assemble_horizontal_diffusion(g, idx)
        loop = _assemble_horizontal_diffusion_loop(g, idx)
        assert _max_abs_diff(vec, loop) < 1e-15

    def test_varying_k(self, small_grid_params):
        params = dict(small_grid_params)
        rng = np.random.RandomState(789)
        params["Kxxyy"] = np.abs(rng.randn(*params["Kxxyy"].shape)) * 200 + 10
        params["K_meander_u"] = np.abs(rng.randn(*params["K_meander_u"].shape)) * 100
        params["K_meander_v"] = np.abs(rng.randn(*params["K_meander_v"].shape)) * 100
        g = _make_grid(params)
        idx = CellIndexer(g.nz, g.ny, g.nx)
        vec = assemble_horizontal_diffusion(g, idx)
        loop = _assemble_horizontal_diffusion_loop(g, idx)
        assert _max_abs_diff(vec, loop) < 1e-15


