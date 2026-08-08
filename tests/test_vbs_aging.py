"""Tests for VBS aging K-block construction in operator.py."""
from __future__ import annotations


import numpy as np
import pytest
from orbit.core.deposition import (
    IDX_VBS_C100, IDX_VBS_C10, IDX_VBS_C1, IDX_VBS_C01, IDX_VBS_C1000,
    VBS_AGING_PAIRS,
)


class TestAgingPairs:
    def test_chain_top_to_bottom(self):
        """VBS_AGING_PAIRS encodes C1000 → C100 → C10 → C1 → C01."""
        sources = [pair[0] for pair in VBS_AGING_PAIRS]
        destinations = [pair[1] for pair in VBS_AGING_PAIRS]
        assert sources == [IDX_VBS_C1000, IDX_VBS_C100, IDX_VBS_C10, IDX_VBS_C1]
        assert destinations == [IDX_VBS_C100, IDX_VBS_C10, IDX_VBS_C1, IDX_VBS_C01]

    def test_no_self_aging(self):
        for src, dst in VBS_AGING_PAIRS:
            assert src != dst

    def test_C01_terminal(self):
        """No aging entry has C01 as a source (it's the lowest bin)."""
        for src, _ in VBS_AGING_PAIRS:
            assert src != IDX_VBS_C01


@pytest.fixture
def grid_with_OH(small_grid_params):
    """Build a small grid with a synthetic archive_OH field."""
    from tests.test_advection import _make_grid
    g = _make_grid(small_grid_params)
    # Synthetic OH field: 1e6 molec/cm³ daytime tropics.
    g.archive_OH = np.full((g.nz, g.ny, g.nx), 1.0e6, dtype=np.float64)
    # Need archive_NO2 for full operator wiring (the existing nox_to_no3
    # builder also reads it).
    g.archive_NO2 = np.full((g.nz, g.ny, g.nx), 1.0e10, dtype=np.float64)
    return g


class TestAgingKBlocks:
    def test_K_blocks_present(self, grid_with_OH, monkeypatch):
        """When archive_OH is populated, the operator has 4 aging K-blocks."""
        monkeypatch.setenv("ORBIT_VBS_K_AGE", "4.0e-11")
        monkeypatch.setenv("ORBIT_VBS_FRAG", "0.75")
        from orbit.core.operator import assemble_species_operators
        from orbit.core.indexing import CellIndexer
        idx = CellIndexer(grid_with_OH.nz, grid_with_OH.ny, grid_with_OH.nx)
        nox_to_no3_zero = np.zeros(
            (grid_with_OH.nz, grid_with_OH.ny, grid_with_OH.nx))
        L_species, K_sources, T, d = assemble_species_operators(
            grid_with_OH, idx, scheme="exp",
            nox_to_no3_rate=nox_to_no3_zero,
        )
        for src, dst in VBS_AGING_PAIRS:
            assert (dst, src) in K_sources, \
                f"missing aging K-block ({dst}, {src})"

    def test_K_block_signs(self, grid_with_OH, monkeypatch):
        """K_age_src has negative diagonals; aging-loss adds positive
        rate to the source bin's loss diagonal."""
        monkeypatch.setenv("ORBIT_VBS_K_AGE", "4.0e-11")
        monkeypatch.setenv("ORBIT_VBS_FRAG", "0.75")
        from orbit.core.operator import assemble_species_operators
        from orbit.core.indexing import CellIndexer
        idx = CellIndexer(grid_with_OH.nz, grid_with_OH.ny, grid_with_OH.nx)
        # Suppress auto-build of NOx→NO3 (needs archive_M which the
        # synthetic grid doesn't have). Pass explicit zeros instead.
        nox_to_no3_zero = np.zeros(
            (grid_with_OH.nz, grid_with_OH.ny, grid_with_OH.nx))
        _, K_sources, _, _ = assemble_species_operators(
            grid_with_OH, idx, scheme="exp",
            nox_to_no3_rate=nox_to_no3_zero,
        )
        for src, dst in VBS_AGING_PAIRS:
            K = K_sources[(dst, src)]
            # K_age_src is -α_frag × rate_diag; should be ≤ 0 everywhere.
            d = K.diagonal()
            assert np.all(d <= 0), \
                f"K_age_src({dst},{src}) has positive diagonal entries"

    def test_no_aging_when_OH_absent(self, monkeypatch, small_grid_params):
        """Grid without archive_OH → no VBS aging K-blocks."""
        monkeypatch.setenv("ORBIT_VBS_K_AGE", "4.0e-11")
        from tests.test_advection import _make_grid
        from orbit.core.operator import assemble_species_operators
        from orbit.core.indexing import CellIndexer
        g = _make_grid(small_grid_params)
        # Don't populate archive_OH.
        idx = CellIndexer(g.nz, g.ny, g.nx)
        _, K_sources, _, _ = assemble_species_operators(
            g, idx, scheme="exp",
        )
        for src, dst in VBS_AGING_PAIRS:
            assert (dst, src) not in K_sources

    def test_alpha_frag_scales_destination(self, grid_with_OH, monkeypatch):
        """α_frag = 0.5 should give half the destination rate vs α_frag = 1.0."""
        from orbit.core.operator import assemble_species_operators
        from orbit.core.indexing import CellIndexer
        idx = CellIndexer(grid_with_OH.nz, grid_with_OH.ny, grid_with_OH.nx)

        nox_to_no3_zero = np.zeros(
            (grid_with_OH.nz, grid_with_OH.ny, grid_with_OH.nx))

        monkeypatch.setenv("ORBIT_VBS_FRAG", "1.0")
        _, K_sources_full, _, _ = assemble_species_operators(
            grid_with_OH, idx, scheme="exp",
            nox_to_no3_rate=nox_to_no3_zero,
        )
        K_full = K_sources_full[(IDX_VBS_C100, IDX_VBS_C1000)].diagonal()

        monkeypatch.setenv("ORBIT_VBS_FRAG", "0.5")
        _, K_sources_half, _, _ = assemble_species_operators(
            grid_with_OH, idx, scheme="exp",
            nox_to_no3_rate=nox_to_no3_zero,
        )
        K_half = K_sources_half[(IDX_VBS_C100, IDX_VBS_C1000)].diagonal()

        # K_half = 0.5 × K_full
        np.testing.assert_allclose(K_half, 0.5 * K_full, rtol=1e-12)
