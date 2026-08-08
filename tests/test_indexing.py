"""Tests for CellIndexer."""

from orbit.core.indexing import CellIndexer


class TestCellIndexer:
    def test_roundtrip(self):
        idx = CellIndexer(3, 4, 5)
        for k in range(3):
            for j in range(4):
                for i in range(5):
                    n = idx.to_flat(k, j, i)
                    assert idx.to_3d(n) == (k, j, i)

    def test_flat_ordering(self):
        idx = CellIndexer(2, 3, 4)
        # k-major: n = k*(ny*nx) + j*nx + i
        assert idx.to_flat(0, 0, 0) == 0
        assert idx.to_flat(0, 0, 3) == 3
        assert idx.to_flat(0, 1, 0) == 4
        assert idx.to_flat(1, 0, 0) == 12  # 1 * 3*4

    def test_total_cells(self):
        idx = CellIndexer(5, 10, 8)
        assert idx.N == 5 * 10 * 8

    def test_species_offset(self):
        idx = CellIndexer(3, 4, 5)
        N = 60
        assert idx.species_offset(0) == 0
        assert idx.species_offset(1) == N
        assert idx.species_offset(5) == 5 * N

    def test_coupled_index(self):
        idx = CellIndexer(3, 4, 5)
        N = 60
        # species 2, cell (1, 2, 3) -> 2*N + to_flat(1,2,3)
        n = idx.to_flat(1, 2, 3)
        assert idx.coupled_index(2, 1, 2, 3) == 2 * N + n

    def test_is_valid(self):
        idx = CellIndexer(3, 4, 5)
        assert idx.is_valid(0, 0, 0)
        assert idx.is_valid(2, 3, 4)
        assert not idx.is_valid(-1, 0, 0)
        assert not idx.is_valid(3, 0, 0)
        assert not idx.is_valid(0, 4, 0)
        assert not idx.is_valid(0, 0, 5)
