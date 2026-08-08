"""Cell indexing: (k, j, i) <-> flat index mapping for sparse matrix assembly."""



class CellIndexer:
    """Maps between 3D grid indices (k, j, i) and flat indices for sparse matrices.

    Uses k-major ordering: n = k * (ny * nx) + j * nx + i

    Parameters
    ----------
    nz, ny, nx : int
        Grid dimensions (levels, lat, lon)
    """

    def __init__(self, nz: int, ny: int, nx: int):
        self.nz = nz
        self.ny = ny
        self.nx = nx
        self.N = nz * ny * nx
        self._ny_nx = ny * nx

    def to_flat(self, k: int, j: int, i: int) -> int:
        """Convert (k, j, i) to flat index."""
        return k * self._ny_nx + j * self.nx + i

    def to_3d(self, n: int) -> tuple:
        """Convert flat index to (k, j, i)."""
        k = n // self._ny_nx
        rem = n % self._ny_nx
        j = rem // self.nx
        i = rem % self.nx
        return (k, j, i)

    def species_offset(self, s: int) -> int:
        """Offset for species s in coupled vector: s * N."""
        return s * self.N

    def coupled_index(self, s: int, k: int, j: int, i: int) -> int:
        """Index in the coupled (6*N) vector for species s at cell (k,j,i)."""
        return self.species_offset(s) + self.to_flat(k, j, i)

    def is_valid(self, k: int, j: int, i: int) -> bool:
        """Check if (k, j, i) is within domain."""
        return 0 <= k < self.nz and 0 <= j < self.ny and 0 <= i < self.nx
