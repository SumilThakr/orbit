"""Tests for advection stencil classification."""

import numpy as np
from orbit.core.stencil import compute_stencil
from orbit.core.grid_data import GridData


class TestStencilSynthetic:
    def test_uniform_wind_all_land(self, small_grid_params):
        """Uniform wind, all land -> interior cells should have most bits set."""
        g = GridData()
        p = small_grid_params
        g.nz, g.ny, g.nx = p["nz"], p["ny"], p["nx"]
        g.UAvg = p["UAvg"]
        g.VAvg = p["VAvg"]
        g.is_land = p["is_land"]

        stencil = compute_stencil(g)

        assert stencil.shape == (g.nz, g.ny, g.nx)
        assert stencil.dtype == np.uint8

        # Interior cell (k=1, j=2, i=2): should have structural bits set
        # i>=2 -> bit 0x01 (west pos); i<nx-1=3 -> bit 0x02 (west neg)
        # i>=1 -> bit 0x04 (east pos); i<nx-2=2 -> NOPE (i=2, nx-2=2, not <)
        mask = stencil[1, 2, 2]
        assert mask & 0x01  # west pos (i=2 >= 2)
        assert mask & 0x02  # west neg (i=2 < 3)
        assert mask & 0x04  # east pos (i=2 >= 1)
        # bit 3 (0x08, east neg): i < nx-2 = 2, i=2 -> NOT set
        assert not (mask & 0x08)

    def test_boundary_cells_limited(self, small_grid_params):
        """Boundary cells should have fewer bits set."""
        g = GridData()
        p = small_grid_params
        g.nz, g.ny, g.nx = p["nz"], p["ny"], p["nx"]
        g.UAvg = p["UAvg"]
        g.VAvg = p["VAvg"]
        g.is_land = p["is_land"]

        stencil = compute_stencil(g)

        # Corner cell (0, 0, 0): no west/south neighbor for second-order
        mask = stencil[0, 0, 0]
        assert not (mask & 0x01)  # west pos needs i>=2
        assert not (mask & 0x10)  # south pos needs j>=2
