"""Tests for full operator assembly."""

import numpy as np
from orbit.core.indexing import CellIndexer
from orbit.core.operator import assemble_transport_block
from tests.test_advection import _make_grid


class TestTransportBlock:
    def test_column_mass_conservation(self, small_grid_params):
        """Column-wise sums of transport block row sums should be >= 0.

        Each column (j,i) summed over all k should be non-negative
        (mass is conserved or lost at boundaries, not created).
        """
        g = _make_grid(small_grid_params)
        idx = CellIndexer(g.nz, g.ny, g.nx)

        T = assemble_transport_block(g, idx, scheme="exp")
        row_sums = np.array(T.sum(axis=1)).ravel()

        # Column-wise sum: for each (j, i), sum row_sums over all k
        for j in range(g.ny):
            for i in range(g.nx):
                col_sum = sum(row_sums[idx.to_flat(k, j, i)] for k in range(g.nz))
                assert col_sum >= -1e-8, f"Column ({j},{i}) sum: {col_sum}"

    def test_positive_diagonal(self, small_grid_params):
        """All diagonal entries should be positive (every cell has some loss)."""
        g = _make_grid(small_grid_params)
        idx = CellIndexer(g.nz, g.ny, g.nx)

        T = assemble_transport_block(g, idx, scheme="exp")
        diag = T.diagonal()
        assert np.all(diag > 0), f"Min diagonal: {diag.min()}"

    def test_single_measure_conservation_dp_area(self, small_grid_params):
        """Full transport block conserves a single mass measure W = dP·area.

        After reconciling vertical diffusion and CMFMC convection to the dP
        measure, all four blocks — horizontal convection-diffusion, vertical
        advection, vertical diffusion, CMFMC convection — conserve Σ dP·area·c,
        so W^T·T = 0 for every fully-interior source cell. The fixture has
        non-uniform density (uniform Dz but Bp-varying dP), so this column-sum
        would NOT vanish under any single measure before the fix (vdiff + CMFMC
        were on the Dz measure). This is the regression certificate for the
        vertical-measure-mismatch fix.
        """
        nz = small_grid_params["nz"]
        ny = small_grid_params["ny"]
        nx = small_grid_params["nx"]
        params = dict(small_grid_params)
        # Downward motion to exercise vertical advection (fixture omega is 0).
        params["omega"] = np.full((nz, ny, nx), 0.02)
        g = _make_grid(params)
        # Convective mass flux with a CLOSED lid (CMFMC[nz]=0 → no top export),
        # so T_conv is active but leaks no mass out of the domain top.
        CMFMC = np.zeros((nz + 1, ny, nx))
        CMFMC[1:nz, :, :] = 0.02
        g.CMFMC = CMFMC
        idx = CellIndexer(g.nz, g.ny, g.nx)

        T = assemble_transport_block(g, idx, scheme="exp")

        # Single conserved weight: W = dP·area, area = dx·dy (dy constant → omit).
        dx_3d = np.broadcast_to(g.dx[None, :, None], (nz, ny, nx))
        W = (dx_3d * g.dP).ravel()
        WT = (W @ T.toarray()).reshape(nz, ny, nx)

        # Fully-interior cells: all six face fluxes stay in-domain, so the
        # weighted column sum cancels to machine precision.
        interior = WT[1:nz - 1, 1:ny - 1, 1:nx - 1]
        assert np.allclose(interior, 0.0, atol=1e-9), (
            f"Max interior |W^T T|: {np.max(np.abs(interior))}"
        )
