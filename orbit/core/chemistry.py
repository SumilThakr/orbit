"""Chemistry coupling: SO2 oxidation to pSO4.

The only off-diagonal coupling block in the 6-species merged system.
SO2 loses mass at rate SO2oxidation[k,j,i], pSO4 gains that same mass.
"""

import numpy as np
import scipy.sparse as sp
from orbit.core.indexing import CellIndexer
from orbit.core.grid_data import GridData


def assemble_so2_oxidation_loss(grid: GridData, indexer: CellIndexer) -> sp.csc_matrix:
    """Diagonal matrix of SO2 oxidation loss rate.

    Added to SO2 block diagonal: L_SO2[n,n] += SO2oxidation[k,j,i]

    Returns
    -------
    K_ox : csc_matrix, shape (N, N)
    """
    N = indexer.N
    diag = grid.SO2oxidation.ravel()
    indices = np.arange(N, dtype=np.int64)
    return sp.csc_matrix((diag, (indices, indices)), shape=(N, N))


def assemble_so2_to_pso4_source(grid: GridData, indexer: CellIndexer) -> sp.csc_matrix:
    """Off-diagonal coupling: pSO4 gains from SO2 at the same cell.

    In the coupled system, this appears as:
      L_coupled[pSO4_row_n, SO2_col_n] = -SO2oxidation[k,j,i]

    The negative sign means mass transfers FROM SO2 TO pSO4.

    Returns
    -------
    K_source : csc_matrix, shape (N, N)
        Off-diagonal block to be placed at (pSO4_rows, SO2_cols) in coupled system
    """
    N = indexer.N
    diag = -grid.SO2oxidation.ravel()
    indices = np.arange(N, dtype=np.int64)
    return sp.csc_matrix((diag, (indices, indices)), shape=(N, N))


# --- Loop versions for equivalence testing ---

def _assemble_so2_oxidation_loss_loop(grid: GridData, indexer: CellIndexer) -> sp.csc_matrix:
    N = indexer.N
    diag = np.zeros(N, dtype=np.float64)
    for k in range(grid.nz):
        for j in range(grid.ny):
            for i in range(grid.nx):
                diag[indexer.to_flat(k, j, i)] = grid.SO2oxidation[k, j, i]
    indices = np.arange(N, dtype=np.int64)
    return sp.csc_matrix((diag, (indices, indices)), shape=(N, N))


def _assemble_so2_to_pso4_source_loop(grid: GridData, indexer: CellIndexer) -> sp.csc_matrix:
    N = indexer.N
    diag = np.zeros(N, dtype=np.float64)
    for k in range(grid.nz):
        for j in range(grid.ny):
            for i in range(grid.nx):
                diag[indexer.to_flat(k, j, i)] = -grid.SO2oxidation[k, j, i]
    indices = np.arange(N, dtype=np.int64)
    return sp.csc_matrix((diag, (indices, indices)), shape=(N, N))
