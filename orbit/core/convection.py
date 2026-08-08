"""Convective mass-flux transport operator assembly.

Adds MERRA-2 CMFMC (upward convective mass flux) as a vertical transport term.
CMFMC is an air mass flux (kg/m²/s) at layer interfaces that carries tracer
upward at the source cell's mixing ratio c_k/ρ_k.

Interface convention (same as LayerHeights):
    CMFMC[0]   = ground (should be zero)
    CMFMC[k]   = interface between layer k-1 and layer k
    CMFMC[nz]  = top of domain (nonzero → export loss to free troposphere)

ORBIT L-convention: positive diagonal (loss), negative off-diagonal (gain).

See notes/2026-04-16_convective_transport_design.md for full derivation.
"""

import numpy as np
import scipy.sparse as sp
from orbit.core.indexing import CellIndexer
from orbit.core.grid_data import GridData

GRAVITY = 9.80665  # m/s², standard gravity


def assemble_cmfmc_transport(grid: GridData, indexer: CellIndexer) -> sp.csc_matrix:
    """Assemble convective mass-flux transport operator.

    For each layer k, the top interface flux F = CMFMC[k+1] drives:
      - Loss from k:   diag[k] += F / (ρ_k · Dz_k)       = F·g / dP_k
      - Gain at k+1:   off_diag[k+1,k] -= F / (ρ_{k+1} · Dz_{k+1}) = F·g / dP_{k+1}
    where ρ_k = dP_k / (g · Dz_k) is the layer-mean air density.

    The gain uses the RECEIVER density ρ_{k+1}, so loss and gain both reduce to a
    flux per unit pressure thickness (F·g/dP). This makes the operator conserve
    the Σ dP_k·c_k mass measure used by horizontal convection-diffusion and
    vertical advection — i.e. the whole transport operator conserves a single
    weight W = dP·area. Using the source density ρ_k for the gain (the
    previous form) instead conserved Σ Dz_k·c_k, an
    O(density-gradient) inconsistency with the other blocks.
    At the domain top (k = nz-1), loss only — no receiver above.

    Parameters
    ----------
    grid : GridData
    indexer : CellIndexer

    Returns
    -------
    T_conv : csc_matrix, shape (N, N)
        Zero matrix when grid.CMFMC is empty (backward compatible).
    """
    nz, ny, nx = grid.nz, grid.ny, grid.nx
    N = indexer.N

    # Guard: return zero matrix when CMFMC is absent
    if grid.CMFMC.size == 0:
        return sp.csc_matrix((N, N))

    CMFMC = grid.CMFMC   # (nz+1, ny, nx)
    dP = grid.dP          # (nz, ny, nx) Pa
    Dz = grid.Dz          # (nz, ny, nx) m

    n3d = np.arange(N, dtype=np.int64).reshape(nz, ny, nx)

    # Air density per layer: ρ_k = dP_k / (g · Dz_k)
    safe_Dz = np.where(Dz > 0, Dz, 1.0)
    rho = dP / (GRAVITY * safe_Dz)  # (nz, ny, nx)
    safe_rho = np.where(rho > 0, rho, 1.0)

    valid = (Dz > 0) & (dP > 0)

    diag = np.zeros((nz, ny, nx), dtype=np.float64)
    rows_list = []
    cols_list = []
    vals_list = []

    # Process each layer k: top interface flux is CMFMC[k+1]
    for k in range(nz):
        F = CMFMC[k + 1]  # (ny, nx), flux at top of layer k
        F = np.maximum(F, 0.0)  # Guard against negatives

        active = valid[k] & (F > 0)
        if not np.any(active):
            continue

        rate_source = np.where(active, F / (safe_rho[k] * safe_Dz[k]), 0.0)

        # Loss from layer k (positive diagonal in L-convention)
        diag[k] += rate_source

        # Gain at layer k+1 (if not top layer). Divide by the RECEIVER density
        # ρ_{k+1} so the gain is F·g/dP_{k+1} — conserving the Σ dP·c measure.
        if k < nz - 1:
            safe_Dz_above = np.where(Dz[k + 1] > 0, Dz[k + 1], 1.0)
            valid_above = valid[k + 1]
            gain_active = active & valid_above

            if np.any(gain_active):
                rate_receiver = np.where(
                    gain_active,
                    F / (safe_rho[k + 1] * safe_Dz_above),
                    0.0,
                )
                idx = np.where(gain_active)
                rows_list.append(n3d[k + 1][idx])
                cols_list.append(n3d[k][idx])
                vals_list.append(-rate_receiver[idx])  # negative = gain

    # Assemble diagonal
    diag_pos = diag > 0
    if not np.any(diag_pos) and not rows_list:
        return sp.csc_matrix((N, N))

    idx_diag = np.where(diag_pos)
    rows_list.append(n3d[idx_diag])
    cols_list.append(n3d[idx_diag])
    vals_list.append(diag[idx_diag])

    all_rows = np.concatenate(rows_list)
    all_cols = np.concatenate(cols_list)
    all_vals = np.concatenate(vals_list)

    return sp.csc_matrix((all_vals, (all_rows, all_cols)), shape=(N, N))


def _assemble_cmfmc_transport_loop(
    grid: GridData, indexer: CellIndexer,
) -> sp.csc_matrix:
    """Loop version of assemble_cmfmc_transport for equivalence testing."""
    nz, ny, nx = grid.nz, grid.ny, grid.nx
    N = indexer.N

    if grid.CMFMC.size == 0:
        return sp.csc_matrix((N, N))

    CMFMC = grid.CMFMC
    dP = grid.dP
    Dz = grid.Dz

    rows, cols, vals = [], [], []

    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                dz_k = Dz[k, j, i]
                dp_k = dP[k, j, i]
                if dz_k <= 0 or dp_k <= 0:
                    continue

                rho_k = dp_k / (GRAVITY * dz_k)
                if rho_k <= 0:
                    continue

                F = max(CMFMC[k + 1, j, i], 0.0)
                if F <= 0:
                    continue

                n = indexer.to_flat(k, j, i)
                loss_rate = F / (rho_k * dz_k)

                # Loss from k
                rows.append(n)
                cols.append(n)
                vals.append(loss_rate)

                # Gain at k+1 (if not top layer). Receiver density ρ_{k+1} so the
                # gain is F·g/dP_{k+1} (Σ dP·c measure).
                if k < nz - 1:
                    dz_above = Dz[k + 1, j, i]
                    dp_above = dP[k + 1, j, i]
                    if dz_above > 0 and dp_above > 0:
                        n_above = indexer.to_flat(k + 1, j, i)
                        rho_above = dp_above / (GRAVITY * dz_above)
                        gain_rate = F / (rho_above * dz_above)
                        rows.append(n_above)
                        cols.append(n)
                        vals.append(-gain_rate)

    if len(vals) == 0:
        return sp.csc_matrix((N, N))

    return sp.csc_matrix(
        (np.array(vals), (np.array(rows, dtype=np.int64), np.array(cols, dtype=np.int64))),
        shape=(N, N),
    )
