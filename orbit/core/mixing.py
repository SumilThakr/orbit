"""Mixing operator assembly: vertical diffusion, horizontal diffusion.

Ports mixing physics from physics_uniform.pyx into sparse matrix rows.
"""

import numpy as np
import scipy.sparse as sp
from orbit.core.indexing import CellIndexer
from orbit.core.grid_data import GridData

GRAVITY = 9.80665  # m/s², standard gravity


def assemble_vertical_diffusion(grid: GridData, indexer: CellIndexer) -> sp.csc_matrix:
    """Assemble vertical diffusion operator (pressure-thickness mass measure).

    Diffusive flux uses the geometric gradient:
        flux = K_face * dC / center_dist          [center_dist = 0.5*(Dz_k+Dz_{k+1})]
    but the flux divergence is normalised by the *pressure-thickness* mass
    measure dP rather than the geometric height Dz, so this block conserves the
    same Σ dP_k·c_k weight as horizontal convection-diffusion and vertical
    advection. Concretely each face coefficient carries a ρ_face/ρ_cell factor
    (ρ = dP/(g·Dz)):

        coeff_k = (K_face / center_dist) * ρ_face * g / dP_k

    This reduces to the old Σ Dz_k·c_k form exactly when ρ_face = ρ_cell (no
    vertical density gradient); the correction is O(density-gradient) and
    matters in the strong-mixing PBL over orography. Reconciling this block (and
    CMFMC convection) to dP makes the full transport operator conserve a single
    mass measure W = dP·area.

    K_face = harmonic_mean(Kzz[k], Kzz[k+1]).
    """
    nz, ny, nx = grid.nz, grid.ny, grid.nx
    N = indexer.N
    Kzz = grid.Kzz
    Dz = grid.Dz
    dP = grid.dP

    n3d = np.arange(N, dtype=np.int64).reshape(nz, ny, nx)

    # Layer-mean air density ρ_k = dP_k / (g·Dz_k); 0 where the layer is
    # degenerate (handled by the ratio guard below, which falls back to 1.0).
    safe_Dz_all = np.where(Dz > 0, Dz, 1.0)
    rho_all = np.where((Dz > 0) & (dP > 0), dP / (GRAVITY * safe_Dz_all), 0.0)

    diag = np.zeros((nz, ny, nx), dtype=np.float64)
    rows_list = []
    cols_list = []
    vals_list = []

    # For each interface between layer k and k+1 (k = 0..nz-2)
    for k in range(nz - 1):
        dz_k = Dz[k]       # (ny, nx)
        dz_k1 = Dz[k + 1]  # (ny, nx)

        valid = (dz_k > 0) & (dz_k1 > 0)
        if not np.any(valid):
            continue

        K_k = Kzz[k]
        K_k1 = Kzz[k + 1]

        K_face = np.where(valid, 2.0 * K_k * K_k1 / (K_k + K_k1 + 1e-30), 0.0)
        dist = np.where(valid, 0.5 * (dz_k + dz_k1), 1.0)

        # ρ_face/ρ_cell reweight: moves the flux divergence onto the dP mass
        # measure. Falls back to 1.0 (old Dz-measure form) where either layer
        # density is undefined.
        rho_k = rho_all[k]
        rho_k1 = rho_all[k + 1]
        rho_face = 0.5 * (rho_k + rho_k1)
        both_rho = valid & (rho_k > 0) & (rho_k1 > 0)
        ratio_k = np.where(both_rho, rho_face / np.where(rho_k > 0, rho_k, 1.0), 1.0)
        ratio_k1 = np.where(both_rho, rho_face / np.where(rho_k1 > 0, rho_k1, 1.0), 1.0)

        # Coefficient for layer k looking at k+1
        safe_dz_k = np.where(dz_k > 0, dz_k, 1.0)
        coeff_k = np.where(valid, K_face / (safe_dz_k * dist) * ratio_k, 0.0)

        # Coefficient for layer k+1 looking at k
        safe_dz_k1 = np.where(dz_k1 > 0, dz_k1, 1.0)
        coeff_k1 = np.where(valid, K_face / (safe_dz_k1 * dist) * ratio_k1, 0.0)

        idx = np.where(valid)
        if idx[0].size == 0:
            continue

        n_k = n3d[k][idx]
        n_k1 = n3d[k + 1][idx]

        # Layer k: off-diag to k+1
        rows_list.append(n_k); cols_list.append(n_k1); vals_list.append(-coeff_k[idx])
        # Layer k+1: off-diag to k
        rows_list.append(n_k1); cols_list.append(n_k); vals_list.append(-coeff_k1[idx])

        # Diagonal contributions
        diag[k][idx] += coeff_k[idx]
        diag[k + 1][idx] += coeff_k1[idx]

    # Add diagonal entries where diag > 0
    diag_pos = diag > 0
    if np.any(diag_pos) or rows_list:
        idx_d = np.where(diag_pos)
        rows_list.append(n3d[idx_d])
        cols_list.append(n3d[idx_d])
        vals_list.append(diag[idx_d])

    if not rows_list:
        return sp.csc_matrix((N, N))

    return sp.csc_matrix(
        (np.concatenate(vals_list),
         (np.concatenate(rows_list), np.concatenate(cols_list))),
        shape=(N, N),
    )


def assemble_horizontal_diffusion(grid: GridData, indexer: CellIndexer) -> sp.csc_matrix:
    """Assemble horizontal diffusion operator.

    K_meander_u/v are face-centered (staggered): K_meander_u[:,:,i] is the meander
    diffusivity at face i (west face of cell i). Kxxyy is cell-centered.

    K_face = harmonic_mean(Kxxyy_L, Kxxyy_R) + K_meander_face
    Flux = K_face * dC / (dx * dx)  [uniform grid: center_dist = dx]

    When split fluxes are present, K_meander is already in UAvg_plus/minus,
    so only the cell-centered Kxxyy contributes to diffusive K.

    No terrain ratio in sigma-native mode (matching physics_uniform.pyx:629).
    """
    nz, ny, nx = grid.nz, grid.ny, grid.nx
    N = indexer.N
    Kxxyy = grid.Kxxyy

    n3d = np.arange(N, dtype=np.int64).reshape(nz, ny, nx)

    # dx varies with latitude
    dx_3d = np.broadcast_to(grid.dx[np.newaxis, :, np.newaxis], (nz, ny, nx)).copy()
    dy = grid.dy

    valid = dx_3d > 0
    use_split = grid.has_split_fluxes

    diag = np.zeros((nz, ny, nx), dtype=np.float64)
    rows_list = []
    cols_list = []
    vals_list = []

    # West neighbor (i-1): face i for i=1..nx-1
    if nx > 1:
        K_cell = Kxxyy[:, :, 1:]     # (nz, ny, nx-1)
        K_nb = Kxxyy[:, :, :-1]
        K_face = 2.0 * K_cell * K_nb / (K_cell + K_nb + 1e-30)
        if not use_split:
            K_face = K_face + grid.K_meander_u[:, :, 1:]
        dx_slice = dx_3d[:, :, 1:]
        coeff = K_face / (dx_slice * dx_slice)

        v = valid[:, :, 1:]
        idx = np.where(v)
        if idx[0].size > 0:
            rows_list.append(n3d[:, :, 1:][idx])
            cols_list.append(n3d[:, :, :-1][idx])
            vals_list.append(-coeff[idx])
            diag[:, :, 1:] += np.where(v, coeff, 0.0)

    # East neighbor (i+1): same faces i=1..nx-1
    if nx > 1:
        K_cell = Kxxyy[:, :, :-1]
        K_nb = Kxxyy[:, :, 1:]
        K_face = 2.0 * K_cell * K_nb / (K_cell + K_nb + 1e-30)
        if not use_split:
            K_face = K_face + grid.K_meander_u[:, :, 1:]
        dx_slice = dx_3d[:, :, :-1]
        coeff = K_face / (dx_slice * dx_slice)

        v = valid[:, :, :-1]
        idx = np.where(v)
        if idx[0].size > 0:
            rows_list.append(n3d[:, :, :-1][idx])
            cols_list.append(n3d[:, :, 1:][idx])
            vals_list.append(-coeff[idx])
            diag[:, :, :-1] += np.where(v, coeff, 0.0)

    # Wrap neighbor (periodic lon: cell 0 <-> cell nx-1)
    if grid.periodic_lon and nx > 1:
        K_c0 = Kxxyy[:, :, 0]
        K_cnx = Kxxyy[:, :, -1]
        K_face_wrap = 2.0 * K_c0 * K_cnx / (K_c0 + K_cnx + 1e-30)
        if not use_split:
            K_face_wrap = K_face_wrap + grid.K_meander_u_wrap
        dx_wrap = dx_3d[:, :, 0]  # same latitude row
        coeff_wrap = K_face_wrap / (dx_wrap * dx_wrap)

        v_wrap = valid[:, :, 0] & valid[:, :, -1]
        idx_w = np.where(v_wrap)
        if idx_w[0].size > 0:
            # cell 0 -> cell nx-1
            rows_list.append(n3d[:, :, 0][idx_w])
            cols_list.append(n3d[:, :, -1][idx_w])
            vals_list.append(-coeff_wrap[idx_w])
            diag[:, :, 0][idx_w] += coeff_wrap[idx_w]
            # cell nx-1 -> cell 0
            rows_list.append(n3d[:, :, -1][idx_w])
            cols_list.append(n3d[:, :, 0][idx_w])
            vals_list.append(-coeff_wrap[idx_w])
            diag[:, :, -1][idx_w] += coeff_wrap[idx_w]

    # South neighbor (j-1): face j for j=1..ny-1
    if ny > 1:
        K_cell = Kxxyy[:, 1:, :]
        K_nb = Kxxyy[:, :-1, :]
        K_face = 2.0 * K_cell * K_nb / (K_cell + K_nb + 1e-30)
        if not use_split:
            K_face = K_face + grid.K_meander_v[:, 1:, :]
        coeff = K_face / (dy * dy)

        v = valid[:, 1:, :]
        idx = np.where(v)
        if idx[0].size > 0:
            rows_list.append(n3d[:, 1:, :][idx])
            cols_list.append(n3d[:, :-1, :][idx])
            vals_list.append(-coeff[idx])
            diag[:, 1:, :] += np.where(v, coeff, 0.0)

    # North neighbor (j+1): same faces j=1..ny-1
    if ny > 1:
        K_cell = Kxxyy[:, :-1, :]
        K_nb = Kxxyy[:, 1:, :]
        K_face = 2.0 * K_cell * K_nb / (K_cell + K_nb + 1e-30)
        if not use_split:
            K_face = K_face + grid.K_meander_v[:, 1:, :]
        coeff = K_face / (dy * dy)

        v = valid[:, :-1, :]
        idx = np.where(v)
        if idx[0].size > 0:
            rows_list.append(n3d[:, :-1, :][idx])
            cols_list.append(n3d[:, 1:, :][idx])
            vals_list.append(-coeff[idx])
            diag[:, :-1, :] += np.where(v, coeff, 0.0)

    # Diagonal
    diag_pos = diag > 0
    if np.any(diag_pos):
        idx_d = np.where(diag_pos)
        rows_list.append(n3d[idx_d])
        cols_list.append(n3d[idx_d])
        vals_list.append(diag[idx_d])

    if not rows_list:
        return sp.csc_matrix((N, N))

    return sp.csc_matrix(
        (np.concatenate(vals_list),
         (np.concatenate(rows_list), np.concatenate(cols_list))),
        shape=(N, N),
    )


# --- Loop versions for equivalence testing ---

def _assemble_vertical_diffusion_loop(grid: GridData, indexer: CellIndexer) -> sp.csc_matrix:
    nz, ny, nx = grid.nz, grid.ny, grid.nx
    N = indexer.N
    Kzz = grid.Kzz
    Dz = grid.Dz
    dP = grid.dP
    rows, cols, vals = [], [], []

    def _rho(kk, jj, ii):
        dz = Dz[kk, jj, ii]
        dp = dP[kk, jj, ii]
        if dz > 0 and dp > 0:
            return dp / (GRAVITY * dz)
        return 0.0

    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                dz = Dz[k, j, i]
                if dz <= 0:
                    continue
                n = indexer.to_flat(k, j, i)
                diag = 0.0
                rho_kc = _rho(k, j, i)
                if k < nz - 1:
                    dz_above = Dz[k + 1, j, i]
                    if dz_above > 0:
                        K_face = 2.0 * Kzz[k, j, i] * Kzz[k + 1, j, i] / (
                            Kzz[k, j, i] + Kzz[k + 1, j, i] + 1e-30)
                        dist = 0.5 * (dz + dz_above)
                        rho_ab = _rho(k + 1, j, i)
                        ratio = (0.5 * (rho_kc + rho_ab) / rho_kc
                                 if rho_kc > 0 and rho_ab > 0 else 1.0)
                        coeff = K_face / (dz * dist) * ratio
                        n_above = indexer.to_flat(k + 1, j, i)
                        rows.append(n); cols.append(n_above); vals.append(-coeff)
                        diag += coeff
                if k > 0:
                    dz_below = Dz[k - 1, j, i]
                    if dz_below > 0:
                        K_face = 2.0 * Kzz[k, j, i] * Kzz[k - 1, j, i] / (
                            Kzz[k, j, i] + Kzz[k - 1, j, i] + 1e-30)
                        dist = 0.5 * (dz + dz_below)
                        rho_be = _rho(k - 1, j, i)
                        ratio = (0.5 * (rho_kc + rho_be) / rho_kc
                                 if rho_kc > 0 and rho_be > 0 else 1.0)
                        coeff = K_face / (dz * dist) * ratio
                        n_below = indexer.to_flat(k - 1, j, i)
                        rows.append(n); cols.append(n_below); vals.append(-coeff)
                        diag += coeff
                if diag > 0:
                    rows.append(n); cols.append(n); vals.append(diag)

    return sp.csc_matrix(
        (np.array(vals), (np.array(rows, dtype=np.int64), np.array(cols, dtype=np.int64))),
        shape=(N, N),
    )


def _assemble_horizontal_diffusion_loop(grid: GridData, indexer: CellIndexer) -> sp.csc_matrix:
    nz, ny, nx = grid.nz, grid.ny, grid.nx
    N = indexer.N
    Kxxyy = grid.Kxxyy
    K_meander_u = grid.K_meander_u
    K_meander_v = grid.K_meander_v
    use_split = grid.has_split_fluxes
    rows, cols, vals = [], [], []

    for k in range(nz):
        for j in range(ny):
            dx_j = grid.dx[j]
            if dx_j <= 0:
                continue
            dy = grid.dy
            for i in range(nx):
                n = indexer.to_flat(k, j, i)
                diag = 0.0

                # West neighbor: face i
                if i > 0:
                    K_face = 2.0 * Kxxyy[k, j, i] * Kxxyy[k, j, i - 1] / (
                        Kxxyy[k, j, i] + Kxxyy[k, j, i - 1] + 1e-30)
                    if not use_split:
                        K_face += K_meander_u[k, j, i]
                    coeff = K_face / (dx_j * dx_j)
                    n_west = indexer.to_flat(k, j, i - 1)
                    rows.append(n); cols.append(n_west); vals.append(-coeff)
                    diag += coeff
                elif grid.periodic_lon and nx > 1:
                    # i=0: west wrap neighbor is nx-1
                    K_face = 2.0 * Kxxyy[k, j, 0] * Kxxyy[k, j, nx - 1] / (
                        Kxxyy[k, j, 0] + Kxxyy[k, j, nx - 1] + 1e-30)
                    if not use_split:
                        K_face += grid.K_meander_u_wrap[k, j]
                    coeff = K_face / (dx_j * dx_j)
                    n_west = indexer.to_flat(k, j, nx - 1)
                    rows.append(n); cols.append(n_west); vals.append(-coeff)
                    diag += coeff
                # East neighbor: face i+1
                if i < nx - 1:
                    K_face = 2.0 * Kxxyy[k, j, i] * Kxxyy[k, j, i + 1] / (
                        Kxxyy[k, j, i] + Kxxyy[k, j, i + 1] + 1e-30)
                    if not use_split:
                        K_face += K_meander_u[k, j, i + 1]
                    coeff = K_face / (dx_j * dx_j)
                    n_east = indexer.to_flat(k, j, i + 1)
                    rows.append(n); cols.append(n_east); vals.append(-coeff)
                    diag += coeff
                elif grid.periodic_lon and nx > 1:
                    # i=nx-1: east wrap neighbor is 0
                    K_face = 2.0 * Kxxyy[k, j, nx - 1] * Kxxyy[k, j, 0] / (
                        Kxxyy[k, j, nx - 1] + Kxxyy[k, j, 0] + 1e-30)
                    if not use_split:
                        K_face += grid.K_meander_u_wrap[k, j]
                    coeff = K_face / (dx_j * dx_j)
                    n_east = indexer.to_flat(k, j, 0)
                    rows.append(n); cols.append(n_east); vals.append(-coeff)
                    diag += coeff
                # South neighbor: face j
                if j > 0:
                    K_face = 2.0 * Kxxyy[k, j, i] * Kxxyy[k, j - 1, i] / (
                        Kxxyy[k, j, i] + Kxxyy[k, j - 1, i] + 1e-30)
                    if not use_split:
                        K_face += K_meander_v[k, j, i]
                    coeff = K_face / (dy * dy)
                    n_south = indexer.to_flat(k, j - 1, i)
                    rows.append(n); cols.append(n_south); vals.append(-coeff)
                    diag += coeff
                # North neighbor: face j+1
                if j < ny - 1:
                    K_face = 2.0 * Kxxyy[k, j, i] * Kxxyy[k, j + 1, i] / (
                        Kxxyy[k, j, i] + Kxxyy[k, j + 1, i] + 1e-30)
                    if not use_split:
                        K_face += K_meander_v[k, j + 1, i]
                    coeff = K_face / (dy * dy)
                    n_north = indexer.to_flat(k, j + 1, i)
                    rows.append(n); cols.append(n_north); vals.append(-coeff)
                    diag += coeff
                if diag > 0:
                    rows.append(n); cols.append(n); vals.append(diag)

    return sp.csc_matrix(
        (np.array(vals), (np.array(rows, dtype=np.int64), np.array(cols, dtype=np.int64))),
        shape=(N, N),
    )
