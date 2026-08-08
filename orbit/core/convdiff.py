"""Combined horizontal convection-diffusion operator (Patankar exponential scheme).

Replaces separate horizontal advection + diffusion assembly when scheme="exp".
Each face uses the exact 1D convection-diffusion solution, producing an M-matrix
by construction (zero negatives guaranteed for any Peclet number).

Reference: Patankar, S.V. (1980) Numerical Heat Transfer and Fluid Flow, Ch. 5.
"""

import numpy as np
import scipy.sparse as sp
from orbit.core.indexing import CellIndexer
from orbit.core.grid_data import GridData, EARTH_RADIUS, DEG_TO_RAD


def _patankar_A(pe_abs: np.ndarray) -> np.ndarray:
    """Compute A(|Pe|) = |Pe| / (exp(|Pe|) - 1), vectorized with guards.

    Limits:
    - Pe -> 0: A -> 1 - Pe/2 (Taylor expansion, avoids 0/0)
    - Pe -> inf: A -> 0 (pure upwind, avoids exp overflow)

    Parameters
    ----------
    pe_abs : ndarray
        Absolute Peclet numbers (must be >= 0)

    Returns
    -------
    A : ndarray, same shape as pe_abs
    """
    result = np.empty_like(pe_abs)
    small = pe_abs < 1e-6
    large = pe_abs > 500.0
    mid = ~small & ~large
    result[small] = 1.0 - 0.5 * pe_abs[small]
    result[large] = 0.0
    result[mid] = pe_abs[mid] / np.expm1(pe_abs[mid])
    return result


def assemble_horizontal_convdiff(
    grid: GridData, indexer: CellIndexer,
) -> sp.csc_matrix:
    """Assemble combined horizontal convection-diffusion (Patankar exponential).

    Replaces separate advection + diffusion assembly. Each face uses the exact 1D
    convection-diffusion solution to compute combined coefficients. The resulting
    matrix is an M-matrix (non-negative diagonal, non-positive off-diagonal).

    Face coefficient formulation (for face between cell L and cell R, u pointing L->R):
        D = K_face / dx                          [m/s]
        Pe = u * dx / K_face                      [-]
        a_L = D * A(|Pe|) + max(u, 0)            [m/s]  (always >= 0)
        a_R = D * A(|Pe|) + max(-u, 0)           [m/s]  (always >= 0)

    Matrix entry = a / dx  [1/s]

    Parameters
    ----------
    grid : GridData
    indexer : CellIndexer

    Returns
    -------
    T_hcd : csc_matrix, shape (N, N)
    """
    nz, ny, nx = grid.nz, grid.ny, grid.nx
    N = indexer.N
    U = grid.UAvg
    V = grid.VAvg
    use_split = grid.has_split_fluxes

    Kxxyy = grid.Kxxyy

    n3d = np.arange(N, dtype=np.int64).reshape(nz, ny, nx)

    dx_3d = np.broadcast_to(grid.dx[np.newaxis, :, np.newaxis], (nz, ny, nx)).copy()
    dy = grid.dy
    valid = dx_3d > 0

    diag = np.zeros((nz, ny, nx), dtype=np.float64)
    rows_list = []
    cols_list = []
    vals_list = []

    # ========== X-FACES (between cell i-1 and cell i, for i=1..nx-1) ==========
    if nx > 1:
        u_face = U[:, :, 1:]  # (nz, ny, nx-1)
        dx_face = dx_3d[:, :, 1:]

        # K_meander_u is face-centered: K_meander_u[:,:,i] = meander K at face i.
        # Interior x-faces are i=1..nx-1. Harmonic-mean the cell-centered Kxxyy,
        # then add the face-centered K_meander at its correct position.
        K_L = Kxxyy[:, :, :-1]
        K_R = Kxxyy[:, :, 1:]
        K_face = 2.0 * K_L * K_R / (K_L + K_R + 1e-30)
        if not use_split:
            K_face = K_face + grid.K_meander_u[:, :, 1:]

        safe_dx = np.where(dx_face > 0, dx_face, 1.0)
        D_face = K_face / safe_dx

        safe_K = np.where(K_face > 1e-30, K_face, 1e-30)
        Pe = u_face * safe_dx / safe_K
        Pe_abs = np.minimum(np.abs(Pe), 500.0)
        A_pe = _patankar_A(Pe_abs)

        # Face coefficients (Patankar): a_L/dx and a_R/dx
        diff_rate = D_face * A_pe / safe_dx
        if use_split:
            adv_L_rate = grid.UAvg_plus[:, :, 1:] / safe_dx
            adv_R_rate = grid.UAvg_minus[:, :, 1:] / safe_dx
        else:
            adv_L_rate = np.maximum(u_face, 0.0) / safe_dx
            adv_R_rate = np.maximum(-u_face, 0.0) / safe_dx

        rate_L = diff_rate + adv_L_rate
        rate_R = diff_rate + adv_R_rate

        # Mass-space terrain correction: dP_face / dP_cell
        # Both diagonal (loss) and off-diagonal (gain) get the same factor,
        # ensuring face-by-face mass conservation: W_R*tr_R = W_L*tr_L
        # Guard: when either cell has non-positive dP (degenerate layer), use tr=1.0
        dP_xL = grid.dP[:, :, :-1]
        dP_xR = grid.dP[:, :, 1:]
        dP_face_x = 0.5 * (dP_xL + dP_xR)
        valid_dP_x = (dP_xL > 0) & (dP_xR > 0)
        tr_R = np.where(valid_dP_x, dP_face_x / dP_xR, 1.0)
        tr_L = np.where(valid_dP_x, dP_face_x / dP_xL, 1.0)

        v = valid[:, :, 1:] & valid[:, :, :-1]
        n_right = n3d[:, :, 1:]
        n_left = n3d[:, :, :-1]

        idx = np.where(v)
        if idx[0].size > 0:
            # Right cell (i): diag += rate_R * tr_R, off(R,L) = -rate_L * tr_R
            diag[:, :, 1:][idx] += (rate_R * tr_R)[idx]
            rows_list.append(n_right[idx])
            cols_list.append(n_left[idx])
            vals_list.append(-(rate_L * tr_R)[idx])

            # Left cell (i-1): diag += rate_L * tr_L, off(L,R) = -rate_R * tr_L
            diag[:, :, :-1][idx] += (rate_L * tr_L)[idx]
            rows_list.append(n_left[idx])
            cols_list.append(n_right[idx])
            vals_list.append(-(rate_R * tr_L)[idx])

    # ========== X WRAP FACE (between cell nx-1 and cell 0, periodic) ==========
    if grid.periodic_lon and nx > 1:
        u_wrap = grid.UAvg_wrap  # (nz, ny)
        dx_wrap = dx_3d[:, :, 0]  # dx at the wrap face (same latitude)

        K_L_wrap = Kxxyy[:, :, -1]
        K_R_wrap = Kxxyy[:, :, 0]
        K_face_wrap = 2.0 * K_L_wrap * K_R_wrap / (K_L_wrap + K_R_wrap + 1e-30)
        if not use_split:
            K_face_wrap = K_face_wrap + grid.K_meander_u_wrap

        safe_dx_wrap = np.where(dx_wrap > 0, dx_wrap, 1.0)
        D_wrap = K_face_wrap / safe_dx_wrap

        safe_K_wrap = np.where(K_face_wrap > 1e-30, K_face_wrap, 1e-30)
        Pe_wrap = u_wrap * safe_dx_wrap / safe_K_wrap
        Pe_abs_wrap = np.minimum(np.abs(Pe_wrap), 500.0)
        A_pe_wrap = _patankar_A(Pe_abs_wrap)

        diff_rate_wrap = D_wrap * A_pe_wrap / safe_dx_wrap
        if use_split:
            adv_L_wrap = grid.UAvg_plus_wrap / safe_dx_wrap
            adv_R_wrap = grid.UAvg_minus_wrap / safe_dx_wrap
        else:
            adv_L_wrap = np.maximum(u_wrap, 0.0) / safe_dx_wrap
            adv_R_wrap = np.maximum(-u_wrap, 0.0) / safe_dx_wrap

        rate_L_wrap = diff_rate_wrap + adv_L_wrap  # loss rate for L=nx-1
        rate_R_wrap = diff_rate_wrap + adv_R_wrap  # loss rate for R=0

        # Terrain correction
        dP_wL = grid.dP[:, :, -1]
        dP_wR = grid.dP[:, :, 0]
        dP_face_wrap = 0.5 * (dP_wL + dP_wR)
        valid_dP_wrap = (dP_wL > 0) & (dP_wR > 0)
        tr_R_wrap = np.where(valid_dP_wrap, dP_face_wrap / dP_wR, 1.0)
        tr_L_wrap = np.where(valid_dP_wrap, dP_face_wrap / dP_wL, 1.0)

        v_wrap = (dx_wrap > 0)
        idx_w = np.where(v_wrap)
        if idx_w[0].size > 0:
            n_cell0 = n3d[:, :, 0]
            n_cellnx = n3d[:, :, -1]
            # Right cell (0): diag += rate_R * tr_R, off(0, nx-1) = -rate_L * tr_R
            diag[:, :, 0][idx_w] += (rate_R_wrap * tr_R_wrap)[idx_w]
            rows_list.append(n_cell0[idx_w])
            cols_list.append(n_cellnx[idx_w])
            vals_list.append(-(rate_L_wrap * tr_R_wrap)[idx_w])
            # Left cell (nx-1): diag += rate_L * tr_L, off(nx-1, 0) = -rate_R * tr_L
            diag[:, :, -1][idx_w] += (rate_L_wrap * tr_L_wrap)[idx_w]
            rows_list.append(n_cellnx[idx_w])
            cols_list.append(n_cell0[idx_w])
            vals_list.append(-(rate_R_wrap * tr_L_wrap)[idx_w])

    # X-boundary loss (only for non-periodic grids)
    if not grid.periodic_lon:
        # X-boundary: i=0, outflux when u < 0 (loss only, no neighbor)
        bnd_w = valid.copy()
        bnd_w[:, :, 1:] = False  # only i=0
        if use_split:
            um_w0 = grid.UAvg_minus[:, :, 0:1]
            bnd_w_loss = bnd_w & np.broadcast_to(um_w0 > 0, (nz, ny, nx))
            if np.any(bnd_w_loss):
                safe_dx_bnd = np.where(dx_3d > 0, dx_3d, 1.0)
                um_bnd = np.zeros((nz, ny, nx), dtype=np.float64)
                um_bnd[:, :, 0] = grid.UAvg_minus[:, :, 0]
                diag += np.where(bnd_w_loss, um_bnd / safe_dx_bnd, 0.0)
        else:
            u_w0 = U[:, :, 0:1]
            bnd_w_loss = bnd_w & np.broadcast_to(u_w0 < 0, (nz, ny, nx))
            if np.any(bnd_w_loss):
                safe_dx_bnd = np.where(dx_3d > 0, dx_3d, 1.0)
                u_abs_bnd = np.zeros((nz, ny, nx), dtype=np.float64)
                u_abs_bnd[:, :, 0] = np.abs(U[:, :, 0])
                diag += np.where(bnd_w_loss, u_abs_bnd / safe_dx_bnd, 0.0)

        # X-boundary: i=nx-1, outflux when u_e >= 0 (loss only, no neighbor)
        bnd_e = valid.copy()
        bnd_e[:, :, :-1] = False  # only i=nx-1
        if use_split:
            up_bnd = np.zeros((nz, ny, nx), dtype=np.float64)
            up_bnd[:, :, -1] = grid.UAvg_plus[:, :, -1]
            bnd_e_loss = bnd_e & (up_bnd > 0)
            if np.any(bnd_e_loss):
                safe_dx_bnd = np.where(dx_3d > 0, dx_3d, 1.0)
                diag += np.where(bnd_e_loss, up_bnd / safe_dx_bnd, 0.0)
        elif nx > 1:
            ue_bnd = np.zeros((nz, ny, nx), dtype=np.float64)
            ue_bnd[:, :, -1] = U[:, :, -1]
            bnd_e_loss = bnd_e & (ue_bnd >= 0)
            if np.any(bnd_e_loss):
                safe_dx_bnd = np.where(dx_3d > 0, dx_3d, 1.0)
                diag += np.where(bnd_e_loss, ue_bnd / safe_dx_bnd, 0.0)
        elif nx == 1:
            ue_bnd = np.zeros((nz, ny, nx), dtype=np.float64)
            ue_bnd[:, :, 0] = U[:, :, 0]
            bnd_e_loss = bnd_e & (ue_bnd >= 0)
            if np.any(bnd_e_loss):
                safe_dx_bnd = np.where(dx_3d > 0, dx_3d, 1.0)
                diag += np.where(bnd_e_loss, ue_bnd / safe_dx_bnd, 0.0)

    # ========== Y-FACES (between cell j-1 and cell j, for j=1..ny-1) ==========
    if ny > 1:
        v_face = V[:, 1:, :]  # (nz, ny-1, nx)
        safe_dy = dy if dy > 0 else 1.0

        # K_meander_v is face-centered: K_meander_v[:,j,:] = meander K at face j.
        # Interior y-faces are j=1..ny-1.
        K_L = Kxxyy[:, :-1, :]
        K_R = Kxxyy[:, 1:, :]
        K_face = 2.0 * K_L * K_R / (K_L + K_R + 1e-30)
        if not use_split:
            K_face = K_face + grid.K_meander_v[:, 1:, :]

        D_face = K_face / safe_dy

        safe_K = np.where(K_face > 1e-30, K_face, 1e-30)
        Pe = v_face * safe_dy / safe_K
        Pe_abs = np.minimum(np.abs(Pe), 500.0)
        A_pe = _patankar_A(Pe_abs)

        diff_rate = D_face * A_pe / safe_dy
        if use_split:
            adv_L_rate = grid.VAvg_plus[:, 1:, :] / safe_dy
            adv_R_rate = grid.VAvg_minus[:, 1:, :] / safe_dy
        else:
            adv_L_rate = np.maximum(v_face, 0.0) / safe_dy
            adv_R_rate = np.maximum(-v_face, 0.0) / safe_dy

        rate_L = diff_rate + adv_L_rate
        rate_R = diff_rate + adv_R_rate

        # Mass-space terrain correction: (dx_face / dx_cell) * (dP_face / dP_cell)
        # Y-faces need dx correction because cells at different latitudes have
        # different zonal widths. This ensures W_R*tr_R = W_L*tr_L at each face.
        # Guard: when either cell has non-positive dP (degenerate layer), use tr=1.0
        dx_south = dx_3d[:, :-1, :]
        dx_north = dx_3d[:, 1:, :]
        dx_face_y = 0.5 * (dx_south + dx_north)

        dP_yS = grid.dP[:, :-1, :]
        dP_yN = grid.dP[:, 1:, :]
        dP_face_y = 0.5 * (dP_yS + dP_yN)

        valid_dP_y = (dP_yS > 0) & (dP_yN > 0)
        safe_dx_n = np.where(dx_north > 0, dx_north, 1.0)
        safe_dx_s = np.where(dx_south > 0, dx_south, 1.0)
        tr_R = np.where(valid_dP_y, (dx_face_y / safe_dx_n) * (dP_face_y / dP_yN), 1.0)
        tr_L = np.where(valid_dP_y, (dx_face_y / safe_dx_s) * (dP_face_y / dP_yS), 1.0)

        v_mask = valid[:, 1:, :] & valid[:, :-1, :]
        n_north = n3d[:, 1:, :]
        n_south = n3d[:, :-1, :]

        idx = np.where(v_mask)
        if idx[0].size > 0:
            # North cell (j): diag += rate_R * tr_R, off(N,S) = -rate_L * tr_R
            diag[:, 1:, :][idx] += (rate_R * tr_R)[idx]
            rows_list.append(n_north[idx])
            cols_list.append(n_south[idx])
            vals_list.append(-(rate_L * tr_R)[idx])

            # South cell (j-1): diag += rate_L * tr_L, off(S,N) = -rate_R * tr_L
            diag[:, :-1, :][idx] += (rate_L * tr_L)[idx]
            rows_list.append(n_south[idx])
            cols_list.append(n_north[idx])
            vals_list.append(-(rate_R * tr_L)[idx])

    # Y-boundary: j=0, outflux when v < 0
    # Apply dx_face/dx_cell correction (face is at lat[0] - dlat/2).
    # Interior Y-faces apply this correction; boundary faces must too.
    bnd_s = valid.copy()
    bnd_s[:, 1:, :] = False  # only j=0
    lat_face_s = grid.lat[0] - grid.dlat / 2.0
    dx_face_s = EARTH_RADIUS * np.cos(lat_face_s * DEG_TO_RAD) * grid.dlon * DEG_TO_RAD
    safe_dx_s0 = grid.dx[0] if grid.dx[0] > 0 else 1.0
    tr_bnd_s = dx_face_s / safe_dx_s0
    if use_split:
        # Split-flux: outflux at south boundary = VAvg_minus (southward component)
        vm_s0 = grid.VAvg_minus[:, 0:1, :]
        bnd_s_loss = bnd_s & np.broadcast_to(vm_s0 > 0, (nz, ny, nx))
        if np.any(bnd_s_loss):
            safe_dy_val = dy if dy > 0 else 1.0
            diag += np.where(bnd_s_loss, grid.VAvg_minus / safe_dy_val * tr_bnd_s, 0.0)
    else:
        v_s0 = V[:, 0:1, :]
        bnd_s_loss = bnd_s & np.broadcast_to(v_s0 < 0, (nz, ny, nx))
        if np.any(bnd_s_loss):
            safe_dy_val = dy if dy > 0 else 1.0
            diag += np.where(bnd_s_loss, np.abs(V) / safe_dy_val * tr_bnd_s, 0.0)

    # Y-boundary: j=ny-1, outflux when v_n >= 0
    # Apply dx_face/dx_cell correction (face is at lat[-1] + dlat/2).
    bnd_n = valid.copy()
    bnd_n[:, :-1, :] = False  # only j=ny-1
    lat_face_n = grid.lat[-1] + grid.dlat / 2.0
    dx_face_n = EARTH_RADIUS * np.cos(lat_face_n * DEG_TO_RAD) * grid.dlon * DEG_TO_RAD
    safe_dx_nend = grid.dx[-1] if grid.dx[-1] > 0 else 1.0
    tr_bnd_n = dx_face_n / safe_dx_nend
    if use_split:
        # Split-flux: outflux at north boundary = VAvg_plus (northward component)
        vp_bnd = np.zeros((nz, ny, nx), dtype=np.float64)
        vp_bnd[:, -1, :] = grid.VAvg_plus[:, -1, :]
        bnd_n_loss = bnd_n & (vp_bnd > 0)
        if np.any(bnd_n_loss):
            safe_dy_val = dy if dy > 0 else 1.0
            diag += np.where(bnd_n_loss, vp_bnd / safe_dy_val * tr_bnd_n, 0.0)
    elif ny > 1:
        vn_bnd = np.zeros((nz, ny, nx), dtype=np.float64)
        vn_bnd[:, -1, :] = V[:, -1, :]
        bnd_n_loss = bnd_n & (vn_bnd >= 0)
        if np.any(bnd_n_loss):
            safe_dy_val = dy if dy > 0 else 1.0
            diag += np.where(bnd_n_loss, vn_bnd / safe_dy_val * tr_bnd_n, 0.0)
    elif ny == 1:
        vn_bnd = np.zeros((nz, ny, nx), dtype=np.float64)
        vn_bnd[:, 0, :] = V[:, 0, :]
        bnd_n_loss = bnd_n & (vn_bnd >= 0)
        if np.any(bnd_n_loss):
            safe_dy_val = dy if dy > 0 else 1.0
            diag += np.where(bnd_n_loss, vn_bnd / safe_dy_val * tr_bnd_n, 0.0)

    # ========== Assemble diagonal + off-diagonal ==========
    diag_pos = diag > 0
    if np.any(diag_pos):
        idx_d = np.where(diag_pos)
        rows_list.append(n3d[idx_d])
        cols_list.append(n3d[idx_d])
        vals_list.append(diag[idx_d])

    if not rows_list:
        return sp.csc_matrix((N, N))

    all_rows = np.concatenate(rows_list)
    all_cols = np.concatenate(cols_list)
    all_vals = np.concatenate(vals_list)

    return sp.csc_matrix((all_vals, (all_rows, all_cols)), shape=(N, N))


def assemble_vertical_convdiff(
    grid: GridData, indexer: CellIndexer,
) -> sp.csc_matrix:
    """PROTOTYPE unified vertical convection-diffusion (Patankar exponential).

    Replaces the separately-assembled vertical advection (`assemble_vertical_
    advection`, omega/dP first-order upwind) and vertical diffusion
    (`assemble_vertical_diffusion`, Kzz/Dz) with a single Patankar exponential
    operator at each vertical interface, so the scheme blends toward centred
    (2nd-order) where physical diffusion competes and collapses to upwind only
    where advection dominates — removing the implicit numerical diffusion
    `K_num = |w|·dz/2` of the standalone upwind omega advection.

    This is gated off by default (used only when
    `grid.unified_vertical_patankar` is True) and is a research prototype, not
    production physics.

    Coordinate frame
    ----------------
    Worked entirely in HEIGHT coordinates (Kzz [m²/s], Dz [m]) so the diffusion
    part reduces *exactly* to `assemble_vertical_diffusion` when the advective
    velocity is zero. The advective velocity is derived from the SAME hybrid,
    terrain-corrected `grid.omega` that production advects with (NOT the raw
    physical `WAvg`), so the omega-terrain-bug fix is preserved:

        rho_g_face = (dP_k + dP_{k+1}) / (Dz_k + Dz_{k+1})   [Pa/m]
        w_up   = omega_minus[k+1] / rho_g_face               [m/s, upward,  L->R]
        w_down = omega_plus[k+1]  / rho_g_face               [m/s, downward, R->L]

    where L = lower cell k, R = upper cell k+1, and omega>0 is downward.

    Face coefficients (Patankar), per interface between layer k and k+1:
        K_face = harmonic_mean(Kzz[k], Kzz[k+1])
        dist   = 0.5 (Dz[k] + Dz[k+1])
        Pe     = (w_up - w_down) * dist / K_face
        diff_flux = K_face / dist * A(|Pe|)                  [m/s]
        coeff_L   = diff_flux + w_up                          [m/s]  (mult. C_L)
        coeff_R   = diff_flux + w_down                        [m/s]  (mult. C_R)
    Matrix entries (each flux coefficient divided by its own cell thickness):
        lower L (k):   diag += coeff_L / Dz[k];   off(L,R) = -coeff_R / Dz[k]
        upper R (k+1): diag += coeff_R / Dz[k+1]; off(R,L) = -coeff_L / Dz[k+1]

    M-matrix by construction (diag >= 0, off-diag <= 0). Conserves Σ Dz_k·c_k —
    the same height-thickness weight the standalone vertical diffusion operator
    conserves. (NOTE: the standalone omega advection conserves the pressure-
    thickness weight Σ dP_k·c_k; unifying into one height-coordinate operator
    moves the advective part to the Dz weight too. This is a prototype
    simplification — an O(density-gradient) change to the advective mass
    measure.)

    Returns
    -------
    T_vcd : csc_matrix, shape (N, N)
    """
    nz, ny, nx = grid.nz, grid.ny, grid.nx
    N = indexer.N
    Kzz = grid.Kzz
    Dz = grid.Dz
    dP = grid.dP
    omega_plus = grid.omega_plus     # (nz, ny, nx) downward >= 0
    omega_minus = grid.omega_minus   # (nz, ny, nx) upward >= 0

    n3d = np.arange(N, dtype=np.int64).reshape(nz, ny, nx)

    diag = np.zeros((nz, ny, nx), dtype=np.float64)
    rows_list = []
    cols_list = []
    vals_list = []

    for k in range(nz - 1):
        dz_k = Dz[k]
        dz_k1 = Dz[k + 1]
        dP_k = dP[k]
        dP_k1 = dP[k + 1]

        valid = (dz_k > 0) & (dz_k1 > 0) & (dP_k > 0) & (dP_k1 > 0)
        if not np.any(valid):
            continue

        K_k = Kzz[k]
        K_k1 = Kzz[k + 1]
        K_face = np.where(valid, 2.0 * K_k * K_k1 / (K_k + K_k1 + 1e-30), 0.0)
        dist = np.where(valid, 0.5 * (dz_k + dz_k1), 1.0)

        # Hybrid omega at the interface = bottom face of layer k+1 (terrain-
        # corrected cross-level velocity). Convert to physical w via the
        # face-averaged rho*g = dP_dist / dz_dist.
        rho_g_face = np.where(valid,
                              (dP_k + dP_k1) / (dz_k + dz_k1 + 1e-30),
                              1.0)
        w_up = np.where(valid, omega_minus[k + 1] / rho_g_face, 0.0)    # L->R
        w_down = np.where(valid, omega_plus[k + 1] / rho_g_face, 0.0)   # R->L
        u_face = w_up - w_down                                          # net up

        safe_K = np.where(K_face > 1e-30, K_face, 1e-30)
        Pe_abs = np.minimum(np.abs(u_face) * dist / safe_K, 500.0)
        A_pe = _patankar_A(Pe_abs)

        diff_flux = np.where(valid, K_face / dist * A_pe, 0.0)          # m/s
        coeff_L = diff_flux + w_up    # multiplies C_L (lower cell k)
        coeff_R = diff_flux + w_down  # multiplies C_R (upper cell k+1)

        safe_dz_k = np.where(dz_k > 0, dz_k, 1.0)
        safe_dz_k1 = np.where(dz_k1 > 0, dz_k1, 1.0)

        idx = np.where(valid)
        if idx[0].size == 0:
            continue
        n_k = n3d[k][idx]
        n_k1 = n3d[k + 1][idx]

        # Lower cell L = k
        diag[k][idx] += (coeff_L / safe_dz_k)[idx]
        rows_list.append(n_k)
        cols_list.append(n_k1)
        vals_list.append(-(coeff_R / safe_dz_k)[idx])

        # Upper cell R = k+1
        diag[k + 1][idx] += (coeff_R / safe_dz_k1)[idx]
        rows_list.append(n_k1)
        cols_list.append(n_k)
        vals_list.append(-(coeff_L / safe_dz_k1)[idx])

    # Ground face (k=0 bottom): downward omega_plus[0] is a loss from the
    # surface layer through the ground, exactly as assemble_vertical_advection
    # applies it. Pure upwind advective boundary flux (no diffusion through the
    # ground). With the height-coords conversion, omega_plus[0]/rho_g/Dz[0]
    # reduces to omega_plus[0]/dP[0] (production's form). The upward ground
    # component omega_minus[0] has no source below and is skipped (matching
    # production). Without this term the prototype would silently drop a real
    # surface-layer sink and the comparison would not be like-for-like.
    valid_g = (Dz[0] > 0) & (dP[0] > 0)
    safe_dP0 = np.where(dP[0] > 0, dP[0], 1.0)
    diag[0] += np.where(valid_g & (omega_plus[0] > 0),
                        omega_plus[0] / safe_dP0, 0.0)

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


def _assemble_horizontal_convdiff_loop(
    grid: GridData, indexer: CellIndexer,
) -> sp.csc_matrix:
    """Loop-based reference implementation for equivalence testing."""
    nz, ny, nx = grid.nz, grid.ny, grid.nx
    N = indexer.N
    U = grid.UAvg
    V = grid.VAvg
    use_split = grid.has_split_fluxes

    Kxxyy = grid.Kxxyy
    K_meander_u = grid.K_meander_u
    K_meander_v = grid.K_meander_v

    rows, cols, vals = [], [], []

    for k in range(nz):
        for j in range(ny):
            dx_j = grid.dx[j]
            if dx_j <= 0:
                continue
            for i in range(nx):
                n = indexer.to_flat(k, j, i)
                diag_val = 0.0

                # === X-faces ===
                # West face (face i, between cell i-1 and cell i)
                if i > 0:
                    u = U[k, j, i]
                    K_face = 2.0 * Kxxyy[k, j, i] * Kxxyy[k, j, i - 1] / (
                        Kxxyy[k, j, i] + Kxxyy[k, j, i - 1] + 1e-30)
                    if not use_split:
                        K_face += K_meander_u[k, j, i]  # face i

                    D = K_face / dx_j
                    safe_K = max(K_face, 1e-30)
                    Pe = u * dx_j / safe_K
                    Pe_abs = min(abs(Pe), 500.0)

                    if Pe_abs < 1e-6:
                        A = 1.0 - 0.5 * Pe_abs
                    elif Pe_abs > 500.0:
                        A = 0.0
                    else:
                        A = Pe_abs / np.expm1(Pe_abs)

                    # L = i-1, R = i, u points L->R
                    diff_rate = D * A / dx_j
                    if use_split:
                        adv_L_rate = grid.UAvg_plus[k, j, i] / dx_j
                        adv_R_rate = grid.UAvg_minus[k, j, i] / dx_j
                    else:
                        adv_L_rate = max(u, 0.0) / dx_j
                        adv_R_rate = max(-u, 0.0) / dx_j

                    # Mass-space terrain correction: dP_face / dP_cell
                    dP_cell = grid.dP[k, j, i]
                    dP_nb = grid.dP[k, j, i - 1]
                    if dP_cell > 0 and dP_nb > 0:
                        dP_face = 0.5 * (dP_nb + dP_cell)
                        tr_here = dP_face / dP_cell
                    else:
                        tr_here = 1.0

                    # This cell (i, the "right" cell) perspective:
                    diag_val += (diff_rate + adv_R_rate) * tr_here
                    n_west = indexer.to_flat(k, j, i - 1)
                    rows.append(n); cols.append(n_west)
                    vals.append(-(diff_rate + adv_L_rate) * tr_here)

                # East face (between i and i+1)
                # East face (face i+1, between cell i and cell i+1)
                if i < nx - 1:
                    u = U[k, j, i + 1]  # velocity at east face
                    K_face = 2.0 * Kxxyy[k, j, i] * Kxxyy[k, j, i + 1] / (
                        Kxxyy[k, j, i] + Kxxyy[k, j, i + 1] + 1e-30)
                    if not use_split:
                        K_face += K_meander_u[k, j, i + 1]  # face i+1

                    D = K_face / dx_j
                    safe_K = max(K_face, 1e-30)
                    Pe = u * dx_j / safe_K
                    Pe_abs = min(abs(Pe), 500.0)

                    if Pe_abs < 1e-6:
                        A = 1.0 - 0.5 * Pe_abs
                    elif Pe_abs > 500.0:
                        A = 0.0
                    else:
                        A = Pe_abs / np.expm1(Pe_abs)

                    # L = i, R = i+1, u points L->R
                    diff_rate = D * A / dx_j
                    if use_split:
                        adv_L_rate = grid.UAvg_plus[k, j, i + 1] / dx_j
                        adv_R_rate = grid.UAvg_minus[k, j, i + 1] / dx_j
                    else:
                        adv_L_rate = max(u, 0.0) / dx_j
                        adv_R_rate = max(-u, 0.0) / dx_j

                    # Mass-space terrain correction: dP_face / dP_cell
                    dP_cell = grid.dP[k, j, i]
                    dP_nb = grid.dP[k, j, i + 1]
                    if dP_cell > 0 and dP_nb > 0:
                        dP_face = 0.5 * (dP_nb + dP_cell)
                        tr_here = dP_face / dP_cell
                    else:
                        tr_here = 1.0

                    # This cell (i, the "left" cell) perspective:
                    diag_val += (diff_rate + adv_L_rate) * tr_here
                    n_east = indexer.to_flat(k, j, i + 1)
                    rows.append(n); cols.append(n_east)
                    vals.append(-(diff_rate + adv_R_rate) * tr_here)
                elif not grid.periodic_lon:
                    # East boundary: outflux proxy (non-periodic only)
                    if use_split:
                        u_e = grid.UAvg_plus[k, j, i]
                        if u_e > 0:
                            diag_val += u_e / dx_j
                    else:
                        if i > 0:
                            u_e = U[k, j, i]
                        else:
                            u_e = U[k, j, 0]
                        if u_e >= 0:
                            diag_val += u_e / dx_j

                # Wrap face (periodic): between cell 0 and cell nx-1
                # Process once per (k,j) pair, at i=0
                if grid.periodic_lon and i == 0 and nx > 1:
                    u_w = grid.UAvg_wrap[k, j]
                    K_fw = 2.0 * Kxxyy[k, j, 0] * Kxxyy[k, j, nx - 1] / (
                        Kxxyy[k, j, 0] + Kxxyy[k, j, nx - 1] + 1e-30)
                    if not use_split:
                        K_fw += grid.K_meander_u_wrap[k, j]
                    D_w = K_fw / dx_j
                    safe_Kw = max(K_fw, 1e-30)
                    Pe_w = u_w * dx_j / safe_Kw
                    Pe_abs_w = min(abs(Pe_w), 500.0)
                    if Pe_abs_w < 1e-6:
                        A_w = 1.0 - 0.5 * Pe_abs_w
                    elif Pe_abs_w > 500.0:
                        A_w = 0.0
                    else:
                        A_w = Pe_abs_w / np.expm1(Pe_abs_w)
                    diff_rate_w = D_w * A_w / dx_j
                    if use_split:
                        adv_L_w = grid.UAvg_plus_wrap[k, j] / dx_j
                        adv_R_w = grid.UAvg_minus_wrap[k, j] / dx_j
                    else:
                        adv_L_w = max(u_w, 0.0) / dx_j
                        adv_R_w = max(-u_w, 0.0) / dx_j

                    # Terrain correction
                    dP_c0 = grid.dP[k, j, 0]
                    dP_cnx = grid.dP[k, j, nx - 1]
                    if dP_c0 > 0 and dP_cnx > 0:
                        dP_fw = 0.5 * (dP_cnx + dP_c0)
                        tr_c0 = dP_fw / dP_c0
                        tr_cnx = dP_fw / dP_cnx
                    else:
                        tr_c0 = 1.0
                        tr_cnx = 1.0

                    # Cell 0 (R): sees wrap face from nx-1 (L)
                    diag_val += (diff_rate_w + adv_R_w) * tr_c0
                    n_wrap = indexer.to_flat(k, j, nx - 1)
                    rows.append(n); cols.append(n_wrap)
                    vals.append(-(diff_rate_w + adv_L_w) * tr_c0)

                if grid.periodic_lon and i == nx - 1 and nx > 1:
                    # Cell nx-1 (L): sees wrap face to cell 0 (R)
                    u_w = grid.UAvg_wrap[k, j]
                    K_fw = 2.0 * Kxxyy[k, j, 0] * Kxxyy[k, j, nx - 1] / (
                        Kxxyy[k, j, 0] + Kxxyy[k, j, nx - 1] + 1e-30)
                    if not use_split:
                        K_fw += grid.K_meander_u_wrap[k, j]
                    D_w = K_fw / dx_j
                    safe_Kw = max(K_fw, 1e-30)
                    Pe_w = u_w * dx_j / safe_Kw
                    Pe_abs_w = min(abs(Pe_w), 500.0)
                    if Pe_abs_w < 1e-6:
                        A_w = 1.0 - 0.5 * Pe_abs_w
                    elif Pe_abs_w > 500.0:
                        A_w = 0.0
                    else:
                        A_w = Pe_abs_w / np.expm1(Pe_abs_w)
                    diff_rate_w = D_w * A_w / dx_j
                    if use_split:
                        adv_L_w = grid.UAvg_plus_wrap[k, j] / dx_j
                        adv_R_w = grid.UAvg_minus_wrap[k, j] / dx_j
                    else:
                        adv_L_w = max(u_w, 0.0) / dx_j
                        adv_R_w = max(-u_w, 0.0) / dx_j

                    dP_c0 = grid.dP[k, j, 0]
                    dP_cnx = grid.dP[k, j, nx - 1]
                    if dP_c0 > 0 and dP_cnx > 0:
                        dP_fw = 0.5 * (dP_cnx + dP_c0)
                        tr_cnx = dP_fw / dP_cnx
                    else:
                        tr_cnx = 1.0

                    diag_val += (diff_rate_w + adv_L_w) * tr_cnx
                    n_wrap = indexer.to_flat(k, j, 0)
                    rows.append(n); cols.append(n_wrap)
                    vals.append(-(diff_rate_w + adv_R_w) * tr_cnx)

                # West boundary (i=0): outflux when u < 0 (non-periodic only)
                if i == 0 and not grid.periodic_lon:
                    if use_split:
                        u_m = grid.UAvg_minus[k, j, 0]
                        if u_m > 0:
                            diag_val += u_m / dx_j
                    else:
                        u_w = U[k, j, 0]
                        if u_w < 0:
                            diag_val += -u_w / dx_j

                # === Y-faces ===
                dy = grid.dy
                safe_dy = dy if dy > 0 else 1.0

                # South face (face j, between cell j-1 and j)
                if j > 0:
                    v = V[k, j, i]
                    K_face = 2.0 * Kxxyy[k, j, i] * Kxxyy[k, j - 1, i] / (
                        Kxxyy[k, j, i] + Kxxyy[k, j - 1, i] + 1e-30)
                    if not use_split:
                        K_face += K_meander_v[k, j, i]  # face j

                    D = K_face / safe_dy
                    safe_K = max(K_face, 1e-30)
                    Pe = v * safe_dy / safe_K
                    Pe_abs = min(abs(Pe), 500.0)

                    if Pe_abs < 1e-6:
                        A = 1.0 - 0.5 * Pe_abs
                    elif Pe_abs > 500.0:
                        A = 0.0
                    else:
                        A = Pe_abs / np.expm1(Pe_abs)

                    diff_rate = D * A / safe_dy
                    if use_split:
                        adv_L_rate = grid.VAvg_plus[k, j, i] / safe_dy
                        adv_R_rate = grid.VAvg_minus[k, j, i] / safe_dy
                    else:
                        adv_L_rate = max(v, 0.0) / safe_dy
                        adv_R_rate = max(-v, 0.0) / safe_dy

                    # Mass-space terrain correction: (dx_face/dx_cell) * (dP_face/dP_cell)
                    dP_cell = grid.dP[k, j, i]
                    dP_nb = grid.dP[k, j - 1, i]
                    if dP_cell > 0 and dP_nb > 0:
                        dP_face = 0.5 * (dP_nb + dP_cell)
                        dx_face = 0.5 * (grid.dx[j - 1] + grid.dx[j])
                        tr_here = (dx_face / dx_j) * (dP_face / dP_cell)
                    else:
                        tr_here = 1.0

                    # This cell (j, "right" cell) perspective:
                    diag_val += (diff_rate + adv_R_rate) * tr_here
                    n_south = indexer.to_flat(k, j - 1, i)
                    rows.append(n); cols.append(n_south)
                    vals.append(-(diff_rate + adv_L_rate) * tr_here)

                # North face (face j+1, between cell j and j+1)
                if j < ny - 1:
                    v = V[k, j + 1, i]
                    K_face = 2.0 * Kxxyy[k, j, i] * Kxxyy[k, j + 1, i] / (
                        Kxxyy[k, j, i] + Kxxyy[k, j + 1, i] + 1e-30)
                    if not use_split:
                        K_face += K_meander_v[k, j + 1, i]  # face j+1

                    D = K_face / safe_dy
                    safe_K = max(K_face, 1e-30)
                    Pe = v * safe_dy / safe_K
                    Pe_abs = min(abs(Pe), 500.0)

                    if Pe_abs < 1e-6:
                        A = 1.0 - 0.5 * Pe_abs
                    elif Pe_abs > 500.0:
                        A = 0.0
                    else:
                        A = Pe_abs / np.expm1(Pe_abs)

                    diff_rate = D * A / safe_dy
                    if use_split:
                        adv_L_rate = grid.VAvg_plus[k, j + 1, i] / safe_dy
                        adv_R_rate = grid.VAvg_minus[k, j + 1, i] / safe_dy
                    else:
                        adv_L_rate = max(v, 0.0) / safe_dy
                        adv_R_rate = max(-v, 0.0) / safe_dy

                    # Mass-space terrain correction: (dx_face/dx_cell) * (dP_face/dP_cell)
                    dP_cell = grid.dP[k, j, i]
                    dP_nb = grid.dP[k, j + 1, i]
                    if dP_cell > 0 and dP_nb > 0:
                        dP_face = 0.5 * (dP_nb + dP_cell)
                        dx_face = 0.5 * (grid.dx[j] + grid.dx[j + 1])
                        tr_here = (dx_face / dx_j) * (dP_face / dP_cell)
                    else:
                        tr_here = 1.0

                    # This cell (j, "left" cell) perspective:
                    diag_val += (diff_rate + adv_L_rate) * tr_here
                    n_north = indexer.to_flat(k, j + 1, i)
                    rows.append(n); cols.append(n_north)
                    vals.append(-(diff_rate + adv_R_rate) * tr_here)
                else:
                    # North boundary: outflux proxy
                    # dx correction: face at lat[-1]+dlat/2
                    lat_fn = grid.lat[-1] + grid.dlat / 2.0
                    dx_fn = EARTH_RADIUS * np.cos(lat_fn * DEG_TO_RAD) * grid.dlon * DEG_TO_RAD
                    tr_n = dx_fn / dx_j if dx_j > 0 else 1.0
                    if use_split:
                        v_n = grid.VAvg_plus[k, j, i]
                        if v_n > 0:
                            diag_val += v_n / safe_dy * tr_n
                    else:
                        if j > 0:
                            v_n = V[k, j, i]
                        else:
                            v_n = V[k, 0, i]
                        if v_n >= 0:
                            diag_val += v_n / safe_dy * tr_n

                # South boundary (j=0): outflux when v < 0
                # dx correction: face at lat[0]-dlat/2
                if j == 0:
                    lat_fs = grid.lat[0] - grid.dlat / 2.0
                    dx_fs = EARTH_RADIUS * np.cos(lat_fs * DEG_TO_RAD) * grid.dlon * DEG_TO_RAD
                    tr_s = dx_fs / dx_j if dx_j > 0 else 1.0
                    if use_split:
                        v_m = grid.VAvg_minus[k, 0, i]
                        if v_m > 0:
                            diag_val += v_m / safe_dy * tr_s
                    else:
                        v_s = V[k, 0, i]
                        if v_s < 0:
                            diag_val += -v_s / safe_dy * tr_s

                if diag_val > 0:
                    rows.append(n); cols.append(n); vals.append(diag_val)

    if len(vals) == 0:
        return sp.csc_matrix((N, N))

    return sp.csc_matrix(
        (np.array(vals), (np.array(rows, dtype=np.int64), np.array(cols, dtype=np.int64))),
        shape=(N, N),
    )
