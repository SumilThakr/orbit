"""Advection operator assembly: horizontal + vertical.

Ports the advection physics from physics_uniform.pyx into sparse matrix rows.
Supports first-order and hybrid second-order upwind (controlled by stencil bitmask).
"""

import numpy as np
import scipy.sparse as sp
from orbit.core.indexing import CellIndexer
from orbit.core.grid_data import GridData
from orbit.core.stencil import (
    SO2_WEST_POS, SO2_WEST_NEG, SO2_EAST_POS, SO2_EAST_NEG,
    SO2_SOUTH_POS, SO2_SOUTH_NEG, SO2_NORTH_POS, SO2_NORTH_NEG,
)


def assemble_horizontal_advection(
    grid: GridData, indexer: CellIndexer, stencil: np.ndarray,
    use_second_order: bool = True,
) -> sp.csc_matrix:
    """Assemble horizontal advection operator (X + Y directions).

    For each cell, encodes advective fluxes through west/east/south/north faces.
    Uses the stencil bitmask to select first-order or second-order reconstruction.

    Symmetric terrain ratio (sigma-native transport):
    - Influx from any neighbor: multiply by dP_ratio (neighbor/cell thickness ratio)
    - Outflux: no terrain ratio (loss rate unscaled)

    Parameters
    ----------
    grid : GridData
    indexer : CellIndexer
    stencil : ndarray of uint8
        Per-cell bitmask from compute_stencil()
    use_second_order : bool
        If False, ignore stencil and use first-order everywhere

    Returns
    -------
    T_hadv : csc_matrix, shape (N, N)
    """
    nz, ny, nx = grid.nz, grid.ny, grid.nx
    N = indexer.N
    U = grid.UAvg       # (nz, ny, nx) west-face velocity
    V = grid.VAvg       # (nz, ny, nx) south-face velocity
    dP_ratio_west = grid.dP_ratio_west
    dP_ratio_south = grid.dP_ratio_south
    dP_ratio_east = grid.dP_ratio_east
    dP_ratio_north = grid.dP_ratio_north

    # 3D flat index array
    n3d = np.arange(N, dtype=np.int64).reshape(nz, ny, nx)

    # dx varies by latitude: broadcast to (nz, ny, nx)
    dx_3d = np.broadcast_to(grid.dx[np.newaxis, :, np.newaxis], (nz, ny, nx)).copy()
    dy = grid.dy

    # Skip cells where dx <= 0
    valid = dx_3d > 0

    mask_3d = stencil.astype(np.uint8) if use_second_order else np.zeros((nz, ny, nx), dtype=np.uint8)

    # Diagonal accumulator
    diag = np.zeros((nz, ny, nx), dtype=np.float64)

    rows_list = []
    cols_list = []
    vals_list = []

    # ========== WEST FACE ==========
    u_w = U  # west-face velocity at (k, j, i)

    # --- i > 0: have a west neighbor ---
    interior_w = np.zeros((nz, ny, nx), dtype=bool)
    interior_w[:, :, 1:] = True
    has_west = valid & interior_w

    n_west = np.zeros_like(n3d)
    n_west[:, :, 1:] = n3d[:, :, :-1]

    # u >= 0: influx from west
    u_pos = has_west & (u_w >= 0)
    coeff_w = np.where(valid, u_w / dx_3d, 0.0)

    # Second-order: west pos, i >= 2
    so2_wp = u_pos & ((mask_3d & SO2_WEST_POS) != 0)
    can_ww = np.zeros((nz, ny, nx), dtype=bool)
    can_ww[:, :, 2:] = True
    so2_wp_ok = so2_wp & can_ww

    n_ww = np.zeros_like(n3d)
    n_ww[:, :, 2:] = n3d[:, :, :-2]

    tr_w = dP_ratio_west

    # Second-order influx from west
    idx_so2wp = np.where(so2_wp_ok)
    if idx_so2wp[0].size > 0:
        r = n3d[idx_so2wp]
        c_coeff = coeff_w[idx_so2wp]
        tr = tr_w[idx_so2wp]
        rows_list.append(r); cols_list.append(n_west[idx_so2wp]); vals_list.append(-1.5 * c_coeff * tr)
        rows_list.append(r); cols_list.append(n_ww[idx_so2wp]); vals_list.append(0.5 * c_coeff * tr)

    # First-order influx from west
    fo_wp = u_pos & ~so2_wp_ok
    idx_fowp = np.where(fo_wp)
    if idx_fowp[0].size > 0:
        r = n3d[idx_fowp]
        rows_list.append(r); cols_list.append(n_west[idx_fowp])
        vals_list.append(-coeff_w[idx_fowp] * tr_w[idx_fowp])

    # Diagonal for u >= 0 influx from west
    diag += np.where(u_pos, coeff_w, 0.0)

    # u < 0: outflux to west
    u_neg = has_west & (u_w < 0)
    neg_coeff_w = np.where(valid, -u_w / dx_3d, 0.0)  # positive

    # Second-order outflux: west neg, i < nx - 1
    so2_wn = u_neg & ((mask_3d & SO2_WEST_NEG) != 0)
    can_east_for_wn = np.zeros((nz, ny, nx), dtype=bool)
    can_east_for_wn[:, :, :-1] = True
    so2_wn_ok = so2_wn & can_east_for_wn

    n_east = np.zeros_like(n3d)
    n_east[:, :, :-1] = n3d[:, :, 1:]

    idx_so2wn = np.where(so2_wn_ok)
    if idx_so2wn[0].size > 0:
        r = n3d[idx_so2wn]
        c_coeff = neg_coeff_w[idx_so2wn]
        rows_list.append(r); cols_list.append(n_east[idx_so2wn]); vals_list.append(0.5 * c_coeff)
    diag += np.where(so2_wn_ok, 1.5 * neg_coeff_w, 0.0)

    # First-order outflux to west
    fo_wn = u_neg & ~so2_wn_ok
    diag += np.where(fo_wn, neg_coeff_w, 0.0)

    # Boundary: i = 0, u < 0 -> outflux
    bnd_w = np.zeros((nz, ny, nx), dtype=bool)
    bnd_w[:, :, 0] = True
    bnd_w_loss = valid & bnd_w & (u_w < 0)
    diag += np.where(bnd_w_loss, neg_coeff_w, 0.0)

    # ========== EAST FACE ==========
    # East face velocity is U[k, j, i+1]
    u_e = np.zeros((nz, ny, nx), dtype=np.float64)
    u_e[:, :, :-1] = U[:, :, 1:]

    has_east = np.zeros((nz, ny, nx), dtype=bool)
    has_east[:, :, :-1] = True
    has_east = valid & has_east

    # n_east already computed above (valid for i < nx-1)

    # u_e >= 0: outflux to east (no terrain ratio)
    ue_pos = has_east & (u_e >= 0)

    # Second-order: east pos, i >= 1
    so2_ep = ue_pos & ((mask_3d & SO2_EAST_POS) != 0)
    can_west_for_ep = np.zeros((nz, ny, nx), dtype=bool)
    can_west_for_ep[:, :, 1:] = True
    so2_ep_ok = so2_ep & can_west_for_ep

    idx_so2ep = np.where(so2_ep_ok)
    if idx_so2ep[0].size > 0:
        r = n3d[idx_so2ep]
        c_coeff = u_e[idx_so2ep] / dx_3d[idx_so2ep]
        rows_list.append(r); cols_list.append(n_west[idx_so2ep]); vals_list.append(0.5 * c_coeff)
    diag += np.where(so2_ep_ok, 1.5 * u_e / np.where(dx_3d > 0, dx_3d, 1.0), 0.0)

    # First-order outflux to east
    fo_ep = ue_pos & ~so2_ep_ok
    diag += np.where(fo_ep, u_e / np.where(dx_3d > 0, dx_3d, 1.0), 0.0)

    # u_e < 0: influx from east
    ue_neg = has_east & (u_e < 0)
    neg_ue = -u_e  # positive magnitude

    # Second-order: east neg, i < nx - 2
    so2_en = ue_neg & ((mask_3d & SO2_EAST_NEG) != 0)
    can_ee = np.zeros((nz, ny, nx), dtype=bool)
    can_ee[:, :, :-2] = True
    so2_en_ok = so2_en & can_ee

    n_ee = np.zeros_like(n3d)
    n_ee[:, :, :-2] = n3d[:, :, 2:]

    idx_so2en = np.where(so2_en_ok)
    if idx_so2en[0].size > 0:
        r = n3d[idx_so2en]
        c_coeff = neg_ue[idx_so2en] / dx_3d[idx_so2en]
        tr_e = dP_ratio_east[idx_so2en]
        rows_list.append(r); cols_list.append(n_east[idx_so2en]); vals_list.append(-1.5 * c_coeff * tr_e)
        rows_list.append(r); cols_list.append(n_ee[idx_so2en]); vals_list.append(0.5 * c_coeff * tr_e)

    # First-order influx from east
    fo_en = ue_neg & ~so2_en_ok
    idx_foen = np.where(fo_en)
    if idx_foen[0].size > 0:
        r = n3d[idx_foen]
        # val = u_e / dx * tr_e (u_e < 0, so this is negative = off-diag gain)
        tr_e = dP_ratio_east[idx_foen]
        rows_list.append(r); cols_list.append(n_east[idx_foen])
        vals_list.append(u_e[idx_foen] / dx_3d[idx_foen] * tr_e)

    # Diagonal for influx from east
    diag += np.where(ue_neg, neg_ue / np.where(dx_3d > 0, dx_3d, 1.0), 0.0)

    # East boundary: i = nx-1
    bnd_e = np.zeros((nz, ny, nx), dtype=bool)
    bnd_e[:, :, -1] = True
    bnd_e = valid & bnd_e

    # Fallback east-boundary velocity
    ue_bnd = np.zeros((nz, ny, nx), dtype=np.float64)
    # For i = nx-1 and i > 0: use U[k,j,i] (west-face velocity)
    ue_bnd[:, :, -1] = U[:, :, -1]
    # For i = nx-1 and i == 0 (only when nx=1): use u_w
    if nx == 1:
        ue_bnd[:, :, -1] = U[:, :, 0]

    bnd_e_loss = bnd_e & (ue_bnd >= 0)
    diag += np.where(bnd_e_loss, ue_bnd / np.where(dx_3d > 0, dx_3d, 1.0), 0.0)

    # ========== SOUTH FACE ==========
    v_s = V  # south-face velocity at (k, j, i)
    tr_s = dP_ratio_south

    has_south = np.zeros((nz, ny, nx), dtype=bool)
    has_south[:, 1:, :] = True
    has_south = valid & has_south

    n_south = np.zeros_like(n3d)
    n_south[:, 1:, :] = n3d[:, :-1, :]

    coeff_s = np.where(valid, v_s / dy, 0.0)
    neg_coeff_s = np.where(valid, -v_s / dy, 0.0)

    # v >= 0: influx from south
    v_pos = has_south & (v_s >= 0)

    # Second-order: south pos, j >= 2
    so2_sp = v_pos & ((mask_3d & SO2_SOUTH_POS) != 0)
    can_ss = np.zeros((nz, ny, nx), dtype=bool)
    can_ss[:, 2:, :] = True
    so2_sp_ok = so2_sp & can_ss

    n_ss = np.zeros_like(n3d)
    n_ss[:, 2:, :] = n3d[:, :-2, :]

    idx_so2sp = np.where(so2_sp_ok)
    if idx_so2sp[0].size > 0:
        r = n3d[idx_so2sp]
        c = coeff_s[idx_so2sp]
        tr = tr_s[idx_so2sp]
        rows_list.append(r); cols_list.append(n_south[idx_so2sp]); vals_list.append(-1.5 * c * tr)
        rows_list.append(r); cols_list.append(n_ss[idx_so2sp]); vals_list.append(0.5 * c * tr)

    # First-order influx from south
    fo_sp = v_pos & ~so2_sp_ok
    idx_fosp = np.where(fo_sp)
    if idx_fosp[0].size > 0:
        r = n3d[idx_fosp]
        rows_list.append(r); cols_list.append(n_south[idx_fosp])
        vals_list.append(-coeff_s[idx_fosp] * tr_s[idx_fosp])

    # Diagonal for v >= 0 influx from south
    diag += np.where(v_pos, coeff_s, 0.0)

    # v < 0: outflux to south
    v_neg = has_south & (v_s < 0)

    so2_sn = v_neg & ((mask_3d & SO2_SOUTH_NEG) != 0)
    n_north = np.zeros_like(n3d)
    n_north[:, :-1, :] = n3d[:, 1:, :]
    can_north_for_sn = np.zeros((nz, ny, nx), dtype=bool)
    can_north_for_sn[:, :-1, :] = True
    so2_sn_ok = so2_sn & can_north_for_sn

    idx_so2sn = np.where(so2_sn_ok)
    if idx_so2sn[0].size > 0:
        r = n3d[idx_so2sn]
        c = neg_coeff_s[idx_so2sn]
        rows_list.append(r); cols_list.append(n_north[idx_so2sn]); vals_list.append(0.5 * c)
    diag += np.where(so2_sn_ok, 1.5 * neg_coeff_s, 0.0)

    fo_sn = v_neg & ~so2_sn_ok
    diag += np.where(fo_sn, neg_coeff_s, 0.0)

    # South boundary: j = 0, v < 0
    bnd_s = np.zeros((nz, ny, nx), dtype=bool)
    bnd_s[:, 0, :] = True
    bnd_s_loss = valid & bnd_s & (v_s < 0)
    diag += np.where(bnd_s_loss, neg_coeff_s, 0.0)

    # ========== NORTH FACE ==========
    v_n = np.zeros((nz, ny, nx), dtype=np.float64)
    v_n[:, :-1, :] = V[:, 1:, :]

    has_north = np.zeros((nz, ny, nx), dtype=bool)
    has_north[:, :-1, :] = True
    has_north = valid & has_north

    safe_dy = dy if dy > 0 else 1.0

    # v_n >= 0: outflux to north
    vn_pos = has_north & (v_n >= 0)

    so2_np = vn_pos & ((mask_3d & SO2_NORTH_POS) != 0)
    can_south_for_np = np.zeros((nz, ny, nx), dtype=bool)
    can_south_for_np[:, 1:, :] = True
    so2_np_ok = so2_np & can_south_for_np

    idx_so2np = np.where(so2_np_ok)
    if idx_so2np[0].size > 0:
        r = n3d[idx_so2np]
        c = v_n[idx_so2np] / safe_dy
        rows_list.append(r); cols_list.append(n_south[idx_so2np]); vals_list.append(0.5 * c)
    diag += np.where(so2_np_ok, 1.5 * v_n / safe_dy, 0.0)

    fo_np = vn_pos & ~so2_np_ok
    diag += np.where(fo_np, v_n / safe_dy, 0.0)

    # v_n < 0: influx from north
    vn_neg = has_north & (v_n < 0)
    neg_vn = -v_n

    so2_nn = vn_neg & ((mask_3d & SO2_NORTH_NEG) != 0)
    can_nn = np.zeros((nz, ny, nx), dtype=bool)
    can_nn[:, :-2, :] = True
    so2_nn_ok = so2_nn & can_nn

    n_nn = np.zeros_like(n3d)
    n_nn[:, :-2, :] = n3d[:, 2:, :]

    idx_so2nn = np.where(so2_nn_ok)
    if idx_so2nn[0].size > 0:
        r = n3d[idx_so2nn]
        c = neg_vn[idx_so2nn] / safe_dy
        tr_n = dP_ratio_north[idx_so2nn]
        rows_list.append(r); cols_list.append(n_north[idx_so2nn]); vals_list.append(-1.5 * c * tr_n)
        rows_list.append(r); cols_list.append(n_nn[idx_so2nn]); vals_list.append(0.5 * c * tr_n)

    fo_nn = vn_neg & ~so2_nn_ok
    idx_fonn = np.where(fo_nn)
    if idx_fonn[0].size > 0:
        r = n3d[idx_fonn]
        tr_n = dP_ratio_north[idx_fonn]
        rows_list.append(r); cols_list.append(n_north[idx_fonn])
        vals_list.append(v_n[idx_fonn] / safe_dy * tr_n)  # v_n < 0, so negative = gain

    diag += np.where(vn_neg, neg_vn / safe_dy, 0.0)

    # North boundary: j = ny-1
    bnd_n = np.zeros((nz, ny, nx), dtype=bool)
    bnd_n[:, -1, :] = True
    bnd_n = valid & bnd_n

    vn_bnd = np.zeros((nz, ny, nx), dtype=np.float64)
    # For j = ny-1 and j > 0: use V[k,j,i]
    vn_bnd[:, -1, :] = V[:, -1, :]
    if ny == 1:
        vn_bnd[:, -1, :] = V[:, 0, :]

    bnd_n_loss = bnd_n & (vn_bnd >= 0)
    diag += np.where(bnd_n_loss, vn_bnd / safe_dy, 0.0)

    # ========== Assemble diagonal + off-diagonal ==========
    rows_list.append(n3d.ravel())
    cols_list.append(n3d.ravel())
    vals_list.append(diag.ravel())

    all_rows = np.concatenate(rows_list)
    all_cols = np.concatenate(cols_list)
    all_vals = np.concatenate(vals_list)

    return sp.csc_matrix((all_vals, (all_rows, all_cols)), shape=(N, N))


def assemble_vertical_advection(grid: GridData, indexer: CellIndexer) -> sp.csc_matrix:
    """Assemble vertical advection operator (sigma-native, first-order only).

    Uses omega (Pa/s) with dP as the vertical "thickness".
    omega > 0: downward (subsidence), omega < 0: upward (convection).

    Interface convention:
    - omega[k] is at the bottom face of layer k (= top face of layer k-1)
    - omega[0] would be the ground surface. It is ignored, because the
      ground is closed (no air crosses it). The preprocessor stores
      cell-centred omega, so omega[0] holds the layer-0 mid-level value,
      not a ground flux; until 2026-09-25 it was applied as a downward loss
      with no receiving cell and removed surface-layer tracer into the
      ground at about three times the dry-deposition rate.
    - Top of domain (above layer nz-1) = zero flux (not stored)

    Parameters
    ----------
    grid : GridData
    indexer : CellIndexer

    Returns
    -------
    T_vadv : csc_matrix, shape (N, N)
    """
    nz, ny, nx = grid.nz, grid.ny, grid.nx
    N = indexer.N
    omega_plus = grid.omega_plus    # (nz, ny, nx) downward component >= 0
    omega_minus = grid.omega_minus  # (nz, ny, nx) upward component >= 0
    dP = grid.dP                    # (nz, ny, nx)

    n3d = np.arange(N, dtype=np.int64).reshape(nz, ny, nx)

    # Safe dP for division
    safe_dP = np.where(dP > 0, dP, 1.0)
    valid = dP > 0

    diag = np.zeros((nz, ny, nx), dtype=np.float64)
    rows_list = []
    cols_list = []
    vals_list = []

    # Bottom face (interface k): omega_plus[k] is downward, omega_minus[k] is
    # upward. Both apply only for k > 0: the bottom face of layer 0 is the
    # ground, which is closed, so omega[0] never enters the operator.
    w_bot_down = omega_plus   # (nz, ny, nx) >= 0
    w_bot_up = omega_minus    # (nz, ny, nx) >= 0
    has_below = np.zeros((nz, ny, nx), dtype=bool)
    has_below[1:, :, :] = True

    # Downward at bottom face: loss from layer k (received by k-1 below)
    bot_down = valid & has_below & (w_bot_down > 0)
    diag += np.where(bot_down, w_bot_down / safe_dP, 0.0)

    # Upward at bottom face: gain at k from k-1
    bot_up = valid & (w_bot_up > 0)
    bot_up_interior = bot_up & has_below

    n_below = np.zeros_like(n3d)
    n_below[1:, :, :] = n3d[:-1, :, :]

    idx_bot_up = np.where(bot_up_interior)
    if idx_bot_up[0].size > 0:
        rows_list.append(n3d[idx_bot_up])
        cols_list.append(n_below[idx_bot_up])
        vals_list.append(-w_bot_up[idx_bot_up] / safe_dP[idx_bot_up])  # negative = gain

    # Top face (interface k+1): omega_plus[k+1] downward, omega_minus[k+1] upward
    w_top_down = np.zeros((nz, ny, nx), dtype=np.float64)
    w_top_down[:-1, :, :] = omega_plus[1:, :, :]  # k+1 for layers 0..nz-2
    w_top_up = np.zeros((nz, ny, nx), dtype=np.float64)
    w_top_up[:-1, :, :] = omega_minus[1:, :, :]
    # For k = nz-1: zero flux at top of domain

    has_above = np.zeros((nz, ny, nx), dtype=bool)
    has_above[:-1, :, :] = True

    # Downward at top face: gain at k from k+1
    top_down = valid & has_above & (w_top_down > 0)
    n_above = np.zeros_like(n3d)
    n_above[:-1, :, :] = n3d[1:, :, :]

    idx_top_down = np.where(top_down)
    if idx_top_down[0].size > 0:
        rows_list.append(n3d[idx_top_down])
        cols_list.append(n_above[idx_top_down])
        vals_list.append(-w_top_down[idx_top_down] / safe_dP[idx_top_down])  # negative = gain

    # Upward at top face: loss from k through top face
    top_up = valid & has_above & (w_top_up > 0)
    diag += np.where(top_up, w_top_up / safe_dP, 0.0)

    # Filter: only add diagonal where diag > 0
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


# --- Loop versions for equivalence testing ---

def _assemble_horizontal_advection_loop(
    grid: GridData, indexer: CellIndexer, stencil: np.ndarray,
    use_second_order: bool = True,
) -> sp.csc_matrix:
    nz, ny, nx = grid.nz, grid.ny, grid.nx
    N = indexer.N
    U = grid.UAvg
    V = grid.VAvg
    dP_ratio_west = grid.dP_ratio_west
    dP_ratio_south = grid.dP_ratio_south
    dP_ratio_east = grid.dP_ratio_east
    dP_ratio_north = grid.dP_ratio_north

    rows, cols, vals = [], [], []

    for k in range(nz):
        for j in range(ny):
            dx_j = grid.dx[j]
            if dx_j <= 0:
                continue
            for i in range(nx):
                n = indexer.to_flat(k, j, i)
                mask = stencil[k, j, i] if use_second_order else 0
                diag = 0.0

                u = U[k, j, i]
                tr_w = dP_ratio_west[k, j, i]
                if i > 0:
                    n_west = indexer.to_flat(k, j, i - 1)
                    if u >= 0:
                        if mask & SO2_WEST_POS and i >= 2:
                            n_ww = indexer.to_flat(k, j, i - 2)
                            coeff = u / dx_j
                            rows.append(n); cols.append(n_west); vals.append(-1.5 * coeff * tr_w)
                            rows.append(n); cols.append(n_ww); vals.append(0.5 * coeff * tr_w)
                        else:
                            rows.append(n); cols.append(n_west); vals.append(-u / dx_j * tr_w)
                        diag += u / dx_j
                    else:
                        if mask & SO2_WEST_NEG and i < nx - 1:
                            n_east = indexer.to_flat(k, j, i + 1)
                            coeff = -u / dx_j
                            diag += 1.5 * coeff
                            rows.append(n); cols.append(n_east); vals.append(0.5 * coeff)
                        else:
                            diag += -u / dx_j
                else:
                    if u < 0:
                        diag += -u / dx_j

                if i < nx - 1:
                    u_e = U[k, j, i + 1]
                    n_east = indexer.to_flat(k, j, i + 1)
                    if u_e >= 0:
                        if mask & SO2_EAST_POS and i >= 1:
                            n_west = indexer.to_flat(k, j, i - 1)
                            coeff = u_e / dx_j
                            diag += 1.5 * coeff
                            rows.append(n); cols.append(n_west); vals.append(0.5 * coeff)
                        else:
                            diag += u_e / dx_j
                    else:
                        tr_e = dP_ratio_east[k, j, i]
                        if mask & SO2_EAST_NEG and i < nx - 2:
                            n_ee = indexer.to_flat(k, j, i + 2)
                            coeff = -u_e / dx_j
                            rows.append(n); cols.append(n_east); vals.append(-1.5 * coeff * tr_e)
                            rows.append(n); cols.append(n_ee); vals.append(0.5 * coeff * tr_e)
                        else:
                            rows.append(n); cols.append(n_east); vals.append(u_e / dx_j * tr_e)
                        diag += -u_e / dx_j
                else:
                    if i > 0:
                        u_e = U[k, j, i]
                    else:
                        u_e = u
                    if u_e >= 0:
                        diag += u_e / dx_j

                v = V[k, j, i]
                tr_s = dP_ratio_south[k, j, i]
                if j > 0:
                    n_south = indexer.to_flat(k, j - 1, i)
                    if v >= 0:
                        if mask & SO2_SOUTH_POS and j >= 2:
                            n_ss = indexer.to_flat(k, j - 2, i)
                            coeff = v / grid.dy
                            rows.append(n); cols.append(n_south); vals.append(-1.5 * coeff * tr_s)
                            rows.append(n); cols.append(n_ss); vals.append(0.5 * coeff * tr_s)
                        else:
                            rows.append(n); cols.append(n_south); vals.append(-v / grid.dy * tr_s)
                        diag += v / grid.dy
                    else:
                        if mask & SO2_SOUTH_NEG and j < ny - 1:
                            n_north = indexer.to_flat(k, j + 1, i)
                            coeff = -v / grid.dy
                            diag += 1.5 * coeff
                            rows.append(n); cols.append(n_north); vals.append(0.5 * coeff)
                        else:
                            diag += -v / grid.dy
                else:
                    if v < 0:
                        diag += -v / grid.dy

                if j < ny - 1:
                    v_n = V[k, j + 1, i]
                    n_north = indexer.to_flat(k, j + 1, i)
                    if v_n >= 0:
                        if mask & SO2_NORTH_POS and j >= 1:
                            n_south = indexer.to_flat(k, j - 1, i)
                            coeff = v_n / grid.dy
                            diag += 1.5 * coeff
                            rows.append(n); cols.append(n_south); vals.append(0.5 * coeff)
                        else:
                            diag += v_n / grid.dy
                    else:
                        tr_n = dP_ratio_north[k, j, i]
                        if mask & SO2_NORTH_NEG and j < ny - 2:
                            n_nn = indexer.to_flat(k, j + 2, i)
                            coeff = -v_n / grid.dy
                            rows.append(n); cols.append(n_north); vals.append(-1.5 * coeff * tr_n)
                            rows.append(n); cols.append(n_nn); vals.append(0.5 * coeff * tr_n)
                        else:
                            rows.append(n); cols.append(n_north); vals.append(v_n / grid.dy * tr_n)
                        diag += -v_n / grid.dy
                else:
                    if j > 0:
                        v_n = V[k, j, i]
                    else:
                        v_n = v
                    if v_n >= 0:
                        diag += v_n / grid.dy

                rows.append(n); cols.append(n); vals.append(diag)

    return sp.csc_matrix(
        (np.array(vals), (np.array(rows, dtype=np.int64), np.array(cols, dtype=np.int64))),
        shape=(N, N),
    )


def _assemble_vertical_advection_loop(grid: GridData, indexer: CellIndexer) -> sp.csc_matrix:
    nz, ny, nx = grid.nz, grid.ny, grid.nx
    N = indexer.N
    omega_plus = grid.omega_plus    # downward component >= 0
    omega_minus = grid.omega_minus  # upward component >= 0
    dP = grid.dP

    rows, cols, vals = [], [], []

    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                dp = dP[k, j, i]
                if dp <= 0:
                    continue
                n = indexer.to_flat(k, j, i)
                diag = 0.0

                # Bottom face: downward = loss, upward = gain from below.
                # The ground (k = 0) is closed: neither term applies there.
                w_down = omega_plus[k, j, i]
                w_up = omega_minus[k, j, i]
                if w_down > 0 and k > 0:
                    diag += w_down / dp
                if w_up > 0 and k > 0:
                    n_below = indexer.to_flat(k - 1, j, i)
                    rows.append(n); cols.append(n_below); vals.append(-w_up / dp)

                # Top face: downward = gain from above, upward = loss
                if k < nz - 1:
                    w_above_down = omega_plus[k + 1, j, i]
                    w_above_up = omega_minus[k + 1, j, i]
                    if w_above_down > 0:
                        n_above = indexer.to_flat(k + 1, j, i)
                        rows.append(n); cols.append(n_above); vals.append(-w_above_down / dp)
                    if w_above_up > 0:
                        diag += w_above_up / dp

                if diag > 0:
                    rows.append(n); cols.append(n); vals.append(diag)

    if len(vals) == 0:
        return sp.csc_matrix((N, N))

    return sp.csc_matrix(
        (np.array(vals), (np.array(rows, dtype=np.int64), np.array(cols, dtype=np.int64))),
        shape=(N, N),
    )
