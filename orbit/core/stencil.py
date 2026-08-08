"""Advection stencil classification for hybrid second-order advection.

Port of uniform_grid.pyx:805-990. Three-layer face classification:
  Layer 1 (structural): upwind-upwind cell exists in domain
  Layer 1.5 (land mask): upwind-upwind cell is land
  Layer 2 (wind shear): angle between upwind and upwind-upwind < 45 deg
"""

import numpy as np
from orbit.core.grid_data import GridData

# Bit constants 
SO2_WEST_POS = 0x01   # West face, u >= 0
SO2_WEST_NEG = 0x02   # West face, u < 0
SO2_EAST_POS = 0x04   # East face, u_e >= 0
SO2_EAST_NEG = 0x08   # East face, u_e < 0
SO2_SOUTH_POS = 0x10  # South face, v >= 0
SO2_SOUTH_NEG = 0x20  # South face, v < 0
SO2_NORTH_POS = 0x40  # North face, v_n >= 0
SO2_NORTH_NEG = 0x80  # North face, v_n < 0

COS_THRESH = 0.70710678118  # cos(45 deg)
MIN_SPD_SQ = 0.25  # (0.5 m/s)^2


def _cell_center_uv(U, V, nz, ny, nx):
    """Compute cell-center velocities from staggered west-face U and south-face V.

    Returns u_c, v_c each of shape (nz, ny, nx).
    """
    u_c = np.empty((nz, ny, nx), dtype=np.float64)
    # Interior: average of west-face and east-face
    u_c[:, :, :-1] = 0.5 * (U[:, :, :-1] + U[:, :, 1:])
    # Last column: just the west-face value
    u_c[:, :, -1] = U[:, :, -1]

    v_c = np.empty((nz, ny, nx), dtype=np.float64)
    v_c[:, :-1, :] = 0.5 * (V[:, :-1, :] + V[:, 1:, :])
    v_c[:, -1, :] = V[:, -1, :]

    return u_c, v_c


def _wind_shear_mask(u_c, v_c, jA, iA, jC, iC, nz):
    """Check wind shear between cells A[k, jA, iA] and C[k, jC, iC].

    Returns boolean array (nz, ny_sub, nx_sub) where True = shear is acceptable.
    Uses the same shapes as the sliced input arrays.
    """
    uA = u_c[:, jA[0]:jA[1], iA[0]:iA[1]]
    vA = v_c[:, jA[0]:jA[1], iA[0]:iA[1]]
    uC = u_c[:, jC[0]:jC[1], iC[0]:iC[1]]
    vC = v_c[:, jC[0]:jC[1], iC[0]:iC[1]]

    spdA_sq = uA * uA + vA * vA
    spdC_sq = uC * uC + vC * vC

    # Weak winds: skip shear check (acceptable)
    weak = (spdA_sq < MIN_SPD_SQ) | (spdC_sq < MIN_SPD_SQ)

    dot = uA * uC + vA * vC
    mag = np.sqrt(spdA_sq) * np.sqrt(spdC_sq)
    # Avoid division by zero (handled by weak winds mask)
    safe_mag = np.where(mag > 0, mag, 1.0)
    cos_theta = dot / safe_mag

    return weak | (cos_theta >= COS_THRESH)


def compute_stencil(grid: GridData) -> np.ndarray:
    """Compute advection stencil bitmask for all cells.

    Parameters
    ----------
    grid : GridData
        Grid with wind and land mask fields populated

    Returns
    -------
    stencil : ndarray of uint8, shape (nz, ny, nx)
        Per-cell bitmask indicating which face-directions can use second-order
    """
    nz, ny, nx = grid.nz, grid.ny, grid.nx
    U = grid.UAvg
    V = grid.VAvg
    is_land = grid.is_land  # (ny, nx) uint8

    # Pre-compute cell-center velocities for wind shear checks
    u_c, v_c = _cell_center_uv(U, V, nz, ny, nx)

    # Layer 1: structural classification (broadcast to 3D)
    # Each bit needs the upwind-upwind cell to exist
    struct = np.zeros((ny, nx), dtype=np.uint8)

    # West pos: i >= 2
    struct[:, 2:] |= SO2_WEST_POS
    # West neg: i < nx-1
    struct[:, :nx - 1] |= SO2_WEST_NEG
    # East pos: i >= 1
    struct[:, 1:] |= SO2_EAST_POS
    # East neg: i < nx-2
    struct[:, :nx - 2] |= SO2_EAST_NEG
    # South pos: j >= 2
    struct[2:, :] |= SO2_SOUTH_POS
    # South neg: j < ny-1
    struct[:ny - 1, :] |= SO2_SOUTH_NEG
    # North pos: j >= 1
    struct[1:, :] |= SO2_NORTH_POS
    # North neg: j < ny-2
    struct[:ny - 2, :] |= SO2_NORTH_NEG

    # Broadcast to 3D
    mask = np.broadcast_to(struct[np.newaxis, :, :], (nz, ny, nx)).copy()

    # Layer 1.5: land mask -- clear bits where upwind-upwind cell is ocean
    # Use zero-padded shifted is_land arrays

    # West pos (upwind-upwind = i-2): is_land[j, i-2]
    if nx > 2:
        ocean = ~is_land[:, :-2].astype(bool)  # (ny, nx-2), corresponds to cells i=2..nx-1
        mask[:, :, 2:] &= np.where(ocean[np.newaxis, :, :], ~np.uint8(SO2_WEST_POS), np.uint8(0xFF))

    # West neg (upwind-upwind = i+1): is_land[j, i+1]
    if nx > 1:
        ocean = ~is_land[:, 1:].astype(bool)  # (ny, nx-1), corresponds to cells i=0..nx-2
        mask[:, :, :nx - 1] &= np.where(ocean[np.newaxis, :, :], ~np.uint8(SO2_WEST_NEG), np.uint8(0xFF))

    # East pos (upwind-upwind = i-1): is_land[j, i-1]
    if nx > 1:
        ocean = ~is_land[:, :-1].astype(bool)
        mask[:, :, 1:] &= np.where(ocean[np.newaxis, :, :], ~np.uint8(SO2_EAST_POS), np.uint8(0xFF))

    # East neg (upwind-upwind = i+2): is_land[j, i+2]
    if nx > 2:
        ocean = ~is_land[:, 2:].astype(bool)
        mask[:, :, :nx - 2] &= np.where(ocean[np.newaxis, :, :], ~np.uint8(SO2_EAST_NEG), np.uint8(0xFF))

    # South pos (upwind-upwind = j-2): is_land[j-2, i]
    if ny > 2:
        ocean = ~is_land[:-2, :].astype(bool)
        mask[:, 2:, :] &= np.where(ocean[np.newaxis, :, :], ~np.uint8(SO2_SOUTH_POS), np.uint8(0xFF))

    # South neg (upwind-upwind = j+1): is_land[j+1, i]
    if ny > 1:
        ocean = ~is_land[1:, :].astype(bool)
        mask[:, :ny - 1, :] &= np.where(ocean[np.newaxis, :, :], ~np.uint8(SO2_SOUTH_NEG), np.uint8(0xFF))

    # North pos (upwind-upwind = j-1): is_land[j-1, i]
    if ny > 1:
        ocean = ~is_land[:-1, :].astype(bool)
        mask[:, 1:, :] &= np.where(ocean[np.newaxis, :, :], ~np.uint8(SO2_NORTH_POS), np.uint8(0xFF))

    # North neg (upwind-upwind = j+2): is_land[j+2, i]
    if ny > 2:
        ocean = ~is_land[2:, :].astype(bool)
        mask[:, :ny - 2, :] &= np.where(ocean[np.newaxis, :, :], ~np.uint8(SO2_NORTH_NEG), np.uint8(0xFF))

    # Layer 2: wind shear classification
    # For each bit, check shear between cell A and cell C (the upwind-upwind pair)

    # West pos: A=(j, i-1), C=(j, i-2)
    if nx > 2:
        shear_ok = _wind_shear_2d(u_c, v_c, nz,
                                   j_slice=(0, ny), i_slice_A=(1, nx - 1), i_slice_C=(0, nx - 2),
                                   axis='i')
        fail = ~shear_ok
        mask[:, :, 2:] &= np.where(fail, ~np.uint8(SO2_WEST_POS), np.uint8(0xFF))

    # West neg: A=(j, i), C=(j, i+1)
    if nx > 1:
        shear_ok = _wind_shear_2d(u_c, v_c, nz,
                                   j_slice=(0, ny), i_slice_A=(0, nx - 1), i_slice_C=(1, nx),
                                   axis='i')
        fail = ~shear_ok
        mask[:, :, :nx - 1] &= np.where(fail, ~np.uint8(SO2_WEST_NEG), np.uint8(0xFF))

    # East pos: A=(j, i), C=(j, i-1)
    if nx > 1:
        shear_ok = _wind_shear_2d(u_c, v_c, nz,
                                   j_slice=(0, ny), i_slice_A=(1, nx), i_slice_C=(0, nx - 1),
                                   axis='i')
        fail = ~shear_ok
        mask[:, :, 1:] &= np.where(fail, ~np.uint8(SO2_EAST_POS), np.uint8(0xFF))

    # East neg: A=(j, i+1), C=(j, i+2)
    if nx > 2:
        shear_ok = _wind_shear_2d(u_c, v_c, nz,
                                   j_slice=(0, ny), i_slice_A=(1, nx - 1), i_slice_C=(2, nx),
                                   axis='i')
        fail = ~shear_ok
        mask[:, :, :nx - 2] &= np.where(fail, ~np.uint8(SO2_EAST_NEG), np.uint8(0xFF))

    # South pos: A=(j-1, i), C=(j-2, i)
    if ny > 2:
        shear_ok = _wind_shear_2d(u_c, v_c, nz,
                                   j_slice_A=(1, ny - 1), j_slice_C=(0, ny - 2), i_slice=(0, nx),
                                   axis='j')
        fail = ~shear_ok
        mask[:, 2:, :] &= np.where(fail, ~np.uint8(SO2_SOUTH_POS), np.uint8(0xFF))

    # South neg: A=(j, i), C=(j+1, i)
    if ny > 1:
        shear_ok = _wind_shear_2d(u_c, v_c, nz,
                                   j_slice_A=(0, ny - 1), j_slice_C=(1, ny), i_slice=(0, nx),
                                   axis='j')
        fail = ~shear_ok
        mask[:, :ny - 1, :] &= np.where(fail, ~np.uint8(SO2_SOUTH_NEG), np.uint8(0xFF))

    # North pos: A=(j, i), C=(j-1, i)
    if ny > 1:
        shear_ok = _wind_shear_2d(u_c, v_c, nz,
                                   j_slice_A=(1, ny), j_slice_C=(0, ny - 1), i_slice=(0, nx),
                                   axis='j')
        fail = ~shear_ok
        mask[:, 1:, :] &= np.where(fail, ~np.uint8(SO2_NORTH_POS), np.uint8(0xFF))

    # North neg: A=(j+1, i), C=(j+2, i)
    if ny > 2:
        shear_ok = _wind_shear_2d(u_c, v_c, nz,
                                   j_slice_A=(1, ny - 1), j_slice_C=(2, ny), i_slice=(0, nx),
                                   axis='j')
        fail = ~shear_ok
        mask[:, :ny - 2, :] &= np.where(fail, ~np.uint8(SO2_NORTH_NEG), np.uint8(0xFF))

    return mask


def _wind_shear_2d(u_c, v_c, nz,
                    j_slice=None, i_slice=None,
                    j_slice_A=None, j_slice_C=None,
                    i_slice_A=None, i_slice_C=None,
                    axis='i'):
    """Vectorized wind shear check between paired cell slices A and C.

    Returns boolean array where True = shear is acceptable.
    """
    if axis == 'i':
        # Varying i, same j range
        j0, j1 = j_slice
        iA0, iA1 = i_slice_A
        iC0, iC1 = i_slice_C
        uA = u_c[:, j0:j1, iA0:iA1]
        vA = v_c[:, j0:j1, iA0:iA1]
        uC = u_c[:, j0:j1, iC0:iC1]
        vC = v_c[:, j0:j1, iC0:iC1]
    else:
        # Varying j, same i range
        i0, i1 = i_slice
        jA0, jA1 = j_slice_A
        jC0, jC1 = j_slice_C
        uA = u_c[:, jA0:jA1, i0:i1]
        vA = v_c[:, jA0:jA1, i0:i1]
        uC = u_c[:, jC0:jC1, i0:i1]
        vC = v_c[:, jC0:jC1, i0:i1]

    spdA_sq = uA * uA + vA * vA
    spdC_sq = uC * uC + vC * vC

    weak = (spdA_sq < MIN_SPD_SQ) | (spdC_sq < MIN_SPD_SQ)

    dot = uA * uC + vA * vC
    mag = np.sqrt(spdA_sq) * np.sqrt(spdC_sq)
    safe_mag = np.where(mag > 0, mag, 1.0)
    cos_theta = dot / safe_mag

    return weak | (cos_theta >= COS_THRESH)


# --- Loop version for equivalence testing ---

def _cell_u(U, k, j, i, nx):
    if i < nx - 1:
        return 0.5 * (U[k, j, i] + U[k, j, i + 1])
    return float(U[k, j, i])


def _cell_v(V, k, j, i, ny):
    if j < ny - 1:
        return 0.5 * (V[k, j, i] + V[k, j + 1, i])
    return float(V[k, j, i])


def _wind_shear_ok(U, V, k, jA, iA, jC, iC, nx, ny):
    uA = _cell_u(U, k, jA, iA, nx)
    vA = _cell_v(V, k, jA, iA, ny)
    uC = _cell_u(U, k, jC, iC, nx)
    vC = _cell_v(V, k, jC, iC, ny)
    spdA_sq = uA * uA + vA * vA
    spdC_sq = uC * uC + vC * vC
    if spdA_sq < MIN_SPD_SQ or spdC_sq < MIN_SPD_SQ:
        return True
    dot = uA * uC + vA * vC
    cos_theta = dot / (np.sqrt(spdA_sq) * np.sqrt(spdC_sq))
    return cos_theta >= COS_THRESH


def _compute_stencil_loop(grid: GridData) -> np.ndarray:
    nz, ny, nx = grid.nz, grid.ny, grid.nx
    U = grid.UAvg
    V = grid.VAvg
    is_land = grid.is_land
    stencil = np.zeros((nz, ny, nx), dtype=np.uint8)

    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                mask = 0
                if i >= 2:
                    mask |= SO2_WEST_POS
                if i < nx - 1:
                    mask |= SO2_WEST_NEG
                if i >= 1:
                    mask |= SO2_EAST_POS
                if i < nx - 2:
                    mask |= SO2_EAST_NEG
                if j >= 2:
                    mask |= SO2_SOUTH_POS
                if j < ny - 1:
                    mask |= SO2_SOUTH_NEG
                if j >= 1:
                    mask |= SO2_NORTH_POS
                if j < ny - 2:
                    mask |= SO2_NORTH_NEG

                if (mask & SO2_WEST_POS) and not is_land[j, i - 2]:
                    mask &= ~SO2_WEST_POS
                if (mask & SO2_WEST_NEG) and not is_land[j, i + 1]:
                    mask &= ~SO2_WEST_NEG
                if (mask & SO2_EAST_POS) and not is_land[j, i - 1]:
                    mask &= ~SO2_EAST_POS
                if (mask & SO2_EAST_NEG) and not is_land[j, i + 2]:
                    mask &= ~SO2_EAST_NEG
                if (mask & SO2_SOUTH_POS) and not is_land[j - 2, i]:
                    mask &= ~SO2_SOUTH_POS
                if (mask & SO2_SOUTH_NEG) and not is_land[j + 1, i]:
                    mask &= ~SO2_SOUTH_NEG
                if (mask & SO2_NORTH_POS) and not is_land[j - 1, i]:
                    mask &= ~SO2_NORTH_POS
                if (mask & SO2_NORTH_NEG) and not is_land[j + 2, i]:
                    mask &= ~SO2_NORTH_NEG

                if mask & SO2_WEST_POS:
                    if not _wind_shear_ok(U, V, k, j, i - 1, j, i - 2, nx, ny):
                        mask &= ~SO2_WEST_POS
                if mask & SO2_WEST_NEG:
                    if not _wind_shear_ok(U, V, k, j, i, j, i + 1, nx, ny):
                        mask &= ~SO2_WEST_NEG
                if mask & SO2_EAST_POS:
                    if not _wind_shear_ok(U, V, k, j, i, j, i - 1, nx, ny):
                        mask &= ~SO2_EAST_POS
                if mask & SO2_EAST_NEG:
                    if not _wind_shear_ok(U, V, k, j, i + 1, j, i + 2, nx, ny):
                        mask &= ~SO2_EAST_NEG
                if mask & SO2_SOUTH_POS:
                    if not _wind_shear_ok(U, V, k, j - 1, i, j - 2, i, nx, ny):
                        mask &= ~SO2_SOUTH_POS
                if mask & SO2_SOUTH_NEG:
                    if not _wind_shear_ok(U, V, k, j, i, j + 1, i, nx, ny):
                        mask &= ~SO2_SOUTH_NEG
                if mask & SO2_NORTH_POS:
                    if not _wind_shear_ok(U, V, k, j, i, j - 1, i, nx, ny):
                        mask &= ~SO2_NORTH_POS
                if mask & SO2_NORTH_NEG:
                    if not _wind_shear_ok(U, V, k, j + 1, i, j + 2, i, nx, ny):
                        mask &= ~SO2_NORTH_NEG

                stencil[k, j, i] = mask

    return stencil
