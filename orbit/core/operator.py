"""Operator assembly.

Assembles the shared N x N transport block, one N x N operator per
transported species (transport plus that species' deposition and diagonal
chemistry loss), and the off-diagonal N x N coupling blocks between species
(SO2 to pSO4, NOx to TotalNO3, the VBS aging cascade, the ISORROPIA
cross-partials). There is no single stacked matrix: the periodic-orbit
solver factors the per-species operators and applies the couplings through
the right-hand sides.
"""

import os

import numpy as np
import scipy.sparse as sp
from orbit.core.indexing import CellIndexer
from orbit.core.grid_data import GridData, EARTH_RADIUS, DEG_TO_RAD
from orbit.core.advection import assemble_vertical_advection
from orbit.core.mixing import assemble_vertical_diffusion
from orbit.core.convdiff import assemble_horizontal_convdiff
from orbit.core.convection import assemble_cmfmc_transport
from orbit.core.grid_data import density_weights
from orbit.core.deposition import (
    assemble_deposition, N_SPECIES,
    IDX_SO2, IDX_NOX, IDX_PSO4, IDX_TOTAL_NO3, IDX_O3, IDX_CO,
)
from orbit.core.chemistry import assemble_so2_oxidation_loss, assemble_so2_to_pso4_source


def compute_lateral_boundary_loss(
    grid: GridData, indexer: CellIndexer,
) -> np.ndarray:
    """Compute lateral (horizontal) boundary loss rate per cell.

    Replicates the advective boundary loss from convdiff.py (lines 204-359).
    This captures only *explicit outflow* — mass leaving the domain through
    boundary faces where wind blows outward. It does NOT capture:

    - Horizontal diffusion at boundaries (convdiff.py's boundary blocks are
      advection-only; diffusion has no boundary loss terms because boundary
      cells simply have fewer neighbors).
    - Missing inflow at upwind boundaries (e.g., cell i=0 with eastward wind
      loses mass to i=1 but receives no replacement from i=-1). This "implicit
      deficit" is a separate, smaller residual that makes upwind boundary cells
      have slightly positive row sums even after closing. It does not cause the
      Dirichlet-like eigenvector decay — that is driven by the explicit outflow.

    Y-boundary loss includes the dx_face/dx_cell terrain correction that
    interior Y-faces apply, using dx at the face latitude (cell edge).

    Boundary conditions:
    - X boundary: i=0 (west) and i=nx-1 (east), non-periodic grids only
    - Y boundary: j=0 (south) and j=ny-1 (north), always present
    - Periodic grids: x-boundary loss is zero (handled by wrap face)

    Parameters
    ----------
    grid : GridData
    indexer : CellIndexer

    Returns
    -------
    lateral_loss : ndarray, shape (N,)
        Lateral boundary loss rate >= 0 for each cell. Zero for interior cells.
    """
    nz, ny, nx = grid.nz, grid.ny, grid.nx
    U = grid.UAvg
    V = grid.VAvg
    use_split = grid.has_split_fluxes

    dx_3d = np.broadcast_to(
        grid.dx[np.newaxis, :, np.newaxis], (nz, ny, nx),
    ).copy()
    dy = grid.dy
    valid = dx_3d > 0

    loss = np.zeros((nz, ny, nx), dtype=np.float64)

    # === X-boundary (non-periodic only) ===
    if not grid.periodic_lon:
        # West boundary: i=0, outflux when u < 0
        bnd_w = valid.copy()
        bnd_w[:, :, 1:] = False
        if use_split:
            um_w0 = grid.UAvg_minus[:, :, 0:1]
            bnd_w_loss = bnd_w & np.broadcast_to(um_w0 > 0, (nz, ny, nx))
            if np.any(bnd_w_loss):
                safe_dx_bnd = np.where(dx_3d > 0, dx_3d, 1.0)
                um_bnd = np.zeros((nz, ny, nx), dtype=np.float64)
                um_bnd[:, :, 0] = grid.UAvg_minus[:, :, 0]
                loss += np.where(bnd_w_loss, um_bnd / safe_dx_bnd, 0.0)
        else:
            u_w0 = U[:, :, 0:1]
            bnd_w_loss = bnd_w & np.broadcast_to(u_w0 < 0, (nz, ny, nx))
            if np.any(bnd_w_loss):
                safe_dx_bnd = np.where(dx_3d > 0, dx_3d, 1.0)
                u_abs_bnd = np.zeros((nz, ny, nx), dtype=np.float64)
                u_abs_bnd[:, :, 0] = np.abs(U[:, :, 0])
                loss += np.where(bnd_w_loss, u_abs_bnd / safe_dx_bnd, 0.0)

        # East boundary: i=nx-1, outflux when u >= 0
        bnd_e = valid.copy()
        bnd_e[:, :, :-1] = False
        if use_split:
            up_bnd = np.zeros((nz, ny, nx), dtype=np.float64)
            up_bnd[:, :, -1] = grid.UAvg_plus[:, :, -1]
            bnd_e_loss = bnd_e & (up_bnd > 0)
            if np.any(bnd_e_loss):
                safe_dx_bnd = np.where(dx_3d > 0, dx_3d, 1.0)
                loss += np.where(bnd_e_loss, up_bnd / safe_dx_bnd, 0.0)
        elif nx > 1:
            ue_bnd = np.zeros((nz, ny, nx), dtype=np.float64)
            ue_bnd[:, :, -1] = U[:, :, -1]
            bnd_e_loss = bnd_e & (ue_bnd >= 0)
            if np.any(bnd_e_loss):
                safe_dx_bnd = np.where(dx_3d > 0, dx_3d, 1.0)
                loss += np.where(bnd_e_loss, ue_bnd / safe_dx_bnd, 0.0)
        elif nx == 1:
            ue_bnd = np.zeros((nz, ny, nx), dtype=np.float64)
            ue_bnd[:, :, 0] = U[:, :, 0]
            bnd_e_loss = bnd_e & (ue_bnd >= 0)
            if np.any(bnd_e_loss):
                safe_dx_bnd = np.where(dx_3d > 0, dx_3d, 1.0)
                loss += np.where(bnd_e_loss, ue_bnd / safe_dx_bnd, 0.0)

    # === Y-boundary (always present) ===
    # Apply dx_face/dx_cell correction at boundary faces, matching convdiff.py.
    # South boundary: j=0, face at lat[0] - dlat/2
    bnd_s = valid.copy()
    bnd_s[:, 1:, :] = False
    lat_face_s = grid.lat[0] - grid.dlat / 2.0
    dx_face_s = EARTH_RADIUS * np.cos(lat_face_s * DEG_TO_RAD) * grid.dlon * DEG_TO_RAD
    safe_dx_s0 = grid.dx[0] if grid.dx[0] > 0 else 1.0
    tr_bnd_s = dx_face_s / safe_dx_s0
    if use_split:
        vm_s0 = grid.VAvg_minus[:, 0:1, :]
        bnd_s_loss = bnd_s & np.broadcast_to(vm_s0 > 0, (nz, ny, nx))
        if np.any(bnd_s_loss):
            safe_dy_val = dy if dy > 0 else 1.0
            loss += np.where(bnd_s_loss, grid.VAvg_minus / safe_dy_val * tr_bnd_s, 0.0)
    else:
        v_s0 = V[:, 0:1, :]
        bnd_s_loss = bnd_s & np.broadcast_to(v_s0 < 0, (nz, ny, nx))
        if np.any(bnd_s_loss):
            safe_dy_val = dy if dy > 0 else 1.0
            loss += np.where(bnd_s_loss, np.abs(V) / safe_dy_val * tr_bnd_s, 0.0)

    # North boundary: j=ny-1, face at lat[-1] + dlat/2
    bnd_n = valid.copy()
    bnd_n[:, :-1, :] = False
    lat_face_n = grid.lat[-1] + grid.dlat / 2.0
    dx_face_n = EARTH_RADIUS * np.cos(lat_face_n * DEG_TO_RAD) * grid.dlon * DEG_TO_RAD
    safe_dx_nend = grid.dx[-1] if grid.dx[-1] > 0 else 1.0
    tr_bnd_n = dx_face_n / safe_dx_nend
    if use_split:
        vp_bnd = np.zeros((nz, ny, nx), dtype=np.float64)
        vp_bnd[:, -1, :] = grid.VAvg_plus[:, -1, :]
        bnd_n_loss = bnd_n & (vp_bnd > 0)
        if np.any(bnd_n_loss):
            safe_dy_val = dy if dy > 0 else 1.0
            loss += np.where(bnd_n_loss, vp_bnd / safe_dy_val * tr_bnd_n, 0.0)
    elif ny > 1:
        vn_bnd = np.zeros((nz, ny, nx), dtype=np.float64)
        vn_bnd[:, -1, :] = V[:, -1, :]
        bnd_n_loss = bnd_n & (vn_bnd >= 0)
        if np.any(bnd_n_loss):
            safe_dy_val = dy if dy > 0 else 1.0
            loss += np.where(bnd_n_loss, vn_bnd / safe_dy_val * tr_bnd_n, 0.0)
    elif ny == 1:
        vn_bnd = np.zeros((nz, ny, nx), dtype=np.float64)
        vn_bnd[:, 0, :] = V[:, 0, :]
        bnd_n_loss = bnd_n & (vn_bnd >= 0)
        if np.any(bnd_n_loss):
            safe_dy_val = dy if dy > 0 else 1.0
            loss += np.where(bnd_n_loss, vn_bnd / safe_dy_val * tr_bnd_n, 0.0)

    return loss.ravel()


def build_detection_operator(
    T: sp.csc_matrix, lateral_loss: np.ndarray,
) -> tuple:
    """Build lateral-closed detection operator.

    Removes lateral boundary loss from T's diagonal. Off-diagonals unchanged.
    Operates on T in ORBIT convention (positive diagonal = loss rate).
    To get the Markov generator Q_det for Schur decomposition, apply
    transport_to_generator() to the returned T_det.

    Parameters
    ----------
    T : csc_matrix (N, N) -- transport operator with absorbing BCs
    lateral_loss : ndarray (N,) -- from compute_lateral_boundary_loss()

    Returns
    -------
    T_det : csc_matrix (N, N) -- lateral boundaries closed
    n_closed : int -- number of cells whose diagonal was adjusted
    total_lateral_flux : float -- total lateral flux removed (diagnostic)
    """
    T_det = T.copy()
    T_det -= sp.diags(lateral_loss)
    n_closed = int(np.count_nonzero(lateral_loss > 0))
    total_lateral_flux = float(np.sum(lateral_loss))
    return T_det, n_closed, total_lateral_flux


def validate_detection_operator(
    T_det: sp.csc_matrix, T: sp.csc_matrix, indexer: CellIndexer,
) -> dict:
    """Validate the lateral-closed detection operator.

    Checks:
    1. Interior cells unchanged (row sums identical)
    2. Boundary cells: diagonal reduced or unchanged
    3. Off-diagonal entries identical
    4. No negative diagonals

    Parameters
    ----------
    T_det : csc_matrix (N, N) -- lateral-closed operator
    T : csc_matrix (N, N) -- original operator
    indexer : CellIndexer

    Returns
    -------
    dict with validation results (bool values + diagnostics)
    """
    nz, ny, nx = indexer.nz, indexer.ny, indexer.nx
    N = indexer.N

    # Build masks using flat-index arithmetic
    n3d = np.arange(N).reshape(nz, ny, nx)
    is_lateral = np.zeros(N, dtype=bool)
    # i=0, i=nx-1
    is_lateral[n3d[:, :, 0].ravel()] = True
    is_lateral[n3d[:, :, -1].ravel()] = True
    # j=0, j=ny-1
    is_lateral[n3d[:, 0, :].ravel()] = True
    is_lateral[n3d[:, -1, :].ravel()] = True
    is_interior = ~is_lateral

    # Row sums
    T_csr = T.tocsr()
    T_det_csr = T_det.tocsr()
    rs_T = np.array(T_csr.sum(axis=1)).ravel()
    rs_det = np.array(T_det_csr.sum(axis=1)).ravel()

    # 1. Interior cells: row sums unchanged
    interior_ok = np.allclose(rs_det[is_interior], rs_T[is_interior], atol=1e-12)

    # 2. Boundary cells: diagonal reduced (T_det diagonal <= T diagonal)
    diag_T = np.array(T.diagonal())
    diag_det = np.array(T_det.diagonal())
    boundary_diag_ok = np.all(diag_det[is_lateral] <= diag_T[is_lateral] + 1e-12)

    # 3. Off-diagonal entries identical
    diff = T_det - T
    diff_csr = diff.tocsr()
    # Off-diag: zero out diagonal of diff
    diff_offdiag = diff_csr.copy()
    diff_offdiag.setdiag(0)
    diff_offdiag.eliminate_zeros()
    offdiag_ok = diff_offdiag.nnz == 0

    # 4. No negative diagonals
    no_neg_diag = np.all(diag_det >= -1e-12)

    return {
        "interior_unchanged": interior_ok,
        "boundary_diag_reduced": boundary_diag_ok,
        "offdiag_identical": offdiag_ok,
        "no_negative_diag": no_neg_diag,
        "n_lateral_cells": int(np.sum(is_lateral)),
        "n_interior_cells": int(np.sum(is_interior)),
        "max_rowsum_diff_interior": float(np.max(np.abs(
            rs_det[is_interior] - rs_T[is_interior]))) if np.any(is_interior) else 0.0,
    }


def assemble_transport_block(
    grid: GridData, indexer: CellIndexer, scheme: str = "exp",
    return_boundary_fluxes: bool = False,
) -> sp.csc_matrix:
    """Assemble the transport block T (shared across all species).

    T = T_hcd + T_vadv + T_vdiff + T_conv   (Patankar exponential scheme)

    T_hcd is the horizontal convection-diffusion operator
    (assemble_horizontal_convdiff), T_vadv/T_vdiff the vertical
    advection/diffusion, and T_conv the CMFMC convective mass-flux operator
    (orbit.core.convection.assemble_cmfmc_transport; a zero matrix if
    grid.CMFMC is empty). The legacy first/second-order upwind schemes
    (scheme="fo"/"so") and their stencil machinery were retired in the
    production-branch cleanup (2026-05-22); only "exp" remains.

    Parameters
    ----------
    grid : GridData
    indexer : CellIndexer
    scheme : str
        Retained for call-site compatibility; must be "exp".
    return_boundary_fluxes : bool
        If True, also return the lateral boundary loss array
        (compute_lateral_boundary_loss). Consumed by the lateral-closed
        detection operator (Q_det); see build_detection_operator.

    Returns
    -------
    T : csc_matrix, shape (N, N)
        or (T, lateral_loss) if return_boundary_fluxes=True
    """
    if scheme != "exp":
        raise ValueError(
            f"Only the Patankar exponential scheme is supported (got scheme={scheme!r}); "
            "the legacy fo/so upwind schemes were retired (2026-05-22)."
        )
    T_hcd = assemble_horizontal_convdiff(grid, indexer)
    if getattr(grid, "unified_vertical_patankar", False):
        # PROTOTYPE (gated off by default): single Patankar exponential
        # operator unifying omega advection + Kzz diffusion vertically, in
        # place of the standalone T_vadv + T_vdiff. See
        # convdiff.assemble_vertical_convdiff.
        from orbit.core.convdiff import assemble_vertical_convdiff
        T_vert = assemble_vertical_convdiff(grid, indexer)
    else:
        T_vert = (assemble_vertical_advection(grid, indexer)
                  + assemble_vertical_diffusion(grid, indexer))
    T_conv = assemble_cmfmc_transport(grid, indexer)
    T = T_hcd + T_vert + T_conv

    # The four blocks conserve the pressure measure dP*area and leave a
    # uniform mixing ratio alone. The state the rest of the model reads is a
    # concentration at local density, c = rho x with rho = dP/(g Dz), so the
    # block that acts on c is diag(rho) T diag(1/rho): it conserves the
    # volume measure and leaves c = rho * const alone. Until 2026-09-26 the
    # untransformed block acted on c directly, which read the transported
    # variable as a mixing ratio in transport and as a concentration
    # everywhere else, a source-to-receptor density ratio in every result
    # (under 1% on the plains, about 20% on the Tibetan plateau).
    if getattr(grid, "concentration_state", True):
        T = to_concentration_form(T, grid)

    if not return_boundary_fluxes:
        return T
    lateral_loss = compute_lateral_boundary_loss(grid, indexer)
    return T, lateral_loss


def to_concentration_form(T: sp.csc_matrix, grid: GridData) -> sp.csc_matrix:
    """diag(rho) T diag(1/rho) with rho = dP/(g Dz): the block on concentrations."""
    rho = density_weights(grid)
    return (sp.diags(rho) @ T @ sp.diags(1.0 / rho)).tocsc()


def assemble_single_species_L(T: sp.csc_matrix, D: sp.csc_matrix) -> sp.csc_matrix:
    """Assemble L for a single species: L_s = T + D.

    Parameters
    ----------
    T : transport block (shared)
    D : deposition diagonal (species-specific, dry + wet combined)

    Returns
    -------
    L_s : csc_matrix, shape (N, N)
    """
    return T + D


def assemble_species_operators(
    grid: GridData, indexer: CellIndexer,
    verbose: bool = True,
    scheme: str = "exp",
    nox_to_no3_rate: np.ndarray = None,
    so2_ox_rate: np.ndarray = None,
    o3_loss_rate: np.ndarray = None,
    co_loss_rate: np.ndarray = None,
    return_D_per_species: bool = False,
) -> tuple:
    """Assemble N_SPECIES independent single-species operators + chemistry couplings.

    14-species inventory (see deposition.py for indices; the first nine are):
        0 SoA, 1 PrimaryPM25, 2 TotalNH, 3 SO2, 4 NOx,
        5 pSO4, 6 TotalNO3, 7 O3, 8 CO.

    Two off-diagonal kinetic couplings:
        SO2 -> pSO4    (via grid.SO2oxidation or so2_ox_rate override)
        NOx -> TotalNO3 (new, via nox_to_no3_rate; zero if not supplied)

    Parameters
    ----------
    grid : GridData
    indexer : CellIndexer
    verbose : bool
    scheme : str
        Must be "exp" (Patankar exponential); the legacy fo/so upwind
        schemes and their stencil machinery were retired (2026-05-22).
    nox_to_no3_rate : ndarray (nz, ny, nx) or None
        Per-cell NOx -> TotalNO3 conversion rate (1/s). Built by the
        oxidants module from diagnostic OH and N2O5 chemistry. If None,
        the NOx -> TotalNO3 coupling is inactive (zero-rate placeholder).
    so2_ox_rate : ndarray (nz, ny, nx) or None
        Per-cell total SO2 -> pSO4 rate (1/s). When provided, replaces
        grid.SO2oxidation (preprocessor aqueous) with the override. Typical
        use: pass aqueous + gas-phase-from-OH from the oxidants module.
    o3_loss_rate : ndarray (nz, ny, nx) or None
        Per-cell O3 chemistry loss rate (1/s).  From `build_o3_rates`:
        k_NO+O3 * [NO] + jO1D * f_O1D->OH.  The corresponding production
        term (ug/m3/s, HO2+NO + RO2+NO) does NOT go into the operator — it
        is fed to the orbit RHS via extra_rhs_per_bin_per_species since O3
        production has no [O3]-dependence.
    co_loss_rate : ndarray (nz, ny, nx) or None
        Per-cell CO chemistry loss rate (1/s): k_OH_CO(T, M) * [OH].
        CO has no transported source species; production from VOC
        oxidation is small and handled (if desired) via the orbit RHS.
        When None, CO is transport-only (no chemistry loss).

    Returns
    -------
    L_species : list of 8 csc_matrix, each (N, N)
        Includes K_ox for SO2 and (if nox_to_no3_rate given) K_nox_loss for NOx.
    K_sources : dict[(target, source)] -> csc_matrix
        Off-diagonal coupling blocks indexed by (target_species, source_species).
        Always contains (IDX_PSO4, IDX_SO2). Contains (IDX_TOTAL_NO3, IDX_NOX)
        only if nox_to_no3_rate is non-zero.
    T : csc_matrix, (N, N)
        Shared transport block
    d_species : ndarray, (N_SPECIES, N)
        Diagonal loss vectors per species.
    """
    if scheme != "exp":
        raise ValueError(
            f"Only scheme='exp' is supported (got {scheme!r}); the legacy fo/so "
            "upwind schemes and their stencil machinery were retired (2026-05-22)."
        )

    if verbose:
        print("Assembling transport block T...")
    T = assemble_transport_block(grid, indexer, scheme=scheme)

    if verbose:
        print("Assembling deposition blocks...")
    D = [assemble_deposition(grid, indexer, s) for s in range(N_SPECIES)]

    if verbose:
        print("Assembling chemistry coupling...")
    if so2_ox_rate is not None:
        N = indexer.N
        rate_flat = np.asarray(so2_ox_rate, dtype=np.float64).ravel()
        idx = np.arange(N, dtype=np.int64)
        K_ox_so2 = sp.csc_matrix((rate_flat, (idx, idx)), shape=(N, N))
        K_so2_to_pso4 = sp.csc_matrix((-rate_flat, (idx, idx)), shape=(N, N))
    else:
        K_ox_so2 = assemble_so2_oxidation_loss(grid, indexer)
        K_so2_to_pso4 = assemble_so2_to_pso4_source(grid, indexer)

    K_sources = {(IDX_PSO4, IDX_SO2): K_so2_to_pso4}

    # NOx -> TotalNO3 coupling: loss on NOx diagonal, source for TotalNO3.
    # On feat/orbit-isorropia-archive, when no explicit rate is provided
    # and the grid carries bin-resolved archive oxidants, fall back to the
    # archive-driven rate (k_OH+NO2 · [OH] · fNO2 daytime + N2O5 hydrolysis
    # night via NO2+NO3rad thermal equilibrium). Mirrors the so2_ox_rate
    # fallback to grid.SO2oxidation above. DCOMP callers can still pass
    # an explicit oxidants-module rate to override.
    if nox_to_no3_rate is None and getattr(grid, "archive_OH", np.array([])).size > 0:
        from orbit.core.nox_to_no3_rate import build_nox_to_no3_rate
        nox_to_no3_rate = build_nox_to_no3_rate(grid)

    K_ox_nox = None
    if nox_to_no3_rate is not None:
        N = indexer.N
        rate_flat = np.asarray(nox_to_no3_rate, dtype=np.float64).ravel()
        idx = np.arange(N, dtype=np.int64)
        K_ox_nox = sp.csc_matrix((rate_flat, (idx, idx)), shape=(N, N))
        K_nox_to_no3 = sp.csc_matrix((-rate_flat, (idx, idx)), shape=(N, N))
        K_sources[(IDX_TOTAL_NO3, IDX_NOX)] = K_nox_to_no3

    # O3 chemistry loss diagonal: NO+O3 + jO1D*f_O1D->OH from build_o3_rates.
    # O3 production (HO2+NO, RO2+NO) is handled in the orbit RHS, not here,
    # because it has no [O3] dependence.
    K_ox_o3 = None
    if o3_loss_rate is not None:
        N = indexer.N
        rate_flat = np.asarray(o3_loss_rate, dtype=np.float64).ravel()
        idx = np.arange(N, dtype=np.int64)
        K_ox_o3 = sp.csc_matrix((rate_flat, (idx, idx)), shape=(N, N))

    # CO chemistry loss diagonal: k_OH_CO * [OH] (linear in [CO]).
    K_ox_co = None
    if co_loss_rate is not None:
        N = indexer.N
        rate_flat = np.asarray(co_loss_rate, dtype=np.float64).ravel()
        idx = np.arange(N, dtype=np.int64)
        K_ox_co = sp.csc_matrix((rate_flat, (idx, idx)), shape=(N, N))

    # VBS aging: OH-driven first-order cascade C1000 → C100 → C10 → C1 → C01.
    # Linear couplings parallel to SO2 → pSO4 (one off-diagonal per source-
    # destination pair). Source loses at rate k_age × [OH] per molecule;
    # destination gains at α_frag × (loss rate), routing the (1 - α_frag)
    # fraction to fragmentation (mass loss to gas-phase residual that we
    # don't transport). archive_OH is in molec/cm³; k_age in cm³/molec/s,
    # so the product is in 1/s as required.
    #
    # Defaults: k_age = 4e-11 (Tsimpidi 2010 / CMAQ); α_frag = 0.75
    # (Lane 2008 central).
    K_age_loss = {}        # dict[src_idx] -> csc_matrix
    K_age_src = {}         # dict[(dst_idx, src_idx)] -> csc_matrix
    if getattr(grid, "archive_OH", np.array([])).size > 0:
        from orbit.core.deposition import VBS_AGING_PAIRS
        K_OH_AGE = float(os.environ.get("ORBIT_VBS_K_AGE", "4.0e-11"))
        ALPHA_FRAG = float(os.environ.get("ORBIT_VBS_FRAG", "0.75"))
        N = indexer.N
        idx = np.arange(N, dtype=np.int64)
        # Surface OH only — aging happens in PBL where OH is meaningful.
        # Above k=0, archive OH is effectively zero in practice, but we
        # use the full 3D field to be physically consistent.
        rate_flat = (K_OH_AGE * grid.archive_OH).ravel()
        for src_idx, dst_idx in VBS_AGING_PAIRS:
            K_age_loss[src_idx] = sp.csc_matrix(
                (rate_flat, (idx, idx)), shape=(N, N))
            K_age_src[(dst_idx, src_idx)] = sp.csc_matrix(
                (-ALPHA_FRAG * rate_flat, (idx, idx)), shape=(N, N))
            K_sources[(dst_idx, src_idx)] = K_age_src[(dst_idx, src_idx)]

    # Extract diagonal loss vectors per species
    d_species = np.empty((N_SPECIES, indexer.N), dtype=np.float64)
    for s in range(N_SPECIES):
        d_species[s] = np.asarray(D[s].diagonal()).ravel()
        if s == IDX_SO2:
            d_species[s] += np.asarray(K_ox_so2.diagonal()).ravel()
        if s == IDX_NOX and K_ox_nox is not None:
            d_species[s] += np.asarray(K_ox_nox.diagonal()).ravel()
        if s == IDX_O3 and K_ox_o3 is not None:
            d_species[s] += np.asarray(K_ox_o3.diagonal()).ravel()
        if s == IDX_CO and K_ox_co is not None:
            d_species[s] += np.asarray(K_ox_co.diagonal()).ravel()
        if s in K_age_loss:
            d_species[s] += np.asarray(K_age_loss[s].diagonal()).ravel()

    L_species = []
    for s in range(N_SPECIES):
        L_s = T + D[s]
        if s == IDX_SO2:
            L_s = L_s + K_ox_so2
        if s == IDX_NOX and K_ox_nox is not None:
            L_s = L_s + K_ox_nox
        if s == IDX_O3 and K_ox_o3 is not None:
            L_s = L_s + K_ox_o3
        if s == IDX_CO and K_ox_co is not None:
            L_s = L_s + K_ox_co
        if s in K_age_loss:
            L_s = L_s + K_age_loss[s]
        L_species.append(L_s)

    total_nnz = sum(L_s.nnz for L_s in L_species) + sum(K.nnz for K in K_sources.values())
    if verbose:
        N = indexer.N
        print(f"{N_SPECIES} species operators: {N:,} x {N:,} each, "
              f"total nnz = {total_nnz:,}, couplings = {len(K_sources)}")

    if return_D_per_species:
        return L_species, K_sources, T, d_species, D, K_age_loss
    return L_species, K_sources, T, d_species


def mass_balance_diagnostics(grids, indexer: CellIndexer, blocks=("transport",)) -> dict:
    """Row sums of the transport block over laterally interior cells, per day.

    For a uniform mixing ratio the transport block changes cell i at the
    rate -rowsum_i, so a non-zero interior row sum is air appearing or
    vanishing: a spurious source or sink of every tracer. This is the check
    that would have caught the vertical-advection ground leak of 2022
    (surface rows summing to about 1 per day). Lateral boundary cells are
    excluded (zero inflow, free outflow by design); the top layer is
    reported separately, since a downward flux through the domain top
    brings in tracer-free air and reads as a sink there.

    Returns a dict with, over all bins pooled, the median, 90th and 99th
    percentile and maximum of |row sum| per day for the interior layers,
    the same for the top layer, the per-layer 90th percentile, and the
    fraction of interior cells above 0.01 and 0.1 per day.
    """
    DAY = 86400.0
    nz, ny, nx = grids[0].nz, grids[0].ny, grids[0].nx
    inner = np.zeros((nz, ny, nx), dtype=bool)
    inner[:, 1:-1, 1:-1] = True
    rows_interior, rows_top, per_layer = [], [], []
    for g in grids:
        T = assemble_transport_block(g, indexer)
        # The state a mass-consistent operator leaves alone is a uniform
        # mixing ratio: x = 1 in the pressure form, c = rho in the
        # concentration form. (T w) / w is the row sum of the pressure form
        # either way.
        w = density_weights(g) if getattr(g, "concentration_state", True) else np.ones(indexer.N)
        r = (np.asarray(T @ w).ravel() / w).reshape(nz, ny, nx) * DAY
        r = np.where(inner, r, np.nan)
        rows_interior.append(r[:-1].ravel())
        rows_top.append(r[-1].ravel())
        per_layer.append(r)
    ri = np.abs(np.concatenate(rows_interior)); ri = ri[np.isfinite(ri)]
    rt = np.abs(np.concatenate(rows_top)); rt = rt[np.isfinite(rt)]
    pl = np.abs(np.stack(per_layer))                      # (bins, nz, ny, nx)
    layer_p90 = [float(np.nanpercentile(pl[:, k], 90)) for k in range(nz)]
    layer_mean = []
    for k in range(nz):
        v = np.stack(per_layer)[:, k]
        layer_mean.append(float(np.nanmean(v)))
    return {
        "units": "per day, |row sum| of the transport block, laterally interior cells, all bins",
        "interior_median": float(np.median(ri)),
        "interior_p90": float(np.percentile(ri, 90)),
        "interior_p99": float(np.percentile(ri, 99)),
        "interior_max": float(ri.max()),
        "interior_frac_above_0.01": float(np.mean(ri > 0.01)),
        "interior_frac_above_0.1": float(np.mean(ri > 0.1)),
        "top_median": float(np.median(rt)),
        "top_p90": float(np.percentile(rt, 90)),
        "layer_p90": layer_p90,
        "layer_mean_signed": layer_mean,
    }


def operator_diagnostics(L: sp.csc_matrix, indexer: CellIndexer) -> dict:
    """Compute diagnostics for the operator matrix.

    Parameters
    ----------
    L : csc_matrix
        Operator matrix (N x N; a per-species operator or the transport block)
    indexer : CellIndexer

    Returns
    -------
    dict with keys:
        row_sums_min, row_sums_max, row_sums_mean
        diag_dominant_frac: fraction of rows that are diagonally dominant
        nnz_per_row_mean, nnz_per_row_max
        positive_diagonal_frac: fraction of diagonal entries > 0
    """
    L_csr = L.tocsr()

    # Row sums
    row_sums = np.array(L_csr.sum(axis=1)).ravel()

    # Diagonal
    diag = np.array(L.diagonal())

    # Diagonal dominance
    abs_offdiag_sum = np.array(np.abs(L_csr).sum(axis=1)).ravel() - np.abs(diag)
    diag_dominant = np.abs(diag) >= abs_offdiag_sum

    # nnz per row
    nnz_per_row = np.diff(L_csr.indptr)

    return {
        "shape": L.shape,
        "nnz": L.nnz,
        "row_sums_min": float(np.min(row_sums)),
        "row_sums_max": float(np.max(row_sums)),
        "row_sums_mean": float(np.mean(row_sums)),
        "negative_row_sum_count": int(np.sum(row_sums < -1e-10)),
        "diag_dominant_frac": float(np.mean(diag_dominant)),
        "positive_diagonal_frac": float(np.mean(diag > 0)),
        "nnz_per_row_mean": float(np.mean(nnz_per_row)),
        "nnz_per_row_max": int(np.max(nnz_per_row)),
    }
