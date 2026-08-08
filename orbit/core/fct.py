"""Flux-corrected transport (FCT) for horizontal advection.

Phase 1: the grid-agnostic core — high-order (van Leer MUSCL) face flux,
first-order upwind low-order flux, the anti-diffusive flux A = F_high - F_low,
and the Zalesak multidimensional flux limiter. These are validated on idealized
advection (tests/test_fct.py) before being wired into the orbit solve.

Why FCT: the production horizontal scheme is pinned at first-order upwind by
Godunov's barrier (median face |Pe| ~ 1.7e5), whose implicit numerical diffusion
K_num = |U|*dx/2 ~ 6e4 m^2/s is 50-500x the model's intended sub-grid mixing and
spreads a localized source over ~1.5-4 grid cells at steady state. FCT removes
that O(dx) diffusion in smooth flow (recovering 2nd order) while a solution-
dependent limiter keeps monotonicity at fronts — the nonlinear escape from
Godunov.

This module deliberately works on plain arrays with explicit time stepping so
the spatial scheme + limiter can be proven correct (2nd-order smooth, monotone
at discontinuities, conservative) independently of ORBIT's implicit orbit
machinery. Phase 2 reuses `antidiffusive_flux` / `zalesak_coefficients` as a
deferred-correction source inside the backward-Euler step.

Conventions (1-D, uniform spacing for the harness; generalized to the metric
grid in Phase 2):
  - cell-centred c[i]; face i+1/2 lies between cell i and cell i+1.
  - face velocity u_face[i] is the velocity at face i+1/2, positive = +x
    (cell i -> cell i+1).
  - flux F[i] is the signed flux through face i+1/2 (units: c * velocity).
  - periodic boundaries via np.roll for the idealized tests.
"""
from __future__ import annotations
import numpy as np
import scipy.sparse as sp


# --------------------------------------------------------------------------- #
# Limiter
# --------------------------------------------------------------------------- #
def vanleer(r: np.ndarray) -> np.ndarray:
    """van Leer flux limiter phi(r) = (r + |r|) / (1 + |r|).

    Second-order TVD; phi(r)=0 for r<=0 (extrema -> upwind), phi->2 as r->inf,
    phi(1)=1 (smooth -> centred). Lipschitz with kinks only at r=0 (extrema),
    which is the smoothness the frozen-coefficient SR tangent-linear relies on.
    """
    r = np.asarray(r, dtype=np.float64)
    out = (r + np.abs(r)) / (1.0 + np.abs(r))
    # r = +inf (zero downwind gradient) -> phi -> 2; nan-guard handled by caller
    return out


# --------------------------------------------------------------------------- #
# 1-D fluxes (periodic), explicit harness
# --------------------------------------------------------------------------- #
def low_order_flux_1d(c: np.ndarray, u_face: np.ndarray) -> np.ndarray:
    """First-order upwind flux at each face i+1/2 (periodic).

    F_low[i] = max(u,0) c[i] + min(u,0) c[i+1]
    """
    cL = c
    cR = np.roll(c, -1)            # c[i+1]
    return np.maximum(u_face, 0.0) * cL + np.minimum(u_face, 0.0) * cR


def _gradient_ratio(c: np.ndarray, upwind_is_left: np.ndarray) -> np.ndarray:
    """Consecutive-gradient ratio r at each face i+1/2 (periodic).

    For a face with the upwind cell on the left (u>0):
        r = (c[i] - c[i-1]) / (c[i+1] - c[i])
    with the upwind cell on the right (u<0):
        r = (c[i+2] - c[i+1]) / (c[i+1] - c[i])
    """
    cim1 = np.roll(c, 1)
    ci = c
    cip1 = np.roll(c, -1)
    cip2 = np.roll(c, -2)

    denom = cip1 - ci                      # downwind-side gradient (face local)
    num_L = ci - cim1
    num_R = cip2 - cip1
    num = np.where(upwind_is_left, num_L, num_R)
    # guard the 0/0: where denom ~ 0 the face is locally flat -> r large so
    # phi -> its r->inf limit; sign chosen to keep r finite & positive there.
    eps = 1e-30
    safe_denom = np.where(np.abs(denom) < eps, eps, denom)
    r = num / safe_denom
    # flat downwind gradient: treat as smooth (r=1 -> centred) only if the
    # upwind gradient is also ~0; otherwise large r (phi->2). Use np.where:
    r = np.where(np.abs(denom) < eps, np.where(np.abs(num) < eps, 1.0, 1e6), r)
    return r


def high_order_flux_1d(c: np.ndarray, u_face: np.ndarray) -> np.ndarray:
    """van Leer MUSCL high-order flux at each face i+1/2 (periodic).

    c_face = c_upwind + 0.5 phi(r) (c_downwind - c_upwind);  F_high = u * c_face
    """
    ci = c
    cip1 = np.roll(c, -1)
    upwind_is_left = u_face >= 0.0
    r = _gradient_ratio(c, upwind_is_left)
    phi = vanleer(r)
    c_upwind = np.where(upwind_is_left, ci, cip1)
    c_downwind = np.where(upwind_is_left, cip1, ci)
    c_face = c_upwind + 0.5 * phi * (c_downwind - c_upwind)
    return u_face * c_face


def antidiffusive_flux(f_high: np.ndarray, f_low: np.ndarray) -> np.ndarray:
    """Anti-diffusive flux A = F_high - F_low."""
    return f_high - f_low


# --------------------------------------------------------------------------- #
# Zalesak limiter (1-D periodic harness form)
# --------------------------------------------------------------------------- #
def zalesak_coefficients_1d(c_prev: np.ndarray, c_td: np.ndarray,
                            A: np.ndarray, dx: float, dt: float) -> np.ndarray:
    """Per-face Zalesak coefficient C in [0,1] (Zalesak 1979), 1-D periodic.

    Guarantees c_td + dt/dx * (limited anti-diffusive update) introduces no new
    extrema relative to {c_prev, c_td} and their neighbours.

    Parameters
    ----------
    c_prev : pre-step solution c^n
    c_td   : transported-and-diffused (low-order) solution
    A      : anti-diffusive face flux at i+1/2
    dx, dt : uniform spacing / time step

    Returns
    -------
    C : per-face coefficient at i+1/2
    """
    # Allowed bounds from c_prev and c_td and their neighbours
    stack = np.vstack([c_prev, c_td,
                       np.roll(c_prev, 1), np.roll(c_prev, -1),
                       np.roll(c_td, 1), np.roll(c_td, -1)])
    c_max = stack.max(axis=0)
    c_min = stack.min(axis=0)

    # Anti-diffusive flux into cell i: +A through left face (i-1/2), -A through
    # right face (i+1/2). A is indexed by right face i+1/2, so the left face of
    # cell i is A[i-1].
    A_right = A                 # flux through i+1/2 (leaves i to the right)
    A_left = np.roll(A, 1)      # flux through i-1/2 (enters i from the left)

    # net into i = A_left - A_right
    P_plus = np.maximum(A_left, 0.0) + np.maximum(-A_right, 0.0)
    P_minus = np.maximum(-A_left, 0.0) + np.maximum(A_right, 0.0)

    Q_plus = (c_max - c_td) * dx / dt
    Q_minus = (c_td - c_min) * dx / dt

    R_plus = np.where(P_plus > 0, np.minimum(1.0, Q_plus / np.where(P_plus > 0, P_plus, 1.0)), 0.0)
    R_minus = np.where(P_minus > 0, np.minimum(1.0, Q_minus / np.where(P_minus > 0, P_minus, 1.0)), 0.0)

    # Face i+1/2: A>0 raises the right cell (i+1) and lowers the left cell (i).
    # C = min(R_plus[i+1], R_minus[i]) if A>=0 else min(R_minus[i+1], R_plus[i])
    Rp_R = np.roll(R_plus, -1)   # R_plus at cell i+1
    Rm_R = np.roll(R_minus, -1)  # R_minus at cell i+1
    C_pos = np.minimum(Rp_R, R_minus)
    C_neg = np.minimum(Rm_R, R_plus)
    C = np.where(A >= 0.0, C_pos, C_neg)
    return np.clip(C, 0.0, 1.0)


# --------------------------------------------------------------------------- #
# Explicit single-step drivers (idealized harness only)
# --------------------------------------------------------------------------- #
def _divergence_1d(F: np.ndarray, dx: float) -> np.ndarray:
    """d/dx of a face flux: (F[i+1/2] - F[i-1/2]) / dx, periodic."""
    return (F - np.roll(F, 1)) / dx


def upwind_step_1d(c: np.ndarray, u_face: np.ndarray, dx: float, dt: float) -> np.ndarray:
    """One explicit first-order upwind step (the low-order reference)."""
    F = low_order_flux_1d(c, u_face)
    return c - dt * _divergence_1d(F, dx)


def fct_step_1d(c: np.ndarray, u_face: np.ndarray, dx: float, dt: float) -> np.ndarray:
    """One explicit FCT step: low-order transport + limited anti-diffusion."""
    F_low = low_order_flux_1d(c, u_face)
    c_td = c - dt * _divergence_1d(F_low, dx)
    F_high = high_order_flux_1d(c, u_face)
    A = antidiffusive_flux(F_high, F_low)
    C = zalesak_coefficients_1d(c, c_td, A, dx, dt)
    return c_td - dt * _divergence_1d(C * A, dx)


def high_order_step_1d(c: np.ndarray, u_face: np.ndarray, dx: float, dt: float) -> np.ndarray:
    """One explicit van Leer step WITHOUT the Zalesak guard (for convergence
    measurement in smooth flow; not monotone at fronts)."""
    F = high_order_flux_1d(c, u_face)
    return c - dt * _divergence_1d(F, dx)


# --------------------------------------------------------------------------- #
# 2-D constant-velocity advection (periodic) for the metric-free shape test
# --------------------------------------------------------------------------- #
def fct_step_2d(c: np.ndarray, ux: float, uy: float, dx: float, dy: float,
                dt: float) -> np.ndarray:
    """One explicit dimensionally-split FCT step on a periodic 2-D field.

    c has shape (ny, nx). Constant velocities (ux, uy). Strang-free simple
    split (x then y); sufficient for the mass-conservation / no-new-extrema
    shape test. axis=1 is x, axis=0 is y.
    """
    def step_axis(field, vel, d, axis):
        # move the target axis to the last position, treat rows independently
        f = np.moveaxis(field, axis, -1)
        shp = f.shape
        f2 = f.reshape(-1, shp[-1])
        uface = np.full(shp[-1], vel, dtype=np.float64)
        out = np.empty_like(f2)
        for k in range(f2.shape[0]):
            out[k] = fct_step_1d(f2[k], uface, d, dt)
        return np.moveaxis(out.reshape(shp), -1, axis)

    c1 = step_axis(c, ux, dx, axis=1)
    c2 = step_axis(c1, uy, dy, axis=0)
    return c2


# --------------------------------------------------------------------------- #
# Grid-aware horizontal FCT source (Phase 2)
# --------------------------------------------------------------------------- #
# d_AD = -div(C ⊙ A) on the real (metric, terrain, split-flux) horizontal grid,
# mirroring orbit.core.convdiff exactly so the low-order flux FCT subtracts is
# the same upwind flux the production operator builds. Interior x/y faces only;
# regional open boundaries fall back to first-order (no neighbour to
# reconstruct from), which is standard and conservative. Returns a per-cell
# tendency (same units as -L c) to be used as a deferred-correction RHS source.

def _vanleer_phi(c_up, c_far_up, c_down):
    """Frozen van Leer limiter value phi(r) at a face. r = (c_up - c_far_up) /
    (c_down - c_up). Bounded in [0, 2]; phi=0 at extrema. Used both for the
    forward reconstruction increment and as the frozen coefficient of the
    tangent-linear operator (where it must stay bounded — no division by the
    gradient)."""
    num = c_up - c_far_up
    den = c_down - c_up
    eps = 1e-30
    safe_den = np.where(np.abs(den) < eps, eps, den)
    r = num / safe_den
    r = np.where(np.abs(den) < eps, np.where(np.abs(num) < eps, 1.0, 1e6), r)
    return vanleer(r)


def _vanleer_increment(c_up, c_far_up, c_down):
    """0.5 * phi(r) * (c_down - c_up), the van Leer high-order face correction
    added to the upwind value c_up."""
    return 0.5 * _vanleer_phi(c_up, c_far_up, c_down) * (c_down - c_up)


def _horizontal_antidiffusive(grid, c):
    """Per-face anti-diffusive fluxes A_x (x-faces) and A_y (y-faces) and the
    metric weights needed to assemble their divergence.

    c : (nz, ny, nx) single-species concentration.
    Returns dict with, for each axis, the signed face flux A (high-order minus
    low-order upwind advective flux, positive in +x/+y), and the per-face
    terrain/metric weights (tr_L, tr_R) and inverse spacing.
    Boundary faces are excluded (interior faces only).
    """
    nz, ny, nx = grid.nz, grid.ny, grid.nx
    use_split = grid.has_split_fluxes
    out = {}

    # ---- x interior faces m = 0..nx-2 (between cell m and m+1) ----
    if nx > 1:
        cL = c[:, :, :-1]                      # cell m
        cR = c[:, :, 1:]                        # cell m+1
        # far-upwind neighbours (shifted), boundary-zeroed below
        cLL = np.empty_like(cL); cLL[:, :, 1:] = c[:, :, :-2]; cLL[:, :, 0] = cL[:, :, 0]
        cRR = np.empty_like(cR); cRR[:, :, :-1] = c[:, :, 2:]; cRR[:, :, -1] = cR[:, :, -1]

        phi_L = _vanleer_phi(cL, cLL, cR)            # eastward (L upwind)
        phi_R = _vanleer_phi(cR, cRR, cL)            # westward (R upwind)
        # boundary faces: no far-upwind cell -> drop to first order
        phi_L[:, :, 0] = 0.0                         # face m=0: L is boundary cell 0
        phi_R[:, :, -1] = 0.0                        # face m=nx-2: R is boundary cell nx-1

        if use_split:
            up = grid.UAvg_plus[:, :, 1:]            # eastward speed at face
            um = grid.UAvg_minus[:, :, 1:]           # westward speed at face
        else:
            u = grid.UAvg[:, :, 1:]
            up = np.maximum(u, 0.0)
            um = np.maximum(-u, 0.0)
        # A = beta*(c_R - c_L); beta = frozen reconstruction gain (bounded).
        beta_x = 0.5 * (up * phi_L + um * phi_R)
        A_x = beta_x * (cR - cL)                     # high - low (advective)

        dx_face = np.broadcast_to(grid.dx[None, :, None], (nz, ny, nx))[:, :, 1:]
        safe_dx = np.where(dx_face > 0, dx_face, 1.0)
        dP_L = grid.dP[:, :, :-1]; dP_R = grid.dP[:, :, 1:]
        dP_face = 0.5 * (dP_L + dP_R)
        valid = (dP_L > 0) & (dP_R > 0)
        tr_R = np.where(valid, dP_face / np.where(dP_R > 0, dP_R, 1.0), 1.0)
        tr_L = np.where(valid, dP_face / np.where(dP_L > 0, dP_L, 1.0), 1.0)
        out["x"] = dict(A=A_x, beta=beta_x, tr_L=tr_L, tr_R=tr_R,
                        inv=1.0 / safe_dx, valid=valid)

    # ---- y interior faces m = 0..ny-2 (between cell m=south and m+1=north) ----
    if ny > 1:
        cS = c[:, :-1, :]
        cN = c[:, 1:, :]
        cSS = np.empty_like(cS); cSS[:, 1:, :] = c[:, :-2, :]; cSS[:, 0, :] = cS[:, 0, :]
        cNN = np.empty_like(cN); cNN[:, :-1, :] = c[:, 2:, :]; cNN[:, -1, :] = cN[:, -1, :]

        phi_S = _vanleer_phi(cS, cSS, cN)            # northward (S upwind)
        phi_N = _vanleer_phi(cN, cNN, cS)            # southward (N upwind)
        phi_S[:, 0, :] = 0.0
        phi_N[:, -1, :] = 0.0

        dy = grid.dy
        if use_split:
            vp = grid.VAvg_plus[:, 1:, :]
            vm = grid.VAvg_minus[:, 1:, :]
        else:
            vv = grid.VAvg[:, 1:, :]
            vp = np.maximum(vv, 0.0)
            vm = np.maximum(-vv, 0.0)
        beta_y = 0.5 * (vp * phi_S + vm * phi_N)
        A_y = beta_y * (cN - cS)

        safe_dy = dy if dy > 0 else 1.0
        dx3 = np.broadcast_to(grid.dx[None, :, None], (nz, ny, nx))
        dx_S = dx3[:, :-1, :]; dx_N = dx3[:, 1:, :]
        dx_face = 0.5 * (dx_S + dx_N)
        dP_S = grid.dP[:, :-1, :]; dP_N = grid.dP[:, 1:, :]
        dP_face = 0.5 * (dP_S + dP_N)
        valid = (dP_S > 0) & (dP_N > 0)
        tr_N = np.where(valid, (dx_face / np.where(dx_N > 0, dx_N, 1.0))
                        * (dP_face / np.where(dP_N > 0, dP_N, 1.0)), 1.0)
        tr_S = np.where(valid, (dx_face / np.where(dx_S > 0, dx_S, 1.0))
                        * (dP_face / np.where(dP_S > 0, dP_S, 1.0)), 1.0)
        out["y"] = dict(A=A_y, beta=beta_y, tr_L=tr_S, tr_R=tr_N,
                        inv=safe_dy and (1.0 / safe_dy), valid=valid)
    return out


def _net_antidiffusive_into_cells(grid, faces, Cx=None, Cy=None):
    """Assemble the per-cell tendency d = -div(C⊙A) from face fluxes, with the
    convdiff terrain weighting. If Cx/Cy are None, use C=1 (unlimited)."""
    nz, ny, nx = grid.nz, grid.ny, grid.nx
    d = np.zeros((nz, ny, nx), dtype=np.float64)
    if "x" in faces:
        f = faces["x"]
        C = 1.0 if Cx is None else Cx
        flux = C * f["A"] * f["inv"]                 # per-face signed (+x)
        # east cell (R = m+1) gains tr_R*flux; west cell (L = m) loses tr_L*flux
        d[:, :, 1:] += np.where(f["valid"], f["tr_R"] * flux, 0.0)
        d[:, :, :-1] -= np.where(f["valid"], f["tr_L"] * flux, 0.0)
    if "y" in faces:
        f = faces["y"]
        C = 1.0 if Cy is None else Cy
        flux = C * f["A"] * f["inv"]
        d[:, 1:, :] += np.where(f["valid"], f["tr_R"] * flux, 0.0)
        d[:, :-1, :] -= np.where(f["valid"], f["tr_L"] * flux, 0.0)
    return d


def _zalesak_2d(grid, c, faces, dt):
    """Zalesak coefficients C_x, C_y in [0,1] so c + dt*d_AD makes no new
    extrema relative to c and its 4-neighbourhood. Steady-state form: bounds
    taken from the current iterate c (= c_td at convergence)."""
    nz, ny, nx = grid.nz, grid.ny, grid.nx
    # neighbour max/min (4-neighbourhood, edge-clamped)
    stack = [c]
    stack.append(np.concatenate([c[:, :, :1], c[:, :, :-1]], axis=2))   # west
    stack.append(np.concatenate([c[:, :, 1:], c[:, :, -1:]], axis=2))   # east
    stack.append(np.concatenate([c[:, :1, :], c[:, :-1, :]], axis=1))   # south
    stack.append(np.concatenate([c[:, 1:, :], c[:, -1:, :]], axis=1))   # north
    s = np.stack(stack, axis=0)
    c_max = s.max(axis=0)
    c_min = s.min(axis=0)

    # net anti-diffusive flux into each cell, split by sign, using C=1
    # contributions per face: into east cell +tr_R*flux, into west cell -tr_L*flux
    P_plus = np.zeros((nz, ny, nx)); P_minus = np.zeros((nz, ny, nx))

    def accum(axis_key, lo_slc, hi_slc, axis):
        if axis_key not in faces:
            return
        f = faces[axis_key]
        flux = f["A"] * f["inv"]
        into_hi = np.where(f["valid"], f["tr_R"] * flux, 0.0)   # into R/N cell
        into_lo = np.where(f["valid"], -f["tr_L"] * flux, 0.0)  # into L/S cell
        # accumulate onto cell grids
        sl_hi = [slice(None)] * 3; sl_hi[axis] = hi_slc
        sl_lo = [slice(None)] * 3; sl_lo[axis] = lo_slc
        P_plus[tuple(sl_hi)] += np.maximum(into_hi, 0.0)
        P_minus[tuple(sl_hi)] += np.maximum(-into_hi, 0.0)
        P_plus[tuple(sl_lo)] += np.maximum(into_lo, 0.0)
        P_minus[tuple(sl_lo)] += np.maximum(-into_lo, 0.0)

    accum("x", slice(0, nx - 1), slice(1, nx), axis=2)
    accum("y", slice(0, ny - 1), slice(1, ny), axis=1)

    Q_plus = (c_max - c) / dt
    Q_minus = (c - c_min) / dt
    R_plus = np.where(P_plus > 0, np.minimum(1.0, Q_plus / np.where(P_plus > 0, P_plus, 1.0)), 0.0)
    R_minus = np.where(P_minus > 0, np.minimum(1.0, Q_minus / np.where(P_minus > 0, P_minus, 1.0)), 0.0)

    def face_C(axis_key, lo_slc, hi_slc, axis):
        if axis_key not in faces:
            return None
        f = faces[axis_key]
        flux = f["A"] * f["inv"]
        into_hi = np.where(f["valid"], f["tr_R"] * flux, 0.0)
        sl_hi = [slice(None)] * 3; sl_hi[axis] = hi_slc
        sl_lo = [slice(None)] * 3; sl_lo[axis] = lo_slc
        Rp_hi = R_plus[tuple(sl_hi)]; Rm_hi = R_minus[tuple(sl_hi)]
        Rp_lo = R_plus[tuple(sl_lo)]; Rm_lo = R_minus[tuple(sl_lo)]
        # flux>0 (into hi): limited by room-to-rise in hi and room-to-fall in lo
        C_pos = np.minimum(Rp_hi, Rm_lo)
        C_neg = np.minimum(Rm_hi, Rp_lo)
        C = np.where(into_hi >= 0.0, C_pos, C_neg)
        return np.clip(C, 0.0, 1.0)

    Cx = face_C("x", slice(0, nx - 1), slice(1, nx), axis=2)
    Cy = face_C("y", slice(0, ny - 1), slice(1, ny), axis=1)
    return Cx, Cy


def compute_horizontal_fct_source(grid, indexer, c_flat, dt, limited=True):
    """Deferred-correction FCT source d_AD = -div(C⊙A) for one species.

    c_flat : (N,) concentration in indexer order.
    dt     : the step used for the Zalesak bound (the orbit DTAU).
    Returns d_AD as a flat (N,) tendency (units of -L c). Gated entirely by the
    caller; production never calls this. With limited=False, returns the
    unlimited anti-diffusive source (for diagnostics only — not monotone).
    """
    nz, ny, nx = grid.nz, grid.ny, grid.nx
    c = np.asarray(c_flat, dtype=np.float64).reshape(nz, ny, nx)
    faces = _horizontal_antidiffusive(grid, c)
    if limited:
        Cx, Cy = _zalesak_2d(grid, c, faces, dt)
    else:
        Cx = Cy = None
    d = _net_antidiffusive_into_cells(grid, faces, Cx, Cy)
    return d.reshape(-1)


def assemble_fct_linear_operator(grid, indexer, c_baseline_flat, dt):
    """Frozen-coefficient FCT tangent-linear operator L_AD (Phase 4).

    Freezes BOTH the Zalesak coefficients C and the van Leer reconstruction at
    the baseline state c_baseline, giving a FIXED LINEAR operator. The marginal/
    adjoint SR tiers use L_low + L_AD: a wider-physics but same-sparsity (5-point)
    operator that factors once. (The next-nearest c_LL dependence lives only in
    φ(r), which is frozen, so L_AD couples nearest neighbours only — same stencil
    as L_low.)

    Construction: the limited anti-diffusive face flux is C·A. At the baseline,
    A is a linear-in-c reconstruction increment proportional to the face gradient
    Δc, so the frozen per-face GAIN is

        g_face = C_baseline · A_baseline / Δc_baseline   (0 where Δc≈0; A→0 there too)

    and the linear flux is g_face·(c_R − c_L). Distributing −div with the same
    convdiff terrain/metric weights (tr_L, tr_R, 1/dx) gives L_AD such that
    L_AD @ c_baseline == −compute_horizontal_fct_source(..., c_baseline) by
    construction (validated in scripts/diag_verify_fct_linop.py).

    NOTE: L_AD is intentionally NOT an M-matrix (anti-diffusion has negative
    diagonal / positive off-diagonal); L_low + L_AD is the meaningful operator.

    Returns
    -------
    L_AD : csc_matrix (N, N)
    """
    nz, ny, nx = grid.nz, grid.ny, grid.nx
    N = indexer.N
    c = np.asarray(c_baseline_flat, dtype=np.float64).reshape(nz, ny, nx)
    n3d = np.arange(N, dtype=np.int64).reshape(nz, ny, nx)
    faces = _horizontal_antidiffusive(grid, c)
    Cx, Cy = _zalesak_2d(grid, c, faces, dt)

    rows, cols, vals = [], [], []

    if "x" in faces:
        f = faces["x"]
        # Frozen gain g = C * beta (bounded: beta = 0.5(u+ phi_L + u- phi_R)).
        # Using beta directly (not A/Δc) avoids huge coefficients + catastrophic
        # cancellation in the matvec at small-gradient faces.
        g = Cx * f["beta"]
        w = np.where(f["valid"], g * f["inv"], 0.0)      # gain * 1/dx
        nL = n3d[:, :, :-1]; nR = n3d[:, :, 1:]
        wR = (f["tr_R"] * w); wL = (f["tr_L"] * w)
        # L_AD c = -d_AD. d_R = +wR(c_R-c_L), d_L = -wL(c_R-c_L), so
        # (L_AD c)_R = -wR(c_R-c_L): row R -> (R,R)=-wR, (R,L)=+wR
        # (L_AD c)_L = +wL(c_R-c_L): row L -> (L,R)=+wL, (L,L)=-wL
        rows.append(nR.ravel()); cols.append(nR.ravel()); vals.append((-wR).ravel())
        rows.append(nR.ravel()); cols.append(nL.ravel()); vals.append((wR).ravel())
        rows.append(nL.ravel()); cols.append(nR.ravel()); vals.append((wL).ravel())
        rows.append(nL.ravel()); cols.append(nL.ravel()); vals.append((-wL).ravel())

    if "y" in faces:
        f = faces["y"]
        g = Cy * f["beta"]
        w = np.where(f["valid"], g * f["inv"], 0.0)
        nS = n3d[:, :-1, :]; nN = n3d[:, 1:, :]
        wN = (f["tr_R"] * w); wS = (f["tr_L"] * w)
        # N = R (north), S = L (south): (N,N)=-wN, (N,S)=+wN, (S,N)=+wS, (S,S)=-wS
        rows.append(nN.ravel()); cols.append(nN.ravel()); vals.append((-wN).ravel())
        rows.append(nN.ravel()); cols.append(nS.ravel()); vals.append((wN).ravel())
        rows.append(nS.ravel()); cols.append(nN.ravel()); vals.append((wS).ravel())
        rows.append(nS.ravel()); cols.append(nS.ravel()); vals.append((-wS).ravel())

    if not rows:
        return sp.csc_matrix((N, N))
    return sp.csc_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(N, N),
    )
