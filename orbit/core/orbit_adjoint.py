"""Periodic-orbit adjoint solver — the mirror image of orbit.py.

Solves the per-species adjoint periodic-orbit equation

    (I − M^T) λ_0 = s_adj

where M = P_{N_BINS-1} · … · P_1 · P_0 is the forward monodromy
(same per-bin propagators P_τ = (I + DTAU · L_τ)^{-1} the forward solver
uses) and s_adj is the backward accumulation of per-bin receptors
R_τ = G_{(τ-1) mod N_BINS}^T S_{(τ-1) mod N_BINS}.

Full recursion:

    λ_τ = R_τ + P_τ^T · λ_{τ+1}             (mod-N_BINS periodic)

so the per-bin emission gradient (no chemistry coupling) is

    ∂J/∂e_τ = DTAU · P_τ^T λ_{τ+1}
            = DTAU · (λ_τ − R_τ)               ← cheap to read off.

This file handles the SINGLE-SPECIES adjoint with no off-diagonal
chemistry coupling (the K^T propagation is a follow-up; needed once
NOx/SO2/VBS adjoint chains are wired into the driver). For each
species' adjoint we reuse the same forward LU list `lu_list_8` and call
`.solve(b, trans='T')` — no re-factorisation cost.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse.linalg as spla

from orbit.core.orbit import N_BINS, DTAU


def apply_monodromy_adjoint(v: np.ndarray, lu_list_8) -> np.ndarray:
    """Apply M^T = P_0^T P_1^T … P_{N_BINS-1}^T to v.

    The forward `apply_monodromy` builds M = P_{N_BINS-1} … P_0 by
    iterating τ = 0 → N_BINS-1 and calling lu_list_8[τ].solve(x). The
    transpose flips operator order: iterate τ = N_BINS-1 → 0 and call
    .solve(x, trans='T').
    """
    x = v.copy()
    for tau in range(N_BINS - 1, -1, -1):
        x = lu_list_8[tau].solve(x, trans="T")
    return x


def compute_s_adjoint(lu_list_8, R_list) -> np.ndarray:
    """Backward-iterate from μ_{N_BINS} = 0 to get s_adj.

    s_adj = R_0 + P_0^T R_1 + P_0^T P_1^T R_2 + …
                + (P_0^T … P_{N_BINS-2}^T) R_{N_BINS-1}.

    Parameters
    ----------
    lu_list_8 : list of N_BINS LU factorizations (each with .solve(b, trans))
    R_list : list of N_BINS per-bin receptor vectors, shape (N,)
        R_τ is paired with λ_τ in the recursion. The mapping from a
        deaths receptor "at end of bin τ-1" is handled by the caller.
    """
    N = lu_list_8[0].shape[0]
    x = np.zeros(N, dtype=np.float64)
    for tau in range(N_BINS - 1, -1, -1):
        x = lu_list_8[tau].solve(x, trans="T") + R_list[tau]
    return x


def solve_orbit_adjoint_one_species(
    lu_list_8, R_list, *,
    lam_warm=None, tol=1e-6, maxiter=200,
):
    """GMRES on (I − M^T) λ_0 = s_adj, then unroll the adjoint orbit.

    Parameters
    ----------
    lu_list_8 : 8 LU factorizations (forward); used via .solve(b, trans='T').
    R_list : 8 receptor vectors (N,), one per bin (R_τ in the recursion).
    lam_warm : optional warm-start for λ_0.
    tol, maxiter : GMRES tolerance and iteration cap.

    Returns
    -------
    adjoint : list of N_BINS+1 ndarrays
        [λ_0, λ_1, …, λ_{N_BINS-1}, λ_{N_BINS}=λ_0]
    info : dict
        gmres_iters, residual, periodicity.
    """
    N = lu_list_8[0].shape[0]
    s_adj = compute_s_adjoint(lu_list_8, R_list)
    s_norm = np.linalg.norm(s_adj)

    if s_norm < 1e-30:
        zero = np.zeros(N, dtype=np.float64)
        return [zero.copy() for _ in range(N_BINS + 1)], dict(
            gmres_iters=0, residual=0.0, periodicity=0.0,
        )

    def matvec(v):
        return v - apply_monodromy_adjoint(v, lu_list_8)

    op = spla.LinearOperator((N, N), matvec=matvec, dtype=np.float64)

    x0 = lam_warm if lam_warm is not None else s_adj.copy()
    iter_count = [0]

    def _counter(_rk):
        iter_count[0] += 1

    lam_0, info_code = spla.gmres(
        op, s_adj, x0=x0,
        rtol=1e-30, atol=tol * s_norm,
        restart=30, maxiter=maxiter,
        callback=_counter, callback_type="pr_norm",
    )
    if info_code != 0:
        print(f"    WARNING: adjoint GMRES did not converge "
              f"(info={info_code}, iters={iter_count[0]})")

    # Unroll the adjoint orbit by backward-recursion from λ_8 = λ_0.
    adjoint = [None] * (N_BINS + 1)
    adjoint[0] = lam_0.copy()
    adjoint[N_BINS] = lam_0.copy()
    x = lam_0.copy()
    for tau in range(N_BINS - 1, 0, -1):
        x = lu_list_8[tau].solve(x, trans="T") + R_list[tau]
        adjoint[tau] = x.copy()

    # Residual + periodicity diagnostics.
    residual_vec = op.matvec(lam_0) - s_adj
    residual = np.linalg.norm(residual_vec) / s_norm
    # Periodicity check: re-derive λ_0 from λ_1.
    lam_0_check = lu_list_8[0].solve(adjoint[1], trans="T") + R_list[0]
    lam_0_norm = np.linalg.norm(lam_0)
    periodicity = (
        np.linalg.norm(lam_0_check - lam_0) / (lam_0_norm + 1e-30)
    )

    return adjoint, dict(
        gmres_iters=iter_count[0],
        residual=float(residual),
        periodicity=float(periodicity),
    )


def gradient_per_emission_bin(adjoint, R_list):
    """Convert adjoint orbit to ∂J/∂e_τ via the identity
    DTAU · P_τ^T λ_{τ+1} = DTAU · (λ_τ − R_τ).

    Returns a list of N_BINS arrays of shape (N,), one per bin.
    """
    return [DTAU * (adjoint[tau] - R_list[tau]) for tau in range(N_BINS)]
