"""Volatility Basis Set (VBS) partitioning closure for SoA.

For each VBS bin i (C* = 0.1, 1, 10, 100, 1000 µg/m³), the gas/particle
equilibrium is governed by Pankow (1994):

    F_p,i = 1 / (1 + C*_i / M_OA)

where M_OA is total organic aerosol mass acting as the absorbing medium:

    M_OA = Σ_i F_p,i × C_i

This is implicit in M_OA (it appears on both sides), so we solve the
fixed-point per cell via vectorized Newton iteration. The function is
strictly monotonic + contractive in M_OA, so 3–5 iterations converge to
1e-6 relative tolerance.

This module provides:
- pankow_F_p(M_OA, C_star)         — per-cell partitioning fraction
- solve_M_OA(C_bins, ...)          — vectorized Newton fixed-point
- update_vbs_partitioning(grid, c) — closure step that writes
                                     grid.M_OA + grid.F_p_vbs

POA participates as absorbing mass:

    M_OA = C_POA + sum_i F_p,i x C_i

POA is a transported species (IDX_POA) as of 2026-08-02; before that it
was lumped into PrimaryPM25 with BC, dust and sea salt and the term was
omitted. The omission was documented as "slight" and was not: measured
M_OA at polluted IGP cells was 2.1 ug/m3 against a POA of ~14.5, so F_p
collapsed and effective anthropogenic yield fell from a nominal 0.417 to
0.0021.

Adding a non-negative constant leaves the fixed point monotone and
contractive, so the Newton iteration is unchanged apart from the term.

References:
- Pankow, J. F. (1994). An absorption model of gas/particle partitioning
  of organic compounds in the atmosphere. Atmos. Environ. 28, 185–188.
- Donahue, N. M. et al. (2006). Coupled partitioning, dilution, and
  chemical aging of semivolatile organics. ES&T 40, 2635–2643.
- Tsimpidi, A. P. et al. (2010). Evaluation of the volatility basis-set
  approach for the simulation of organic aerosol formation in the Mexico
  City metropolitan area. ACP 10, 525–546.
"""
from __future__ import annotations

import os
import numpy as np

from orbit.core.deposition import (
    IDX_POA,
    C_STAR_VALS,
    IDX_VBS_BINS,
    N_VBS_BINS,
)

# Default closure parameters — overridable via env vars.
# maxiter=15 is a Picard-iteration safety margin; each iter is just one
# Pankow evaluation per bin per cell (~µs/cell), so 15 is cheap. Phase 6
# adds Anderson acceleration on top, dropping to ~5 iters in production.
_DEFAULT_TOL = 1.0e-3
_DEFAULT_MAXITER = 15
_M_OA_BOOTSTRAP = 10.0   # µg/m³, mid-range Indian background


def _force_F_p() -> float | None:
    """If ORBIT_VBS_FORCE_FP is set, return the forced value (typically 1.0
    for the Variant-A regression test). Returns None when Pankow is active.
    """
    raw = os.environ.get("ORBIT_VBS_FORCE_FP", "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _poa_absorb() -> bool:
    """Whether POA counts as absorbing mass. ORBIT_VBS_POA_ABSORB=0 disables
    it, reproducing the pre-2026-08-02 behaviour for the regression gate."""
    return os.environ.get("ORBIT_VBS_POA_ABSORB", "1").strip() not in ("0", "false", "False")


def pankow_F_p(M_OA: np.ndarray, C_star: float) -> np.ndarray:
    """Pankow gas/particle equilibrium fraction in particle phase.

    F_p = 1 / (1 + C* / M_OA),  clipped at M_OA ≥ 1e-6 to avoid division
    by zero in cells with no organic mass.

    Parameters
    ----------
    M_OA : ndarray of any shape
        Total organic aerosol mass (µg/m³).
    C_star : float
        Effective saturation concentration of the bin (µg/m³).

    Returns
    -------
    F_p : ndarray, same shape as M_OA
        Fraction of bin mass in particle phase, in [0, 1].
    """
    M_safe = np.maximum(M_OA, 1.0e-6)
    return 1.0 / (1.0 + C_star / M_safe)


def solve_M_OA(
    C_bins: np.ndarray,
    tol: float = _DEFAULT_TOL,
    maxiter: int = _DEFAULT_MAXITER,
    M_OA_init: np.ndarray | float | None = None,
    C_POA: np.ndarray | float | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Solve the M_OA = C_POA + Σ F_p,i × C_i fixed-point per cell.

    Parameters
    ----------
    C_bins : ndarray, shape (N_VBS_BINS, ...)
        Per-bin total mass (gas + particle) in µg/m³, for each cell.
        First axis is VBS bins in low-to-high C* order, matching
        IDX_VBS_BINS / C_STAR_VALS.
    tol : float
        Relative convergence tolerance on max |ΔM_OA| / M_OA.
    maxiter : int
        Maximum Newton iterations.
    M_OA_init : ndarray or float, optional
        Initial M_OA guess. Defaults to _M_OA_BOOTSTRAP (10 µg/m³)
        broadcast to the trailing shape.
    C_POA : ndarray or float, optional
        Primary organic aerosol acting as absorbing mass, µg/m³, same
        trailing shape as a bin. Non-negative. Defaults to zero, which
        reproduces the pre-2026-08-02 behaviour exactly.

    Returns
    -------
    M_OA : ndarray, shape = C_bins.shape[1:]
        Converged organic aerosol mass per cell (µg/m³).
    F_p : ndarray, shape (N_VBS_BINS, ...)
        Per-bin particle fraction at convergence.
    info : dict
        Diagnostics: iters_run, max_rel_change_history, converged_mask.
    """
    if C_bins.shape[0] != N_VBS_BINS:
        raise ValueError(
            f"C_bins.shape[0] must be {N_VBS_BINS}, got {C_bins.shape[0]}"
        )

    cell_shape = C_bins.shape[1:]
    if C_POA is None:
        poa = np.zeros(cell_shape, dtype=np.float64)
    elif np.isscalar(C_POA):
        poa = np.full(cell_shape, float(C_POA), dtype=np.float64)
    else:
        poa = np.maximum(np.asarray(C_POA, dtype=np.float64), 0.0)
        if poa.shape != cell_shape:
            raise ValueError(
                f"C_POA shape {poa.shape} must match cell shape {cell_shape}"
            )
    if M_OA_init is None:
        M_OA = np.full(cell_shape, _M_OA_BOOTSTRAP, dtype=np.float64)
    elif np.isscalar(M_OA_init):
        M_OA = np.full(cell_shape, float(M_OA_init), dtype=np.float64)
    else:
        M_OA = np.asarray(M_OA_init, dtype=np.float64).copy()
        if M_OA.shape != cell_shape:
            raise ValueError(
                f"M_OA_init shape {M_OA.shape} must match cell shape {cell_shape}"
            )

    # If ORBIT_VBS_FORCE_FP is set (regression / pure-particle limit),
    # short-circuit: F_p = forced value uniformly, M_OA = sum of all C.
    forced = _force_F_p()
    if forced is not None:
        F_p = np.full((N_VBS_BINS,) + cell_shape, forced, dtype=np.float64)
        M_OA = poa + (F_p * C_bins).sum(axis=0)
        return M_OA, F_p, {
            "iters_run": 0, "max_rel_change_history": [],
            "converged_mask": np.ones(cell_shape, dtype=bool),
            "forced_F_p": forced,
        }

    history = []
    for it in range(maxiter):
        # Compute F_p for current M_OA across all bins.
        F_p = np.stack(
            [pankow_F_p(M_OA, C_star) for C_star in C_STAR_VALS], axis=0,
        )
        # New M_OA from the partition step.
        M_OA_new = poa + (F_p * C_bins).sum(axis=0)
        # Relative change for convergence check.
        denom = np.maximum(M_OA_new, 1.0e-6)
        rel_change = np.abs(M_OA_new - M_OA) / denom
        history.append(float(rel_change.max()))
        M_OA = M_OA_new
        if history[-1] < tol:
            break

    # Final F_p at converged M_OA.
    F_p = np.stack(
        [pankow_F_p(M_OA, C_star) for C_star in C_STAR_VALS], axis=0,
    )
    converged_mask = (rel_change < tol)
    return M_OA, F_p, {
        "iters_run": it + 1,
        "max_rel_change_history": history,
        "converged_mask": converged_mask,
        "forced_F_p": None,
    }


def update_vbs_partitioning(
    grid,
    c_mean: np.ndarray,
    indexer,
    tol: float | None = None,
    maxiter: int | None = None,
) -> dict:
    """Closure step: write grid.M_OA_3d + grid.F_p_vbs from c_mean.

    Mirrors update_grid_partitioning() in orbit/core/dcomp_isorropia.py
    in spirit: reads the species concentrations the operator has just
    produced, runs the implicit closure, writes the converged
    partitioning fields into the grid for the next operator rebuild.

    Parameters
    ----------
    grid : GridData
        Mutated in-place. Adds:
          grid.M_OA_3d        (nz, ny, nx) µg/m³
          grid.M_OA_surface   (ny, nx) µg/m³
          grid.F_p_vbs        (5, nz, ny, nx) particle fraction per bin
          grid.F_p_vbs_surface (5, ny, nx)
    c_mean : ndarray
        Either flat shape (N_SPECIES * N,) or 2D (N_SPECIES, N).
    indexer : CellIndexer
        For nz/ny/nx and N.
    tol, maxiter : optional override for closure parameters.

    Returns
    -------
    dict : {iters_run, max_rel_change, converged_fraction, M_OA_p90,
            forced_F_p}
    """
    nz, ny, nx = grid.nz, grid.ny, grid.nx
    N = indexer.N

    # Reshape c_mean to (N_SPECIES, nz, ny, nx).
    c_arr = np.asarray(c_mean)
    if c_arr.ndim == 1:
        c_arr = c_arr.reshape(-1, N)
    elif c_arr.ndim != 2:
        raise ValueError(f"c_mean must be 1D or 2D, got ndim={c_arr.ndim}")

    # Pull per-bin concentrations in low-to-high C* order.
    C_bins = np.stack(
        [c_arr[idx].reshape(nz, ny, nx) for idx in IDX_VBS_BINS], axis=0,
    )
    # Clip negatives that may arise from numerical artifacts.
    C_bins = np.maximum(C_bins, 0.0)

    tol = _DEFAULT_TOL if tol is None else tol
    maxiter = _DEFAULT_MAXITER if maxiter is None else maxiter

    # Initial M_OA: reuse converged value from previous outer iteration
    # if present (warm start), else bootstrap.
    M_OA_init = getattr(grid, "M_OA_3d", None)
    if M_OA_init is None or M_OA_init.size != nz * ny * nx:
        M_OA_init = None  # solve_M_OA will bootstrap

    # POA absorbing mass. IDX_POA may be absent from a truncated c_mean
    # (older outputs, or tests built on the 13-species layout) — treat that
    # as zero, which reproduces the pre-POA behaviour exactly.
    C_POA = None
    if _poa_absorb() and c_arr.shape[0] > IDX_POA:
        C_POA = np.maximum(c_arr[IDX_POA].reshape(nz, ny, nx), 0.0)

    M_OA, F_p, info = solve_M_OA(
        C_bins, tol=tol, maxiter=maxiter, M_OA_init=M_OA_init, C_POA=C_POA,
    )

    # Write back to grid.
    grid.M_OA_3d = M_OA
    grid.M_OA_surface = M_OA[0]
    grid.F_p_vbs = F_p                # (5, nz, ny, nx)
    grid.F_p_vbs_surface = F_p[:, 0]  # (5, ny, nx)

    return {
        "iters_run": info["iters_run"],
        "max_rel_change":
            info["max_rel_change_history"][-1]
            if info["max_rel_change_history"] else 0.0,
        "converged_fraction": float(info["converged_mask"].mean()),
        "M_OA_p90": float(np.percentile(M_OA, 90)),
        "M_OA_surface_mean": float(M_OA[0].mean()),
        "forced_F_p": info["forced_F_p"],
    }
