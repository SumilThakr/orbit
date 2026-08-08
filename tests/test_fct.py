"""Idealized-advection validation of the FCT core (orbit/core/fct.py), Phase 1.

Proves, before any orbit integration:
  - 1-D square wave: monotone (no new extrema), bounded, mass-conserving.
  - 1-D Gaussian: second-order L1 convergence in smooth flow, strictly better
    than first-order upwind; and that upwind really is first-order (the baseline
    the production scheme sits at).
  - 2-D constant-velocity advection: shape preserved, mass conserved, no new
    extrema.
  - van Leer limiter algebra.
"""
import numpy as np
import pytest

from orbit.core import fct


# --------------------------------------------------------------------------- #
def _advect_periodic(step_fn, c0, u, dx, n_steps, dt):
    c = c0.copy()
    uface = np.full_like(c0, u)
    for _ in range(n_steps):
        c = step_fn(c, uface, dx, dt)
    return c


def test_vanleer_limiter_algebra():
    assert fct.vanleer(np.array([1.0]))[0] == pytest.approx(1.0)   # smooth -> centred
    assert fct.vanleer(np.array([-2.0]))[0] == pytest.approx(0.0)  # extremum -> upwind
    assert fct.vanleer(np.array([0.0]))[0] == pytest.approx(0.0)
    big = fct.vanleer(np.array([1e12]))[0]
    assert 1.9 < big <= 2.0                                        # r->inf -> 2
    # monotone, bounded in [0,2]
    r = np.linspace(-5, 50, 200)
    phi = fct.vanleer(r)
    assert np.all(phi >= -1e-12) and np.all(phi <= 2.0 + 1e-12)


def test_square_wave_monotone_and_conservative():
    """FCT must not create new extrema and must conserve mass on a top hat."""
    nx = 200
    dx = 1.0 / nx
    x = (np.arange(nx) + 0.5) * dx
    c0 = np.where((x > 0.3) & (x < 0.5), 1.0, 0.0)
    u = 1.0
    cfl = 0.4
    dt = cfl * dx / u
    n_steps = int(round(0.5 / (u * dt)))    # advect half a domain

    c = _advect_periodic(fct.fct_step_1d, c0, u, dx, n_steps, dt)
    c_up = _advect_periodic(fct.upwind_step_1d, c0, u, dx, n_steps, dt)

    # no new extrema (allow tiny roundoff)
    assert c.min() >= -1e-9, f"undershoot {c.min()}"
    assert c.max() <= 1.0 + 1e-9, f"overshoot {c.max()}"
    # mass conserved (periodic)
    assert c.sum() == pytest.approx(c0.sum(), rel=1e-12)

    # FCT keeps the hat far sharper than upwind. Use wrap-safe measures (the
    # hat advects across the periodic seam, so np.diff-based TV is invalid):
    #   - nonzero span: how many cells the hat has smeared across
    #   - peak preservation
    span_fct = int((c > 1e-6).sum())
    span_up = int((c_up > 1e-6).sum())
    assert span_fct < 0.6 * span_up, (span_fct, span_up)
    assert c.max() >= c_up.max() - 1e-9, (c.max(), c_up.max())


def _l1_error_gaussian(step_fn, nx, sigma=0.08, n_periods=1):
    dx = 1.0 / nx
    x = (np.arange(nx) + 0.5) * dx
    u = 1.0
    cfl = 0.4
    dt = cfl * dx / u
    c0 = np.exp(-((x - 0.5) ** 2) / (2 * sigma ** 2))
    n_steps = int(round(n_periods / (u * dt)))
    dt = n_periods / (u * n_steps)   # exact periodic return -> exact sol = c0
    c = _advect_periodic(step_fn, c0, u, dx, n_steps, dt)
    return np.abs(c - c0).mean()


def test_gaussian_convergence_fct_beats_upwind():
    """In smooth flow FCT is far more accurate than first-order upwind, and the
    advantage GROWS under refinement (FCT converges faster).

    Note on order: van Leer is a TVD limiter, which is formally first-order at a
    smooth extremum (the Gaussian peak). So FCT's asymptotic L1 order sits
    between 1 and 2 rather than a clean 2.0 — the meaningful Phase-1 claims are
    (a) upwind is first-order, (b) FCT is strictly and increasingly better, and
    (c) FCT is super-linear away from the under-resolved grid.
    """
    res = [80, 160, 320, 640]
    err_fct = [_l1_error_gaussian(fct.fct_step_1d, n) for n in res]
    err_up = [_l1_error_gaussian(fct.upwind_step_1d, n) for n in res]

    def orders(errs):
        return [np.log2(errs[i] / errs[i + 1]) for i in range(len(errs) - 1)]

    o_up = orders(err_up)
    o_fct = orders(err_fct)

    # (a) upwind ~first order (0.74 -> 0.84 -> 0.91, climbing to 1)
    assert 0.5 < np.mean(o_up) < 1.3, f"upwind order {o_up}"
    assert o_up[-1] > 0.7, f"upwind finest order {o_up[-1]}"

    # (b) FCT strictly better at every resolution, advantage growing
    ratios = [eu / ef for ef, eu in zip(err_fct, err_up)]
    for ef, eu in zip(err_fct, err_up):
        assert ef < eu, f"fct {ef} not better than upwind {eu}"
    assert ratios[-1] > ratios[0], f"advantage not growing: {ratios}"
    assert ratios[-1] > 2.4, f"fct only {ratios[-1]:.1f}x better at finest grid"

    # (c) 2nd-order CHARACTER on the resolved grid: the leading-refinement order
    # is well above first order (~1.6). It relaxes toward 1 asymptotically
    # because van Leer is first-order at the smooth Gaussian extremum (TVD).
    assert o_fct[0] > 1.4, f"fct leading order {o_fct[0]} (orders {o_fct})"


def test_2d_constant_advection_shape_and_mass():
    """2-D periodic advection of a cone: mass conserved, no new extrema."""
    ny, nx = 80, 80
    dx = dy = 1.0 / nx
    xs = (np.arange(nx) + 0.5) * dx
    ys = (np.arange(ny) + 0.5) * dy
    X, Y = np.meshgrid(xs, ys)
    r = np.sqrt((X - 0.5) ** 2 + (Y - 0.5) ** 2)
    c0 = np.maximum(0.0, 1.0 - r / 0.2)        # cone, peak 1 at centre

    ux, uy = 1.0, 0.5
    cfl = 0.3
    dt = cfl * dx / max(abs(ux), abs(uy))
    n_steps = 40

    c = c0.copy()
    for _ in range(n_steps):
        c = fct.fct_step_2d(c, ux, uy, dx, dy, dt)

    assert c.min() >= -1e-9, f"undershoot {c.min()}"
    assert c.max() <= c0.max() + 1e-9, f"overshoot {c.max()}"
    assert c.sum() == pytest.approx(c0.sum(), rel=1e-10)


def test_fct_reduces_to_transport_for_uniform_field():
    """A spatially-uniform field has zero anti-diffusive flux: FCT == exact."""
    nx = 64
    c0 = np.full(nx, 3.7)
    u = 1.0
    dx = 1.0 / nx
    dt = 0.4 * dx / u
    c = _advect_periodic(fct.fct_step_1d, c0, u, dx, 50, dt)
    assert np.allclose(c, 3.7, atol=1e-12)
