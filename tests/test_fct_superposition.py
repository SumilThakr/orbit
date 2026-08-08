"""FCT is nonlinear: a species split is not additive unless the shapes match.

This is not a defect — Godunov's theorem says any monotone scheme above first
order must be nonlinear, and the van Leer limiter phi(r) = (r+|r|)/(1+|r|)
takes a RATIO of local gradients while the Zalesak coefficients clip on local
extrema.  Both are scale-invariant but shape-sensitive.

It matters because it sets a hard floor on every ORBIT diagnostic that assumes
PM2.5 decomposes linearly into components.  The Phase 3 POA gate was written
expecting `control PM2.5 == gate (PM2.5 + POA)` to ~1e-7; it closed at 6.8e-03
mean per cell instead, and this is why.
"""

import numpy as np

from orbit.core.fct import fct_step_1d


def _setup(n=96):
    """A front plus a smooth hump, so the limiter is genuinely active."""
    x = np.linspace(0.0, 1.0, n, endpoint=False)
    u_face = np.full(n, 0.6)
    dx, dt = 1.0 / n, 0.4 / n / 0.6
    return x, u_face, dx, dt


def _step(c, u_face, dx, dt):
    return fct_step_1d(c, u_face, dx, dt)


def test_proportional_split_is_exact():
    """Same shape, different amplitude: the limiter sees identical ratios."""
    x, u, dx, dt = _setup()
    c = np.where((x > 0.2) & (x < 0.4), 1.0, 0.1) + 0.5 * np.exp(-((x - 0.7) / 0.05) ** 2)
    for f in (0.5, 0.3105, 0.01):
        a, b = f * c, (1.0 - f) * c
        resid = _step(a + b, u, dx, dt) - (_step(a, u, dx, dt) + _step(b, u, dx, dt))
        scale = np.abs(_step(c, u, dx, dt)).max()
        assert np.abs(resid).max() / scale < 1e-12, f"f={f}"


def test_structural_split_is_not_additive():
    """Different shapes: superposition fails at the percent level.

    POA and the remaining primary PM2.5 are exactly this case — the POA share
    ranges from 0.005 to 0.811 across January cells, so the two fields have
    genuinely different structure.
    """
    x, u, dx, dt = _setup()
    a = np.where((x > 0.2) & (x < 0.4), 1.0, 0.02)          # a front
    b = 0.8 * np.exp(-((x - 0.32) / 0.06) ** 2) + 0.02      # a co-located hump
    resid = _step(a + b, u, dx, dt) - (_step(a, u, dx, dt) + _step(b, u, dx, dt))
    rel = np.abs(resid).max() / np.abs(_step(a + b, u, dx, dt)).max()
    assert rel > 1e-3, f"expected non-additivity, got {rel:.2e}"


def test_non_additivity_conserves_mass():
    """The residual redistributes mass; it does not create or destroy it.

    This is why the POA gate still closed to 5e-04 in domain burden while
    disagreeing by 7e-03 per cell.
    """
    x, u, dx, dt = _setup()
    a = np.where((x > 0.2) & (x < 0.4), 1.0, 0.02)
    b = 0.8 * np.exp(-((x - 0.32) / 0.06) ** 2) + 0.02
    resid = _step(a + b, u, dx, dt) - (_step(a, u, dx, dt) + _step(b, u, dx, dt))
    assert abs(resid.sum()) / np.abs(a + b).sum() < 1e-12


def test_uniform_field_split_is_exact():
    """No gradients, no limiter response, so any split is trivially additive."""
    x, u, dx, dt = _setup()
    c = np.full_like(x, 3.0)
    a, b = 0.3 * c, 0.7 * c
    resid = _step(a + b, u, dx, dt) - (_step(a, u, dx, dt) + _step(b, u, dx, dt))
    assert np.abs(resid).max() < 1e-12
