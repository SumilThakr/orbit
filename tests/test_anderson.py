"""Unit tests for AndersonAccelerator (Opt 1 of orbit solver optimizations).

Tests use simple scalar / small-vector fixed points where the expected
convergence behaviour is known analytically, so failures point at
specific algorithm issues rather than system-level problems.
"""

import numpy as np
import pytest

from orbit.core.dcomp_iter import AndersonAccelerator


# ---------------------------------------------------------------------------
# Scalar fixed points
# ---------------------------------------------------------------------------

def _picard_iter(g, x0, tol=1e-6, max_iter=200):
    """Plain Picard: x_{k+1} = g(x_k).  Returns (x_final, n_iter)."""
    x = np.asarray(x0, dtype=np.float64)
    for k in range(max_iter):
        x_new = g(x)
        if np.linalg.norm(x_new - x) < tol:
            return x_new, k + 1
        x = x_new
    return x, max_iter


def _anderson_iter(g, x0, m=3, tol=1e-6, max_iter=200, beta=1.0):
    """Anderson(m) on g.  Returns (x_final, n_iter)."""
    accel = AndersonAccelerator(m=m, beta=beta)
    x = np.asarray(x0, dtype=np.float64)
    for k in range(max_iter):
        g_x = g(x)
        x_next = accel.update(x, g_x)
        if np.linalg.norm(x_next - x) < tol:
            return x_next, k + 1
        x = x_next
    return x, max_iter


def test_anderson_converges_cos_fixed_point():
    """x = cos(x) has fixed point ≈ 0.7390851332.  Picard is slow here
    (ρ ≈ |sin(x*)| ≈ 0.67), Anderson(3) should be substantially faster."""
    g = np.cos
    x_ref = 0.7390851332151607  # canonical root of x - cos(x)

    x_aa, n_aa = _anderson_iter(g, 0.0, m=3, tol=1e-10)
    x_pic, n_pic = _picard_iter(g, 0.0, tol=1e-10)

    assert abs(x_aa - x_ref) < 1e-9, f"Anderson converged to {x_aa}, not {x_ref}"
    assert abs(x_pic - x_ref) < 1e-9
    assert n_aa < n_pic, (
        f"Anderson should be faster than Picard: n_aa={n_aa}, n_pic={n_pic}"
    )
    # Anderson(3) on x=cos(x) typically finishes in <= 10 iters;
    # Picard at tol=1e-10 with ρ ≈ 0.67 needs ~50+.
    assert n_aa <= 10, f"Anderson(3) took {n_aa} iters; expected <= 10"
    assert n_pic >= 20, f"Picard took only {n_pic}; this benchmark is too easy"


def test_anderson_handles_sqrt_via_newton_like():
    """Compute √2 via g(x) = 0.5 * (x + 2/x).  Newton-like, quadratic
    under Picard already — Anderson should be no worse."""
    g = lambda x: 0.5 * (x + 2.0 / x)

    x_aa, n_aa = _anderson_iter(g, 1.0, m=3, tol=1e-12)
    assert abs(x_aa - np.sqrt(2)) < 1e-11
    # With Newton-like quadratic convergence, Anderson should still
    # converge in a handful of iters and not blow up.
    assert n_aa <= 10


def test_anderson_first_iter_is_picard_when_beta_1():
    """On the first call, history is empty and Anderson should fall
    back to x_0 + beta * r_0."""
    accel = AndersonAccelerator(m=3, beta=1.0)
    x0 = np.array([1.0, 2.0, 3.0])
    g_x0 = np.array([1.5, 2.5, 3.5])  # r_0 = [0.5, 0.5, 0.5]
    x1 = accel.update(x0, g_x0)
    np.testing.assert_allclose(x1, g_x0)  # beta=1 Picard step = g(x0)


def test_anderson_reset_clears_history():
    accel = AndersonAccelerator(m=3)
    for _ in range(5):
        accel.update(np.array([1.0]), np.array([0.5]))
    assert len(accel.X) == 4  # m+1 = 4
    accel.reset()
    assert len(accel.X) == 0
    assert len(accel.F) == 0


def test_anderson_vector_state():
    """2-D linear fixed point x = A x + b with a contraction A.
    Anderson should find the fixed point x* = (I - A)^{-1} b."""
    A = np.array([[0.4, 0.3], [0.2, 0.5]])
    b = np.array([1.0, 2.0])
    x_star = np.linalg.solve(np.eye(2) - A, b)

    g = lambda x: A @ x + b
    x_aa, n_aa = _anderson_iter(g, np.zeros(2), m=3, tol=1e-12)

    np.testing.assert_allclose(x_aa, x_star, atol=1e-10)
    # Linear contraction with spectral radius ≈ 0.7 under Picard would
    # need ~80 iters to 1e-12.  Anderson(3) typically <10.
    assert n_aa <= 12, f"Anderson(3) on linear system took {n_aa}; expected <= 12"


def test_safeguard_accepts_good_step():
    """On a well-behaved fixed point, the safeguard should always accept."""
    accel = AndersonAccelerator(m=3, safeguard_factor=1.5)
    g = np.cos
    x = 0.0
    for _ in range(6):
        g_x = g(x)
        x_next, accepted = accel.apply_with_safeguard(x, g_x)
        x = x_next
    assert accel.n_safeguard_falls == 0


def test_memory_depth_cap():
    """With m=3, history should cap at m+1 = 4 entries."""
    accel = AndersonAccelerator(m=3)
    for k in range(10):
        x = np.array([float(k)])
        gx = np.array([float(k) + 0.1])
        accel.update(x, gx)
    assert len(accel.X) == 4
    assert len(accel.F) == 4


def test_degenerate_history_lstsq_rcond():
    """Repeated identical iterates should not crash the LSQ solve;
    the rcond cutoff should drop rank-deficient columns cleanly."""
    accel = AndersonAccelerator(m=3, rcond=1e-10)
    # Feed the same (x, g(x)) pair multiple times - ΔF will be all zeros.
    for _ in range(5):
        x_next = accel.update(np.array([1.0]), np.array([0.9]))
    # Should not have NaN'd out.
    assert np.isfinite(x_next).all()


# ---------------------------------------------------------------------------
# splice_oh_into_chem helper (Opt 1b)
# ---------------------------------------------------------------------------

def _make_dummy_chem(nz, ny, nx, N_BINS):
    """Build a minimal ChemistryPerBin for splice tests.  All 11 required
    OxidantFields entries are zeros of the right shape; only OH is
    inspected by these tests."""
    from orbit.core.dcomp import ChemistryPerBin
    from orbit.core.oxidants import OxidantFields

    def zero3(): return np.zeros((nz, ny, nx))

    ox_list = [
        OxidantFields(
            OH=zero3(), HO2=zero3(), RO2=zero3(), NO=zero3(), NO2=zero3(),
            f_NO2=zero3(), NO3=zero3(), N2O5=zero3(),
            k_n2o5_het=zero3(), k_oh_so2=zero3(), k_oh_no2_for_hno3=zero3(),
            k_oh_co=zero3(),
        )
        for _ in range(N_BINS)
    ]
    return ChemistryPerBin(
        oxidants=ox_list,
        k_so2_rate=[zero3() for _ in range(N_BINS)],
        k_nox_to_no3_rate=[zero3() for _ in range(N_BINS)],
        jNO2=np.zeros((N_BINS, ny, nx)),
        jO1D=np.zeros((N_BINS, ny, nx)),
        jNO3=np.zeros((N_BINS, ny, nx)),
        jHONO=np.zeros((N_BINS, ny, nx)),
    )


def test_splice_oh_into_chem_roundtrip():
    """splice_oh_into_chem writes a flat (N_BINS * N) OH vector back
    into chem.oxidants[tau].OH for each bin."""
    from orbit.core.dcomp import splice_oh_into_chem

    nz, ny, nx = 2, 3, 4
    N = nz * ny * nx
    N_BINS = 8
    chem = _make_dummy_chem(nz, ny, nx, N_BINS)

    # Distinct values per bin so we catch index mix-ups.
    oh_flat = np.concatenate([
        np.full(N, float(tau + 1) * 1e6, dtype=np.float64)
        for tau in range(N_BINS)
    ])
    splice_oh_into_chem(chem, oh_flat, nz, ny, nx)

    for tau in range(N_BINS):
        np.testing.assert_allclose(
            chem.oxidants[tau].OH,
            float(tau + 1) * 1e6,
        )


def test_splice_oh_into_chem_size_mismatch_raises():
    from orbit.core.dcomp import splice_oh_into_chem
    chem = _make_dummy_chem(2, 3, 4, 8)
    with pytest.raises(ValueError, match="expected"):
        splice_oh_into_chem(chem, np.zeros(10), 2, 3, 4)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
