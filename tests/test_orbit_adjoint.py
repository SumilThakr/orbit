"""Verify the periodic-orbit adjoint solver against finite-difference of the
forward orbit. This is the load-bearing correctness test for the marginal-
deaths integration: if ∂J/∂e_τ at every (bin, cell) matches FD to ~1e-5,
we know the adjoint chain (recursion + transposed back-solves + GMRES on
(I − M^T)) is wired correctly.
"""
import numpy as np
import pytest
import scipy.sparse as sp

from orbit.core.solve import _HAS_UMFPACK
from orbit.core.orbit import (
    N_BINS, DTAU, solve_orbit_one_species, _factor_one_bin,
)
from orbit.core.orbit_adjoint import (
    solve_orbit_adjoint_one_species, gradient_per_emission_bin,
)


@pytest.mark.skipif(not _HAS_UMFPACK, reason="UMFPACK not available")
def test_adjoint_matches_forward_finite_difference():
    """Brute-force test: for each bin τ and a perturbation at one cell j,
    the adjoint's predicted ∂J/∂e_τ[j] must match the finite-difference
    of J under a +h perturbation at e_τ[j]."""
    n = 60
    rng = np.random.default_rng(7)

    # Diagonal-dominant asymmetric sparse matrices, one per bin. Scale L
    # to ORBIT-physical magnitudes (~1e-5 /s loss rate) so I + DTAU·L
    # has order-unity entries; otherwise A=I+DTAU·L would be ~10^5·I,
    # P would be ~10^-5·I, M would be ~10^-40·I, and Arnoldi divisions
    # underflow to NaN inside scipy.gmres.
    L_SCALE = 5e-5
    I_N = sp.eye(n, format="csc")
    L_per_bin = []
    lu_list_8 = []
    for tau in range(N_BINS):
        L = sp.random(n, n, density=0.06, format="csc", random_state=rng) * L_SCALE
        L = L + 1.0e-4 * sp.eye(n, format="csc")
        L_per_bin.append(L.tocsc())
        lu_list_8.append(_factor_one_bin(L, I_N, None, None))

    # Per-bin "deaths gradient" — random non-negative weights.
    s_per_bin = [rng.standard_normal(n) ** 2 for _ in range(N_BINS)]

    # Baseline emissions (must not be all zero so the forward orbit is
    # non-trivial; gradient is independent of baseline since orbit is linear).
    e_per_bin = [rng.standard_normal(n) for _ in range(N_BINS)]

    def J_at(e_list):
        """Run forward orbit with given emissions; return Σ_τ ⟨s_τ, c_τ⟩."""
        orbit, _ = solve_orbit_one_species(
            lu_list_8, e_list, DTAU,
            c_warm=None, tol=1e-12, maxiter=500,
        )
        # orbit = [c_0, c_1, ..., c_8]; sum f_τ = ⟨s_τ, c_τ⟩ over τ=0..7.
        return float(sum(np.dot(s_per_bin[t], orbit[t]) for t in range(N_BINS)))

    # Adjoint solve: R_τ = s_τ (with the J = Σ ⟨s_τ, c_τ⟩ convention,
    # the per-bin receptor equals the deaths-gradient vector exactly).
    adjoint, info = solve_orbit_adjoint_one_species(
        lu_list_8, s_per_bin, tol=1e-12, maxiter=500,
    )
    grad = gradient_per_emission_bin(adjoint, s_per_bin)
    # grad[τ] is shape (n,) = ∂J/∂e_τ.

    # Verify by FD at a handful of (τ, cell) pairs.
    J0 = J_at(e_per_bin)
    h = 1e-6
    test_cells = rng.choice(n, size=4, replace=False)
    test_bins = [0, 3, 7]
    for tau in test_bins:
        for j in test_cells:
            e_plus = [e.copy() for e in e_per_bin]
            e_plus[tau][j] += h
            J_plus = J_at(e_plus)
            fd = (J_plus - J0) / h
            pred = grad[tau][j]
            # Linearisation: GMRES tol ~ 1e-12 on both forward and adjoint;
            # FD truncation at h=1e-6 → error ~ h · max(d²J/de²) ~ 1e-6.
            assert abs(fd - pred) < 1e-4 * max(abs(fd), 1.0), (
                f"bin {tau}, cell {j}: pred={pred:.6g}, fd={fd:.6g}, "
                f"diff={pred - fd:.2g}"
            )


@pytest.mark.skipif(not _HAS_UMFPACK, reason="UMFPACK not available")
def test_adjoint_zero_receptor_gives_zero_orbit():
    """Sanity: empty deaths gradient → adjoint orbit is identically zero."""
    n = 20
    L = 1e-4 * sp.eye(n, format="csc") + 5e-5 * sp.random(
        n, n, density=0.1, format="csc", random_state=42
    )
    I_N = sp.eye(n, format="csc")
    lu_list_8 = [_factor_one_bin(L, I_N, None, None) for _ in range(N_BINS)]
    R_list = [np.zeros(n) for _ in range(N_BINS)]
    adjoint, info = solve_orbit_adjoint_one_species(lu_list_8, R_list)
    for lam in adjoint:
        np.testing.assert_array_equal(lam, np.zeros(n))


@pytest.mark.skipif(not _HAS_UMFPACK, reason="UMFPACK not available")
def test_adjoint_orbit_is_periodic():
    """Recovered λ_{N_BINS} must equal λ_0 to GMRES tolerance."""
    n = 30
    rng = np.random.default_rng(123)
    I_N = sp.eye(n, format="csc")
    L_list = [
        5e-5 * sp.random(n, n, density=0.05, format="csc", random_state=rng)
        + 1e-4 * sp.eye(n, format="csc")
        for _ in range(N_BINS)
    ]
    lu_list_8 = [_factor_one_bin(L, I_N, None, None) for L in L_list]
    R_list = [rng.standard_normal(n) for _ in range(N_BINS)]
    adjoint, info = solve_orbit_adjoint_one_species(
        lu_list_8, R_list, tol=1e-12, maxiter=200,
    )
    np.testing.assert_allclose(adjoint[0], adjoint[N_BINS])
    # periodicity diagnostic should also report ≈0.
    assert info["periodicity"] < 1e-8
