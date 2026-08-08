"""Tests for VBS Pankow partitioning + M_OA fixed-point closure."""
from __future__ import annotations


import numpy as np

from orbit.core.dcomp_vbs import pankow_F_p, solve_M_OA
from orbit.core.deposition import C_STAR_VALS, IDX_VBS_BINS, N_VBS_BINS


class TestPankow:
    """Single-bin gas/particle partitioning."""

    def test_high_M_OA_limit(self):
        """M_OA → ∞ should drive F_p → 1."""
        for C_star in C_STAR_VALS:
            F = pankow_F_p(np.array([1.0e8]), C_star)
            assert F[0] > 0.99999, f"F_p = {F[0]} for C*={C_star}"

    def test_zero_M_OA_limit(self):
        """M_OA → 0 should drive F_p → 0."""
        # Use a very small but non-pathological M_OA.
        for C_star in C_STAR_VALS:
            F = pankow_F_p(np.array([1.0e-5]), C_star)
            assert F[0] < 1.0e-3, f"F_p = {F[0]} for C*={C_star}"

    def test_F_p_monotonic_in_M_OA(self):
        """F_p strictly increases in M_OA at fixed C*."""
        M_OA = np.array([0.1, 1.0, 10.0, 100.0, 1000.0])
        for C_star in C_STAR_VALS:
            F = pankow_F_p(M_OA, C_star)
            assert np.all(np.diff(F) > 0), \
                f"F_p not monotonic for C*={C_star}: {F}"

    def test_F_p_monotonic_in_C_star(self):
        """F_p strictly decreases in C* at fixed M_OA."""
        M_OA = np.array([10.0])
        F_vals = [pankow_F_p(M_OA, C_star)[0] for C_star in C_STAR_VALS]
        assert np.all(np.diff(F_vals) < 0), \
            f"F_p not monotonic in C*: {F_vals}"

    def test_F_p_at_C_star_eq_M_OA(self):
        """F_p(M_OA = C*) = 0.5 exactly."""
        for C_star in C_STAR_VALS:
            F = pankow_F_p(np.array([C_star]), C_star)
            np.testing.assert_allclose(F[0], 0.5, rtol=1e-12)


class TestSolveMOA:
    """M_OA fixed-point Newton iteration."""

    def test_fixed_point_holds(self):
        """At convergence, M_OA = sum(F_p,i × C_i)."""
        C = np.array([
            [0.5, 1.0, 2.0],
            [1.0, 2.0, 3.0],
            [3.0, 5.0, 7.0],
            [5.0, 8.0, 11.0],
            [10.0, 15.0, 20.0],
        ])
        M_OA, F_p, info = solve_M_OA(C, tol=1e-8, maxiter=50)
        M_OA_check = (F_p * C).sum(axis=0)
        np.testing.assert_allclose(M_OA, M_OA_check, rtol=1e-6)

    def test_converges_for_bootstrap(self):
        """Default bootstrap (M_OA = 10) should converge in ≤15 iters."""
        C = np.full((5, 4), 1.0)   # 4 cells, 1 µg/m³ in each bin
        _, _, info = solve_M_OA(C)
        assert info["iters_run"] <= 15

    def test_zero_emissions_zero_M_OA(self):
        """If all bins empty, M_OA → 0 (or near zero)."""
        C = np.zeros((5, 3))
        M_OA, F_p, info = solve_M_OA(C, tol=1e-8, maxiter=20)
        assert np.all(M_OA < 1.0e-3)

    def test_force_F_p_one(self, monkeypatch):
        """ORBIT_VBS_FORCE_FP=1.0 short-circuits Pankow."""
        monkeypatch.setenv("ORBIT_VBS_FORCE_FP", "1.0")
        C = np.array([
            [0.5, 1.0],
            [1.0, 2.0],
            [3.0, 5.0],
            [5.0, 8.0],
            [10.0, 15.0],
        ])
        M_OA, F_p, info = solve_M_OA(C)
        assert info["forced_F_p"] == 1.0
        # F_p uniformly 1
        np.testing.assert_allclose(F_p, np.ones_like(F_p))
        # M_OA = sum across bins
        np.testing.assert_allclose(M_OA, C.sum(axis=0))

    def test_force_F_p_half(self, monkeypatch):
        """ORBIT_VBS_FORCE_FP=0.5 → F_p uniformly 0.5."""
        monkeypatch.setenv("ORBIT_VBS_FORCE_FP", "0.5")
        C = np.full((5, 2), 4.0)
        M_OA, F_p, _ = solve_M_OA(C)
        np.testing.assert_allclose(F_p, 0.5 * np.ones_like(F_p))
        np.testing.assert_allclose(M_OA, 0.5 * C.sum(axis=0))


class TestVBSConstants:
    """Sanity: deposition.py constants match Pankow assumptions."""

    def test_bin_count_consistent(self):
        assert len(C_STAR_VALS) == N_VBS_BINS == len(IDX_VBS_BINS) == 5

    def test_C_star_low_to_high(self):
        # C_STAR_VALS in low-to-high order (smallest C* first)
        assert np.all(np.diff(C_STAR_VALS) > 0)

    def test_C_star_decade_spacing(self):
        ratios = C_STAR_VALS[1:] / C_STAR_VALS[:-1]
        np.testing.assert_allclose(ratios, 10.0, rtol=1e-12)
