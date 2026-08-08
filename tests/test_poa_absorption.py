"""POA as absorbing mass in the VBS Pankow partitioning.

Before 2026-08-02 `M_OA` summed the VBS bins only, omitting POA, which
collapsed `F_p` and suppressed anthropogenic SOA by a documented-as-slight
but actually ~200-fold factor. These tests pin the corrected behaviour and,
critically, that omitting POA reproduces the old answer exactly — the
backward-compatibility guarantee the regression gate depends on.
"""

import numpy as np
import pytest

from orbit.core.dcomp_vbs import pankow_F_p, solve_M_OA
from orbit.core.deposition import C_STAR_VALS, IDX_POA, N_SPECIES, N_VBS_BINS


def _bins(value=1.0, shape=(4, 5)):
    """Uniform mass in every VBS bin."""
    return np.full((N_VBS_BINS,) + shape, value, dtype=np.float64)


def test_zero_poa_reproduces_the_old_fixed_point():
    """C_POA=None, 0.0 and an explicit zero array must all agree exactly."""
    C = _bins()
    a, fa, _ = solve_M_OA(C)
    b, fb, _ = solve_M_OA(C, C_POA=0.0)
    c, fc, _ = solve_M_OA(C, C_POA=np.zeros(C.shape[1:]))
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(a, c)
    np.testing.assert_array_equal(fa, fb)
    np.testing.assert_array_equal(fa, fc)


def test_poa_raises_M_OA_and_the_particle_fraction():
    C = _bins()
    m0, f0, _ = solve_M_OA(C)
    m1, f1, _ = solve_M_OA(C, C_POA=20.0)
    assert (m1 > m0).all()
    assert (f1 >= f0).all() and (f1 > f0).any()


def test_M_OA_satisfies_its_own_fixed_point():
    """The returned M_OA must equal C_POA + sum F_p,i C_i to tolerance."""
    C = _bins(value=2.0)
    poa = 15.0
    M, F_p, _ = solve_M_OA(C, C_POA=poa, tol=1e-12, maxiter=500)
    residual = np.abs(M - (poa + (F_p * C).sum(axis=0))) / M
    assert residual.max() < 1e-8


def test_partitioning_matches_pankow_at_the_converged_M_OA():
    C = _bins()
    M, F_p, _ = solve_M_OA(C, C_POA=10.0, tol=1e-12, maxiter=500)
    for i, cstar in enumerate(C_STAR_VALS):
        np.testing.assert_allclose(F_p[i], pankow_F_p(M, cstar), rtol=1e-10)


def test_M_OA_is_at_least_the_poa_floor():
    """POA is non-volatile here, so it is a lower bound on M_OA."""
    M, _, _ = solve_M_OA(_bins(value=0.0), C_POA=7.5)
    assert np.allclose(M, 7.5)


def test_negative_poa_is_clipped_not_propagated():
    C = _bins()
    M, _, _ = solve_M_OA(C, C_POA=np.full(C.shape[1:], -3.0))
    ref, _, _ = solve_M_OA(C, C_POA=0.0)
    np.testing.assert_allclose(M, ref)


def test_poa_shape_mismatch_raises():
    with pytest.raises(ValueError, match="C_POA shape"):
        solve_M_OA(_bins(), C_POA=np.zeros((2, 2)))


def test_scaling_is_monotone_in_poa():
    C = _bins()
    prev = -np.inf
    for poa in (0.0, 1.0, 5.0, 20.0, 100.0):
        M, _, _ = solve_M_OA(C, C_POA=poa)
        assert M.mean() > prev
        prev = M.mean()


def test_idx_poa_is_last_so_existing_indices_are_stable():
    """Appending POA must not renumber any pre-existing species."""
    assert IDX_POA == N_SPECIES - 1 == 13


def test_the_documented_suppression_is_reproduced():
    """Sanity-check the previously measured yield suppression.

    With M_OA ~2 the anthropogenic yield set (0.067 at C*=100, 0.350 at
    C*=1000) realises ~0.002; with POA present it recovers by an order of
    magnitude. This is the motivation for the whole change.
    """
    from orbit.emissions.vbs_yields import VBS_PARENT_YIELDS
    y = VBS_PARENT_YIELDS["anthro_high_nox"]
    eff = lambda m: float((y * np.array([1 / (1 + c / m) for c in C_STAR_VALS])).sum())
    assert eff(2.1) < 0.005
    assert eff(17.0) > 5 * eff(2.1)


# ---------------------------------------------------------------------------
# Species-table consistency
#
# Adding POA broke two hardcoded 13-element lists, each of which failed only
# at run time: _SOLVE_ORDER (KeyError 13, ~23 min in) and _SPECIES_NAMES
# (IndexError, ~23 s in). Both are asserted at import now; these tests state
# the invariant explicitly so the reason is discoverable.
# ---------------------------------------------------------------------------

def test_solve_order_covers_every_species_exactly_once():
    from orbit.core.orbit import _SOLVE_ORDER
    assert sorted(_SOLVE_ORDER) == list(range(N_SPECIES))


def test_species_name_tables_match_the_species_count():
    from orbit.core.orbit import _SPECIES_NAMES
    from orbit.cli import SPECIES_NAMES
    assert len(_SPECIES_NAMES) == N_SPECIES
    assert len(SPECIES_NAMES) == N_SPECIES
    assert _SPECIES_NAMES == SPECIES_NAMES, "the two tables must agree"


def test_poa_is_named_in_both_tables():
    from orbit.core.orbit import _SPECIES_NAMES
    from orbit.cli import SPECIES_NAMES
    assert _SPECIES_NAMES[IDX_POA] == "POA"
    assert SPECIES_NAMES[IDX_POA] == "POA"


def test_poa_deposits_exactly_like_primary_pm25():
    """Identical operators are what makes LU sharing valid."""
    import inspect
    import orbit.core.deposition as dep
    src = inspect.getsource(dep)
    # every PM25 deposition branch must also admit POA
    assert "elif species_idx == IDX_PM25:" not in src, (
        "a PM25-only deposition branch remains; POA would get the wrong rate")
    assert src.count("elif species_idx in (IDX_PM25, IDX_POA):") == 4
