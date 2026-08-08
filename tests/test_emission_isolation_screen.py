"""Spatial-isolation screen for emission allocation artefacts.

The domain-share screen asks "does one cell hold too much of the domain?",
which has no physical meaning — a big domain dilutes a real artefact, and a
small one condemns a legitimate megacity. Isolation asks the physical
question instead: "is this cell unlike its neighbourhood?" A real source sits
in a populated region; a national sector total dropped on one cell does not.
"""

import numpy as np
import pytest

from orbit.emissions.netcdf import (
    _neighbourhood_mean,
    isolation_ratio,
    screen_by_isolation,
)


def _field(n=21, background=1.0):
    return np.full((n, n), background, dtype=np.float64)


def test_neighbourhood_mean_excludes_the_centre():
    a = np.zeros((5, 5))
    a[2, 2] = 100.0
    a[1, 2] = 2.0
    m = _neighbourhood_mean(a, radius=1)
    # centre's 8 neighbours hold 2.0 total
    assert m[2, 2] == pytest.approx(2.0 / 8)


def test_neighbourhood_mean_is_edge_aware():
    """A corner cell must not be diluted by out-of-domain phantom zeros."""
    a = np.ones((5, 5))
    m = _neighbourhood_mean(a, radius=1)
    assert m[0, 0] == pytest.approx(1.0)
    assert m[2, 2] == pytest.approx(1.0)


def test_uniform_field_has_isolation_one():
    r = isolation_ratio(_field(), radius=2)
    np.testing.assert_allclose(r, 1.0)


def test_isolated_spike_scores_high_and_real_cluster_does_not():
    a = _field(background=1.0)
    a[5, 5] = 1000.0                      # lone spike in flat background
    a[15, 15] = 40.0                      # a "city": bright, but in a bright region
    a[14:17, 14:17] = np.maximum(a[14:17, 14:17], 12.0)
    a[15, 15] = 40.0
    r = isolation_ratio(a, radius=2)
    assert r[5, 5] > 200
    assert r[15, 15] < 20


def test_empty_neighbourhood_is_inf_not_nan():
    a = np.zeros((11, 11))
    a[5, 5] = 3.0
    r = isolation_ratio(a, radius=2)
    assert np.isinf(r[5, 5])
    assert not np.isnan(r).any()


def test_screen_conserves_domain_total():
    a = _field(background=1.0)
    a[5, 5] = 1e5
    out, recs = screen_by_isolation(a, label="t")
    assert recs
    assert out.sum() == pytest.approx(a.sum(), rel=1e-12)


def test_screen_reduces_the_flagged_cell_and_raises_others():
    a = _field(background=1.0)
    a[5, 5] = 1e5
    out, _ = screen_by_isolation(a, label="t")
    assert out[5, 5] < a[5, 5]
    assert out[0, 0] > a[0, 0]


def test_mass_floor_protects_tiny_isolated_cells():
    """A lone village in an empty region has infinite ratio and no consequence.

    Without the min_share condition the screen would 'fix' it, which is both
    pointless and a way to lose real emissions. What is asserted is that such
    a cell is never *flagged*; it may still receive redistributed mass as a
    donor, which is correct behaviour.
    """
    a = _field(background=1.0)      # plenty of donors, as in a real field
    a[5, 5] = 1e5                   # isolated AND massive -> flagged
    a[15, 15] = 1e-6                # isolated but negligible -> not flagged
    _, recs = screen_by_isolation(a, min_share=0.005, label="t",
                                  lat=np.arange(21) * 1.0,
                                  lon=np.arange(21) * 1.0)
    flagged = {(r["lat"], r["lon"]) for r in recs}
    assert (5.0, 5.0) in flagged
    assert (15.0, 15.0) not in flagged
    assert all(r["share"] > 0.005 for r in recs)


def test_thin_donor_pool_warns_rather_than_inflating_silently():
    """Redistribution onto very few donors would blow them up; say so.

    Degenerate in practice (real fields have thousands of non-zero cells) but
    a silent 1e6-fold inflation of one cell is exactly the class of failure
    this project keeps getting bitten by.
    """
    a = np.zeros((21, 21))
    a[10, 10] = 1000.0
    a[2, 2] = 1e-6
    with pytest.warns(UserWarning, match="donor"):
        out, _ = screen_by_isolation(a, min_share=0.005, label="t")
    assert out.sum() == pytest.approx(a.sum(), rel=1e-12)


def test_uniform_field_is_untouched():
    a = _field()
    out, recs = screen_by_isolation(a, label="t")
    assert not recs
    np.testing.assert_array_equal(out, a)


def test_disabled_by_nonpositive_threshold():
    a = _field()
    a[5, 5] = 1e6
    out, recs = screen_by_isolation(a, ratio_threshold=0.0, label="t")
    assert not recs
    np.testing.assert_array_equal(out, a)


def test_records_carry_both_scores():
    """The run log must show why a cell was cut, on both criteria."""
    a = _field(background=1.0)
    a[5, 5] = 1e5
    _, recs = screen_by_isolation(a, label="t",
                                  lat=np.arange(21) * 1.0,
                                  lon=np.arange(21) * 1.0)
    assert len(recs) == 1
    r = recs[0]
    assert r["isolation"] > 50 and 0.0 < r["share"] <= 1.0
    assert r["capped"] < r["value"]
    assert r["lat"] == 5.0 and r["lon"] == 5.0


def test_zero_and_degenerate_fields_are_safe():
    for a in (np.zeros((7, 7)), np.full((7, 7), np.nan)):
        out, recs = screen_by_isolation(np.nan_to_num(a), label="t")
        assert not recs
