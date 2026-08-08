"""Spatial-allocation artefact screen on native emission grids."""

import numpy as np
import pytest

from orbit.emissions.netcdf import (
    _DEFAULT_SPIKE_FRAC,
    deconcentrate_allocation_artefacts,
)


def _field(spike=None, n=20, base=1.0):
    a = np.full((n, n), base, dtype=np.float64)
    if spike is not None:
        a[5, 7] = spike
    return a


def test_clean_field_is_untouched():
    a = _field()
    out, recs = deconcentrate_allocation_artefacts(a, threshold=0.05)
    assert recs == []
    assert out is a          # no copy when nothing is flagged
    np.testing.assert_array_equal(out, a)


def test_spike_is_capped_at_the_threshold():
    # one cell holding ~50% of the domain total
    a = _field(spike=399.0)          # 399 vs 399 spread over the rest
    total = a.sum()
    assert a[5, 7] / total == pytest.approx(0.5, rel=1e-6)

    out, recs = deconcentrate_allocation_artefacts(a, threshold=0.05)
    assert len(recs) == 1
    assert recs[0]["share"] == pytest.approx(0.5, rel=1e-6)
    assert out[5, 7] / out.sum() == pytest.approx(0.05, rel=1e-9)


def test_domain_total_is_conserved():
    a = _field(spike=399.0)
    out, _ = deconcentrate_allocation_artefacts(a, threshold=0.05)
    assert out.sum() == pytest.approx(a.sum(), rel=1e-12)


def test_excess_is_redistributed_in_proportion():
    a = _field(spike=399.0)
    a[0, 0] = 3.0                     # a donor three times its neighbours
    out, _ = deconcentrate_allocation_artefacts(a, threshold=0.05)
    # donors keep their relative ordering and ratios exactly
    assert out[0, 0] / out[1, 1] == pytest.approx(a[0, 0] / a[1, 1], rel=1e-12)
    assert out[1, 1] > a[1, 1]        # everyone gained


def test_threshold_zero_disables_the_screen():
    a = _field(spike=399.0)
    out, recs = deconcentrate_allocation_artefacts(a, threshold=0.0)
    assert recs == []
    np.testing.assert_array_equal(out, a)


def test_multiple_spikes_are_capped_together():
    a = _field(spike=399.0)
    a[9, 9] = 399.0
    out, recs = deconcentrate_allocation_artefacts(a, threshold=0.05)
    assert len(recs) == 2
    assert out.sum() == pytest.approx(a.sum(), rel=1e-12)
    for j, i in ((5, 7), (9, 9)):
        assert out[j, i] / out.sum() == pytest.approx(0.05, rel=1e-9)


def test_all_mass_in_one_cell_is_left_alone():
    """Nothing to redistribute onto: warn and leave the field unchanged."""
    a = np.zeros((10, 10))
    a[2, 3] = 5.0
    with pytest.warns(UserWarning, match="no other non-zero cells"):
        out, recs = deconcentrate_allocation_artefacts(a, threshold=0.05)
    assert recs == []
    np.testing.assert_array_equal(out, a)


def test_empty_and_zero_fields_are_safe():
    for a in (np.zeros((5, 5)), np.full((5, 5), np.nan)):
        out, recs = deconcentrate_allocation_artefacts(a, threshold=0.05)
        assert recs == []


def test_records_carry_coordinates():
    a = _field(spike=399.0)
    lat = np.linspace(4.25, 38.75, 20)
    lon = np.linspace(58.25, 99.75, 20)
    _, recs = deconcentrate_allocation_artefacts(a, threshold=0.05,
                                                 lat=lat, lon=lon)
    assert recs[0]["lat"] == pytest.approx(lat[5])
    assert recs[0]["lon"] == pytest.approx(lon[7])


def test_env_threshold_is_honoured(monkeypatch):
    a = _field(spike=399.0)
    monkeypatch.setenv("ORBIT_EMISSION_SPIKE_FRAC", "0.20")
    out, recs = deconcentrate_allocation_artefacts(a)
    assert len(recs) == 1
    assert out[5, 7] / out.sum() == pytest.approx(0.20, rel=1e-9)

    monkeypatch.setenv("ORBIT_EMISSION_SPIKE_FRAC", "0")
    out, recs = deconcentrate_allocation_artefacts(_field(spike=399.0))
    assert recs == []


def test_env_threshold_out_of_range_raises(monkeypatch):
    monkeypatch.setenv("ORBIT_EMISSION_SPIKE_FRAC", "1.5")
    with pytest.raises(ValueError, match=r"must be in \[0, 1\]"):
        deconcentrate_allocation_artefacts(_field(spike=399.0))


def test_default_is_off_to_match_published_baseline():
    """The published SAS 2022 baseline ran with the screen disabled (two
    flagged-but-retained cells, stated in the methods); the shipped default
    must reproduce it. Screening is opt-in via ORBIT_EMISSION_SPIKE_FRAC."""
    assert _DEFAULT_SPIKE_FRAC == 0.0
