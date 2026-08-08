"""Tests for VBS yields table + parent-class lookup."""
from __future__ import annotations

import numpy as np
import pytest

from orbit.emissions.vbs_yields import (
    VBS_PARENT_YIELDS,
    NOX_REGIME_PAIRS,
    get_yields,
    list_parent_classes,
)
from orbit.core.deposition import C_STAR_VALS, N_VBS_BINS


class TestVBSYields:
    def test_each_class_has_5_bins(self):
        for cls, y in VBS_PARENT_YIELDS.items():
            assert y.shape == (N_VBS_BINS,), f"{cls}: shape {y.shape}"

    def test_yields_nonnegative(self):
        for cls, y in VBS_PARENT_YIELDS.items():
            assert np.all(y >= 0), f"{cls}: negative yields {y}"

    def test_known_parent_classes_present(self):
        expected = {
            "anthro_high_nox", "anthro_low_nox",
            "bio_monoterpene", "bio_isoprene",
            "biomass_burning", "ivoc",
        }
        assert expected.issubset(set(list_parent_classes()))

    def test_total_yields_in_chamber_range(self):
        """Sums match published total yields within chamber-uncertainty bands."""
        # Tsimpidi 2010 monoterpene low-NOx total = 1.158
        assert abs(VBS_PARENT_YIELDS["bio_monoterpene"].sum() - 1.158) < 0.05
        # Anthro high-NOx total = 0.417
        assert abs(VBS_PARENT_YIELDS["anthro_high_nox"].sum() - 0.417) < 0.02
        # Grieshop 2009 BB total = 0.890
        assert abs(VBS_PARENT_YIELDS["biomass_burning"].sum() - 0.890) < 0.05

    def test_C_star_decade_alignment(self):
        """Yields are tied to C* = (0.1, 1, 10, 100, 1000) µg/m³ — guard
        against accidental future bin shifts (memory:
        feedback_vbs_bin_alignment)."""
        np.testing.assert_array_equal(
            C_STAR_VALS, np.array([0.1, 1.0, 10.0, 100.0, 1000.0])
        )


class TestNOxRegimePairs:
    def test_anthro_pair_defined(self):
        assert "anthro" in NOX_REGIME_PAIRS
        hi, lo = NOX_REGIME_PAIRS["anthro"]
        assert hi == "anthro_high_nox"
        assert lo == "anthro_low_nox"

    def test_pairs_resolve(self):
        for cls, (hi, lo) in NOX_REGIME_PAIRS.items():
            assert hi in VBS_PARENT_YIELDS
            assert lo in VBS_PARENT_YIELDS

    def test_high_nox_lower_total_yield(self):
        """Anthro high-NOx total < anthro low-NOx total (NOx-regime
        chemistry: high RO2+NO branching reduces SoA mass yield)."""
        y_hi = VBS_PARENT_YIELDS["anthro_high_nox"].sum()
        y_lo = VBS_PARENT_YIELDS["anthro_low_nox"].sum()
        assert y_hi < y_lo, \
            f"high_nox {y_hi:.3f} should be < low_nox {y_lo:.3f}"


class TestGetYields:
    def test_returns_copy(self):
        """get_yields returns an independent array — mutating must not
        affect the table."""
        y = get_yields("anthro_high_nox")
        y[0] = 999.0
        assert VBS_PARENT_YIELDS["anthro_high_nox"][0] != 999.0

    def test_unknown_class_raises(self):
        with pytest.raises(KeyError):
            get_yields("not_a_real_class")
