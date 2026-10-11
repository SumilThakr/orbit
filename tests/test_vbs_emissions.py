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
            "bio_monoterpene_high_nox", "bio_monoterpene_low_nox",
            "bio_isoprene_high_nox", "bio_isoprene_low_nox",
            "biomass_burning_high_nox", "biomass_burning_low_nox",
            "ivoc_high_nox", "ivoc_low_nox",
        }
        assert expected.issubset(set(list_parent_classes()))

    def test_rows_match_the_sources(self):
        """Tsimpidi (2010) Table 2 rows at C* = 1, 10, 100, 1000; nothing at 0.1."""
        t2 = {
            "anthro_high_nox": (0.003, 0.165, 0.300, 0.435),          # ARO1
            "anthro_low_nox": (0.075, 0.225, 0.375, 0.525),
            "bio_monoterpene_high_nox": (0.012, 0.122, 0.201, 0.500),  # TERP
            "bio_monoterpene_low_nox": (0.107, 0.092, 0.359, 0.600),
            "bio_isoprene_high_nox": (0.001, 0.023, 0.015, 0.000),     # ISOP
            "bio_isoprene_low_nox": (0.009, 0.030, 0.015, 0.000),
        }
        for key, row in t2.items():
            assert VBS_PARENT_YIELDS[key][0] == 0.0
            np.testing.assert_allclose(VBS_PARENT_YIELDS[key][1:], row)
        # Hodzic (2016) IVOC: most mass in the lowest bin
        for key in ("ivoc_high_nox", "ivoc_low_nox"):
            y = VBS_PARENT_YIELDS[key]
            assert y[0] == y.max()
        # low-NOx yields are at least the high-NOx yields in total, every class
        for cls, (hi, lo) in NOX_REGIME_PAIRS.items():
            assert VBS_PARENT_YIELDS[lo].sum() >= VBS_PARENT_YIELDS[hi].sum(), cls

    def test_biomass_burning_rows_reproduce_from_gfed_shares(self):
        """scripts/bb_voc_yields.py with the recorded 2022 GFED5.1 shares."""
        import importlib.util, os
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        spec = importlib.util.spec_from_file_location(
            "bb_voc_yields", os.path.join(here, "scripts", "bb_voc_yields.py"))
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        shares = {"C6H6": 0.232, "C7H8": 0.130, "C8H10": 0.035, "Higher_Alkenes": 0.147,
                  "Higher_Alkanes": 0.068, "C5H8": 0.104, "C10H16": 0.285}
        hi, lo = mod.weighted_rows(shares)
        np.testing.assert_allclose(VBS_PARENT_YIELDS["biomass_burning_high_nox"], hi, atol=1.5e-3)
        np.testing.assert_allclose(VBS_PARENT_YIELDS["biomass_burning_low_nox"], lo, atol=1.5e-3)

    def test_regime_yields_interpolates(self):
        from orbit.emissions.vbs_yields import regime_yields
        y_hi = VBS_PARENT_YIELDS["anthro_high_nox"]; y_lo = VBS_PARENT_YIELDS["anthro_low_nox"]
        np.testing.assert_allclose(regime_yields("anthro", 1.0), y_hi)
        np.testing.assert_allclose(regime_yields("anthro", 0.0), y_lo)
        F = np.full((2, 3, 4), 0.25)
        y = regime_yields("anthro", F)
        assert y.shape == (2, 5, 3, 4)
        np.testing.assert_allclose(y[1, :, 0, 0], 0.25 * y_hi + 0.75 * y_lo)

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
