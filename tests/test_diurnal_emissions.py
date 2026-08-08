"""Unit tests for diurnal-resolved emissions in the orbit driver.

Covers:
* `_local_to_utc_bin_factors`: local-time → UTC mapping with the IST
  offset, including the 0.5h fractional tail.
* `DiurnalConfig.from_yaml`: schema parsing, mean=1 enforcement.
* `EmissionsBundle`-style budget conservation (sum of bin emissions
  with day-mean=1 profiles equals N_BINS × bin-flat).
"""

from pathlib import Path

import numpy as np
import pytest

from orbit.emissions.loader import (
    DiurnalConfig,
    _local_to_utc_bin_factors,
)


class TestLocalToUtcBinFactors:
    """Sampler that converts a 24-h local-time profile into N_BINS UTC factors."""

    def test_flat_profile_returns_ones(self):
        flat = np.ones(24)
        for offset in (0.0, 5.5, 12.0, -3.0):
            f = _local_to_utc_bin_factors(flat, offset, n_bins=8)
            assert f.shape == (8,)
            np.testing.assert_allclose(f, 1.0, atol=1e-12)

    def test_mean_preserved_for_arbitrary_profile(self):
        # An arbitrary profile with mean = 1.0 — the bin factors must
        # also have mean = 1.0 (within 1e-12) regardless of IST offset.
        rng = np.random.default_rng(42)
        prof = rng.uniform(0.1, 2.0, size=24)
        prof = prof / prof.mean()  # exact mean=1
        for offset in (0.0, 5.5, 11.5, 23.5):
            f = _local_to_utc_bin_factors(prof, offset, n_bins=8)
            np.testing.assert_allclose(f.mean(), 1.0, atol=1e-12)

    def test_zero_offset_noon_concentrates_at_bin_4(self):
        # Profile is zero everywhere except hour 12 LT, normalised so mean=1.
        prof = np.zeros(24)
        prof[12] = 24.0  # mean = 1.0
        f = _local_to_utc_bin_factors(prof, ist_offset_hours=0.0, n_bins=8)
        # 12 LT = 12 UTC = bin 4 (12-15 UTC).  bin 3 (09-12) has no weight
        # because the 12-13 LT hour falls *inside* bin 4.
        assert int(np.argmax(f)) == 4

    def test_ist_offset_shifts_noon_to_bin_2(self):
        # Same profile, IST offset 5.5h: 12 LT = 06:30 UTC, which falls in
        # bin 2 (06-09 UTC).
        prof = np.zeros(24)
        prof[12] = 24.0
        f = _local_to_utc_bin_factors(prof, ist_offset_hours=5.5, n_bins=8)
        assert int(np.argmax(f)) == 2

    def test_invalid_length_raises(self):
        with pytest.raises(ValueError):
            _local_to_utc_bin_factors(np.ones(23), 0.0)


class TestDiurnalConfigFromYaml:
    """YAML parsing and renormalisation."""

    def test_parses_minimal_yaml(self, tmp_path: Path):
        cfg_path = tmp_path / "diurnal.yaml"
        cfg_path.write_text("""
ist_offset_hours: 5.5
profiles:
  flat:
    hours: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
source_profiles:
  example_source.nc: flat
""")
        cfg = DiurnalConfig.from_yaml(str(cfg_path))
        assert cfg.ist_offset_hours == 5.5
        assert "flat" in cfg.profiles
        assert cfg.profiles["flat"].shape == (24,)
        assert cfg.source_profiles == {"example_source.nc": "flat"}

    def test_renormalises_non_unit_mean(self, tmp_path: Path):
        cfg_path = tmp_path / "diurnal.yaml"
        # All values 2.0 → mean = 2.0; should renormalise to mean=1.0
        body = "[" + ", ".join(["2.0"] * 24) + "]"
        cfg_path.write_text(f"""
profiles:
  doubled:
    hours: {body}
""")
        with pytest.warns(UserWarning, match="mean"):
            cfg = DiurnalConfig.from_yaml(str(cfg_path))
        np.testing.assert_allclose(cfg.profiles["doubled"].mean(), 1.0, atol=1e-12)

    def test_rejects_wrong_length(self, tmp_path: Path):
        cfg_path = tmp_path / "diurnal.yaml"
        body = "[" + ", ".join(["1.0"] * 12) + "]"
        cfg_path.write_text(f"""
profiles:
  short:
    hours: {body}
""")
        with pytest.raises(ValueError, match="length 24"):
            DiurnalConfig.from_yaml(str(cfg_path))

    def test_profile_for_falls_back_to_flat(self):
        cfg = DiurnalConfig()
        prof = cfg.profile_for("/some/missing/file.nc", month=1)
        np.testing.assert_allclose(prof, 1.0)
        assert prof.shape == (24,)

    def test_profile_for_uses_basename_lookup(self, tmp_path: Path):
        cfg_path = tmp_path / "diurnal.yaml"
        body = "[" + ", ".join(["1.0"] * 24) + "]"
        cfg_path.write_text(f"""
profiles:
  flat:
    hours: {body}
source_profiles:
  bar.nc: flat
""")
        cfg = DiurnalConfig.from_yaml(str(cfg_path))
        # Absolute paths must still hit the lookup via basename.
        prof = cfg.profile_for("/abs/path/to/bar.nc", month=1)
        np.testing.assert_allclose(prof, 1.0)

    def test_profile_for_per_month_override(self, tmp_path: Path):
        cfg_path = tmp_path / "diurnal.yaml"
        body_flat = "[" + ", ".join(["1.0"] * 24) + "]"
        body_alt = "[" + ", ".join(["0.5", "1.5"] * 12) + "]"
        cfg_path.write_text(f"""
profiles:
  flat:
    hours: {body_flat}
  alt:
    hours: {body_alt}
source_profiles:
  foo.nc: flat
source_profiles_by_month:
  1:
    foo.nc: alt
""")
        cfg = DiurnalConfig.from_yaml(str(cfg_path))
        # January → alt; July → flat
        np.testing.assert_allclose(
            cfg.profile_for("foo.nc", month=1).mean(), 1.0, atol=1e-12
        )
        assert not np.allclose(cfg.profile_for("foo.nc", month=1), 1.0)
        np.testing.assert_allclose(cfg.profile_for("foo.nc", month=7), 1.0)


class TestBudgetConservation:
    """When every profile has day-mean = 1.0, the bin-mean of per-bin
    emissions equals the bin-flat result element-wise — preserves budget."""

    def test_bin_mean_equals_flat_with_unit_profiles(self):
        # Synthesise a "monthly emission slab" with random per-cell rates,
        # apply the loader's per-bin multiplier path, verify conservation.
        rng = np.random.default_rng(0)
        nz, ny, nx = 2, 3, 4
        N_ORBIT_SPECIES = 6
        emis_4d = rng.uniform(0.0, 5.0, size=(N_ORBIT_SPECIES, nz, ny, nx))

        # Mimic load_emissions_diurnal's per-bin multiply step
        prof = np.ones(24)
        factors = _local_to_utc_bin_factors(prof, 5.5, n_bins=8)

        e_flat = emis_4d.reshape(N_ORBIT_SPECIES, -1)  # (6, N)
        e_per_bin = np.stack(
            [factors[tau] * e_flat for tau in range(8)], axis=0
        )  # (8, 6, N)
        bin_mean = e_per_bin.mean(axis=0)
        np.testing.assert_allclose(bin_mean, e_flat, atol=1e-12)
