"""Diurnal profiles must survive the CEDS stack-height tier split.

`diurnal_sas.yaml` keys `source_profiles` on the pre-split basenames
(`ceds_pm25_anthro_2022_monthly.nc`). On 2026-05-04 the CEDS sector split
renamed the actual files to `..._surface.nc` / `_low` / `_medium` / `_high`,
and exact-basename lookup started returning bin-flat for every CEDS source
without warning. The Dec-Feb residential override -- roughly 10x
peak-to-trough for IGP winter heating -- was therefore never applied to any
production run, and nothing failed.

These tests pin the two properties that make that class of failure loud:
tier stems resolve, and an unmatched source warns.
"""

import warnings

import numpy as np
import pytest

from orbit.emissions.loader import DiurnalConfig, _strip_tier_suffix

TRAFFIC = np.array([0.4, 0.3, 0.3, 0.3, 0.4, 0.7, 1.2, 1.5, 1.6, 1.5, 1.2, 1.0,
                    1.0, 1.0, 1.0, 1.1, 1.3, 1.5, 1.6, 1.6, 1.4, 1.0, 0.7, 0.4])
RESID = np.array([0.3, 0.2, 0.2, 0.2, 0.3, 0.6, 1.4, 1.9, 1.8, 1.2, 0.8, 0.7,
                  0.7, 0.7, 0.8, 1.0, 1.4, 1.8, 2.0, 1.9, 1.5, 1.1, 0.8, 0.7])
INDUST = np.array([0.5, 0.5, 0.5, 0.5, 0.6, 0.7, 0.9, 1.1, 1.3, 1.4, 1.4, 1.3,
                   1.2, 1.2, 1.2, 1.2, 1.3, 1.4, 1.4, 1.3, 1.1, 0.9, 0.6, 0.5])
BASE = "ceds_pm25_anthro_2022_monthly.nc"


def _cfg():
    return DiurnalConfig(
        profiles={"anthro_traffic": TRAFFIC, "anthro_residential": RESID,
                  "anthro_industrial": INDUST, "flat": np.ones(24)},
        source_profiles={BASE: "anthro_traffic",
                         "ceds_pm25_anthro_2022_monthly_high.nc": "anthro_industrial",
                         "merra2_dust_pm25_2022_monthly.nc": "flat"},
        source_profiles_by_month={1: {BASE: "anthro_residential"}},
    )


@pytest.mark.parametrize("tier", ["surface", "low", "medium", "high"])
def test_tier_suffix_strips_to_the_stem(tier):
    assert _strip_tier_suffix(f"ceds_pm25_anthro_2022_monthly_{tier}.nc") == BASE


def test_untiered_names_are_untouched():
    for n in (BASE, "gfed5_pm25_bb_2022_monthly.nc", "a_highway.nc"):
        assert _strip_tier_suffix(n) == n


@pytest.mark.parametrize("tier", ["surface", "low"])
def test_tiers_inherit_the_stem_profile(tier):
    """The bug: these returned bin-flat instead of the traffic shape."""
    got = _cfg().profile_for(f"ceds_pm25_anthro_2022_monthly_{tier}.nc", month=6)
    np.testing.assert_array_equal(got, TRAFFIC)


@pytest.mark.parametrize("tier", ["surface", "low"])
def test_winter_override_reaches_the_tiers(tier):
    """The consequence that mattered: IGP winter heating, Dec-Feb."""
    got = _cfg().profile_for(f"ceds_pm25_anthro_2022_monthly_{tier}.nc", month=1)
    np.testing.assert_array_equal(got, RESID)


def test_exact_key_beats_the_stem_and_the_month_override():
    """Power and heavy industry keep an industrial shape in January."""
    got = _cfg().profile_for("ceds_pm25_anthro_2022_monthly_high.nc", month=1)
    np.testing.assert_array_equal(got, INDUST)


def test_unmatched_source_warns():
    with pytest.warns(UserWarning, match="No diurnal profile matched"):
        _cfg().profile_for("brand_new_inventory_2022.nc", month=1)


def test_explicit_flat_is_silent():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        got = _cfg().profile_for("merra2_dust_pm25_2022_monthly.nc", month=1)
    np.testing.assert_array_equal(got, np.ones(24))


def test_warning_fires_once_per_source():
    cfg = _cfg()
    with pytest.warns(UserWarning):
        cfg.profile_for("unmapped_thing.nc", month=1)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cfg.profile_for("unmapped_thing.nc", month=1)   # 8 bins x 6 iters


def test_resolve_all_reports_names_for_preflight():
    names = dict(_cfg().resolve_all(
        ["ceds_pm25_anthro_2022_monthly_surface.nc",
         "ceds_pm25_anthro_2022_monthly_high.nc",
         "nope.nc"], month=1))
    assert names["ceds_pm25_anthro_2022_monthly_surface.nc"] == "anthro_residential"
    assert names["ceds_pm25_anthro_2022_monthly_high.nc"] == "anthro_industrial"
    assert names["nope.nc"] is None


def test_shipped_config_maps_every_january_source():
    """Guards the real YAML against a future rename going quiet again."""
    from orbit.emissions.loader import DiurnalConfig as DC
    from orbit.cli import _default_diurnal_config_path
    cfg = DC.from_yaml(_default_diurnal_config_path())
    tiers = ["surface", "low", "medium", "high"]
    names = [f"ceds_{sp}_anthro_2022_monthly_{t}.nc"
             for sp in ("pm25", "poa", "bcoth", "nox", "so2") for t in tiers]
    unmapped = [b for b, n in cfg.resolve_all(names, 1) if n is None]
    assert not unmapped, f"unmapped in shipped config: {unmapped}"
