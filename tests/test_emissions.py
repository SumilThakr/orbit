"""Tests for emission loading."""

import numpy as np
import pytest
import os
from datetime import date
from tests.conftest import (
    SOLAR_MAP_PATH, WEEKLY_SUBSET_BASE,
)


EMISSION_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "evaluation", "emissions", "regions", "sas"
)

WEEKLY_EMISSION_DIR = (
    "/path/to/data/preproc/emissions/emissions/regions/sas"
)

# 17 SAS emission files
_EMISSION_FILES = [
    "ceds_voc_anthro_2022_monthly.nc",
    "gfed5_voc_bb_2022_monthly.nc",
    "cams_bio_voc_2022_monthly.nc",
    "ceds_pm25_anthro_2022_monthly.nc",
    "gfed5_pm25_bb_2022_monthly.nc",
    "merra2_dust_pm25_2022_monthly.nc",
    "merra2_seasalt_pm25_2022_monthly.nc",
    "ceds_nh3_anthro_2022_monthly.nc",
    "gfed5_nh3_bb_2022_monthly.nc",
    "geia_nh3_natural_2022_monthly.nc",
    "ceds_so2_anthro_2022_monthly.nc",
    "gfed5_so2_bb_2022_monthly.nc",
    "carn_volcanic_so2_2022_monthly.nc",
    "cams_oce_dms_as_so2_climatology_monthly.nc",
    "ceds_nox_anthro_2022_monthly.nc",
    "gfed5_nox_bb_2022_monthly.nc",
    "cams_soil_nox_climatology_monthly.nc",
]


def _build_emission_sources(month=None):
    """Build EmissionSource list, optionally with a specific month's time_index."""
    from orbit.emissions.sources import EmissionSource

    sources = []
    for filename in _EMISSION_FILES:
        filepath = os.path.join(EMISSION_DIR, filename)
        if os.path.exists(filepath):
            sources.append(EmissionSource(
                path=filepath,
                format="netcdf",
                units="kg/m2/s",
                time_index=month - 1 if month is not None else None,
            ))
    return sources


def _build_weekly_emission_sources():
    """Build EmissionSource list for weekly loading (no time_index)."""
    from orbit.emissions.sources import EmissionSource

    sources = []
    for filename in _EMISSION_FILES:
        filepath = os.path.join(WEEKLY_EMISSION_DIR, filename)
        if os.path.exists(filepath):
            sources.append(EmissionSource(
                path=filepath,
                format="netcdf",
                units="kg/m2/s",
            ))
    return sources


# ── Unit tests (no data needed) ─────────────────────────────────────────


class TestIsoWeeks:
    def test_2016_has_53_weeks(self):
        from orbit.emissions.loader import _generate_iso_weeks

        weeks = _generate_iso_weeks(2016)
        assert len(weeks) == 53

    def test_2016_w01_partial(self):
        """2016 starts on Friday, so the first entry is Jan 1-3.

        The ISO week number is 53 (belonging to ISO year 2015), since
        these days fall in ISO week 53 of 2015. Entries are always in
        chronological order regardless of week number.
        """
        from orbit.emissions.loader import _generate_iso_weeks

        weeks = _generate_iso_weeks(2016)
        start, end_excl, wnum = weeks[0]
        assert start == date(2016, 1, 1)
        assert end_excl == date(2016, 1, 4)  # Monday
        assert wnum == 53  # ISO week 53 of 2015

    def test_2016_full_coverage(self):
        """All days of 2016 are covered exactly once."""
        from orbit.emissions.loader import _generate_iso_weeks
        from datetime import timedelta

        weeks = _generate_iso_weeks(2016)
        all_days = set()
        for start, end_excl, _ in weeks:
            day = start
            while day < end_excl:
                assert day not in all_days, f"Duplicate day: {day}"
                all_days.add(day)
                day += timedelta(days=1)

        # 2016 is a leap year: 366 days
        assert len(all_days) == 366
        assert date(2016, 1, 1) in all_days
        assert date(2016, 12, 31) in all_days

    def test_2022_has_53_entries(self):
        """2022 starts Saturday, so partial first week -> 53 entries."""
        from orbit.emissions.loader import _generate_iso_weeks

        weeks = _generate_iso_weeks(2022)
        assert len(weeks) == 53
        # First entry is partial: Jan 1 (Sat) - Jan 2 (Sun)
        start, end_excl, _ = weeks[0]
        assert start == date(2022, 1, 1)
        assert end_excl == date(2022, 1, 3)


class TestMonthWeights:
    def test_single_month(self):
        """A week entirely within one month has weight 1.0 for that month."""
        from orbit.emissions.loader import _get_month_weights

        weights = _get_month_weights(date(2016, 1, 18), date(2016, 1, 25))
        assert weights == {1: 1.0}

    def test_month_boundary(self):
        """A week crossing Jan/Feb splits weights correctly."""
        from orbit.emissions.loader import _get_month_weights

        # Jan 28 - Feb 3 (7 days: 4 in Jan, 3 in Feb)
        weights = _get_month_weights(date(2016, 1, 28), date(2016, 2, 4))
        assert len(weights) == 2
        assert abs(weights[1] - 4 / 7) < 1e-10
        assert abs(weights[2] - 3 / 7) < 1e-10

    def test_weights_sum_to_one(self):
        """Weights always sum to 1.0."""
        from orbit.emissions.loader import _get_month_weights

        ranges = [
            (date(2016, 1, 1), date(2016, 1, 4)),    # partial week
            (date(2016, 6, 27), date(2016, 7, 4)),    # Jun/Jul boundary
            (date(2016, 12, 26), date(2017, 1, 1)),   # partial last week
        ]
        for start, end in ranges:
            weights = _get_month_weights(start, end)
            assert abs(sum(weights.values()) - 1.0) < 1e-10, \
                f"Weights don't sum to 1 for {start}-{end}: {weights}"


# ── Integration tests (need real data) ───────────────────────────────────


@pytest.mark.slow
class TestEmissions:
    def test_load_january(self, january_preprocessor_path, constants_path):
        if not os.path.exists(january_preprocessor_path):
            pytest.skip("Preprocessor file not found")
        if not os.path.exists(EMISSION_DIR):
            pytest.skip("Emission directory not found")

        from orbit.core.grid_data import load_grid
        from orbit.core.indexing import CellIndexer
        from orbit.emissions.loader import load_emissions

        g = load_grid(january_preprocessor_path, constants_path)
        idx = CellIndexer(g.nz, g.ny, g.nx)

        sources = _build_emission_sources(month=1)
        e = load_emissions(sources, g, idx)
        assert e.shape == (6 * idx.N,)

        # Should have some nonzero emissions
        assert np.any(e > 0)

        # pSO4 (species 4) should have zero direct emissions
        N = idx.N
        assert np.all(e[4 * N:5 * N] == 0)

        # PM2.5 (species 1) should have positive emissions somewhere
        assert np.any(e[1 * N:2 * N] > 0)


@pytest.mark.slow
class TestEmissionsWeekly:
    def _skip_if_missing(self):
        if not os.path.exists(SOLAR_MAP_PATH):
            pytest.skip("Solar map not found")
        subset = os.path.join(WEEKLY_SUBSET_BASE, "sas_2016_W02_day.nc")
        if not os.path.exists(subset):
            pytest.skip("Weekly subset file not found")
        if not os.path.exists(WEEKLY_EMISSION_DIR):
            pytest.skip("Weekly emission directory not found")

    def test_load_weekly_day(self, constants_path):
        self._skip_if_missing()

        from orbit.core.grid_data import load_grid
        from orbit.core.indexing import CellIndexer
        from orbit.emissions.loader import load_emissions_weekly

        subset = os.path.join(WEEKLY_SUBSET_BASE, "sas_2016_W02_day.nc")
        g = load_grid(os.path.abspath(subset), os.path.abspath(constants_path))
        idx = CellIndexer(g.nz, g.ny, g.nx)
        N = idx.N

        sources = _build_weekly_emission_sources()
        e = load_emissions_weekly(
            sources, g, idx,
            week=2, year=2016,
            solar_map_path=SOLAR_MAP_PATH,
            time_bin="day",
        )

        assert e.shape == (6 * N,)
        assert np.any(e > 0)
        # pSO4 (species 4) should be zero
        assert np.all(e[4 * N : 5 * N] == 0)

    def test_day_night_sum(self, constants_path):
        """e_day + e_night should reconstruct unsplit emission rate."""
        self._skip_if_missing()

        from orbit.core.grid_data import load_grid
        from orbit.core.indexing import CellIndexer
        from orbit.emissions.loader import load_emissions_weekly, load_emissions

        subset = os.path.join(WEEKLY_SUBSET_BASE, "sas_2016_W02_day.nc")
        g = load_grid(os.path.abspath(subset), os.path.abspath(constants_path))
        idx = CellIndexer(g.nz, g.ny, g.nx)

        sources = _build_weekly_emission_sources()
        e_day = load_emissions_weekly(
            sources, g, idx,
            week=2, year=2016,
            solar_map_path=SOLAR_MAP_PATH,
            time_bin="day",
        )
        e_night = load_emissions_weekly(
            sources, g, idx,
            week=2, year=2016,
            solar_map_path=SOLAR_MAP_PATH,
            time_bin="night",
        )

        e_sum = e_day + e_night

        # Reconstruct expected unsplit rate via load_emissions with
        # month-weighted sources (week 2 of 2016 is entirely in January)
        jan_sources = _build_weekly_emission_sources()
        for s in jan_sources:
            s.time_index = 0  # January
        e_full = load_emissions(jan_sources, g, idx)

        np.testing.assert_allclose(e_sum, e_full, rtol=1e-12)
