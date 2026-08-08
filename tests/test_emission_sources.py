"""Unit tests for emission source configuration and constants."""

import numpy as np
from orbit.emissions.sources import (
    CONVERT_COLS,
    ELEMENT_CONVERSION,
    N_ORBIT_SPECIES,
    NH3_TO_N,
    NOx_TO_N,
    R_EARTH,
    LEGACY9_TO_ORBIT,
    SPECIES_MAP,
    SOx_TO_S,
    UNIT_CONVERSIONS,
    EmissionSource,
    compute_cell_areas,
    parse_emission_sources,
)


class TestSpeciesMapping:
    """Test legacy 9-species to ORBIT 6-species mapping."""

    def test_all_sigma_indices_mapped(self):
        """All 9 legacy indices have a ORBIT mapping."""
        for legacy_idx in range(9):
            assert legacy_idx in LEGACY9_TO_ORBIT, f"legacy index {legacy_idx} not mapped"

    def test_trace_indices_in_range(self):
        """All ORBIT indices are in [0, 5]."""
        for orbit_idx in LEGACY9_TO_ORBIT.values():
            assert 0 <= orbit_idx < N_ORBIT_SPECIES

    def test_org_merge(self):
        """VOC (0) and SOA (1) both map to SoA (0)."""
        assert LEGACY9_TO_ORBIT[0] == 0
        assert LEGACY9_TO_ORBIT[1] == 0

    def test_pm25_direct(self):
        """PM2.5 (2) maps to PM2.5 (1)."""
        assert LEGACY9_TO_ORBIT[2] == 1

    def test_nh_merge(self):
        """NH3 (3) and pNH4 (4) both map to Total NH (2)."""
        assert LEGACY9_TO_ORBIT[3] == 2
        assert LEGACY9_TO_ORBIT[4] == 2

    def test_sulfur_split(self):
        """SO2 (5) -> SO2 (3), pSO4 (6) -> pSO4 (4) — not merged."""
        assert LEGACY9_TO_ORBIT[5] == 3
        assert LEGACY9_TO_ORBIT[6] == 4

    def test_no_merge(self):
        """NOx (7) and pNO3 (8) both map to Total NO (5)."""
        assert LEGACY9_TO_ORBIT[7] == 5
        assert LEGACY9_TO_ORBIT[8] == 5


class TestElementConversion:
    """Test element conversion factors."""

    def test_nox_to_n(self):
        """NOx -> N conversion factor is MW_N / MW_NOx ~ 0.3045."""
        assert abs(NOx_TO_N - 0.30449) < 0.001

    def test_sox_to_s(self):
        """SO2 -> S conversion factor is MW_S / MW_SO2 ~ 0.5005."""
        assert abs(SOx_TO_S - 0.50052) < 0.001

    def test_nh3_to_n(self):
        """NH3 -> N conversion factor is MW_N / MW_NH3 ~ 0.8224."""
        assert abs(NH3_TO_N - 0.82245) < 0.001

    def test_element_conversion_indices(self):
        """Element conversion applies to correct legacy indices."""
        assert 3 in ELEMENT_CONVERSION  # NH3
        assert 5 in ELEMENT_CONVERSION  # SO2
        assert 7 in ELEMENT_CONVERSION  # NOx
        # Non-precursor indices should NOT have conversion
        for idx in [0, 1, 2, 4, 6, 8]:
            assert idx not in ELEMENT_CONVERSION

    def test_convert_cols(self):
        """CONVERT_COLS has the right column names."""
        assert 'nh3' in CONVERT_COLS
        assert 'sox' in CONVERT_COLS
        assert 'so2' in CONVERT_COLS
        assert 'nox' in CONVERT_COLS
        assert 'no2' in CONVERT_COLS
        # These should NOT trigger conversion
        assert 'pm25' not in CONVERT_COLS
        assert 'voc' not in CONVERT_COLS
        assert 'nh4' not in CONVERT_COLS


class TestColumnDetection:
    """Test species column name detection."""

    def test_standard_names(self):
        """Standard names map to correct legacy indices."""
        assert SPECIES_MAP['pm25'] == 2
        assert SPECIES_MAP['nox'] == 7
        assert SPECIES_MAP['so2'] == 5
        assert SPECIES_MAP['nh3'] == 3
        assert SPECIES_MAP['voc'] == 0

    def test_alternate_names(self):
        """Alternate names map to same indices."""
        assert SPECIES_MAP['pm2.5'] == 2
        assert SPECIES_MAP['pm2_5'] == 2
        assert SPECIES_MAP['sox'] == 5
        assert SPECIES_MAP['no2'] == 7


class TestUnitConversions:
    """Test unit conversion factors."""

    def test_kg_per_year(self):
        """1 kg/year ~ 31.71 ug/s."""
        factor = UNIT_CONVERSIONS['kg/year']
        expected = 1e9 / (365.0 * 24.0 * 3600.0)
        assert abs(factor - expected) < 1e-6

    def test_ug_per_s(self):
        """ug/s is identity."""
        assert UNIT_CONVERSIONS['ug/s'] == 1.0

    def test_kg_per_s(self):
        """kg/s -> 1e9 ug/s."""
        assert UNIT_CONVERSIONS['kg/s'] == 1e9

    def test_flux_units_are_sentinel(self):
        """Flux units have None sentinel (handled specially)."""
        assert UNIT_CONVERSIONS['kg/m2/s'] is None
        assert UNIT_CONVERSIONS['kg/m2'] is None


class TestCellAreas:
    """Test cell area computation."""

    def test_equator(self):
        """At equator, cos(lat) = 1, area = R^2 * dlon * dlat."""
        lat = np.array([0.0])
        area = compute_cell_areas(lat, 0.625, 0.5)
        expected = R_EARTH**2 * np.deg2rad(0.625) * np.deg2rad(0.5)
        assert abs(area[0] - expected) / expected < 1e-10

    def test_midlat(self):
        """At 45 deg, area should be cos(45) ~ 0.707 times equator area."""
        lat_eq = np.array([0.0])
        lat_45 = np.array([45.0])
        area_eq = compute_cell_areas(lat_eq, 0.625, 0.5)
        area_45 = compute_cell_areas(lat_45, 0.625, 0.5)
        ratio = area_45[0] / area_eq[0]
        assert abs(ratio - np.cos(np.deg2rad(45.0))) < 1e-10

    def test_pole(self):
        """At 90 deg, cos(90) ~ 0, area should be near zero."""
        lat = np.array([90.0])
        area = compute_cell_areas(lat, 0.625, 0.5)
        assert area[0] < 1.0  # effectively zero

    def test_shape(self):
        """Output shape matches input lat array."""
        lat = np.array([0.0, 10.0, 20.0, 30.0])
        area = compute_cell_areas(lat, 0.625, 0.5)
        assert area.shape == (4,)


class TestEmissionSource:
    """Test EmissionSource dataclass."""

    def test_defaults(self):
        """Default values are sensible."""
        s = EmissionSource()
        assert s.format == "netcdf"
        assert s.units == "kg/m2/s"
        assert s.elevated is False
        assert s.time_index is None

    def test_explicit_values(self):
        """Values can be set explicitly."""
        s = EmissionSource(
            path="/some/path.nc",
            format="netcdf",
            units="kg/m2/s",
            time_index=5,
            variable_mapping={"pm25": "my_pm25"},
        )
        assert s.path == "/some/path.nc"
        assert s.time_index == 5
        assert s.variable_mapping == {"pm25": "my_pm25"}


class TestParseEmissionSources:
    """Test YAML parsing of emission sources."""

    def test_string_entries(self, tmp_path):
        """String entries become EmissionSource with path resolved."""
        yaml_list = ["file1.nc", "file2.nc"]
        sources = parse_emission_sources(yaml_list, str(tmp_path))
        assert len(sources) == 2
        assert sources[0].path == str(tmp_path / "file1.nc")

    def test_dict_entries(self, tmp_path):
        """Dict entries become EmissionSource with all fields."""
        yaml_list = [{
            "path": "emis.nc",
            "format": "netcdf",
            "units": "kg/m2/s",
            "time_index": 3,
        }]
        sources = parse_emission_sources(yaml_list, str(tmp_path))
        assert len(sources) == 1
        assert sources[0].time_index == 3
        assert sources[0].units == "kg/m2/s"

    def test_absolute_paths_unchanged(self, tmp_path):
        """Absolute paths are not resolved relative to config_dir."""
        yaml_list = [{"path": "/absolute/path.nc"}]
        sources = parse_emission_sources(yaml_list, str(tmp_path))
        assert sources[0].path == "/absolute/path.nc"
