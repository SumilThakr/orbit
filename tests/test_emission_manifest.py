"""Tests for the emission manifest."""

import os

import pytest
import yaml

from orbit.emissions.manifest import (
    EmissionManifest,
    MissingEmissionsError,
    default_manifest_path,
    load_manifest,
)


def _write(tmp_path, body):
    p = tmp_path / "manifest.yaml"
    p.write_text(yaml.safe_dump(body, sort_keys=False))
    return str(p)


def _touch(tmp_path, *names):
    d = tmp_path / "emis"
    d.mkdir(exist_ok=True)
    for n in names:
        (d / n).write_text("")
    return str(d)


class TestShippedManifest:
    def test_default_manifest_exists_and_parses(self):
        m = load_manifest()
        assert m.name == "sas_2022"
        assert len(m.entries) == 30

    def test_ships_inside_the_package(self):
        """Must live under orbit/ so a pip install includes it."""
        p = default_manifest_path()
        assert os.path.exists(p)
        assert os.path.basename(os.path.dirname(p)) == "data"

    def test_voc_classes_are_assigned(self):
        """A VOC source with no class contributes nothing to SOA, so the
        shipped manifest must classify all three VOC inventories."""
        voc = load_manifest().voc_classes()
        assert voc == {
            "ceds_voc_anthro_2022_monthly.nc": "anthro",
            "gfed5_voc_bb_2022_monthly.nc": "biomass_burning",
            "cams_bio_voc_2022_monthly.nc": "bio_voc",
        }

    def test_stack_tiers_present(self):
        """Twelve elevated sources: pm25/so2/nox/poa x low/medium/high."""
        elevated = [e for e in load_manifest().entries if e.stack]
        assert len(elevated) == 12
        heights = sorted({e.stack["height"] for e in elevated})
        assert heights == [30.0, 50.0, 220.0]

    def test_soil_nox_carries_its_own_bins(self):
        e = next(x for x in load_manifest().entries
                 if x.file == "cams_soil_nox_climatology_diurnal.nc")
        assert e.bin_axis is True


class TestMissingFiles:
    def test_missing_required_raises(self, tmp_path):
        path = _write(tmp_path, {"name": "t", "sources": [{"file": "a.nc"}]})
        m = EmissionManifest.from_yaml(path)
        with pytest.raises(MissingEmissionsError, match="a.nc"):
            m.build_sources(str(tmp_path / "nope"), month=1)

    def test_allow_missing_overrides(self, tmp_path):
        path = _write(tmp_path, {"name": "t", "sources": [{"file": "a.nc"}]})
        m = EmissionManifest.from_yaml(path)
        sources, report = m.build_sources(str(tmp_path), 1, allow_missing=True)
        assert sources == []
        assert report.missing_required == ["a.nc"]

    def test_optional_missing_is_not_fatal(self, tmp_path):
        path = _write(tmp_path, {
            "name": "t",
            "sources": [{"file": "a.nc", "required": False}],
        })
        m = EmissionManifest.from_yaml(path)
        sources, report = m.build_sources(str(tmp_path), month=1)
        assert sources == []
        assert report.missing_optional == ["a.nc"]


class TestSchema:
    def test_stack_applies_plume_rise_params(self, tmp_path):
        d = _touch(tmp_path, "a.nc")
        path = _write(tmp_path, {"name": "t", "sources": [
            {"file": "a.nc",
             "stack": {"height": 220, "diameter": 6,
                       "temperature": 410, "velocity": 20}},
        ]})
        sources, _ = EmissionManifest.from_yaml(path).build_sources(d, month=1)
        s = sources[0]
        assert s.elevated is True
        assert (s.stack_height, s.stack_diam) == (220.0, 6.0)
        assert (s.stack_temp, s.stack_vel) == (410.0, 20.0)

    def test_time_index_month_resolves_to_month_minus_one(self, tmp_path):
        d = _touch(tmp_path, "a.nc")
        path = _write(tmp_path, {"name": "t", "sources": [{"file": "a.nc"}]})
        m = EmissionManifest.from_yaml(path)
        assert m.build_sources(d, month=7)[0][0].time_index == 6

    def test_explicit_integer_time_index(self, tmp_path):
        d = _touch(tmp_path, "a.nc")
        path = _write(tmp_path, {"name": "t", "sources": [
            {"file": "a.nc", "time_index": 0}]})
        m = EmissionManifest.from_yaml(path)
        assert m.build_sources(d, month=7)[0][0].time_index == 0

    def test_rejects_unknown_voc_class(self, tmp_path):
        path = _write(tmp_path, {"name": "t", "sources": [
            {"file": "a.nc", "voc_class": "not_a_class"}]})
        with pytest.raises(ValueError, match="unknown voc_class"):
            EmissionManifest.from_yaml(path)

    def test_rejects_partial_stack(self, tmp_path):
        path = _write(tmp_path, {"name": "t", "sources": [
            {"file": "a.nc", "stack": {"height": 30}}]})
        with pytest.raises(ValueError, match="missing"):
            EmissionManifest.from_yaml(path)

    def test_rejects_duplicate_source(self, tmp_path):
        path = _write(tmp_path, {"name": "t", "sources": [
            {"file": "a.nc"}, {"file": "a.nc"}]})
        with pytest.raises(ValueError, match="Duplicate"):
            EmissionManifest.from_yaml(path)

    def test_rejects_empty_manifest(self, tmp_path):
        path = _write(tmp_path, {"name": "t", "sources": []})
        with pytest.raises(ValueError, match="no sources"):
            EmissionManifest.from_yaml(path)

    def test_defaults_apply_to_sources(self, tmp_path):
        _touch(tmp_path, "a.nc")
        path = _write(tmp_path, {
            "name": "t",
            "defaults": {"units": "kg/s"},
            "sources": [{"file": "a.nc"}, {"file": "b.nc", "units": "ug/s"}],
        })
        m = EmissionManifest.from_yaml(path)
        assert m.entries[0].units == "kg/s"
        assert m.entries[1].units == "ug/s"


class TestContentHash:
    def _one(self, tmp_path, **extra):
        body = {"name": "t", "sources": [dict(file="a.nc", **extra)]}
        return EmissionManifest.from_yaml(_write(tmp_path, body))

    def test_stable_for_identical_content(self, tmp_path):
        a = self._one(tmp_path)
        b = self._one(tmp_path)
        assert a.content_hash() == b.content_hash()

    def test_changes_when_units_change(self, tmp_path):
        """Interpretation matters, not just filenames -- a baseline must not
        be reusable across manifests that read the same file differently."""
        a = self._one(tmp_path)
        b = self._one(tmp_path, units="kg/s")
        assert a.content_hash() != b.content_hash()

    def test_changes_when_stack_changes(self, tmp_path):
        a = self._one(tmp_path, stack={"height": 30, "diameter": 1.5,
                                       "temperature": 380, "velocity": 10})
        b = self._one(tmp_path, stack={"height": 220, "diameter": 6,
                                       "temperature": 410, "velocity": 20})
        assert a.content_hash() != b.content_hash()

    def test_changes_when_voc_class_changes(self, tmp_path):
        a = self._one(tmp_path, voc_class="anthro")
        b = self._one(tmp_path, voc_class="biomass_burning")
        assert a.content_hash() != b.content_hash()

    def test_insensitive_to_source_ordering(self, tmp_path):
        p1 = _write(tmp_path, {"name": "t", "sources": [
            {"file": "a.nc"}, {"file": "b.nc"}]})
        m1 = EmissionManifest.from_yaml(p1)
        p2 = _write(tmp_path, {"name": "t", "sources": [
            {"file": "b.nc"}, {"file": "a.nc"}]})
        m2 = EmissionManifest.from_yaml(p2)
        assert m1.content_hash() == m2.content_hash()
