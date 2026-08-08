"""Tests for run logging, fingerprinting and warnings."""

import json
import sys

import pytest

from orbit.runlog import (
    RunRecord,
    full_fingerprint,
    quick_fingerprint,
    unique_path,
)


class TestFingerprint:
    def _write(self, tmp_path, name, data):
        p = tmp_path / name
        p.write_bytes(data)
        return str(p)

    def test_quick_detects_different_content(self, tmp_path):
        a = self._write(tmp_path, "a", b"x" * 4096)
        b = self._write(tmp_path, "b", b"y" * 4096)
        assert quick_fingerprint(a)["digest"] != quick_fingerprint(b)["digest"]

    def test_quick_detects_truncation(self, tmp_path):
        """The realistic failure mode: a half-downloaded file."""
        a = self._write(tmp_path, "a", b"z" * 10000)
        b = self._write(tmp_path, "b", b"z" * 5000)
        assert quick_fingerprint(a)["digest"] != quick_fingerprint(b)["digest"]

    def test_quick_is_stable_for_identical_content(self, tmp_path):
        a = self._write(tmp_path, "a", b"q" * 9999)
        b = self._write(tmp_path, "b", b"q" * 9999)
        assert quick_fingerprint(a)["digest"] == quick_fingerprint(b)["digest"]

    def test_quick_misses_middle_edit_but_full_catches_it(self, tmp_path):
        """Documents the tradeoff the flag exists for."""
        data = bytearray(b"m" * (4 << 20))
        a = self._write(tmp_path, "a", bytes(data))
        data[2 << 20] = ord("X")           # edit the middle only
        b = self._write(tmp_path, "b", bytes(data))
        assert quick_fingerprint(a)["digest"] == quick_fingerprint(b)["digest"]
        assert full_fingerprint(a)["digest"] != full_fingerprint(b)["digest"]

    def test_methods_are_labelled(self, tmp_path):
        a = self._write(tmp_path, "a", b"1234")
        assert quick_fingerprint(a)["method"] == "quick"
        assert full_fingerprint(a)["method"] == "full"

    def test_missing_file_is_reported_not_raised(self, tmp_path):
        from orbit.runlog import fingerprint
        rec = fingerprint(str(tmp_path / "nope"))
        assert rec["method"] == "unavailable"
        assert "error" in rec


class TestNeverClobber:
    def test_unique_path_increments(self, tmp_path):
        p = tmp_path / "orbit_M01.log"
        assert unique_path(str(p)) == str(p)
        p.write_text("first run")
        assert unique_path(str(p)) == str(p) + ".2"
        (tmp_path / "orbit_M01.log.2").write_text("second")
        assert unique_path(str(p)) == str(p) + ".3"

    def test_attach_does_not_overwrite(self, tmp_path):
        p = str(tmp_path / "run.log")
        rec = RunRecord(command=["orbit"])
        with rec.attach(p):
            print("first")
        with rec.attach(p):
            print("second")
        assert "first" in open(p).read()
        assert "second" in open(p + ".2").read()


class TestTee:
    def test_output_goes_to_both(self, tmp_path, capsys):
        p = str(tmp_path / "run.log")
        rec = RunRecord(command=["orbit"])
        with rec.attach(p):
            print("hello log")
        assert "hello log" in open(p).read()
        assert "hello log" in capsys.readouterr().out

    def test_stdout_restored_after(self, tmp_path):
        before = sys.stdout
        rec = RunRecord(command=["orbit"])
        with rec.attach(str(tmp_path / "r.log")):
            pass
        assert sys.stdout is before

    def test_stdout_restored_on_exception(self, tmp_path):
        before = sys.stdout
        rec = RunRecord(command=["orbit"])
        with pytest.raises(RuntimeError):
            with rec.attach(str(tmp_path / "r.log")):
                raise RuntimeError("boom")
        assert sys.stdout is before

    def test_partial_log_survives_a_crash(self, tmp_path):
        """A job killed mid-run must still leave a usable record."""
        p = str(tmp_path / "r.log")
        rec = RunRecord(command=["orbit"])
        with pytest.raises(RuntimeError):
            with rec.attach(p):
                print("got this far")
                raise RuntimeError("killed")
        assert "got this far" in open(p).read()

    def test_none_path_is_a_noop(self, capsys):
        rec = RunRecord(command=["orbit"])
        with rec.attach(None) as written:
            print("terminal only")
        assert written is None
        assert "terminal only" in capsys.readouterr().out


class TestWarnings:
    def test_warning_records_impact_and_fix(self, capsys):
        rec = RunRecord(command=["orbit"])
        rec.warn("solver", "UMFPACK missing", impact="slower",
                 fix="conda install scikit-umfpack", silence="set X=y")
        out = capsys.readouterr().out
        assert "UMFPACK missing" in out
        assert "Impact:" in out and "Fix:" in out and "Silence:" in out

    def test_warnings_repeat_in_footer(self):
        """A warning at minute 2 of a 50-minute run must not be lost."""
        rec = RunRecord(command=["orbit"])
        rec.warn("solver", "UMFPACK missing", fix="install it", echo=False)
        assert "UMFPACK missing" in rec.footer()
        assert "install it" in rec.footer()

    def test_no_warnings_says_none(self):
        rec = RunRecord(command=["orbit"])
        assert "none" in rec.warnings_block()


class TestConfigSplit:
    def test_results_and_performance_are_separated(self):
        rec = RunRecord(command=["orbit"])
        rec.set_config("horizontal transport", "FCT", True, "--horizontal-fct")
        rec.set_config("species threads", 4, False, "ORBIT_SPECIES_THREADS")
        block = rec.config_block()
        assert "affects results" in block and "affects performance only" in block
        assert rec.config_results["horizontal transport"]["value"] == "FCT"
        assert "species threads" in rec.config_perf

    def test_default_vs_overridden_is_marked(self):
        rec = RunRecord(command=["orbit"])
        rec.set_config("VBS k_age", "4.0e-11", True)
        rec.set_config("VBS frag", "0.9", True, "ORBIT_VBS_FRAG")
        block = rec.config_block()
        assert "[default]" in block
        assert "[ORBIT_VBS_FRAG]" in block


class TestSidecar:
    def test_json_round_trips(self, tmp_path):
        rec = RunRecord(command=["orbit", "--month", "1"])
        rec.set_config("chemistry iterations", 0, True)
        rec.warn("solver", "test", echo=False)
        rec.emissions["by_species_tg_yr"] = {"TotalNH": 9.68}
        rec.status = "completed"
        p = str(tmp_path / "run.json")
        rec.write_json(p, "1.0.0")
        d = json.load(open(p))
        assert d["status"] == "completed"
        assert d["orbit_version"] == "1.0.0"
        assert d["config"]["results_affecting"]["chemistry iterations"]["value"] == "0"
        assert d["emissions"]["by_species_tg_yr"]["TotalNH"] == pytest.approx(9.68)
        assert len(d["warnings"]) == 1

    def test_run_id_is_stable_within_a_record(self):
        rec = RunRecord(command=["orbit"])
        assert rec.run_id() == rec.run_id()

    def test_header_carries_provenance(self):
        rec = RunRecord(command=["orbit", "--month", "1"])
        h = rec.header("1.0.0")
        for token in ("ORBIT 1.0.0", "command", "code", "host", "python", "run id"):
            assert token in h
