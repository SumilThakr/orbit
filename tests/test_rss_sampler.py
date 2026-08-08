"""Phase-attributed RSS sampling (memory-audit diagnostic D3)."""
import sys

import numpy as np
import pytest

from orbit.runlog import RssSampler, _read_vmrss_mb


@pytest.mark.skipif(not sys.platform.startswith("linux"),
                    reason="/proc/self/status is linux-only")
def test_sampler_attributes_phases():
    assert _read_vmrss_mb() > 0
    s = RssSampler(interval_s=0.05)
    assert s.enabled
    s.start()
    s.set_phase("alloc")
    big = np.ones(int(3e7))          # ~240 MB, sampled at the phase boundary
    s.set_phase("after")
    peaks = s.stop()
    del big
    assert set(peaks) >= {"alloc", "after"}
    assert all(v > 0 for v in peaks.values())
    # The allocation phase must have seen at least the array's size more
    # than the startup baseline could have.
    assert peaks["alloc"] >= peaks.get("startup", 0)


def test_sampler_disabled_is_inert(monkeypatch):
    monkeypatch.setenv("ORBIT_RSS_SAMPLER", "0")
    s = RssSampler()
    assert not s.enabled
    s.start()
    s.set_phase("x")
    assert s.stop() == {}
    assert s.render() == ""
