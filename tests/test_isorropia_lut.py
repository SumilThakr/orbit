"""Tests for the ISORROPIA 7D LUT loader (orbit.core.isorropia_lut).

Marked ``slow`` because loading the full 7D LUT (~1 GB) causes OOM kills on
login nodes with limited memory.  Run with ``pytest -m slow`` explicitly, or
on a compute node via SLURM.

TODO: refactor to test on a small synthetic sub-LUT so these can rejoin the
default suite.
"""

import numpy as np
import pytest
import os

from orbit.core.isorropia_lut import IsorropiaLUT

pytestmark = pytest.mark.slow


# --- Grid spec matching generate_lut.py ---
SO4_AXIS = np.logspace(np.log10(0.001), np.log10(80.0), 25)
NH_AXIS = np.logspace(np.log10(0.001), np.log10(100.0), 25)
NO3_AXIS = np.logspace(np.log10(0.001), np.log10(100.0), 25)
CA_AXIS = np.array([0.0, 0.1, 0.3, 1.0, 2.5, 5.0])
NA_AXIS = np.array([0.0, 0.1, 0.3, 1.0, 2.5, 5.0])
T_AXIS = np.linspace(200.0, 320.0, 20)
RH_AXIS = np.linspace(0.01, 0.99, 20)
SHAPE = (25, 25, 25, 6, 6, 20, 20)

LUT_PATH = "/path/to/data/preproc/output/LUT/isorropia_lut_7d.npz"


def _make_synthetic_lut(tmpdir):
    """Create a small synthetic LUT for unit tests (no HETP dependency)."""
    # Use tiny grids so tests are fast
    so4 = np.array([0.1, 1.0, 10.0])
    nh = np.array([0.1, 1.0, 10.0])
    no3 = np.array([0.1, 1.0, 10.0])
    ca = np.array([0.0, 1.0])
    na = np.array([0.0, 1.0])
    t = np.array([250.0, 300.0])
    rh = np.array([0.3, 0.8])
    shape = (3, 3, 3, 2, 2, 2, 2)

    rng = np.random.default_rng(123)
    path = os.path.join(str(tmpdir), "test_lut.npz")
    np.savez_compressed(
        path,
        f_nh4=rng.uniform(0, 1, shape).astype(np.float32),
        f_no3=rng.uniform(0, 1, shape).astype(np.float32),
        aerosol_water=rng.uniform(0, 100, shape).astype(np.float32),
        ph=rng.uniform(0, 8, shape).astype(np.float32),
        so4_axis=so4, nh_axis=nh, no3_axis=no3,
        ca_axis=ca, na_axis=na, t_axis=t, rh_axis=rh,
    )
    return path


_ALL_FIELDS = ("f_nh4", "f_no3", "aerosol_water", "ph")


@pytest.fixture
def synthetic_lut(tmp_path):
    """Fixture providing a small synthetic LUT (all 4 fields, to exercise aw/ph)."""
    path = _make_synthetic_lut(tmp_path)
    return IsorropiaLUT(path, fields=_ALL_FIELDS)


@pytest.fixture
def real_lut():
    """Fixture loading the real 7D LUT (all 4 fields; skips if not available)."""
    if not os.path.exists(LUT_PATH):
        pytest.skip(f"Real LUT not found at {LUT_PATH}")
    return IsorropiaLUT(LUT_PATH, fields=_ALL_FIELDS)


class TestIsorropiaLUTSynthetic:
    """Unit tests using a small synthetic LUT (no external dependencies)."""

    def test_load(self, synthetic_lut):
        """LUT loads without error and has expected interpolators (all 4 requested)."""
        assert not synthetic_lut.loaded          # read on first query, not construction
        synthetic_lut._ensure_loaded()
        assert synthetic_lut.loaded
        assert len(synthetic_lut._interp) == 4
        assert set(synthetic_lut._interp.keys()) == {"f_nh4", "f_no3", "aerosol_water", "ph"}

    def test_default_fields_are_lean(self, tmp_path):
        """Default load is the two production fields only; aw/ph come back NaN."""
        path = _make_synthetic_lut(tmp_path)
        lut = IsorropiaLUT(path)  # default fields
        lut._ensure_loaded()
        assert set(lut._interp.keys()) == {"f_nh4", "f_no3"}
        f_nh4, f_no3, aw, ph = lut.query(1.0, 1.0, 1.0, 0.0, 0.0, 275.0, 0.5)
        assert np.all(np.isfinite(f_nh4)) and np.all(np.isfinite(f_no3))
        assert np.all(np.isnan(aw)) and np.all(np.isnan(ph))

    def test_release_and_reload(self, synthetic_lut):
        """release() drops the table; the next query reloads it and answers the same."""
        args = (1.0, 1.0, 1.0, 0.0, 0.0, 275.0, 0.5)
        first = synthetic_lut.query(*args)
        assert synthetic_lut.loaded and synthetic_lut.n_loads == 1
        synthetic_lut.release()
        assert not synthetic_lut.loaded
        second = synthetic_lut.query(*args)
        assert synthetic_lut.n_loads == 2
        for a, b in zip(first, second):
            np.testing.assert_array_equal(a, b)

    def test_disk_cache_matches_archive(self, tmp_path, monkeypatch):
        """The decompressed cache is written on the first load and read on the
        next; both give identical answers."""
        path = _make_synthetic_lut(tmp_path)
        cache_root = tmp_path / "lut_cache"
        monkeypatch.setenv("ORBIT_LUT_CACHE_DIR", str(cache_root))
        args = (0.7, 1.3, 0.9, 0.0, 0.0, 280.0, 0.6)
        lut = IsorropiaLUT(path)
        from_archive = lut.query(*args)
        assert (cache_root / lut._cache_dir.split("/")[-1] / "complete").exists()
        lut2 = IsorropiaLUT(path)
        from_cache = lut2.query(*args)
        for a, b in zip(from_archive, from_cache):
            np.testing.assert_array_equal(a, b)

    def test_scalar_query(self, synthetic_lut):
        """Query with scalar inputs returns length-1 arrays."""
        f_nh4, f_no3, aw, ph = synthetic_lut.query(1.0, 1.0, 1.0, 0.0, 0.0, 275.0, 0.5)
        assert f_nh4.shape == (1,)
        assert f_no3.shape == (1,)

    def test_array_query(self, synthetic_lut):
        """Query with array inputs returns matching shape."""
        n = 50
        so4 = np.full(n, 1.0)
        nh = np.full(n, 1.0)
        no3 = np.full(n, 1.0)
        ca = np.zeros(n)
        na = np.zeros(n)
        t = np.full(n, 275.0)
        rh = np.full(n, 0.5)
        f_nh4, f_no3, aw, ph = synthetic_lut.query(so4, nh, no3, ca, na, t, rh)
        assert f_nh4.shape == (n,)

    def test_extrapolation_no_error(self, synthetic_lut):
        """Out-of-bounds queries should not raise (nearest extrapolation)."""
        f_nh4, f_no3, aw, ph = synthetic_lut.query(
            0.001, 0.001, 0.001, 0.0, 0.0, 200.0, 0.01
        )
        assert np.all(np.isfinite(f_nh4))

    def test_deterministic(self, synthetic_lut):
        """Same query returns same result."""
        args = (5.0, 5.0, 5.0, 0.5, 0.5, 280.0, 0.6)
        r1 = synthetic_lut.query(*args)
        r2 = synthetic_lut.query(*args)
        for a, b in zip(r1, r2):
            np.testing.assert_array_equal(a, b)

    def test_oob_outputs_clamped(self, synthetic_lut):
        """Out-of-bounds queries should return clamped outputs."""
        # Query far beyond the grid edges to trigger linear extrapolation
        f_nh4, f_no3, aw, ph = synthetic_lut.query(
            1000.0, 1000.0, 1000.0, 100.0, 100.0, 400.0, 1.5
        )
        assert np.all(f_nh4 >= 0.0) and np.all(f_nh4 <= 1.0), (
            f"f_nh4 out of [0, 1]: {f_nh4}"
        )
        assert np.all(f_no3 >= 0.0) and np.all(f_no3 <= 1.0), (
            f"f_no3 out of [0, 1]: {f_no3}"
        )
        assert np.all(aw >= 0.0), f"aerosol_water negative: {aw}"

        # Also test the low end
        f_nh4, f_no3, aw, ph = synthetic_lut.query(
            0.0001, 0.0001, 0.0001, 0.0, 0.0, 100.0, -0.5
        )
        assert np.all(f_nh4 >= 0.0) and np.all(f_nh4 <= 1.0)
        assert np.all(f_no3 >= 0.0) and np.all(f_no3 <= 1.0)
        assert np.all(aw >= 0.0)


class TestIsorropiaLUTReal:
    """Integration tests against the full 7D LUT (skip if not generated yet)."""

    def test_shape(self, real_lut):
        """Axes match expected grid dimensions."""
        assert len(real_lut.axes) == 7
        assert len(real_lut.axes[0]) == 25   # SO4
        assert len(real_lut.axes[1]) == 25   # NH
        assert len(real_lut.axes[2]) == 25   # NO3
        assert len(real_lut.axes[3]) == 6    # Ca
        assert len(real_lut.axes[4]) == 6    # Na
        assert len(real_lut.axes[5]) == 20   # T
        assert len(real_lut.axes[6]) == 20   # RH

    def test_fractions_bounded(self, real_lut):
        """Partitioning fractions should be in [0, 1] at grid nodes."""
        # Sample grid-node queries (no interpolation error)
        so4 = real_lut.axes[0][[0, 5, 12, -1]]
        nh = real_lut.axes[1][[0, 5, 12, -1]]
        no3 = real_lut.axes[2][[0, 5, 12, -1]]
        ca = np.zeros(4)
        na = np.zeros(4)
        t = np.full(4, 298.0)
        rh = np.full(4, 0.50)
        f_nh4, f_no3, aw, ph = real_lut.query(so4, nh, no3, ca, na, t, rh)
        assert np.all(f_nh4 >= 0) and np.all(f_nh4 <= 1)
        assert np.all(f_no3 >= 0) and np.all(f_no3 <= 1)
        assert np.all(aw >= 0)

    def test_no3_decreases_with_temperature(self, real_lut):
        """f_NO3 should generally decrease as temperature rises (NH4NO3 volatility)."""
        n = len(real_lut.axes[5])
        so4 = np.full(n, 5.0)    # moderate SO4
        nh = np.full(n, 10.0)    # NH3-rich
        no3 = np.full(n, 10.0)   # abundant NO3
        ca = np.zeros(n)
        na = np.zeros(n)
        t = real_lut.axes[5]     # sweep temperature
        rh = np.full(n, 0.70)
        f_nh4, f_no3, aw, ph = real_lut.query(so4, nh, no3, ca, na, t, rh)
        # At least 70% of consecutive T steps should show f_NO3 decreasing
        decreasing = np.sum(np.diff(f_no3) < 0)
        assert decreasing >= 0.7 * (n - 1), (
            f"f_NO3 vs T: only {decreasing}/{n-1} decreasing steps"
        )

    def test_no3_increases_with_rh(self, real_lut):
        """f_NO3 should generally increase with RH (aqueous nitrate favored)."""
        n = len(real_lut.axes[6])
        so4 = np.full(n, 5.0)
        nh = np.full(n, 10.0)
        no3 = np.full(n, 10.0)
        ca = np.zeros(n)
        na = np.zeros(n)
        t = np.full(n, 298.0)
        rh = real_lut.axes[6]
        f_nh4, f_no3, aw, ph = real_lut.query(so4, nh, no3, ca, na, t, rh)
        increasing = np.sum(np.diff(f_no3) > 0)
        assert increasing >= 0.7 * (n - 1), (
            f"f_NO3 vs RH: only {increasing}/{n-1} increasing steps"
        )

    def test_crustal_shifts_partitioning(self, real_lut):
        """Adding crustal Ca should shift sulfate ratio, affecting NH4 partitioning."""
        n = len(real_lut.axes[3])
        so4 = np.full(n, 5.0)
        nh = np.full(n, 5.0)
        no3 = np.full(n, 5.0)
        ca = real_lut.axes[3]   # sweep Ca
        na = np.zeros(n)
        t = np.full(n, 298.0)
        rh = np.full(n, 0.70)
        f_nh4, f_no3, aw, ph = real_lut.query(so4, nh, no3, ca, na, t, rh)
        # Ca claims SO4 as CaSO4, freeing NH3 -> f_nh4 should decrease at high Ca
        # At minimum, there should be a measurable difference between Ca=0 and Ca=5
        assert f_nh4[0] != f_nh4[-1], "Ca has no effect on f_nh4"

    def test_zero_ca_na_consistency(self, real_lut):
        """Ca=0, Na=0 should give finite, physically reasonable results."""
        f_nh4, f_no3, aw, ph = real_lut.query(5.0, 10.0, 5.0, 0.0, 0.0, 298.0, 0.70)
        assert np.all(np.isfinite(f_nh4))
        assert np.all(np.isfinite(f_no3))
        assert 0 < f_nh4[0] < 1
        assert 0 < f_no3[0] < 1

    def test_water_increases_with_rh(self, real_lut):
        """Aerosol water should increase with RH."""
        n = len(real_lut.axes[6])
        so4 = np.full(n, 10.0)
        nh = np.full(n, 10.0)
        no3 = np.full(n, 10.0)
        ca = np.zeros(n)
        na = np.zeros(n)
        t = np.full(n, 298.0)
        rh = real_lut.axes[6]
        _, _, aw, _ = real_lut.query(so4, nh, no3, ca, na, t, rh)
        increasing = np.sum(np.diff(aw) > 0)
        assert increasing >= 0.8 * (n - 1)

    def test_query_speed(self, real_lut):
        """10K queries should complete in < 1 second."""
        import time
        n = 10000
        rng = np.random.default_rng(99)
        so4 = rng.uniform(0.01, 50, n)
        nh = rng.uniform(0.01, 50, n)
        no3 = rng.uniform(0.01, 50, n)
        ca = rng.uniform(0, 3, n)
        na = rng.uniform(0, 3, n)
        t = rng.uniform(220, 310, n)
        rh = rng.uniform(0.1, 0.9, n)
        t0 = time.time()
        real_lut.query(so4, nh, no3, ca, na, t, rh)
        elapsed = time.time() - t0
        assert elapsed < 1.0, f"10K queries took {elapsed:.2f}s (target < 1s)"
