"""Tests for deposition flux maps."""

import numpy as np
import pytest

from orbit.core.deposition_maps import (
    UG_M2_S_TO_KG_HA_YR,
    compute_deposition_maps,
    summarise,
)
from orbit.core.deposition import N_SPECIES, IDX_PM25
from orbit.core.indexing import CellIndexer


class _Grid:
    """Minimal grid stub exposing what the deposition helpers read."""

    def __init__(self, nz, ny, nx, vd=0.01, wd=0.0, dz=100.0):
        self.nz, self.ny, self.nx = nz, ny, nx
        self.Dz = np.full((nz, ny, nx), dz)
        # Names must match what orbit.core.deposition actually reads.
        self.particle_dry_dep = np.full((ny, nx), vd)
        self.NH3_dry_dep = np.full((ny, nx), vd)
        self.SO2_dry_dep = np.full((ny, nx), vd)
        self.NOx_dry_dep = np.full((ny, nx), vd)
        self.VOC_dry_dep = np.full((ny, nx), vd)
        self.particle_wet_dep = np.full((nz, ny, nx), wd)
        self.other_gas_wet_dep = np.full((nz, ny, nx), wd)
        self.SO2_wet_dep = np.full((nz, ny, nx), wd)
        self.NHPartitioning = np.zeros((nz, ny, nx))
        self.NO3Partitioning = np.zeros((nz, ny, nx))
        self.NO3PartitioningEq = np.zeros((nz, ny, nx))


def _orbits(nz, ny, nx, n_bins, value, species=(IDX_PM25,)):
    N = nz * ny * nx
    out = {}
    for s in species:
        out[s] = [np.full(N, value) for _ in range(n_bins + 1)]
    return out


@pytest.fixture
def setup():
    nz, ny, nx, n_bins = 3, 4, 5, 8
    idx = CellIndexer(nz, ny, nx)
    return nz, ny, nx, n_bins, idx


class TestDryDeposition:
    def test_dry_flux_equals_vd_times_surface_conc(self, setup):
        """The layer thickness must cancel: flux = vd * c[0]."""
        nz, ny, nx, n_bins, idx = setup
        vd, conc = 0.02, 5.0
        grids = [_Grid(nz, ny, nx, vd=vd, wd=0.0) for _ in range(n_bins)]
        maps = compute_deposition_maps(
            _orbits(nz, ny, nx, n_bins, conc), grids, idx)
        assert np.allclose(maps["dry"][IDX_PM25], vd * conc)

    def test_dry_is_independent_of_layer_thickness(self, setup):
        """Doubling Dz must not change the per-area dry flux."""
        nz, ny, nx, n_bins, idx = setup
        o = _orbits(nz, ny, nx, n_bins, 5.0)
        a = compute_deposition_maps(
            o, [_Grid(nz, ny, nx, dz=100.0) for _ in range(n_bins)], idx)["dry"]
        b = compute_deposition_maps(
            o, [_Grid(nz, ny, nx, dz=250.0) for _ in range(n_bins)], idx)["dry"]
        assert np.allclose(a, b)


class TestWetDeposition:
    def test_wet_flux_is_column_integral(self, setup):
        """flux = sum_z wd * c * Dz, so it scales with column depth."""
        nz, ny, nx, n_bins, idx = setup
        wd, conc, dz = 1e-5, 5.0, 100.0
        grids = [_Grid(nz, ny, nx, vd=0.0, wd=wd, dz=dz) for _ in range(n_bins)]
        maps = compute_deposition_maps(
            _orbits(nz, ny, nx, n_bins, conc), grids, idx)
        assert np.allclose(maps["wet"][IDX_PM25], nz * wd * conc * dz)

    def test_wet_scales_with_layer_thickness(self, setup):
        nz, ny, nx, n_bins, idx = setup
        o = _orbits(nz, ny, nx, n_bins, 5.0)
        a = compute_deposition_maps(
            o, [_Grid(nz, ny, nx, vd=0, wd=1e-5, dz=100.0) for _ in range(n_bins)], idx)
        b = compute_deposition_maps(
            o, [_Grid(nz, ny, nx, vd=0, wd=1e-5, dz=200.0) for _ in range(n_bins)], idx)
        assert np.allclose(b["wet"], 2.0 * a["wet"])


class TestStructure:
    def test_total_is_dry_plus_wet(self, setup):
        nz, ny, nx, n_bins, idx = setup
        grids = [_Grid(nz, ny, nx, vd=0.02, wd=1e-5) for _ in range(n_bins)]
        m = compute_deposition_maps(_orbits(nz, ny, nx, n_bins, 5.0), grids, idx)
        assert np.allclose(m["total"], m["dry"] + m["wet"])

    def test_shapes_and_species_coverage(self, setup):
        nz, ny, nx, n_bins, idx = setup
        grids = [_Grid(nz, ny, nx) for _ in range(n_bins)]
        m = compute_deposition_maps(_orbits(nz, ny, nx, n_bins, 1.0), grids, idx)
        for k in ("dry", "wet", "total"):
            assert m[k].shape == (N_SPECIES, ny, nx)

    def test_species_absent_from_orbit_is_zero(self, setup):
        """Only species actually solved contribute."""
        nz, ny, nx, n_bins, idx = setup
        grids = [_Grid(nz, ny, nx, vd=0.02) for _ in range(n_bins)]
        m = compute_deposition_maps(
            _orbits(nz, ny, nx, n_bins, 5.0, species=(IDX_PM25,)), grids, idx)
        others = [s for s in range(N_SPECIES) if s != IDX_PM25]
        assert np.allclose(m["total"][others], 0.0)

    def test_negative_flux_is_clipped_and_reported(self, setup):
        """Negative concentrations must not produce negative deposition."""
        nz, ny, nx, n_bins, idx = setup
        grids = [_Grid(nz, ny, nx, vd=0.02) for _ in range(n_bins)]
        m = compute_deposition_maps(
            _orbits(nz, ny, nx, n_bins, -5.0), grids, idx)
        assert (m["dry"] >= 0).all()
        assert float(m["clipped_negative_ug_m2_s"]) < 0

    def test_orbit_mean_not_sum(self, setup):
        """Result is the mean over bins, so it is bin-count independent."""
        nz, ny, nx, idx = 3, 4, 5, CellIndexer(3, 4, 5)
        a = compute_deposition_maps(
            _orbits(nz, ny, nx, 8, 5.0), [_Grid(nz, ny, nx) for _ in range(8)], idx)
        b = compute_deposition_maps(
            _orbits(nz, ny, nx, 4, 5.0), [_Grid(nz, ny, nx) for _ in range(4)], idx)
        assert np.allclose(a["dry"], b["dry"])

    def test_empty_grids_raises(self, setup):
        nz, ny, nx, n_bins, idx = setup
        with pytest.raises(ValueError):
            compute_deposition_maps(_orbits(nz, ny, nx, n_bins, 1.0), [], idx)


class TestUnits:
    def test_kg_ha_yr_conversion(self):
        """1 ug/m2/s over a year is 31.5576 g/m2 = 315.576 kg/ha."""
        seconds_per_year = 365.0 * 24 * 3600
        expected = 1e-9 * seconds_per_year * 1e4  # kg/m2/yr -> kg/ha/yr
        assert UG_M2_S_TO_KG_HA_YR == pytest.approx(expected, rel=1e-3)

    def test_summarise_runs_and_names_species(self, setup):
        nz, ny, nx, n_bins, idx = setup
        grids = [_Grid(nz, ny, nx, vd=0.02) for _ in range(n_bins)]
        m = compute_deposition_maps(_orbits(nz, ny, nx, n_bins, 5.0), grids, idx)
        names = [f"sp{i}" for i in range(N_SPECIES)]
        out = summarise(m, names)
        assert f"sp{IDX_PM25}" in out
