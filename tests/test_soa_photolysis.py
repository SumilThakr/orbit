"""Unit tests for the VBS SoA photolytic-loss sink.

Guards the spring SOA over-production fix: a first-order, particle-phase,
daytime SOA photolytic loss Lambda = A_PHOTO * j(NO2) * F_p, driven by the
TUV clear-sky j(NO2) LUT.
"""
import numpy as np

from orbit.core.indexing import CellIndexer
from orbit.core.deposition import (
    assemble_deposition, IDX_SOA, IDX_PM25, IDX_VBS_BINS, N_VBS_BINS,
)
from orbit.core.photolysis import PhotolysisLUT
from orbit.core.soa_photolysis import (
    soa_photolysis_rate, compute_jno2_per_bin, attach_jno2_to_grid,
    get_a_photo, A_PHOTO_HODZIC,
)
from tests.test_advection import _make_grid

A = A_PHOTO_HODZIC  # 4e-4, the recommended ON value


# --- pure rate function -------------------------------------------------

def test_night_no_loss():
    """j(NO2) = 0 (night) -> zero photolytic loss."""
    assert np.all(soa_photolysis_rate(0.0, 1.0, A) == 0.0)


def test_particle_phase_weighting():
    """Loss scales with F_p: gas-dominated bin (F_p=0) -> 0; full particle -> full."""
    j = 1.0e-2
    assert soa_photolysis_rate(j, 0.0, A) == 0.0
    assert np.isclose(soa_photolysis_rate(j, 1.0, A), A * j)
    assert np.isclose(soa_photolysis_rate(j, 0.5, A), 0.5 * A * j)


def test_monotonic_in_jno2():
    """Heavier actinic flux -> faster loss (monotonic)."""
    rates = [soa_photolysis_rate(j, 0.7, A) for j in (1e-4, 1e-3, 5e-3, 1e-2)]
    assert all(b > a for a, b in zip(rates, rates[1:]))


def test_closed_form():
    """Lambda = A_PHOTO * j(NO2) * F_p, elementwise."""
    j = np.array([1e-3, 5e-3, 1e-2])
    fp = np.array([0.2, 0.5, 0.9])
    assert np.allclose(soa_photolysis_rate(j, fp, A), A * j * fp)


def test_noon_lifetime_about_three_days():
    """At solar-noon j(NO2)~1.0e-2 /s and A=4e-4, particle-phase SOA photolytic
    lifetime is ~3 days (Hodzic 2016), not hours or weeks."""
    rate = soa_photolysis_rate(1.0e-2, 1.0, A)        # F_p = 1
    efold_days = 1.0 / rate / 86400.0
    assert 2.0 < efold_days < 4.0


def test_a_photo_off_by_default(monkeypatch):
    """Sink is OFF unless ORBIT_VBS_A_PHOTO is set explicitly (no silent on)."""
    monkeypatch.delenv("ORBIT_VBS_A_PHOTO", raising=False)
    assert get_a_photo() == 0.0
    assert soa_photolysis_rate(1.0e-2, 1.0) == 0.0      # a_photo=None -> reads env


def test_a_photo_env_override(monkeypatch):
    """ORBIT_VBS_A_PHOTO overrides the default; 0 disables the sink."""
    monkeypatch.setenv("ORBIT_VBS_A_PHOTO", "0")
    assert get_a_photo() == 0.0
    assert soa_photolysis_rate(1.0e-2, 1.0) == 0.0      # a_photo=None -> reads env
    monkeypatch.setenv("ORBIT_VBS_A_PHOTO", "8.0e-4")
    assert get_a_photo() == 8.0e-4


# --- per-bin j(NO2) from the LUT (MCM fallback, no file needed) ---------

def _mcm_lut():
    return PhotolysisLUT.from_mcm()


def test_jno2_night_bin_is_zero():
    """A deep-night UTC bin over the SAS grid -> j(NO2) ~ 0 everywhere."""
    lut = _mcm_lut()
    lat = np.linspace(8.0, 36.0, 20)
    lon = np.linspace(68.0, 96.0, 20)
    # bin 6 center = 1170 min = 19:30 UTC ~ 01:00-02:00 IST local: night over SAS.
    j = compute_jno2_per_bin(lut, lat, lon, 2016, 3, 6)
    assert np.all(j < 1.0e-6)


def test_jno2_noon_bin_positive():
    """A midday bin over SAS -> positive, physical j(NO2) (< ~1.2e-2 /s)."""
    lut = _mcm_lut()
    lat = np.linspace(8.0, 36.0, 20)
    lon = np.linspace(68.0, 96.0, 20)
    # bin 2 center = 450 min = 07:30 UTC ~ 13:00 IST: local midday over SAS.
    j = compute_jno2_per_bin(lut, lat, lon, 2016, 3, 2)
    assert j.max() > 1.0e-3
    assert j.max() < 1.2e-2


# --- integration: the loss enters the VBS-bin deposition diagonal -------

def test_deposition_adds_photolysis_for_vbs_only(small_grid_params, monkeypatch):
    """assemble_deposition adds A_PHOTO*j*F_p to VBS-bin diagonals and leaves
    non-VBS species unchanged."""
    monkeypatch.setenv("ORBIT_VBS_A_PHOTO", str(A))
    g = _make_grid(small_grid_params)
    idx = CellIndexer(g.nz, g.ny, g.nx)
    nz, ny, nx = g.nz, g.ny, g.nx

    # Known F_p (per VBS bin) and j(NO2) fields.
    g.F_p_vbs = np.full((N_VBS_BINS, nz, ny, nx), 0.6)
    jval = 5.0e-3
    j3d = np.full((nz, ny, nx), jval)

    soa_idx = IDX_SOA                       # a VBS bin
    bin_pos = IDX_VBS_BINS.index(soa_idx)

    D_off = assemble_deposition(g, idx, soa_idx)        # j_no2_soa absent -> no sink
    g.j_no2_soa = j3d
    D_on = assemble_deposition(g, idx, soa_idx)

    added = (D_on - D_off).diagonal()
    expected = A * jval * g.F_p_vbs[bin_pos].ravel()
    assert np.allclose(added, expected)

    # PM2.5 (non-VBS) must be untouched by the photolytic sink.
    D_pm_off = assemble_deposition(g, idx, IDX_PM25)
    D_pm_on = assemble_deposition(g, idx, IDX_PM25)     # j_no2_soa now set
    assert np.allclose(D_pm_off.diagonal(), D_pm_on.diagonal())


def test_deposition_disabled_when_a_photo_zero(small_grid_params, monkeypatch):
    """ORBIT_VBS_A_PHOTO=0 -> deposition identical with/without j_no2_soa."""
    monkeypatch.setenv("ORBIT_VBS_A_PHOTO", "0")
    g = _make_grid(small_grid_params)
    idx = CellIndexer(g.nz, g.ny, g.nx)
    g.F_p_vbs = np.full((N_VBS_BINS, g.nz, g.ny, g.nx), 0.6)
    D_off = assemble_deposition(g, idx, IDX_SOA)
    g.j_no2_soa = np.full((g.nz, g.ny, g.nx), 5.0e-3)
    D_on = assemble_deposition(g, idx, IDX_SOA)
    assert np.allclose(D_off.diagonal(), D_on.diagonal())


def test_attach_jno2_sets_3d_field(small_grid_params):
    """attach_jno2_to_grid populates grid.j_no2_soa as (nz, ny, nx); no-op if LUT None."""
    g = _make_grid(small_grid_params)
    attach_jno2_to_grid(g, None, 2016, 3, 2)            # None LUT -> no-op
    assert getattr(g, "j_no2_soa", np.array([])).size == 0
    attach_jno2_to_grid(g, _mcm_lut(), 2016, 3, 2)
    assert g.j_no2_soa.shape == (g.nz, g.ny, g.nx)
