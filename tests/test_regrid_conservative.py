"""Conservative emission regridding (2026-08-02).

The previous bilinear point sampling changed domain-emitted mass by -21%
to +25% across the production inventory. These tests pin the properties
that matter: mass conservation in both refinement directions, exactness
on constant fields, and identity when the grids already match.
"""
import numpy as np
import pytest

from orbit.emissions.netcdf import (
    _cell_edges, _regrid_emission_bilinear, _regrid_emission_conservative,
)

R = 6371000.0


def _mass(field, lat, lon):
    """Spherical-exact domain mass of an intensity field (R^2 factored out)."""
    lat_e = np.deg2rad(_cell_edges(lat))
    lon_e = np.deg2rad(_cell_edges(lon))
    dsin = np.diff(np.sin(lat_e))
    dlon = np.diff(lon_e)
    return float((field * dsin[:, None] * dlon[None, :]).sum())


def _grid(n_lat, n_lon, lat0=5.0, lat1=38.0, lon0=60.0, lon1=100.0):
    """Cell-centred grid whose inferred edges span exactly [lat0, lat1] x
    [lon0, lon1], so grids of different resolution share an extent and
    conservation is testable independently of domain coverage."""
    lat = lat0 + (np.arange(n_lat) + 0.5) * (lat1 - lat0) / n_lat
    lon = lon0 + (np.arange(n_lon) + 0.5) * (lon1 - lon0) / n_lon
    return lat, lon


def test_constant_field_is_preserved():
    lat_s, lon_s = _grid(120, 140)
    lat_d, lon_d = _grid(30, 35)
    data = np.full((lat_s.size, lon_s.size), 3.5)
    out = _regrid_emission_conservative(data, lon_s, lat_s, lon_d, lat_d)
    np.testing.assert_allclose(out, 3.5, rtol=1e-12)


def test_uncovered_destination_drops_outside_mass():
    """A destination extending beyond the source gets zero where there is
    no source data — the correct behaviour for a regional domain cut out
    of a larger inventory, and what the old fill_value=0.0 also did."""
    lat_s, lon_s = _grid(20, 20, lat0=10.0, lat1=20.0, lon0=70.0, lon1=80.0)
    lat_d, lon_d = _grid(40, 40, lat0=5.0, lat1=38.0, lon0=60.0, lon1=100.0)
    data = np.ones((lat_s.size, lon_s.size))
    out = _regrid_emission_conservative(data, lon_s, lat_s, lon_d, lat_d)
    assert out[0, 0] == 0.0 and out[-1, -1] == 0.0
    # All the source mass that fits inside the destination is retained.
    ratio = _mass(out, lat_d, lon_d) / _mass(data, lat_s, lon_s)
    assert abs(ratio - 1.0) < 1e-12


def test_mass_conserved_fine_to_coarse():
    rng = np.random.default_rng(20260802)
    lat_s, lon_s = _grid(350, 420)
    lat_d, lon_d = _grid(71, 65)
    # Spiky, fire-like field: a few hot cells on a quiet background.
    data = rng.random((lat_s.size, lon_s.size)) * 1e-3
    for _ in range(40):
        data[rng.integers(20, 330), rng.integers(20, 400)] = rng.random() * 50
    out = _regrid_emission_conservative(data, lon_s, lat_s, lon_d, lat_d)
    m_src = _mass(data, lat_s, lon_s)
    m_dst = _mass(out, lat_d, lon_d)
    assert abs(m_dst / m_src - 1.0) < 1e-12     # shared extent: exact

    # The bug this replaces: bilinear on the same spiky field is far off.
    bil = _regrid_emission_bilinear(data, lon_s, lat_s, lon_d, lat_d)
    assert abs(_mass(bil, lat_d, lon_d) / m_src - 1.0) > 0.05


def test_mass_conserved_coarse_to_fine():
    rng = np.random.default_rng(7)
    lat_s, lon_s = _grid(18, 17)
    lat_d, lon_d = _grid(71, 65)
    data = rng.random((lat_s.size, lon_s.size))
    out = _regrid_emission_conservative(data, lon_s, lat_s, lon_d, lat_d)
    ratio = _mass(out, lat_d, lon_d) / _mass(data, lat_s, lon_s)
    assert abs(ratio - 1.0) < 1e-12


def test_identical_grids_round_trip():
    rng = np.random.default_rng(11)
    lat, lon = _grid(40, 45)
    data = rng.random((lat.size, lon.size))
    out = _regrid_emission_conservative(data, lon, lat, lon, lat)
    np.testing.assert_allclose(out, data, rtol=1e-10, atol=1e-12)


def test_descending_latitude_input():
    rng = np.random.default_rng(3)
    lat, lon = _grid(60, 70)
    data = rng.random((lat.size, lon.size))
    lat_d, lon_d = _grid(25, 30)
    asc = _regrid_emission_conservative(data, lon, lat, lon_d, lat_d)
    desc = _regrid_emission_conservative(data[::-1, :], lon, lat[::-1],
                                        lon_d, lat_d)
    np.testing.assert_allclose(asc, desc, rtol=1e-10, atol=1e-12)


def test_nans_treated_as_zero():
    lat_s, lon_s = _grid(40, 40)
    lat_d, lon_d = _grid(12, 12)
    data = np.ones((lat_s.size, lon_s.size))
    data[5, 5] = np.nan
    out = _regrid_emission_conservative(data, lon_s, lat_s, lon_d, lat_d)
    assert np.isfinite(out).all()


def test_unknown_scheme_rejected(monkeypatch):
    from orbit.emissions.netcdf import _regrid_emission
    monkeypatch.setenv("ORBIT_REGRID", "nearest")
    lat, lon = _grid(10, 10)
    with pytest.raises(ValueError, match="Unknown ORBIT_REGRID"):
        _regrid_emission(np.ones((10, 10)), lon, lat, lon, lat)
