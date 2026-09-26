"""The nitrate cross-coupling derivative carries the wet-deposition contrast.

deposition.py scavenges the HNO3 branch of TotalNO3 at the soluble-gas rate
and the particle branch at the particle rate (since 2026-08-02), so the
derivative of the TotalNO3 loss rate with respect to the particle fraction
has a wet part, particle_wet_dep - other_gas_wet_dep, in every layer. The
NH derivative always had it; the NO3 one was zero until 2026-09-26.
"""
import numpy as np

from orbit.core.grid_data import GridData
from orbit.modes.iso_coupling import _delta_k_nh, _delta_k_no3


def _grid(nz=3, ny=2, nx=2):
    n = (nz, ny, nx)
    particle_wet = np.full(n, 1.0e-4)
    particle_wet[1] = 3.0e-4                       # a wetter middle layer
    gas_wet = np.full(n, 4.0e-5)
    return GridData(
        nz=nz, ny=ny, nx=nx,
        Dz=np.full(n, 50.0),
        particle_wet_dep=particle_wet,
        other_gas_wet_dep=gas_wet,
        particle_dry_dep=np.full(n, 0.002),
        NH3_dry_dep=np.full(n, 0.010),
        HNO3_dry_dep=np.full(n, 0.030),
    )


def test_no3_wet_contrast_is_particle_minus_soluble_gas_aloft():
    g = _grid()
    dk = _delta_k_no3(g)
    assert dk.shape == (3, 2, 2)
    # Above the surface only the wet part remains.
    np.testing.assert_allclose(dk[1], 3.0e-4 - 4.0e-5)
    np.testing.assert_allclose(dk[2], 1.0e-4 - 4.0e-5)


def test_no3_surface_adds_dry_contrast_over_dz():
    g = _grid()
    dk = _delta_k_no3(g)
    expected = (1.0e-4 - 4.0e-5) + (0.002 - 0.030) / 50.0
    np.testing.assert_allclose(dk[0], expected)


def test_no3_and_nh_share_the_wet_part():
    g = _grid()
    wet_no3 = _delta_k_no3(g)[1:]
    wet_nh = _delta_k_nh(g)[1:]
    np.testing.assert_allclose(wet_no3, wet_nh)
    assert np.all(wet_no3 > 0)


def test_no3_wet_part_is_not_identically_zero_on_a_realistic_contrast():
    g = _grid()
    g.particle_dry_dep[:] = g.HNO3_dry_dep      # remove the dry contrast
    dk = _delta_k_no3(g)
    assert np.all(dk != 0.0)
