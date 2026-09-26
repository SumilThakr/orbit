"""The baseline VBS field that feeds the Pankow feedback factor 1/D.

Marginal mode and adjoint mode must build it from the forward output's
c_orbit in the same way; adjoint mode left it None until 2026-09-26, so its
POA receptor equalled the primary-PM2.5 receptor everywhere.
"""
import numpy as np

from orbit.core.deposition import IDX_VBS_BINS, N_SPECIES
from orbit.core.orbit import N_BINS
from orbit.modes.marginal import baseline_vbs_from_c_orbit

NZ, NY, NX = 2, 3, 4
N = NZ * NY * NX


def _c_orbit():
    rng = np.random.default_rng(0)
    c = rng.normal(size=(N_SPECIES, N_BINS + 1, N))
    return c


def test_shape_and_bin_order():
    c = _c_orbit()
    out = baseline_vbs_from_c_orbit(c, NZ, NY, NX)
    assert out.shape == (N_BINS, len(IDX_VBS_BINS), NZ, NY, NX)
    for tau in range(N_BINS):
        for k, s in enumerate(IDX_VBS_BINS):
            np.testing.assert_array_equal(
                out[tau, k], np.maximum(c[s, tau + 1], 0.0).reshape(NZ, NY, NX))


def test_uses_the_end_of_bin_state_not_the_start():
    c = np.zeros((N_SPECIES, N_BINS + 1, N))
    c[IDX_VBS_BINS[0], 0] = 7.0          # start of the orbit
    c[IDX_VBS_BINS[0], 1] = 2.0          # end of bin 0
    out = baseline_vbs_from_c_orbit(c, NZ, NY, NX)
    assert np.all(out[0, 0] == 2.0)


def test_negative_concentrations_clip_to_zero():
    c = -np.ones((N_SPECIES, N_BINS + 1, N))
    out = baseline_vbs_from_c_orbit(c, NZ, NY, NX)
    assert np.all(out == 0.0)


def test_adjoint_mode_fills_the_field_from_c_orbit():
    """adjoint.py must call the same helper marginal.py uses."""
    import inspect
    import orbit.modes.adjoint as adj
    src = inspect.getsource(adj)
    assert "baseline_vbs_from_c_orbit(" in src
    assert 'if "c_orbit" in orbit_npz.files' in src
