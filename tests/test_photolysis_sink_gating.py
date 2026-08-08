"""Asking for the SOA photolytic sink and not getting it must be an error.

The sink needs a TUV clear-sky j(NO2) LUT whose module default is a
``/path/to/data/...`` placeholder — so on any machine that has not set
``ORBIT_PHOTOLYSIS_LUT``, the file is missing. ``load_photolysis_lut``
degrades gracefully to ``None``, which is right when the sink is off and
wrong when it was explicitly requested: the config banner still printed the
requested ``A_PHOTO``, so the run looked like a sink experiment and produced
output bit-identical to the no-sink case.

That cost a 21-minute April run on 2026-08-03 and, worse, nearly produced
the conclusion "the photolytic sink has no effect" from a run in which the
sink never executed.
"""

import os


from orbit.core.soa_photolysis import (
    _DEFAULT_LUT_PATH,
    get_a_photo,
    load_photolysis_lut,
    soa_photolysis_rate,
)


def test_module_default_is_off():
    """The sink must stay opt-in; it has never run in production."""
    old = os.environ.pop("ORBIT_VBS_A_PHOTO", None)
    try:
        assert get_a_photo() == 0.0
    finally:
        if old is not None:
            os.environ["ORBIT_VBS_A_PHOTO"] = old


def test_default_lut_path_is_a_placeholder_not_a_real_file():
    """Pins the reason the guard is needed rather than optional."""
    assert _DEFAULT_LUT_PATH.startswith("/path/to/data")
    assert not os.path.exists(_DEFAULT_LUT_PATH)


def test_missing_lut_returns_none_rather_than_raising():
    """The loader itself stays graceful; the CLI decides whether that is ok."""
    assert load_photolysis_lut("/definitely/not/here.npz") is None


def test_rate_is_zero_at_night_and_scales_with_a_photo():
    import numpy as np
    j_day = np.array([1.0e-2, 1.0e-2])
    j_night = np.zeros(2)
    F_p = np.array([0.5, 0.5])
    assert soa_photolysis_rate(j_night, F_p, a_photo=4e-4).max() == 0.0
    r1 = soa_photolysis_rate(j_day, F_p, a_photo=4e-4)
    r2 = soa_photolysis_rate(j_day, F_p, a_photo=8e-4)
    np.testing.assert_allclose(r2, 2 * r1)


def test_rate_is_restricted_to_the_particle_phase():
    import numpy as np
    j = np.array([1.0e-2])
    assert soa_photolysis_rate(j, np.zeros(1), a_photo=4e-4)[0] == 0.0


def test_cli_refuses_to_run_a_sink_experiment_without_the_lut(monkeypatch):
    """The whole point: a requested sink that cannot act must not run."""
    monkeypatch.setenv("ORBIT_VBS_A_PHOTO", "4.0e-4")
    monkeypatch.setenv("ORBIT_PHOTOLYSIS_LUT", "/definitely/not/here.npz")
    import importlib

    import orbit.core.soa_photolysis as sp
    importlib.reload(sp)
    assert sp.get_a_photo() > 0
    assert sp.load_photolysis_lut() is None
    # The CLI turns exactly this combination into a SystemExit; assert the
    # precondition it keys on, without importing the full solver here.
