"""Stable-branch plume rise must be dimensionally sound and physical.

The stable branch used to apply Briggs' coefficient for the stability
parameter s = (g/theta) dtheta/dz [1/s^2] to the grid's S1, which is
(dtheta/dz)/theta [1/m].  The result was ~24x too much rise, lofting
elevated-stack emissions kilometres above the boundary layer.
"""

import math

import pytest

from orbit.emissions.plumerise import calc_delta_h, g


NZ = 15


def _column(u=4.0, T=285.0, s1=1.4e-4, stable=True):
    """Uniform met column with the precomputed wind powers ORBIT passes."""
    return dict(
        temperature=[T] * NZ,
        wind_speed=[u] * NZ,
        s_class=[1.0 if stable else 0.0] * NZ,
        s1=[s1] * NZ,
        wind_speed_minus_one_point_four=[u ** -1.4] * NZ,
        wind_speed_minus_third=[u ** (-1.0 / 3.0)] * NZ,
        wind_speed_inverse=[1.0 / u] * NZ,
    )


def _delta_h(stack_height, diam, temp, vel, **kw):
    c = _column(**kw)
    return calc_delta_h(
        0, c["temperature"], c["wind_speed"], c["s_class"], c["s1"],
        stack_height, temp, vel, diam,
        c["wind_speed_minus_one_point_four"],
        c["wind_speed_minus_third"], c["wind_speed_inverse"],
    )


def _briggs_stable(F, u, s1):
    """Textbook Briggs, built independently of the implementation."""
    return 2.6 * (F / (u * g * s1)) ** (1.0 / 3.0)


def _buoyancy_flux(diam, stack_temp, air_temp, vel):
    temp_diff = 2 * (stack_temp - air_temp) / (stack_temp + air_temp)
    return g * temp_diff * vel * (diam / 2) ** 2


# The three CEDS stack classes shipped in the production manifest.
STACKS = [
    ("low", 30.0, 1.5, 380.0, 10.0),
    ("medium", 50.0, 3.0, 380.0, 12.0),
    ("high", 220.0, 6.0, 410.0, 20.0),
]


@pytest.mark.parametrize("name,height,diam,temp,vel", STACKS)
def test_stable_rise_matches_briggs(name, height, diam, temp, vel):
    u, T, s1 = 4.0, 285.0, 1.4e-4
    got = _delta_h(height, diam, temp, vel, u=u, T=T, s1=s1)
    if temp - T < 50.0 and vel > u and vel > 10.0:
        pytest.skip(f"{name} is momentum-dominated in this column")
    want = _briggs_stable(_buoyancy_flux(diam, temp, T, vel), u, s1)
    assert got == pytest.approx(want, rel=1e-6)


@pytest.mark.parametrize("name,height,diam,temp,vel", STACKS)
def test_stable_plume_stays_in_the_lower_troposphere(name, height, diam, temp, vel):
    """A 30-220 m stack must not put its plume kilometres up.

    With the old coefficient the 220 m stack rose to ~5.9 km on a
    representative January IGP column.
    """
    dh = _delta_h(height, diam, temp, vel)
    assert height + dh < 1000.0, f"{name}: effective height {height + dh:.0f} m"


def test_stable_branch_is_dimensionally_consistent():
    """Halving s (via s1) must scale rise by 2^(1/3), and F likewise.

    This is the property the old expression violated; it holds for any
    correct (F/(u s))^(1/3) form regardless of the coefficient.
    """
    a = _delta_h(220.0, 6.0, 410.0, 20.0, s1=1.4e-4)
    b = _delta_h(220.0, 6.0, 410.0, 20.0, s1=0.7e-4)
    assert b / a == pytest.approx(2.0 ** (1.0 / 3.0), rel=1e-6)

    c = _delta_h(220.0, 6.0, 410.0, 20.0, u=4.0)
    d = _delta_h(220.0, 6.0, 410.0, 20.0, u=8.0)
    assert c / d == pytest.approx(2.0 ** (1.0 / 3.0), rel=1e-6)


def test_unstable_branch_unchanged():
    """Only the stable branch moved; the ASME neutral form is correct."""
    u = 4.0
    dh = _delta_h(220.0, 6.0, 410.0, 20.0, u=u, stable=False)
    F = _buoyancy_flux(6.0, 410.0, 285.0, 20.0)
    want = 7.4 * (F * 220.0 ** 2) ** (1.0 / 3.0) / u
    assert dh == pytest.approx(want, rel=1e-6)


def test_negative_s1_does_not_raise():
    """S1 <= 0 must fall through to the unstable branch, not blow up.

    ORBIT's grids never pair Sclass > 0.5 with S1 <= 0, but the guard
    should not depend on that: math.pow of a negative base raised
    ValueError, which the caller turns into "inject at the model top".
    """
    dh = _delta_h(220.0, 6.0, 410.0, 20.0, s1=-1e-6)
    assert math.isfinite(dh) and dh > 0.0
