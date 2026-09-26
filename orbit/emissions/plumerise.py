"""Plume rise calculations for elevated emission sources.

Implements the ASME (1973) plume rise algorithm: buoyancy and momentum
rise from stack parameters and local meteorology.

Falls back to simple cumulative Dz lookup when met fields are unavailable
or stack parameters are not provided.

Reference:
    ASME (1973), as described in Seinfeld and Pandis,
    "Atmospheric Chemistry and Physics - From Air Pollution to Climate Change"
"""

import math
from typing import List, Tuple

# Gravitational acceleration [m/s^2]
g = 9.80665

# Briggs stable-plume coefficient, expressed against the grid's S1
# ((dtheta/dz)/theta, [1/m]) rather than the stability parameter
# s = g * S1 [1/s^2]:  2.6 / g^(1/3).
_STABLE_COEFF = 2.6 / g ** (1.0 / 3.0)


class PlumeAboveModelTop(Exception):
    """Raised when the calculated plume height exceeds the model top."""
    pass


def find_layer(layer_heights: List[float], height: float) -> Tuple[int, bool]:
    """Find the layer index for a given height.

    Uses binary search (equivalent to Go's sort.SearchFloat64s).

    Parameters
    ----------
    layer_heights : list of float
        Layer edge heights [m] from ground (staggered grid).
        layer_heights[0] = 0 (ground), layer_heights[n] = top of layer n-1.
    height : float
        Height to locate [m]

    Returns
    -------
    layer_index : int
        Index of the layer containing this height
    above_top : bool
        True if height exceeds model top
    """
    n = len(layer_heights)
    if n == 0:
        return 0, True

    # Binary search for insertion point (like sort.SearchFloat64s)
    lo, hi = 0, n
    while lo < hi:
        mid = (lo + hi) // 2
        if layer_heights[mid] < height:
            lo = mid + 1
        else:
            hi = mid

    stack_layer = lo

    if stack_layer == n:
        return n - 2, True

    if stack_layer != 0:
        stack_layer -= 1

    return stack_layer, False


def calc_delta_h(
    stack_layer: int,
    temperature: List[float],
    wind_speed: List[float],
    s_class: List[float],
    s1: List[float],
    stack_height: float,
    stack_temp: float,
    stack_vel: float,
    stack_diam: float,
    wind_speed_minus_one_point_four: List[float],
    wind_speed_minus_third: List[float],
    wind_speed_inverse: List[float],
) -> float:
    """Calculate plume rise deltaH using ASME (1973).

    Two regimes:
    1. Momentum-dominated (tempDiff < 50K, stackVel > windSpd, stackVel > 10):
       deltaH = D * Vs^1.4 * windSpeedMinusOnePointFour

    2. Buoyancy-dominated:
       F = g * tempDiff * Vs * (D/2)^2  (buoyancy flux)
       - Stable (sClass > 0.5, s1 != 0, F > 0):
         deltaH = 2.6 * (F / (u * s))^(1/3),  s = g * s1
       - Unstable (F > 0):
         deltaH = 7.4 * (F * H^2)^(1/3) * windSpeedInverse
       - F <= 0: deltaH = 0

    The stable branch previously used ``29 * (F/s1)^(1/3) * u^(-1/3)``.
    That is dimensionally inconsistent: the grid's ``S1`` is
    ``(dtheta/dz) / theta`` [1/m] (see the preprocessor), not the
    Brunt-Vaisala stability parameter ``s = (g/theta) dtheta/dz`` [1/s^2]
    that Briggs' stable-plume formula takes, so the expression evaluated
    to m^(4/3)/s^(2/3) rather than a length. Substituting ``s = g * s1``
    gives the coefficient ``2.6 / g^(1/3) = 1.213`` against the
    precomputed ``u^(-1/3)``; the old 29 overstated stable plume rise by
    roughly 24x, lofting elevated-stack emissions kilometres above the
    boundary layer.

    Parameters
    ----------
    stack_layer : int
        Layer index of the stack
    temperature : list of float
        Air temperature at each layer [K]
    wind_speed : list of float
        Wind speed at each layer [m/s]
    s_class : list of float
        Stability class at each layer (0=unstable, 1=stable)
    s1 : list of float
        Stability parameter at each layer
    stack_height : float
        Stack height [m]
    stack_temp : float
        Stack exit temperature [K]
    stack_vel : float
        Stack exit velocity [m/s]
    stack_diam : float
        Stack diameter [m]
    wind_speed_minus_one_point_four : list of float
        Precomputed wind_speed^(-1.4) at each layer
    wind_speed_minus_third : list of float
        Precomputed wind_speed^(-1/3) at each layer
    wind_speed_inverse : list of float
        Precomputed 1/wind_speed at each layer

    Returns
    -------
    float
        Plume rise [m]

    Raises
    ------
    ValueError
        If calculation results in NaN
    """
    delta_h = 0.0

    air_temp = temperature[stack_layer]
    wind_spd = wind_speed[stack_layer]

    # Check for momentum-dominated regime
    if ((stack_temp - air_temp) < 50.0 and
            stack_vel > wind_spd and stack_vel > 10.0):
        delta_h = (stack_diam *
                   math.pow(stack_vel, 1.4) *
                   wind_speed_minus_one_point_four[stack_layer])

        if math.isnan(delta_h):
            raise ValueError(
                f"plumerise: momentum-dominated deltaH is NaN. "
                f"stackDiam: {stack_diam}, stackVel: {stack_vel}, "
                f"windSpeedMinusOnePointFour: {wind_speed_minus_one_point_four[stack_layer]}"
            )

    else:
        # Buoyancy-dominated
        if stack_temp - air_temp == 0:
            temp_diff = 0.0
        else:
            temp_diff = 2 * (stack_temp - air_temp) / (stack_temp + air_temp)

        F = g * temp_diff * stack_vel * math.pow(stack_diam / 2, 2)

        if s_class[stack_layer] > 0.5 and s1[stack_layer] > 0 and F > 0:
            # Stable conditions.  Briggs: deltaH = 2.6 (F/(u s))^(1/3) with
            # s = (g/theta) dtheta/dz = g * S1, so the coefficient against
            # the precomputed u^(-1/3) is 2.6 / g^(1/3).
            delta_h = (_STABLE_COEFF *
                       math.pow(F / s1[stack_layer], 0.333333333) *
                       wind_speed_minus_third[stack_layer])

            if math.isnan(delta_h):
                raise ValueError(
                    f"plumerise: stable buoyancy-dominated deltaH is NaN. "
                    f"F: {F}, s1: {s1[stack_layer]}, "
                    f"windSpeedMinusThird: {wind_speed_minus_third[stack_layer]}"
                )

        elif F > 0:
            # Unstable conditions
            delta_h = (7.4 *
                       math.pow(F * math.pow(stack_height, 2), 0.333333333) *
                       wind_speed_inverse[stack_layer])

            if math.isnan(delta_h):
                raise ValueError(
                    f"plumerise: unstable buoyancy-dominated deltaH is NaN. "
                    f"F: {F}, stackHeight: {stack_height}, "
                    f"windSpeedInverse: {wind_speed_inverse[stack_layer]}"
                )

        else:
            delta_h = 0.0

    return delta_h


def asme_plume_rise(
    stack_height: float,
    stack_diam: float,
    stack_temp: float,
    stack_vel: float,
    layer_heights: List[float],
    temperature: List[float],
    wind_speed: List[float],
    s_class: List[float],
    s1: List[float],
    wind_speed_minus_one_point_four: List[float],
    wind_speed_minus_third: List[float],
    wind_speed_inverse: List[float],
) -> Tuple[int, float]:
    """Calculate plume rise using ASME (1973).

    Parameters
    ----------
    stack_height : float
        Stack height [m]
    stack_diam : float
        Stack diameter [m]
    stack_temp : float
        Stack exit temperature [K]
    stack_vel : float
        Stack exit velocity [m/s]
    layer_heights : list of float
        Layer edge heights [m] (staggered grid)
    temperature : list of float
        Air temperature at each layer [K]
    wind_speed : list of float
        Wind speed at each layer [m/s]
    s_class : list of float
        Stability class at each layer
    s1 : list of float
        Stability parameter at each layer
    wind_speed_minus_one_point_four : list of float
        Precomputed wind_speed^(-1.4)
    wind_speed_minus_third : list of float
        Precomputed wind_speed^(-1/3)
    wind_speed_inverse : list of float
        Precomputed 1/wind_speed

    Returns
    -------
    plume_layer : int
        Layer index where plume ends
    plume_height : float
        Final plume height [m]

    Raises
    ------
    PlumeAboveModelTop
        If the plume exceeds the model top
    """
    stack_layer, above_top = find_layer(layer_heights, stack_height)
    if above_top:
        raise PlumeAboveModelTop(
            f"Stack height {stack_height}m exceeds model top"
        )

    delta_h = calc_delta_h(
        stack_layer, temperature, wind_speed, s_class, s1,
        stack_height, stack_temp, stack_vel, stack_diam,
        wind_speed_minus_one_point_four, wind_speed_minus_third,
        wind_speed_inverse
    )

    plume_height = stack_height + delta_h

    plume_layer, above_top = find_layer(layer_heights, plume_height)
    if above_top:
        raise PlumeAboveModelTop(
            f"Plume height {plume_height}m exceeds model top"
        )

    return plume_layer, plume_height


def _has_met_fields(grid) -> bool:
    """Check if grid has all met fields needed for ASME plume rise."""
    for attr in ("Temperature", "WindSpeed", "Sclass", "S1",
                 "WindSpeedInverse", "WindSpeedMinusThird",
                 "WindSpeedMinusOnePointFour"):
        arr = getattr(grid, attr, None)
        if arr is None or (hasattr(arr, 'size') and arr.size == 0):
            return False
    return True


def _get_layer_heights(grid, i: int, j: int) -> List[float]:
    """Build layer heights list for column (i, j).

    Uses LayerHeights if available, otherwise builds from cumulative Dz.
    """
    if hasattr(grid, 'LayerHeights') and hasattr(grid.LayerHeights, 'size') and grid.LayerHeights.size > 0:
        return [grid.LayerHeights[k, j, i] for k in range(grid.nz + 1)]

    heights = [0.0]
    for k in range(grid.nz):
        heights.append(heights[-1] + grid.Dz[k, j, i])
    return heights


def _asme_injection_layer(grid, i: int, j: int, height: float,
                          diam: float, temp: float, vel: float) -> int:
    """Compute ASME plume rise and return injection layer.

    Extracts column met data from GridData and calls asme_plume_rise().
    On errors (PlumeAboveModelTop, ValueError from NaN), returns top layer.
    """
    nz = grid.nz
    layer_heights = _get_layer_heights(grid, i, j)

    temperature = [grid.Temperature[k, j, i] for k in range(nz)]
    wind_speed = [grid.WindSpeed[k, j, i] for k in range(nz)]
    s_class = [grid.Sclass[k, j, i] for k in range(nz)]
    s1_col = [grid.S1[k, j, i] for k in range(nz)]
    ws_inv = [grid.WindSpeedInverse[k, j, i] for k in range(nz)]
    ws_m3 = [grid.WindSpeedMinusThird[k, j, i] for k in range(nz)]
    ws_m14 = [grid.WindSpeedMinusOnePointFour[k, j, i] for k in range(nz)]

    try:
        layer, _ = asme_plume_rise(
            height, diam, temp, vel,
            layer_heights, temperature, wind_speed, s_class, s1_col,
            ws_m14, ws_m3, ws_inv
        )
        return layer
    except (PlumeAboveModelTop, ValueError):
        return nz - 1


def find_injection_layer(grid, i: int, j: int, height: float,
                         stack_diam: float = 0.0, stack_temp: float = 0.0,
                         stack_vel: float = 0.0) -> int:
    """Find the grid layer index for a given injection height.

    When stack parameters are provided and the grid has met fields,
    uses the full ASME (1973) plume rise algorithm to compute the
    effective injection height from buoyancy/momentum rise.

    Otherwise falls back to simple cumulative Dz lookup.

    Parameters
    ----------
    grid : GridData
        Model grid (needs Dz[k, j, i] and nz; optionally Temperature,
        WindSpeed, Sclass, S1 and precomputed wind power arrays)
    i, j : int
        Horizontal grid indices
    height : float
        Physical stack height above ground level [m]
    stack_diam : float, optional
        Stack diameter [m]. Default 0 (triggers fallback).
    stack_temp : float, optional
        Stack exit temperature [K]. Default 0 (triggers fallback).
    stack_vel : float, optional
        Stack exit velocity [m/s]. Default 0 (triggers fallback).

    Returns
    -------
    int
        Layer index k (0 = surface)
    """
    if height <= 0:
        return 0

    # Use ASME if stack params provided and met fields available
    has_stack_params = stack_diam > 0 and stack_temp > 0 and stack_vel > 0
    if has_stack_params and _has_met_fields(grid):
        return _asme_injection_layer(grid, i, j, height,
                                     stack_diam, stack_temp, stack_vel)

    # Fallback: simple cumulative Dz lookup
    cumulative = 0.0
    for k in range(grid.nz):
        cumulative += grid.Dz[k, j, i]
        if cumulative >= height:
            return k

    return grid.nz - 1
