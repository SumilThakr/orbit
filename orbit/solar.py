"""
Solar geometry for SZA-based day/night classification of MERRA-2 A3 windows.

Provides analytic computation of the sun-up fraction for each 3-hour A3
averaging window at every grid cell, using NOAA-style solar position formulas.

Functions:
    decl_eqtime_noaa  — solar declination + equation of time
    hour_angle_center — hour angle at A3 center time per longitude
    day_fraction_3h   — fraction of 3h window with SZA < 90°
    day_mask_3h       — boolean day/night assignment (majority rule)

Reference: NOAA Solar Calculator
    https://gml.noaa.gov/grad/solcalc/solareqns.PDF
"""

import numpy as np
from datetime import timezone

# 3-hour window halfwidth in hour-angle space:
# 1 hour = 15° = pi/12 rad; 1.5 hours = 22.5° = pi/8 rad
DELTA_H = np.pi / 8.0

# A3 center times in minutes past midnight UTC
A3_CENTER_MINUTES = [90, 270, 450, 630, 810, 990, 1170, 1350]


def decl_eqtime_noaa(dt_utc):
    """NOAA-style solar declination and equation of time.

    Parameters
    ----------
    dt_utc : datetime
        UTC datetime (naive datetimes treated as UTC).

    Returns
    -------
    decl : float
        Solar declination in radians.
    eqtime : float
        Equation of time in minutes.
    """
    if dt_utc.tzinfo is None:
        dt_utc = dt_utc.replace(tzinfo=timezone.utc)
    dt_utc = dt_utc.astimezone(timezone.utc)

    doy = dt_utc.timetuple().tm_yday
    hour = dt_utc.hour + dt_utc.minute / 60.0 + dt_utc.second / 3600.0

    # Fractional year (radians)
    gamma = 2.0 * np.pi / 365.0 * (doy - 1 + (hour - 12.0) / 24.0)

    # Equation of time (minutes)
    eqtime = 229.18 * (
        0.000075
        + 0.001868 * np.cos(gamma) - 0.032077 * np.sin(gamma)
        - 0.014615 * np.cos(2 * gamma) - 0.040849 * np.sin(2 * gamma)
    )

    # Solar declination (radians)
    decl = (
        0.006918
        - 0.399912 * np.cos(gamma) + 0.070257 * np.sin(gamma)
        - 0.006758 * np.cos(2 * gamma) + 0.000907 * np.sin(2 * gamma)
        - 0.002697 * np.cos(3 * gamma) + 0.00148 * np.sin(3 * gamma)
    )
    return decl, eqtime


def hour_angle_center(lon_deg_1d, dt_utc, eqtime_min):
    """Hour angle at UTC datetime for each longitude.

    Parameters
    ----------
    lon_deg_1d : array_like, shape (nlon,)
        Longitudes in degrees east (negative = west).
    dt_utc : datetime
        UTC datetime.
    eqtime_min : float
        Equation of time in minutes.

    Returns
    -------
    ha : ndarray, shape (nlon,)
        Hour angle in radians, wrapped to [-pi, pi].
    """
    utc_minutes = dt_utc.hour * 60.0 + dt_utc.minute + dt_utc.second / 60.0
    # True solar time (minutes)
    tst = (utc_minutes + eqtime_min + 4.0 * np.asarray(lon_deg_1d, dtype=np.float64)) % 1440.0
    ha_deg = tst / 4.0 - 180.0
    ha = np.deg2rad(ha_deg)
    ha = (ha + np.pi) % (2.0 * np.pi) - np.pi
    return ha


def day_fraction_3h(lat_deg_1d, lon_deg_1d, t_center_utc):
    """Fraction of a 3-hour A3 window when the sun is above the horizon.

    Uses analytic overlap between the hour-angle window [Hc-dH, Hc+dH]
    and the daylight interval [-H0, +H0], handling date-line wraps in
    hour-angle space.

    Parameters
    ----------
    lat_deg_1d : array_like, shape (nlat,)
        Latitudes in degrees.
    lon_deg_1d : array_like, shape (nlon,)
        Longitudes in degrees east.
    t_center_utc : datetime
        A3 center time in UTC.

    Returns
    -------
    frac : ndarray, shape (nlat, nlon)
        Day fraction in [0, 1].
    """
    decl, eqt = decl_eqtime_noaa(t_center_utc)

    lat = np.deg2rad(np.asarray(lat_deg_1d, dtype=np.float64))    # (nlat,)
    ha_c = hour_angle_center(lon_deg_1d, t_center_utc, eqt)[None, :]  # (1, nlon)

    # Sunrise/sunset half-angle H0 per latitude
    x = -np.tan(lat) * np.tan(decl)  # (nlat,)
    H0 = np.empty_like(x)

    polar_day = x <= -1.0
    polar_night = x >= 1.0
    mid = (~polar_day) & (~polar_night)

    H0[polar_day] = np.pi
    H0[polar_night] = 0.0
    H0[mid] = np.arccos(x[mid])

    H0 = H0[:, None]  # (nlat, 1)

    # Window interval in hour-angle space
    a = ha_c - DELTA_H
    b = ha_c + DELTA_H

    # Overlap between [a, b] and [-H0, +H0], handling [-pi, pi] wraps

    # Main part clipped to [-pi, pi]
    a1 = np.maximum(a, -np.pi)
    b1 = np.minimum(b, np.pi)
    len1 = np.maximum(0.0, np.minimum(b1, H0) - np.maximum(a1, -H0))

    # Wrap left: window extends below -pi → wrap by +2pi
    a_wrap_left = a < -np.pi
    len_left = np.zeros_like(len1)
    if np.any(a_wrap_left):
        a2 = a + 2.0 * np.pi
        b2_val = np.pi  # upper bound
        len_left = np.maximum(0.0, np.minimum(b2_val, H0) - np.maximum(a2, -H0))

    # Wrap right: window extends above +pi → wrap by -2pi
    b_wrap_right = b > np.pi
    len_right = np.zeros_like(len1)
    if np.any(b_wrap_right):
        a2_val = -np.pi  # lower bound
        b2 = b - 2.0 * np.pi
        len_right = np.maximum(0.0, np.minimum(b2, H0) - np.maximum(a2_val, -H0))

    total_len = (
        len1
        + len_left * a_wrap_left.astype(np.float64)
        + len_right * b_wrap_right.astype(np.float64)
    )

    return np.clip(total_len / (2.0 * DELTA_H), 0.0, 1.0)


def day_mask_3h(lat_deg_1d, lon_deg_1d, t_center_utc):
    """Two-bin day/night assignment for a 3-hour A3 record.

    Day if majority (>= 50%) of the window has SZA < 90 deg.

    Parameters
    ----------
    lat_deg_1d : array_like, shape (nlat,)
        Latitudes in degrees.
    lon_deg_1d : array_like, shape (nlon,)
        Longitudes in degrees east.
    t_center_utc : datetime
        A3 center time in UTC.

    Returns
    -------
    mask : ndarray of bool, shape (nlat, nlon)
        True where classified as day.
    """
    frac = day_fraction_3h(lat_deg_1d, lon_deg_1d, t_center_utc)
    return frac >= 0.5
