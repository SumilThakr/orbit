"""Per-bin j(NO2) actinic field for the SOA photolytic-loss sink.

The VBS SoA scheme (orbit/core/dcomp_vbs.py + the aging cascade in
orbit/core/operator.py) has OH-driven aging + fragmentation but no
photolytic / heterogeneous condensed-phase loss. GC limits biomass-burning
SOA via SOA photolysis; without it ORBIT over-produces spring SOA ~4x.

This module supplies the actinic-flux proxy for that sink:

    Lambda_photo,i = A_PHOTO * j_NO2(lat, lon, bin) * F_p,i      [1/s]

j_NO2 is taken from the existing TUV 5.4 clear-sky lookup table
(orbit/core/photolysis.py), evaluated per orbit bin from analytic solar
geometry. Only the j_NO2 *field* is used here — NOT the dormant DCOMP OH
PSS / O3 chemistry that the LUT once fed (that arc was scoped out on
2026-04-24). The j_NO2
itself validated to ~+/-15% vs HEMCO FAST-JX, which is well inside the
factor-of-few uncertainty of the literature photolysis-rate scaling.

Design choices:
- CLEAR SKY (cloud_fraction=None): avoids the Jensen-inequality error of
  applying monthly-mean cloud fraction to a clear-sky field, and the
  production bin files carry qCloud but not a ready CLDTOT field. The mean
  cloud effect folds into the A_PHOTO calibration.
- Mid-month day (15) as the monthly representative SZA, matching the
  monthly-mean periodic-orbit philosophy.
- Overhead O3 column fixed at 300 DU (j_NO2 is only weakly O3-column
  dependent).
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np

from orbit.core.photolysis import PhotolysisLUT, j_per_bin

# ORBIT bin centres (UTC minutes past midnight), mirroring
# orbit/core/dcomp.py:_BIN_CENTER_MIN (8 UTC 3h bins).
_BIN_CENTER_MIN = (90, 270, 450, 630, 810, 990, 1170, 1350)

# Default TUV clear-sky LUT (jNO2_clear over SZA x O3-column).
_DEFAULT_LUT_PATH = "/path/to/data/preproc/output/LUT/photolysis_tuv.npz"

# SOA photolysis scaling: Lambda_photo = A_PHOTO * j(NO2) * F_p. Hodzic et al.
# (2016, ACP) "0.04% of j(NO2)" -> ~3-day photolytic lifetime at solar-noon
# j(NO2) ~ 1.0e-2 /s. This is the recommended value to use when the sink is ON.
A_PHOTO_HODZIC = 4.0e-4

# The sink is OFF by default and must be enabled EXPLICITLY by setting the
# ORBIT_VBS_A_PHOTO env var (e.g. =4.0e-4 = A_PHOTO_HODZIC). Defaulting to 0
# ensures a re-run of any production/sweep script that does not set the var
# cannot silently change physics. 0 (or unset) disables the sink entirely.
A_PHOTO_DEFAULT = 0.0


def get_a_photo() -> float:
    """SOA photolysis scaling A_PHOTO; OFF (0.0) unless ORBIT_VBS_A_PHOTO is set."""
    return float(os.environ.get("ORBIT_VBS_A_PHOTO", A_PHOTO_DEFAULT))


def soa_photolysis_rate(j_no2, F_p_i, a_photo: Optional[float] = None) -> np.ndarray:
    """First-order particle-phase SOA photolytic loss rate [1/s].

        Lambda_photo,i = A_PHOTO * j(NO2) * F_p,i

    j(NO2) is zero at night, so the rate is daytime-only. F_p,i restricts the
    loss to the particle phase of VBS bin i (photolysis acts on condensed-phase
    SOA), exactly like the F_p-weighted deposition rates.
    """
    a = get_a_photo() if a_photo is None else a_photo
    return a * np.asarray(j_no2, dtype=np.float64) * np.asarray(F_p_i, dtype=np.float64)


def bin_center_utc(year: int, month: int, bin_idx: int, day: int = 15) -> datetime:
    """UTC datetime at the centre of orbit bin ``bin_idx`` (0..7)."""
    base = datetime(year, month, day, tzinfo=timezone.utc)
    return base + timedelta(minutes=int(_BIN_CENTER_MIN[bin_idx]))


def load_photolysis_lut(path: Optional[str] = None) -> Optional[PhotolysisLUT]:
    """Load the TUV clear-sky photolysis LUT.

    Resolution order: explicit ``path`` -> ``ORBIT_PHOTOLYSIS_LUT`` env ->
    module default. Returns None (with a printed warning) if the file is
    absent, so callers can degrade gracefully to "no photolytic sink".
    """
    p = path or os.environ.get("ORBIT_PHOTOLYSIS_LUT", _DEFAULT_LUT_PATH)
    if not os.path.exists(p):
        print(f"[soa_photolysis] LUT not found at {p}; SOA photolysis disabled.")
        return None
    return PhotolysisLUT.load(p)


def compute_jno2_per_bin(
    lut: PhotolysisLUT,
    lat_deg: np.ndarray,
    lon_deg: np.ndarray,
    year: int,
    month: int,
    bin_idx: int,
    day: int = 15,
    o3_column_du: float = 300.0,
) -> np.ndarray:
    """Clear-sky j(NO2) [1/s] on the (ny, nx) grid for one orbit bin.

    Night cells (SZA >= 90) are zero, consistent with the LUT night edge.
    """
    t_utc = bin_center_utc(year, month, bin_idx, day=day)
    jNO2 = j_per_bin(
        lut, lat_deg, lon_deg, t_utc,
        cloud_fraction_2d=None,          # clear sky (see module docstring)
        o3_column_du=o3_column_du,
    )[0]
    return np.asarray(jNO2, dtype=np.float64)


def attach_jno2_to_grid(
    grid,
    lut: Optional[PhotolysisLUT],
    year: int,
    month: int,
    bin_idx: int,
    day: int = 15,
    o3_column_du: float = 300.0,
) -> None:
    """Set ``grid.j_no2_soa`` (nz, ny, nx) for the photolytic SOA sink.

    The same clear-sky 2D j(NO2) is broadcast to every level (j(NO2) varies
    only modestly with altitude; SOA mass is mostly in the PBL, so this
    slightly under-estimates aloft -> conservative). No-op if ``lut`` is None.
    """
    if lut is None:
        return
    j2d = compute_jno2_per_bin(
        lut, grid.lat, grid.lon, year, month, bin_idx,
        day=day, o3_column_du=o3_column_du,
    )
    grid.j_no2_soa = np.broadcast_to(
        j2d[None, :, :], (grid.nz, grid.ny, grid.nx)
    ).copy()
