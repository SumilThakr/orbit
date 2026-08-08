"""Photolysis rates for DCOMP diagnostic chemistry.

Provides per-bin j_NO2 and j_O1D (and related) photolysis rates from:
  (a) a clear-sky TUV 5.4 lookup table, j_clear(SZA, O3_column)
  (b) per-cell, per-bin solar zenith angle from analytic solar geometry
  (c) per-bin cloud attenuation, j = j_clear * (1 - 0.7 * CF)

The default lookup table is generated offline by TUV 5.4 (Madronich et al.,
NCAR ACOM) via scripts/photolysis/generate_tuv_lut.py, tabulated on a
91-point SZA axis x 13 overhead-O3-column values. The MCM analytic fits
(Saunders et al. 2003) are retained as a functional fallback for
environments without TUV and as a sensitivity comparison for the paper.

Validation notes (Jan 2016 SAS surface, clear sky, 300 DU, surface
albedo alsurf=0.10, tauaer=0):

  jO1D : TUV day-mean / HEMCO FAST-JX monthly = 1.020  (std 0.14 across cells)
  jNO2 : TUV day-mean / HEMCO FAST-JX monthly = 1.155  (std 0.13 across cells)

The +15 percent jNO2 bias is not eliminated by reasonable surface albedo
variation (0.10 -> 0.20 increases TUV j by ~17 percent, widening the gap,
not narrowing it), by aerosol (at visible wavelengths AOD ~0.5 gives 3-7
percent attenuation, insufficient), or by applying 1 - 0.7 * CF_monthly
attenuation (that overshoots due to Jensen's inequality across the mix of
clear and cloudy days in a monthly mean). The residual is most likely the
known FAST-JX underprediction of full TUV jNO2 by ~10-15 percent driven
by its reduced-wavelength binning near 420 nm (Wild et al. 2000, J. Atmos.
Chem. 37:245). In that case TUV is the reference and the benchmark carries
the bias; we carry the +15 percent forward and quantify its downstream
effect through the OH/PSS validation rather than attempting to adjust it.

The jO1D 2 percent agreement should be interpreted cautiously: it may
reflect coincidental cancellation of small TUV and FAST-JX biases rather
than either being individually accurate. The cleaner statement is that
jO1D day-mean agrees with the best available benchmark to within
observational uncertainty.

Usage
-----
    from orbit.core.photolysis import PhotolysisLUT, j_per_bin
    lut = PhotolysisLUT.load("/path/to/photolysis_lut.npz")
    j_NO2_tau, j_O1D_tau = j_per_bin(lut, grid_lat, grid_lon,
                                      bin_center_utc, cloud_frac_2d,
                                      o3_column_du=300.0)

SZA is computed analytically in `solar.py` from (lat, lon, UTC datetime).
Cloud fraction is expected per bin (CLDTOT from preprocessor). Overhead
O3 column is taken as a scalar or a per-cell (ny, nx) field.

Night bins (cos(SZA) <= 0 for all cells) return zero photolysis rates,
and the LUT is not queried.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple, Union

import numpy as np

from orbit.solar import decl_eqtime_noaa, hour_angle_center


# MCM (Saunders et al. 2003, Atmos. Chem. Phys. 3: 161-180) photolysis rate
# coefficients, fit against TUV at a reference atmosphere (300 DU O3,
# surface albedo 0.1, aerosol-free). Form: j = l * cos(SZA)^m * exp(-n * sec(SZA)).
# These are used to populate the lookup table when a direct TUV run is
# unavailable. The tabulation is linearly interpolated at use time, so
# the accuracy limitations of the MCM fit at high SZA are captured in the
# table's high-SZA entries rather than in extrapolation.
MCM_COEFFS = {
    # j_NO2 -> NO + O(3P) (MCM reaction #4)
    "jNO2":  (1.165e-2, 0.244, 0.267),
    # j_O3 -> O(1D) + O2 (MCM reaction #1)
    "jO1D":  (6.073e-5, 1.743, 0.474),
    # j_O3 -> O(3P) + O2 (MCM reaction #2) — minor, not currently used
    "jO3P":  (4.775e-4, 0.298, 0.080),
    # j_HNO3 -> OH + NO2 (MCM reaction #8) — for validation
    "jHNO3": (1.165e-6, 0.244, 0.267),
    # j_NO3 -> NO + O2 + NO2 (MCM reactions #5/#6 combined)
    "jNO3":  (2.485e-2, 0.168, 0.108),
}


def _mcm_j(sza_rad: np.ndarray, l: float, m: float, n: float) -> np.ndarray:
    """MCM analytic photolysis rate: j = l * cos(SZA)^m * exp(-n * sec(SZA))."""
    cos_sza = np.cos(sza_rad)
    j = np.zeros_like(cos_sza, dtype=np.float64)
    # Only evaluate for sun-above-horizon cells (cos > small epsilon to avoid
    # divide-by-zero in sec(SZA)).
    mask = cos_sza > 1e-6
    if np.any(mask):
        cs = cos_sza[mask]
        j[mask] = l * (cs ** m) * np.exp(-n / cs)
    return j


@dataclass
class PhotolysisLUT:
    """Tabulated clear-sky photolysis rates.

    Axes:
      sza_axis : (n_sza,) degrees, 0..90 inclusive
      o3_col_axis : (n_o3,) Dobson Units, e.g. 150..450
    Tables (all 1/s):
      jNO2_clear : (n_sza, n_o3)    — NO2 -> NO + O(3P) photolysis
      jO1D_clear : (n_sza, n_o3)    — O3 -> O(1D) + O2 (dominant OH source)
      jNO3_clear : (n_sza, n_o3)    — NO3 total photolysis (NO3 -> NO + O2
                                       plus NO3 -> NO2 + O(3P)); dominates
                                       NO3 daytime loss.
    """
    sza_axis: np.ndarray
    o3_col_axis: np.ndarray
    jNO2_clear: np.ndarray
    jO1D_clear: np.ndarray
    jNO3_clear: np.ndarray = None
    jHONO_clear: np.ndarray = None   # HONO + hv -> OH + NO
    jHCHO_clear: np.ndarray = None   # CH2O + hv -> H + HCO (radical channel; 2 HO2 yield)
    jH2O2_clear: np.ndarray = None   # H2O2 + hv -> 2 OH

    @classmethod
    def from_mcm(cls, sza_step: float = 1.0,
                 o3_columns_du=(150, 175, 200, 225, 250, 275, 300,
                                325, 350, 375, 400, 425, 450)) -> "PhotolysisLUT":
        """Populate the table from MCM analytic fits across SZA x O3 column.

        Fallback only — prefer the TUV-generated LUT from
        scripts/photolysis/generate_tuv_lut.py when possible. MCM coefficients
        under-predict j_O1D by ~20 percent against FAST-JX benchmarks and do
        not capture twilight (SZA>85) correctly. The column dependence is
        approximated by Lambert-Beer sqrt(300/DU) attenuation on j_O1D; j_NO2
        uses the MCM baseline unchanged across columns.
        """
        sza = np.arange(0.0, 90.0 + sza_step / 2.0, sza_step, dtype=np.float64)
        o3_col = np.asarray(o3_columns_du, dtype=np.float64)
        sza_rad = np.deg2rad(sza)

        # j_NO2: weak O3-column dependence; same baseline at every column.
        l, m, n = MCM_COEFFS["jNO2"]
        jNO2_base = _mcm_j(sza_rad, l, m, n)
        jNO2_table = np.broadcast_to(jNO2_base[:, None], (len(sza), len(o3_col))).copy()

        # j_O1D: scale with column via Lambert-Beer-style sqrt attenuation.
        l, m, n = MCM_COEFFS["jO1D"]
        jO1D_base = _mcm_j(sza_rad, l, m, n)
        scale = np.sqrt(300.0 / o3_col)  # (n_o3,)
        jO1D_table = jO1D_base[:, None] * scale[None, :]

        # j_NO3: weak O3-column dependence at 660 nm (Chappuis band), treat
        # as column-invariant to match MCM's 1D fit.
        l, m, n = MCM_COEFFS["jNO3"]
        jNO3_base = _mcm_j(sza_rad, l, m, n)
        jNO3_table = np.broadcast_to(jNO3_base[:, None], (len(sza), len(o3_col))).copy()

        return cls(sza_axis=sza, o3_col_axis=o3_col,
                   jNO2_clear=jNO2_table.astype(np.float64),
                   jO1D_clear=jO1D_table.astype(np.float64),
                   jNO3_clear=jNO3_table.astype(np.float64))

    def save(self, path: str) -> None:
        """Persist the lookup table as a compressed NPZ."""
        d = {
            "sza_axis": self.sza_axis,
            "o3_col_axis": self.o3_col_axis,
            "jNO2_clear": self.jNO2_clear,
            "jO1D_clear": self.jO1D_clear,
        }
        if self.jNO3_clear is not None:
            d["jNO3_clear"] = self.jNO3_clear
        if self.jHONO_clear is not None:
            d["jHONO_clear"] = self.jHONO_clear
        if self.jHCHO_clear is not None:
            d["jHCHO_clear"] = self.jHCHO_clear
        if self.jH2O2_clear is not None:
            d["jH2O2_clear"] = self.jH2O2_clear
        np.savez_compressed(path, **d)

    @classmethod
    def load(cls, path: str) -> "PhotolysisLUT":
        """Load a lookup table from NPZ (round-trip with .save)."""
        d = np.load(path)
        def _get(name):
            return d[name] if name in d.files else None
        return cls(
            sza_axis=d["sza_axis"],
            o3_col_axis=d["o3_col_axis"],
            jNO2_clear=d["jNO2_clear"],
            jO1D_clear=d["jO1D_clear"],
            jNO3_clear=_get("jNO3_clear"),
            jHONO_clear=_get("jHONO_clear"),
            jHCHO_clear=_get("jHCHO_clear"),
            jH2O2_clear=_get("jH2O2_clear"),
        )


def solar_zenith_angle(lat_deg: np.ndarray, lon_deg: np.ndarray,
                       t_utc: datetime) -> np.ndarray:
    """Solar zenith angle (degrees) at bin-center UTC time on a lat x lon grid.

    Night cells (sun below horizon) get SZA = 90 deg, consistent with the
    LUT's night edge. The analytic formula follows NOAA:
        cos(SZA) = sin(phi) sin(decl) + cos(phi) cos(decl) cos(H)
    with phi = latitude, decl = solar declination, H = hour angle at t_utc.

    Parameters
    ----------
    lat_deg, lon_deg : 1D arrays (ny,), (nx,)
    t_utc : datetime (timezone-naive treated as UTC)

    Returns
    -------
    sza_deg : ndarray (ny, nx)
    """
    decl, eqt = decl_eqtime_noaa(t_utc)
    ha = hour_angle_center(lon_deg, t_utc, eqt)  # (nx,) radians
    phi = np.deg2rad(np.asarray(lat_deg, dtype=np.float64))  # (ny,)
    sin_phi = np.sin(phi)[:, None]
    cos_phi = np.cos(phi)[:, None]
    cos_h = np.cos(ha)[None, :]

    cos_sza = sin_phi * np.sin(decl) + cos_phi * np.cos(decl) * cos_h
    # Night cells: set SZA=90 rather than >90 so lookup falls at LUT edge.
    cos_sza = np.clip(cos_sza, 0.0, 1.0)
    return np.rad2deg(np.arccos(cos_sza))


def _interp_lut(lut: PhotolysisLUT, sza_deg: np.ndarray,
                o3_col_du: Union[float, np.ndarray],
                table: str) -> np.ndarray:
    """Bilinear interpolation over (SZA, O3 column) from a LUT table."""
    arr = getattr(lut, table)          # (n_sza, n_o3)
    sza_clip = np.clip(sza_deg, lut.sza_axis[0], lut.sza_axis[-1])
    if np.isscalar(o3_col_du):
        col_idx = float(np.searchsorted(lut.o3_col_axis, o3_col_du))
        col_idx = max(0.0, min(col_idx, len(lut.o3_col_axis) - 1))
        col_lo = int(np.floor(col_idx))
        col_hi = min(col_lo + 1, len(lut.o3_col_axis) - 1)
        if col_hi == col_lo:
            return np.interp(sza_clip, lut.sza_axis, arr[:, col_lo])
        j_lo = np.interp(sza_clip, lut.sza_axis, arr[:, col_lo])
        j_hi = np.interp(sza_clip, lut.sza_axis, arr[:, col_hi])
        # Linear column interpolation
        col_denom = lut.o3_col_axis[col_hi] - lut.o3_col_axis[col_lo]
        if col_denom == 0:
            return j_lo
        w = (o3_col_du - lut.o3_col_axis[col_lo]) / col_denom
        return j_lo * (1.0 - w) + j_hi * w

    # Per-cell O3 column: use scipy for vectorised bilinear interpolation.
    from scipy.interpolate import RegularGridInterpolator
    interp = RegularGridInterpolator(
        (lut.sza_axis, lut.o3_col_axis), arr,
        bounds_error=False, fill_value=None,
    )
    o3_clip = np.clip(np.asarray(o3_col_du, dtype=np.float64),
                      lut.o3_col_axis[0], lut.o3_col_axis[-1])
    pts = np.column_stack([sza_clip.ravel(), o3_clip.ravel()])
    return interp(pts).reshape(sza_clip.shape)


def j_per_bin(lut: PhotolysisLUT,
              lat_deg: np.ndarray,
              lon_deg: np.ndarray,
              bin_center_utc: datetime,
              cloud_fraction_2d: Optional[np.ndarray] = None,
              o3_column_du: Union[float, np.ndarray] = 300.0,
              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                          np.ndarray, np.ndarray, np.ndarray]:
    """Compute j_NO2, j_O1D, j_NO3, j_HONO, j_HCHO, j_H2O2 for one bin.

    Returns
    -------
    jNO2, jO1D, jNO3, jHONO, jHCHO, jH2O2, sza_deg : each (ny, nx).
    jNO3 / jHONO / jHCHO / jH2O2 are 0 if the LUT lacks the corresponding
    field.
    """
    sza_deg = solar_zenith_angle(lat_deg, lon_deg, bin_center_utc)
    day_mask = sza_deg < 90.0

    jNO2 = np.zeros_like(sza_deg)
    jO1D = np.zeros_like(sza_deg)
    jNO3 = np.zeros_like(sza_deg)
    jHONO = np.zeros_like(sza_deg)
    jHCHO = np.zeros_like(sza_deg)
    jH2O2 = np.zeros_like(sza_deg)
    if day_mask.any():
        jNO2_clear = _interp_lut(lut, sza_deg, o3_column_du, "jNO2_clear")
        jO1D_clear = _interp_lut(lut, sza_deg, o3_column_du, "jO1D_clear")
        if cloud_fraction_2d is not None:
            cf = np.clip(np.asarray(cloud_fraction_2d), 0.0, 1.0)
            att = 1.0 - 0.7 * cf
        else:
            att = np.ones_like(sza_deg)
        jNO2 = np.where(day_mask, jNO2_clear * att, 0.0)
        jO1D = np.where(day_mask, jO1D_clear * att, 0.0)
        if lut.jNO3_clear is not None:
            jNO3_clear = _interp_lut(lut, sza_deg, o3_column_du, "jNO3_clear")
            jNO3 = np.where(day_mask, jNO3_clear * att, 0.0)
        if lut.jHONO_clear is not None:
            jHONO_clear = _interp_lut(lut, sza_deg, o3_column_du, "jHONO_clear")
            jHONO = np.where(day_mask, jHONO_clear * att, 0.0)
        if lut.jHCHO_clear is not None:
            jHCHO_clear = _interp_lut(lut, sza_deg, o3_column_du, "jHCHO_clear")
            jHCHO = np.where(day_mask, jHCHO_clear * att, 0.0)
        if lut.jH2O2_clear is not None:
            jH2O2_clear = _interp_lut(lut, sza_deg, o3_column_du, "jH2O2_clear")
            jH2O2 = np.where(day_mask, jH2O2_clear * att, 0.0)

    return jNO2, jO1D, jNO3, jHONO, jHCHO, jH2O2, sza_deg


def j_all_bins(lut: PhotolysisLUT,
               lat_deg: np.ndarray,
               lon_deg: np.ndarray,
               year: int, month: int, day: int,
               cloud_fraction_per_bin: Optional[np.ndarray] = None,
               o3_column_du: Union[float, np.ndarray] = 300.0,
               n_subtimes: int = 7,
               ) -> dict:
    """Compute bin-averaged j_NO2, j_O1D, j_NO3 at the 8 UTC-3h bins.

    Bin centres are A3_CENTER_MINUTES = [90, 270, 450, 630, 810, 990, 1170, 1350]
    minutes past midnight UTC (01:30, 04:30, ..., 22:30).

    Each bin j is averaged over `n_subtimes` equally-spaced sub-instances
    within the 3h window. This is important at twilight bins (dawn/dusk)
    where j(SZA) varies by orders of magnitude across the window; evaluating
    only at the bin centre (`n_subtimes=1`) biases dawn bins low and
    afternoon bins high vs GC's ~10-min chemistry step. The default
    `n_subtimes=7` (approx 25-min sampling across 3h) tracks the rapid
    cos(SZA) swing through sunrise/sunset cleanly.

    Parameters
    ----------
    lut : PhotolysisLUT
    lat_deg, lon_deg : 1D grid arrays
    year, month, day : representative date (typically mid-month)
    cloud_fraction_per_bin : (8, ny, nx) in [0, 1] or None
    o3_column_du : scalar or (ny, nx)
    n_subtimes : int
        Number of equally-spaced sub-sample times within each 3h bin window.
        1 = bin-centre only (legacy behaviour, biased at twilight).

    Returns
    -------
    dict with keys:
        jNO2 : (8, ny, nx) s^-1, bin-averaged
        jO1D : (8, ny, nx) s^-1, bin-averaged
        jNO3 : (8, ny, nx) s^-1, bin-averaged
        sza  : (8, ny, nx) degrees, bin-centre value (reference only)
    """
    from datetime import timedelta, timezone
    from orbit.solar import A3_CENTER_MINUTES

    BIN_WIDTH_MIN = 180  # 3 hours = 180 minutes
    HALF_WIDTH = BIN_WIDTH_MIN / 2.0

    ny, nx = len(lat_deg), len(lon_deg)
    jNO2 = np.zeros((8, ny, nx))
    jO1D = np.zeros((8, ny, nx))
    jNO3 = np.zeros((8, ny, nx))
    jHONO = np.zeros((8, ny, nx))
    jHCHO = np.zeros((8, ny, nx))
    jH2O2 = np.zeros((8, ny, nx))
    sza = np.zeros((8, ny, nx))

    if n_subtimes <= 1:
        offsets_min = np.array([0.0])
    else:
        step = BIN_WIDTH_MIN / n_subtimes
        offsets_min = -HALF_WIDTH + step/2.0 + step * np.arange(n_subtimes)

    base = datetime(year, month, day, tzinfo=timezone.utc)
    for b, minutes in enumerate(A3_CENTER_MINUTES):
        t_center = base + timedelta(minutes=int(minutes))
        cf = None if cloud_fraction_per_bin is None else cloud_fraction_per_bin[b]

        jNO2_sum = np.zeros((ny, nx))
        jO1D_sum = np.zeros((ny, nx))
        jNO3_sum = np.zeros((ny, nx))
        jHONO_sum = np.zeros((ny, nx))
        jHCHO_sum = np.zeros((ny, nx))
        jH2O2_sum = np.zeros((ny, nx))
        for off in offsets_min:
            t_sub = t_center + timedelta(minutes=float(off))
            jn, jo, j3, jh, jhc, jhp, _ = j_per_bin(
                lut, lat_deg, lon_deg, t_sub,
                cloud_fraction_2d=cf, o3_column_du=o3_column_du,
            )
            jNO2_sum += jn
            jO1D_sum += jo
            jNO3_sum += j3
            jHONO_sum += jh
            jHCHO_sum += jhc
            jH2O2_sum += jhp
        jNO2[b] = jNO2_sum / len(offsets_min)
        jO1D[b] = jO1D_sum / len(offsets_min)
        jNO3[b] = jNO3_sum / len(offsets_min)
        jHONO[b] = jHONO_sum / len(offsets_min)
        jHCHO[b] = jHCHO_sum / len(offsets_min)
        jH2O2[b] = jH2O2_sum / len(offsets_min)

        sza[b] = solar_zenith_angle(lat_deg, lon_deg, t_center)

    return {"jNO2": jNO2, "jO1D": jO1D, "jNO3": jNO3, "jHONO": jHONO,
            "jHCHO": jHCHO, "jH2O2": jH2O2, "sza": sza}
