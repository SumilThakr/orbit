"""Config-driven emission loading for ORBIT.

Dispatches to format-specific loaders (netcdf, shapefile/geopackage) and
accumulates into a flat 6*N emission vector.

Also provides weekly day/night emission loading via load_emissions_weekly()
and diurnal-resolved orbit-style loading via load_emissions_diurnal().
"""

import os
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Dict, List, Optional

import numpy as np

from orbit.core.grid_data import GridData
from orbit.core.indexing import CellIndexer
from orbit.emissions.sources import EmissionSource, N_ORBIT_SPECIES


def _accumulate_sources(
    sources: List[EmissionSource],
    grid: GridData,
    verbose: bool = False,
) -> np.ndarray:
    """Load and accumulate all emission sources into a 4D array.

    Slot 0 (legacy TotalOrg / VBS C*=100) is handled specially:
    sources without a ``voc_parent_class`` set contribute nothing to
    slot 0 (their slot-0 mass is dropped here). Sources WITH a parent
    class also have slot 0 dropped at this point — their VOC mass is
    redistributed across the 5 VBS bins downstream of the loader by
    ``run_orbit.py:_distribute_voc_to_vbs_bins``, where the
    13-species solver layout is available.

    Parameters
    ----------
    sources : list of EmissionSource
    grid : GridData
    verbose : bool

    Returns
    -------
    emis_4d : ndarray, shape (6, nz, ny, nx) in ug/m^3/s
        Slot 0 is always zero on return. The downstream caller is
        responsible for VBS distribution from per-source VOC mass.
    """
    emis_4d = np.zeros((N_ORBIT_SPECIES, grid.nz, grid.ny, grid.nx), dtype=np.float64)

    for source in sources:
        if verbose:
            print(f"  Loading {source.path}")
            print(f"    Format: {source.format}, Units: {source.units}, "
                  f"Elevated: {source.elevated}, "
                  f"voc_parent_class: {source.voc_parent_class}")

        if source.format in ('shapefile', 'geopackage'):
            from orbit.emissions.shapefile import load_shapefile_source
            contrib = load_shapefile_source(source, grid, verbose)
        elif source.format == 'netcdf':
            from orbit.emissions.netcdf import load_netcdf_source
            contrib = load_netcdf_source(source, grid, verbose)
        else:
            if verbose:
                print(f"    Skipping unsupported format: {source.format}")
            continue

        # Drop slot-0 contributions: legacy semantics retained nothing here
        # for non-VOC sources; for VOC sources, the VBS distribution
        # happens downstream of this aggregator (per-source identity is
        # required and is preserved in run_orbit.py).
        contrib[0] = 0.0
        emis_4d += contrib

    return emis_4d


def load_emissions(
    sources: List[EmissionSource],
    grid: GridData,
    indexer: CellIndexer,
    verbose: bool = False,
) -> np.ndarray:
    """Load emissions from all configured sources.

    Returns a flat vector e[6*N] in ug/m^3/s, ordered as
    [species_0(k=0,j=0,i=0..nx-1, ..., k=nz-1,...), species_1(...), ...].

    Parameters
    ----------
    sources : list of EmissionSource
        Emission source configurations
    grid : GridData
        Model grid
    indexer : CellIndexer
        Flat index mapper
    verbose : bool
        Print progress

    Returns
    -------
    ndarray, shape (6*N,)
        Emission rates in ug/m^3/s
    """
    emis_4d = _accumulate_sources(sources, grid, verbose)

    # Flatten to e[6*N]
    N = indexer.N
    e = np.zeros(N_ORBIT_SPECIES * N, dtype=np.float64)
    for s in range(N_ORBIT_SPECIES):
        e[s * N : (s + 1) * N] = emis_4d[s].ravel()
    return e


def preload_monthly_emissions(
    sources: List[EmissionSource],
    grid: GridData,
    n_months: int = 12,
    verbose: bool = False,
) -> dict:
    """Pre-load and regrid emissions for all months (one-time cost).

    Returns a cache dict {month_0based: emis_4d} where each emis_4d is
    shape (6, nz, ny, nx).  This avoids re-opening and regridding the
    same 17 NetCDF files for every (week, time_bin) combination.

    Parameters
    ----------
    sources : list of EmissionSource
    grid : GridData
    n_months : int
        Number of months to load (default 12)
    verbose : bool

    Returns
    -------
    dict {int: ndarray}
        Keys are 0-based month indices, values are (6, nz, ny, nx) arrays.
    """
    cache = {}
    for m in range(n_months):
        if verbose:
            print(f"  Loading month {m + 1}/{n_months}...")
        month_sources = []
        for src in sources:
            ms = EmissionSource(
                path=src.path,
                format=src.format,
                units=src.units,
                elevated=src.elevated,
                averaging_period=src.averaging_period,
                variable_mapping=src.variable_mapping,
                height_variable=src.height_variable,
                layer_index=src.layer_index,
                time_index=m,
            )
            month_sources.append(ms)
        cache[m] = _accumulate_sources(month_sources, grid, verbose=False)
    return cache


def apply_weekly_fractions(
    emis_cache: dict,
    grid: GridData,
    indexer: CellIndexer,
    week: int,
    year: int,
    solar_map_data: dict,
    time_bin: str,
) -> np.ndarray:
    """Blend cached monthly emissions for a given week and apply day/night fraction.

    Uses pre-loaded monthly emissions from preload_monthly_emissions() instead
    of re-opening NetCDF files.

    Parameters
    ----------
    emis_cache : dict {int: ndarray}
        Monthly emission cache from preload_monthly_emissions()
    grid : GridData
    indexer : CellIndexer
    week : int
        1-based week index
    year : int
    solar_map_data : dict
        Loaded solar map (from load_solar_map())
    time_bin : str
        "day" or "night"

    Returns
    -------
    e : ndarray, shape (6*N,)
    """
    if time_bin not in ("day", "night"):
        raise ValueError(f"time_bin must be 'day' or 'night', got '{time_bin}'")

    weeks = _generate_iso_weeks(year)
    if week < 1 or week > len(weeks):
        raise ValueError(f"Week {week} out of range (1-{len(weeks)}) for {year}")
    start, end_excl, _ = weeks[week - 1]

    # Month weights for cross-boundary weeks
    month_weights = _get_month_weights(start, end_excl)

    # Blend cached monthly emissions
    emis_4d = np.zeros((N_ORBIT_SPECIES, grid.nz, grid.ny, grid.nx), dtype=np.float64)
    for month, weight in month_weights.items():
        emis_4d += weight * emis_cache[month - 1]

    # Apply solar day/night fraction
    day_frac = solar_map_data["day_frac_weekly"][week - 1]  # (ny, nx)
    if time_bin == "day":
        frac = day_frac
    else:
        frac = 1.0 - day_frac

    N = indexer.N
    e = np.zeros(N_ORBIT_SPECIES * N, dtype=np.float64)
    for s in range(N_ORBIT_SPECIES):
        scaled = emis_4d[s] * frac[np.newaxis, :, :]
        e[s * N : (s + 1) * N] = scaled.ravel()

    return e


# ── Weekly helpers ────────────────────────────────────────────────────────


def _generate_iso_weeks(year: int) -> list:
    """Generate ISO-week-aligned date ranges clipped to a calendar year.

    Each entry covers the days that fall within the calendar year,
    so the first and last entries may be partial weeks. The week_number
    is the ISO 8601 week of the entry's start date, which means the
    first entry for a year like 2016 (starts Friday) will carry the
    ISO week number from the *previous* year (e.g. W53 of 2015 for
    Jan 1-3, 2016). Entries are always in chronological order.

    Parameters
    ----------
    year : int

    Returns
    -------
    list of (start_date, end_date_excl, week_number)
    """
    jan1 = date(year, 1, 1)
    dec31 = date(year, 12, 31)

    # Find Monday of the ISO week containing Jan 1
    monday = jan1 - timedelta(days=jan1.weekday())

    weeks = []
    while monday <= dec31:
        week_end = monday + timedelta(days=7)  # exclusive end (next Monday)
        # Clip to calendar year
        start = max(monday, jan1)
        end = min(week_end, dec31 + timedelta(days=1))  # exclusive
        if start < end:
            _, wnum, _ = start.isocalendar()
            weeks.append((start, end, wnum))
        monday = week_end

    return weeks


def _get_month_weights(start_date: date, end_date_excl: date) -> dict:
    """Compute day-weights for each month overlapping a date range.

    Parameters
    ----------
    start_date : date
    end_date_excl : date (exclusive)

    Returns
    -------
    dict {month: weight} where weights sum to 1.0
    """
    day = start_date
    month_days = defaultdict(int)
    total_days = 0
    while day < end_date_excl:
        month_days[day.month] += 1
        total_days += 1
        day += timedelta(days=1)

    return {m: d / total_days for m, d in month_days.items()}


def load_solar_map(path: str) -> dict:
    """Load a pre-computed solar map NPZ.

    Parameters
    ----------
    path : str
        Path to solar_map_{year}.npz

    Returns
    -------
    dict with keys: day_frac_weekly (n_weeks, ny, nx), lat, lon, weeks, year
    """
    d = np.load(path)
    return {
        "day_frac_weekly": d["day_frac_weekly"],
        "lat": d["lat"],
        "lon": d["lon"],
        "weeks": d["weeks"],
        "year": int(d["year"]),
    }


def load_emissions_weekly(
    sources: List[EmissionSource],
    grid: GridData,
    indexer: CellIndexer,
    week: int,
    year: int,
    solar_map_path: str,
    time_bin: str,
    verbose: bool = False,
) -> np.ndarray:
    """Load weekly day or night emission vector.

    Computes month-weighted emissions for weeks that cross month
    boundaries, then applies solar day/night fraction from a
    pre-computed solar map.

    Parameters
    ----------
    sources : list of EmissionSource
        Emission source configurations. Each source's time_index will be
        overridden per-month internally; pass sources with time_index=None.
    grid : GridData
    indexer : CellIndexer
    week : int
        Week number (1-based index into _generate_iso_weeks output)
    year : int
    solar_map_path : str
        Path to solar_map_{year}.npz
    time_bin : str
        "day" or "night"
    verbose : bool

    Returns
    -------
    e : ndarray, shape (6*N,)
        Emission rates in ug/m^3/s, with day/night fraction applied.
    """
    if time_bin not in ("day", "night"):
        raise ValueError(f"time_bin must be 'day' or 'night', got '{time_bin}'")

    # Get week date range
    weeks = _generate_iso_weeks(year)
    if week < 1 or week > len(weeks):
        raise ValueError(f"Week {week} out of range (1-{len(weeks)}) for {year}")
    start, end_excl, _ = weeks[week - 1]

    # Month weights for cross-boundary weeks
    month_weights = _get_month_weights(start, end_excl)

    # Weight-averaged 4D emissions across months
    emis_4d = np.zeros((N_ORBIT_SPECIES, grid.nz, grid.ny, grid.nx), dtype=np.float64)
    for month, weight in month_weights.items():
        # Override time_index on each source for this month
        month_sources = []
        for src in sources:
            ms = EmissionSource(
                path=src.path,
                format=src.format,
                units=src.units,
                elevated=src.elevated,
                averaging_period=src.averaging_period,
                variable_mapping=src.variable_mapping,
                height_variable=src.height_variable,
                layer_index=src.layer_index,
                time_index=month - 1,
            )
            month_sources.append(ms)
        emis_4d += weight * _accumulate_sources(month_sources, grid, verbose)

    # Load solar map and validate grid alignment
    solar = load_solar_map(solar_map_path)
    day_frac = solar["day_frac_weekly"][week - 1]  # (ny, nx)
    assert day_frac.shape == (grid.ny, grid.nx), \
        f"Solar map shape {day_frac.shape} != grid ({grid.ny}, {grid.nx})"

    # Apply day/night fraction (surface layer only for surface emissions,
    # but apply uniformly to all layers for elevated sources)
    if time_bin == "day":
        frac = day_frac
    else:
        frac = 1.0 - day_frac

    # Apply fraction and flatten to e[6*N]
    N = indexer.N
    e = np.zeros(N_ORBIT_SPECIES * N, dtype=np.float64)
    for s in range(N_ORBIT_SPECIES):
        # frac is (ny, nx), broadcast across nz layers
        scaled = emis_4d[s] * frac[np.newaxis, :, :]  # (nz, ny, nx)
        e[s * N : (s + 1) * N] = scaled.ravel()

    return e


# ── Diurnal-resolved emissions (orbit driver) ─────────────────────────────


_DEFAULT_FLAT_PROFILE = np.ones(24, dtype=np.float64)

# Stack-height tiers appended to the CEDS basenames by the sector split
# (surface = agr/tra/rco/slv/shp, low = wst, medium = ind, high = ene).
_TIER_SUFFIXES = ("_surface", "_low", "_medium", "_high")


def _strip_tier_suffix(basename: str) -> str:
    """``ceds_pm25_anthro_2022_monthly_high.nc`` -> ``..._monthly.nc``."""
    root, ext = os.path.splitext(basename)
    for suffix in _TIER_SUFFIXES:
        if root.endswith(suffix):
            return root[:-len(suffix)] + ext
    return basename


DYNAMIC_PROFILE_KINDS = ("henry_temperature",)


@dataclass(frozen=True)
class DynamicProfile:
    """A diurnal profile computed from the bin meteorology, not a 24-h table."""
    name: str
    kind: str


def henry_temperature_weight(T_kelvin: np.ndarray) -> np.ndarray:
    """Henry's-law temperature factor for NH3 over a livestock/manure surface,
    H(T) = (161500 / T) exp(-10380 / T) (Nemitz et al. 2000, as used by Zhu
    et al. 2015, Eq. 3). Only its diurnal shape matters here."""
    T = np.asarray(T_kelvin, dtype=np.float64)
    return 161500.0 / T * np.exp(-10380.0 / T)


def dynamic_profile_factors(profile: DynamicProfile, grids, n_bins: int) -> np.ndarray:
    """Per-bin, per-cell multipliers (n_bins, ny, nx) with bin-mean 1 per cell.

    ``grids`` is the list of per-bin grids. If a single grid is passed for
    all bins (no per-bin meteorology), the profile degenerates to flat and a
    warning is issued, because the emission budget would otherwise be
    altered by a meteorology that does not vary with the bin.
    """
    if len(grids) != n_bins:
        raise ValueError(f"dynamic profile '{profile.name}' needs {n_bins} bin "
                         f"grids, got {len(grids)}")
    g0 = grids[0]
    if all(g is g0 for g in grids):
        warnings.warn(f"Diurnal profile '{profile.name}' ({profile.kind}) needs "
                      f"per-bin meteorology but one grid was given for every "
                      f"bin; emitting bin-flat.", stacklevel=2)
        return np.ones((n_bins, g0.ny, g0.nx), dtype=np.float64)
    if profile.kind == "henry_temperature":
        w = np.stack([henry_temperature_weight(g.Temperature[0]) for g in grids],
                     axis=0)                                   # (n_bins, ny, nx)
    else:
        raise ValueError(f"unknown dynamic profile kind '{profile.kind}'")
    mean = w.mean(axis=0, keepdims=True)
    return np.where(mean > 0.0, w / np.where(mean > 0.0, mean, 1.0), 1.0)


def scale_by_bin_factor(factor, arr) -> np.ndarray:
    """Multiply an emission slab by a bin factor that is either a scalar or a
    per-cell (ny, nx) map (the last two axes of ``arr`` are (ny, nx))."""
    f = np.asarray(factor, dtype=np.float64)
    if f.ndim == 0:
        return float(f) * arr
    return arr * f.reshape((1,) * (arr.ndim - 2) + f.shape)


@dataclass
class DiurnalConfig:
    """Configuration for diurnal-resolved emissions in the orbit solver.

    Profiles are 24-element local-time multipliers with day-mean = 1.0.
    They are sampled into N_BINS UTC bins via a single SAS-wide IST offset.
    """
    ist_offset_hours: float = 5.5
    profiles: Dict[str, np.ndarray] = field(default_factory=dict)
    # Meteorology-driven profiles: name -> {"kind": ...}. These have no fixed
    # 24-h shape; their per-bin, per-cell factors are computed from the bin
    # grids by ``dynamic_profile_factors``. Kinds:
    #   henry_temperature  -- w(T) = (161500 / T) exp(-10380 / T), the
    #       Henry's-law temperature factor of the GEOS-Chem livestock NH3
    #       diurnal scheme (Zhu et al. 2015, ACP 15, 12823, Eqs. 2-3, after
    #       Nemitz et al. 2000), evaluated on each bin's surface air
    #       temperature and normalised to a bin-mean of 1 per cell. The
    #       aerodynamic-resistance factor of Eq. 2 is not applied.
    dynamic_profiles: Dict[str, Dict[str, str]] = field(default_factory=dict)
    source_profiles: Dict[str, str] = field(default_factory=dict)
    source_profiles_by_month: Dict[int, Dict[str, str]] = field(default_factory=dict)
    # basenames already warned about, so the fall-through warning fires once
    # per source rather than once per (source, bin, closure iteration).
    _unmatched_warned: set = field(default_factory=set, repr=False, compare=False)

    def profile_for(self, source_path: str, month: int) -> np.ndarray:
        """Resolve the 24-h multiplier for a source/month.

        Lookup order, first hit wins:

          1. exact basename, this month's override
          2. exact basename
          3. tier-stripped stem, this month's override
          4. tier-stripped stem
          5. bin-flat, with a warning

        The stem step exists because the CEDS sources were split into
        stack-height tiers (``..._surface.nc``, ``_low``, ``_medium``,
        ``_high``) after this config was written, and the YAML still keys on
        the pre-split basenames. Exact-match-only lookup therefore returned
        bin-flat for *every* CEDS species, silently, so the Dec-Feb
        residential override -- roughly 10x peak-to-trough for IGP winter
        heating -- was never applied to any production run.

        Exact keys still win, so a tier can be given its own profile where
        the sector mix justifies it (``_high`` is CEDS ``ene``, ``_medium``
        is ``ind``; both are industrial rather than traffic-shaped).

        Falling through to flat now warns. Sources that genuinely have no
        diurnal signal should be mapped to the ``flat`` profile explicitly,
        which is silent -- that way a new or renamed file is loud instead of
        quietly losing its profile the way the tier split did.
        """
        basename = os.path.basename(source_path)
        month_overrides = self.source_profiles_by_month.get(int(month), {})
        stem = _strip_tier_suffix(basename)

        name = (month_overrides.get(basename)
                or self.source_profiles.get(basename))
        if name is None and stem != basename:
            name = (month_overrides.get(stem)
                    or self.source_profiles.get(stem))

        if name is None:
            if basename not in self._unmatched_warned:
                self._unmatched_warned.add(basename)
                warnings.warn(
                    f"No diurnal profile matched {basename} (stem {stem}); "
                    f"emitting bin-flat. If that is intended, map it to "
                    f"'flat' explicitly in source_profiles to silence this.",
                    stacklevel=2)
            return _DEFAULT_FLAT_PROFILE
        if name in self.dynamic_profiles:
            return DynamicProfile(name, self.dynamic_profiles[name]["kind"])
        if name not in self.profiles:
            warnings.warn(f"Diurnal profile '{name}' referenced by {basename} "
                          f"not defined; falling back to flat")
            return _DEFAULT_FLAT_PROFILE
        return self.profiles[name]

    def resolve_all(self, source_paths, month: int):
        """Report which profile each source resolves to: [(basename, name)].

        Pre-flight introspection for run scripts -- the check that would have
        caught the tier-split mismatch immediately. ``None`` means bin-flat.
        """
        out = []
        for p in source_paths:
            basename = os.path.basename(p)
            stem = _strip_tier_suffix(basename)
            mo = self.source_profiles_by_month.get(int(month), {})
            name = (mo.get(basename) or self.source_profiles.get(basename)
                    or (mo.get(stem) if stem != basename else None)
                    or (self.source_profiles.get(stem) if stem != basename else None))
            out.append((basename, name))
        return out

    @classmethod
    def from_yaml(cls, path: str) -> "DiurnalConfig":
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        ist = float(data.get("ist_offset_hours", 5.5))
        raw_profiles = data.get("profiles", {}) or {}
        profiles: Dict[str, np.ndarray] = {}
        dynamic: Dict[str, Dict[str, str]] = {}
        for name, body in raw_profiles.items():
            if isinstance(body, dict) and "kind" in body:
                kind = str(body["kind"])
                if kind not in DYNAMIC_PROFILE_KINDS:
                    raise ValueError(
                        f"Diurnal profile '{name}': unknown kind '{kind}'; "
                        f"valid: {sorted(DYNAMIC_PROFILE_KINDS)}")
                dynamic[name] = {"kind": kind}
                continue
            if isinstance(body, dict):
                hours = body.get("hours")
            else:
                hours = body
            if hours is None:
                raise ValueError(f"Diurnal profile '{name}' has no 'hours' entry")
            arr = np.asarray(hours, dtype=np.float64)
            if arr.shape != (24,):
                raise ValueError(
                    f"Diurnal profile '{name}' must be length 24, "
                    f"got shape {arr.shape}"
                )
            mean = float(arr.mean())
            if mean <= 0:
                raise ValueError(f"Diurnal profile '{name}' has non-positive mean")
            if abs(mean - 1.0) > 1e-6:
                warnings.warn(f"Diurnal profile '{name}' mean={mean:.6f} != 1.0; "
                              f"renormalising to preserve emissions budget")
                arr = arr / mean
            profiles[name] = arr
        sp = {k: str(v) for k, v in (data.get("source_profiles") or {}).items()}
        sp_by_m = {}
        for m, mapping in (data.get("source_profiles_by_month") or {}).items():
            sp_by_m[int(m)] = {k: str(v) for k, v in (mapping or {}).items()}
        return cls(
            ist_offset_hours=ist,
            profiles=profiles,
            dynamic_profiles=dynamic,
            source_profiles=sp,
            source_profiles_by_month=sp_by_m,
        )


def _local_to_utc_bin_factors(
    profile_24h: np.ndarray,
    ist_offset_hours: float,
    n_bins: int = 8,
) -> np.ndarray:
    """Convert a 24-hour local-time profile (mean=1) into n_bins UTC factors.

    UTC bin tau spans [3*tau, 3*(tau+1)) UTC, which is
    [3*tau + offset, 3*(tau+1) + offset) in local time (mod 24).
    Returns shape (n_bins,) with mean ≈ 1.0.

    The 0.5-hour IST tail makes the local-time window straddle hour
    boundaries — handled by averaging via fractional weights summing to
    the bin width (3 h).
    """
    profile = np.asarray(profile_24h, dtype=np.float64)
    if profile.shape != (24,):
        raise ValueError(f"profile_24h must be length 24, got {profile.shape}")
    bin_width = 24.0 / n_bins
    factors = np.zeros(n_bins, dtype=np.float64)
    for tau in range(n_bins):
        utc_start = bin_width * tau
        local_start = (utc_start + ist_offset_hours) % 24.0
        # Walk fractional sub-windows of width up to 1h until we've covered
        # `bin_width` hours.  Each sub-window contributes (sub_width *
        # profile[hour_idx]) to the running mean.
        remaining = bin_width
        cursor = local_start
        weighted = 0.0
        while remaining > 1e-12:
            hour_idx = int(np.floor(cursor)) % 24
            next_hour = np.floor(cursor) + 1.0
            sub_width = min(next_hour - cursor, remaining)
            weighted += sub_width * profile[hour_idx]
            cursor = (cursor + sub_width) % 24.0
            remaining -= sub_width
        factors[tau] = weighted / bin_width
    return factors


def as_bin_grids(grid_or_grids, n_bins: int) -> List[GridData]:
    """Normalise a single grid or a per-bin list of grids to a list of n_bins."""
    if isinstance(grid_or_grids, (list, tuple)):
        grids = list(grid_or_grids)
        if len(grids) != n_bins:
            raise ValueError(f"expected {n_bins} bin grids, got {len(grids)}")
        return grids
    return [grid_or_grids] * n_bins


def _source_needs_bin_met(source: EmissionSource) -> bool:
    """True for elevated sources whose injection layer comes from the ASME
    plume rise, which depends on each bin's wind, temperature and stability."""
    return bool(source.elevated and source.layer_index is None
                and source.stack_diam > 0 and source.stack_temp > 0
                and source.stack_vel > 0)


def load_emissions_diurnal(
    sources: List[EmissionSource],
    grid,
    indexer: CellIndexer,
    diurnal_cfg: Optional[DiurnalConfig] = None,
    month: int = 1,
    n_bins: int = 8,
    verbose: bool = False,
) -> np.ndarray:
    """Build per-bin emission vectors for the orbit solver.

    Parameters
    ----------
    grid : GridData or list of GridData
        One grid, or the list of per-bin grids. With the list, elevated
        sources compute their plume rise from each bin's meteorology (1.0.3;
        through 1.0.2 the 00-03 UTC bin's injection layer was used for all
        8 bins) and meteorology-driven diurnal profiles can be evaluated.

    Returns
    -------
    e_per_bin : ndarray, shape (n_bins, N_ORBIT_SPECIES * N), in ug/m^3/s
        Emission rates per UTC bin.  Bin-mean equals the legacy
        ``load_emissions`` output element-wise when every profile has
        day-mean = 1.0 — the budget is preserved.
    """
    N = indexer.N
    grids = as_bin_grids(grid, n_bins)
    grid = grids[0]
    per_bin_met = any(g is not grid for g in grids)
    e_per_bin = np.zeros((n_bins, N_ORBIT_SPECIES * N), dtype=np.float64)

    # Per-source: either read pre-binned (n_bins, lat, lon) slabs and
    # accumulate, or read monthly slab and apply per-bin profile factors.
    # Slot 0 contributions are zeroed (see _accumulate_sources rationale).
    for source in sources:
        if verbose:
            print(f"  Loading {source.path} (diurnal, "
                  f"voc_parent_class={source.voc_parent_class})")

        if source.format in ('shapefile', 'geopackage'):
            from orbit.emissions.shapefile import load_shapefile_source
            monthly_4d = load_shapefile_source(source, grid, verbose)
            monthly_4d[0] = 0.0
            factors = _factors_for_source(source, diurnal_cfg, month, n_bins, grids)
            if verbose:
                print(f"    profile factors (UTC) = {_describe_factors(factors)}")
            for tau in range(n_bins):
                slab = scale_by_bin_factor(factors[tau], monthly_4d)
                _accumulate_into_bin(e_per_bin, tau, slab, N)
            continue

        if source.format != 'netcdf':
            if verbose:
                print(f"    Skipping unsupported format: {source.format}")
            continue

        if source.bin_axis:
            from orbit.emissions.netcdf import load_netcdf_source_per_bin
            slabs = load_netcdf_source_per_bin(source, grid, n_bins, verbose)
            # slabs shape: (n_bins, N_ORBIT_SPECIES, nz, ny, nx)
            slabs[:, 0] = 0.0
            for tau in range(n_bins):
                _accumulate_into_bin(e_per_bin, tau, slabs[tau], N)
        else:
            from orbit.emissions.netcdf import load_netcdf_source
            factors = _factors_for_source(source, diurnal_cfg, month, n_bins, grids)
            if verbose:
                print(f"    profile factors (UTC) = {_describe_factors(factors)}")
            if per_bin_met and _source_needs_bin_met(source):
                # Plume rise per bin: the injection layer (and the cell
                # volume it is divided by) follow this bin's meteorology.
                if verbose:
                    print("    plume rise evaluated per bin")
                for tau in range(n_bins):
                    monthly_4d = load_netcdf_source(source, grids[tau], False)
                    monthly_4d[0] = 0.0
                    slab = scale_by_bin_factor(factors[tau], monthly_4d)
                    _accumulate_into_bin(e_per_bin, tau, slab, N)
            else:
                monthly_4d = load_netcdf_source(source, grid, verbose)
                monthly_4d[0] = 0.0
                for tau in range(n_bins):
                    slab = scale_by_bin_factor(factors[tau], monthly_4d)
                    _accumulate_into_bin(e_per_bin, tau, slab, N)

    return e_per_bin


def _describe_factors(factors: np.ndarray) -> str:
    f = np.asarray(factors)
    if f.ndim == 1:
        return str(np.round(f, 3))
    return (f"per-cell, bin means over the domain "
            f"{np.round(f.mean(axis=(1, 2)), 3)}")


def _factors_for_source(
    source: EmissionSource,
    diurnal_cfg: Optional[DiurnalConfig],
    month: int,
    n_bins: int,
    grids=None,
) -> np.ndarray:
    """Per-bin multipliers for a source: (n_bins,) for a tabulated profile,
    (n_bins, ny, nx) for a meteorology-driven one (needs ``grids``, the
    per-bin grid list; without it the dynamic profile falls back to flat
    with a warning)."""
    if diurnal_cfg is None:
        return np.ones(n_bins, dtype=np.float64)
    profile = diurnal_cfg.profile_for(source.path, month)
    if isinstance(profile, DynamicProfile):
        if grids is None:
            warnings.warn(f"Diurnal profile '{profile.name}' ({profile.kind}) "
                          f"for {os.path.basename(source.path)} needs the bin "
                          f"grids; none given, emitting bin-flat.", stacklevel=2)
            return np.ones(n_bins, dtype=np.float64)
        return dynamic_profile_factors(profile, list(grids), n_bins)
    return _local_to_utc_bin_factors(
        profile, diurnal_cfg.ist_offset_hours, n_bins=n_bins
    )


def _accumulate_into_bin(e_per_bin: np.ndarray, tau: int,
                          emis_4d: np.ndarray, N: int) -> None:
    """In-place add `emis_4d` (N_ORBIT_SPECIES, nz, ny, nx) into e_per_bin[tau]."""
    for s in range(N_ORBIT_SPECIES):
        e_per_bin[tau, s * N:(s + 1) * N] += emis_4d[s].ravel()
