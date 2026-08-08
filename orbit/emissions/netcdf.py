"""NetCDF emission loading for ORBIT.

Loads emissions, adapted for
ORBIT's 6-species merged convention with element conversion applied on the
legacy indices before merging into ORBIT species.
"""

import os
import warnings
from typing import Dict, List, Optional

import numpy as np

from orbit.emissions.sources import (
    CONVERT_COLS,
    ELEMENT_CONVERSION,
    NETCDF_SPECIES_PATTERNS,
    N_ORBIT_SPECIES,
    LEGACY9_TO_ORBIT,
    SPECIES_MAP,
    UNIT_CONVERSIONS,
    EmissionSource,
    compute_cell_areas,
)
from orbit.emissions.plumerise import find_injection_layer


# Share of a source file's domain total above which a single native cell is
# treated as a spatial-allocation artefact rather than a real point source.
# OFF (0) by default: the published South Asia 2022 baseline was produced
# with the screen disabled — two flagged-but-retained cells (offshore
# Mumbai high-tier, India/Myanmar-border low-tier) are stated in the
# methods — and the default configuration must reproduce that baseline
# exactly, including in the marginal mode's perturbation assembly.
# Set ORBIT_EMISSION_SPIKE_FRAC=0.05 to screen your own emission inputs.
_DEFAULT_SPIKE_FRAC = 0.0


def _spike_threshold() -> float:
    raw = os.environ.get("ORBIT_EMISSION_SPIKE_FRAC")
    if raw is None:
        return _DEFAULT_SPIKE_FRAC
    try:
        val = float(raw)
    except ValueError:
        warnings.warn(f"ORBIT_EMISSION_SPIKE_FRAC={raw!r} is not a number; "
                      f"using {_DEFAULT_SPIKE_FRAC}")
        return _DEFAULT_SPIKE_FRAC
    if not 0.0 <= val <= 1.0:
        raise ValueError(f"ORBIT_EMISSION_SPIKE_FRAC must be in [0, 1], got {val}")
    return val


def _neighbourhood_mean(data, radius: int = 2):
    """Mean of the (2r+1)^2 box around each cell, EXCLUDING the cell itself.

    Exact (a plain shifted-sum, no filter approximation) and edge-aware: the
    divisor counts only in-domain neighbours, so a coastal or corner cell is
    not diluted by phantom zeros.
    """
    h, w = data.shape
    pad = np.pad(data, radius, mode="constant", constant_values=0.0)
    cnt = np.pad(np.ones_like(data, dtype=np.float64), radius,
                 mode="constant", constant_values=0.0)
    n = 2 * radius + 1
    s = np.zeros_like(data, dtype=np.float64)
    c = np.zeros_like(data, dtype=np.float64)
    for dj in range(n):
        for di in range(n):
            s += pad[dj:dj + h, di:di + w]
            c += cnt[dj:dj + h, di:di + w]
    s -= data
    c -= 1.0
    return s / np.maximum(c, 1.0)


def isolation_ratio(data, radius: int = 2):
    """Each cell's value divided by the mean of its surrounding neighbourhood.

    The physical discriminator the domain-share screen lacks. A real city or
    power cluster sits in a populated neighbourhood and scores 3-7. A
    spatial-allocation artefact -- a national sector total dropped on one
    cell -- has nothing around it and scores in the hundreds to millions.
    The band between is empty, so the threshold is not delicate.

    Zero-neighbourhood cells return ``inf``; callers must combine this with a
    mass floor, or a single tiny cell in an empty region would be "flagged"
    on a ratio that means nothing.
    """
    neigh = _neighbourhood_mean(np.asarray(data, dtype=np.float64), radius)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(neigh > 0.0, data / neigh,
                         np.where(data > 0.0, np.inf, 0.0))
    return ratio


def screen_by_isolation(data, ratio_threshold=50.0, min_share=0.005,
                        radius=2, label="", lat=None, lon=None,
                        verbose=False):
    """Cap cells that are both spatially isolated AND carry real mass.

    Two conditions, deliberately:

    * ``isolation_ratio > ratio_threshold`` -- the cell is unlike its
      surroundings, which is what an allocation artefact looks like and what
      a real source does not.
    * ``share of domain total > min_share`` -- so the screen only ever
      touches cells that matter. Without this, an isolated village cell in an
      otherwise empty region has an enormous ratio and no consequence.

    A flagged cell is capped at the larger of (its neighbourhood mean x
    ratio_threshold) and ``min_share`` of the domain total, and the excess is
    redistributed proportionally over the remaining non-zero cells. Domain
    total is conserved exactly, as with the domain-share screen.

    Returns ``(screened, records)`` in the same shape as
    ``deconcentrate_allocation_artefacts`` so the two are interchangeable.
    """
    data = np.asarray(data, dtype=np.float64)
    total = float(np.sum(data))
    if not np.isfinite(total) or total <= 0.0 or ratio_threshold <= 0.0:
        return data, []

    ratio = isolation_ratio(data, radius)
    flagged = (ratio > ratio_threshold) & (data > min_share * total)
    if not np.any(flagged):
        return data, []

    donors = (data > 0.0) & ~flagged
    donor_mass = float(np.sum(data[donors]))
    if donor_mass <= 0.0:
        warnings.warn(f"{label}: isolation screen flagged "
                      f"{int(flagged.sum())} cell(s) but no donors exist; "
                      f"leaving the field unchanged")
        return data, []

    # Redistribution is proportional, so a thin donor pool means the excess
    # lands almost entirely on a handful of cells. Harmless in a real
    # inventory (thousands of non-zero cells) but a silent multi-order
    # inflation otherwise, which is precisely the failure mode this project
    # keeps paying for. Warn rather than quietly reshaping the field.
    n_donors = int(donors.sum())
    if n_donors < 10:
        warnings.warn(
            f"{label}: only {n_donors} donor cell(s) available to absorb the "
            f"excess; the redistribution will change them substantially. "
            f"Check that this field is what you think it is.")

    neigh = _neighbourhood_mean(data, radius)
    cap = np.maximum(neigh * ratio_threshold, min_share * total)

    out = data.copy()
    excess = float(np.sum(out[flagged] - cap[flagged]))
    out[flagged] = cap[flagged]
    out[donors] += out[donors] * (excess / donor_mass)

    records = []
    for j, i in zip(*np.nonzero(flagged)):
        rec = {
            "lat": float(lat[j]) if lat is not None else None,
            "lon": float(lon[i]) if lon is not None else None,
            "share": float(data[j, i] / total),
            "isolation": float(ratio[j, i]),
            "value": float(data[j, i]),
            "capped": float(cap[j, i]),
        }
        records.append(rec)
        where = (f" at {rec['lat']:.2f}N {rec['lon']:.2f}E"
                 if rec["lat"] is not None else f" at index ({j}, {i})")
        iso_txt = ("an empty neighbourhood"
                   if not np.isfinite(rec["isolation"])
                   else f"{rec['isolation']:.0f}x its neighbourhood")
        msg = (f"{label}: capped an isolated cell{where} -- "
               f"{iso_txt}, "
               f"{rec['share']:.1%} of the domain total; "
               f"excess redistributed over {int(donors.sum())} cells "
               f"(domain total conserved)")
        warnings.warn(msg)
        if verbose:
            print(f"    {msg}")

    return out, records


def deconcentrate_allocation_artefacts(data, threshold=None, label="", lat=None,
                                       lon=None, verbose=False):
    """Cap implausibly dominant single cells, conserving domain mass.

    Some gridded inventories place a whole administrative total on one grid
    cell when a sector cannot be allocated spatially.  In the CEDS sectoral
    (elevated-stack) files shipped with ORBIT, one 0.5-degree cell on the
    India/Myanmar border holds 51.5% of the *entire South Asian domain* for
    both PM2.5 and SO2 — physically implausible for a real source at that
    resolution and location, and an artefact of the allocation step rather
    than of the emission totals.

    Any cell holding more than ``threshold`` of the field's domain total is
    capped at exactly that share, and the excess is redistributed over the
    remaining non-zero cells in proportion to their existing values.  The
    domain total is unchanged, so the inventory's mass — which is not what
    is in doubt — is preserved.

    Capping rather than deleting is deliberate: a genuinely large source
    (the top cell of the ``high`` file is offshore Mumbai, plausibly the
    Bombay High field) is retained as the largest cell in the field, merely
    not a dominant one.

    Parameters
    ----------
    data : ndarray
        2-D field on its native grid, non-negative.
    threshold : float, optional
        Share of the domain total above which a cell is capped.  Defaults
        to ``ORBIT_EMISSION_SPIKE_FRAC`` or 0.05.  Zero disables the screen.
    label : str
        Name used in the diagnostic message.
    lat, lon : ndarray, optional
        Native coordinates, used only to report where a cell was capped.
    verbose : bool
        Print a line per capped cell.

    Returns
    -------
    (ndarray, list)
        The screened field and a list of records for the run log, one per
        capped cell: ``{"lat", "lon", "share", "value", "capped"}``.
    """
    if threshold is None:
        threshold = _spike_threshold()
    if threshold <= 0.0:
        return data, []

    total = float(np.sum(data))
    if not np.isfinite(total) or total <= 0.0:
        return data, []

    flagged = data > threshold * total
    if not np.any(flagged):
        return data, []

    donors = (data > 0.0) & ~flagged
    donor_mass = float(np.sum(data[donors]))
    if donor_mass <= 0.0:
        warnings.warn(
            f"{label}: {int(flagged.sum())} cell(s) exceed {threshold:.1%} of the "
            f"domain total but no other non-zero cells exist to receive the "
            f"excess; leaving the field unchanged"
        )
        return data, []

    out = np.array(data, dtype=np.float64, copy=True)
    cap = threshold * total
    excess = float(np.sum(out[flagged] - cap))
    out[flagged] = cap
    out[donors] += out[donors] * (excess / donor_mass)

    records = []
    for j, i in zip(*np.nonzero(flagged)):
        rec = {
            "lat": float(lat[j]) if lat is not None else None,
            "lon": float(lon[i]) if lon is not None else None,
            "share": float(data[j, i] / total),
            "value": float(data[j, i]),
            "capped": float(cap),
        }
        records.append(rec)
        where = (f" at {rec['lat']:.2f}N {rec['lon']:.2f}E"
                 if rec["lat"] is not None else f" at index ({j}, {i})")
        msg = (f"{label}: capped an implausible cell{where} holding "
               f"{rec['share']:.1%} of the domain total, down to "
               f"{threshold:.1%}; excess redistributed over "
               f"{int(donors.sum())} cells (domain total conserved)")
        warnings.warn(msg)
        if verbose:
            print(f"    {msg}")

    return out, records


def load_netcdf_source(source: EmissionSource, grid, verbose: bool = False) -> np.ndarray:
    """Load emissions from a NetCDF file into a 4D array.

    Handles coordinate detection, lon 360->180, lat flip, variable
    auto-detection or explicit mapping, time_index selection, NaN/negative
    handling, unit conversion (rate + flux), element conversion, bilinear
    regridding, cell area computation for flux units, volume normalization,
    and elevated injection.

    Parameters
    ----------
    source : EmissionSource
        Source configuration
    grid : GridData
        Target grid (needs lon, lat, nz, ny, nx, Dz, volume, dlon, dlat)
    verbose : bool
        Print progress

    Returns
    -------
    ndarray, shape (6, nz, ny, nx)
        Emission rates in ug/m^3/s, indexed by ORBIT species
    """
    import xarray as xr

    emis_4d = np.zeros((N_ORBIT_SPECIES, grid.nz, grid.ny, grid.nx), dtype=np.float64)

    ds = xr.open_dataset(source.path)

    # Detect coordinate variables
    lon_var = _detect_coord_var(ds, ['lon', 'longitude', 'x', 'LON', 'LONGITUDE'])
    lat_var = _detect_coord_var(ds, ['lat', 'latitude', 'y', 'LAT', 'LATITUDE'])

    if lon_var is None or lat_var is None:
        if verbose:
            print(f"    Warning: Could not detect lon/lat coordinates in {source.path}")
        ds.close()
        return emis_4d

    lon_nc = ds[lon_var].values.copy()
    lat_nc = ds[lat_var].values.copy()

    # Handle lon in [0, 360] -> [-180, 180]
    if np.any(lon_nc > 180):
        lon_nc = np.where(lon_nc > 180, lon_nc - 360, lon_nc)
        if verbose:
            print("    Converted longitude from [0,360] to [-180,180]")

    # Check if lat is reversed (descending)
    lat_reversed = len(lat_nc) > 1 and lat_nc[0] > lat_nc[-1]
    if lat_reversed:
        lat_nc = lat_nc[::-1]
        if verbose:
            print("    Reversed latitude (was north-to-south)")

    # Detect species variables
    if source.variable_mapping:
        species_vars = {spec: var for spec, var in source.variable_mapping.items()
                        if var in ds.data_vars}
    else:
        species_vars = _detect_netcdf_species(ds.data_vars)

    if not species_vars:
        if verbose:
            print(f"    Warning: No species variables detected in {source.path}")
        ds.close()
        return emis_4d

    if verbose:
        print(f"    Detected species variables: {list(species_vars.keys())}")

    # Determine conversion strategy
    is_flux_unit = source.units in ('kg/m2/s', 'kg/m2')
    if source.units == 'kg/m2':
        if source.averaging_period is None or source.averaging_period <= 0:
            raise ValueError(
                f"units='kg/m2' requires a positive averaging_period (days), "
                f"got {source.averaging_period!r} for {source.path}"
            )
        unit_factor = 1.0 / (source.averaging_period * 86400.0)
    elif is_flux_unit:
        unit_factor = 1.0
    else:
        unit_factor = UNIT_CONVERSIONS.get(source.units, 1.0)

    if verbose and source.time_index is not None:
        print(f"    Using time_index={source.time_index}")
    if verbose and is_flux_unit:
        unit_label = source.units
        if source.units == 'kg/m2':
            unit_label += f" (averaging_period={source.averaging_period} days)"
        print(f"    Flux units ({unit_label}): will compute cell areas for conversion")

    # Determine if regridding is needed
    needs_regrid = not _grids_match(lon_nc, lat_nc, grid.lon, grid.lat)
    if needs_regrid and verbose:
        print(f"    Regridding from {len(lon_nc)}x{len(lat_nc)} to {grid.nx}x{grid.ny}")

    # Load height variable if specified
    height_field = None
    if source.elevated and source.height_variable:
        if source.height_variable in ds.data_vars:
            height_field = ds[source.height_variable].values
            if lat_reversed:
                height_field = height_field[::-1, :]
            if verbose:
                print(f"    Using height variable: {source.height_variable}")

    for species_name, var_name in species_vars.items():
        # Get legacy species index
        species_lower = species_name.lower().replace('_', '').replace('-', '')
        legacy_idx = SPECIES_MAP.get(species_lower)
        if legacy_idx is None:
            continue

        # Map to ORBIT species
        orbit_idx = LEGACY9_TO_ORBIT.get(legacy_idx)
        if orbit_idx is None:
            continue

        # Load data
        data = ds[var_name].values.copy()

        # Handle extra dimensions (time, level)
        if data.ndim > 2:
            if source.time_index is not None:
                data = data[source.time_index]
                while data.ndim > 2:
                    data = data[0]
            else:
                while data.ndim > 2:
                    data = data[0]

        # Flip if lat was reversed
        if lat_reversed:
            data = data[::-1, :]

        # Handle NaN
        nan_count = np.sum(np.isnan(data))
        if nan_count > 0:
            warnings.warn(f"Found {nan_count} NaN values in {var_name}, treating as zero")
            data = np.where(np.isnan(data), 0.0, data)

        # Handle negatives
        neg_count = np.sum(data < 0)
        if neg_count > 0:
            warnings.warn(f"Found {neg_count} negative values in {var_name}")

        # Screen spatial-allocation artefacts on the native grid, before
        # regridding, since that is where the artefact lives.
        data, _spikes = deconcentrate_allocation_artefacts(
            data, label=f"{os.path.basename(source.path)}:{var_name}",
            lat=lat_nc, lon=lon_nc, verbose=verbose,
        )

        # Apply unit conversion
        data = data * unit_factor

        # Apply element conversion if needed
        if species_lower in CONVERT_COLS and legacy_idx in ELEMENT_CONVERSION:
            data = data * ELEMENT_CONVERSION[legacy_idx]

        # Regrid if needed
        if needs_regrid:
            data = _regrid_emission(data, lon_nc, lat_nc, grid.lon, grid.lat)

        # Convert flux units (kg/m2/s) to ug/s per cell using model grid areas
        if is_flux_unit:
            dlon = grid.dlon if hasattr(grid, 'dlon') else (
                grid.lon[1] - grid.lon[0] if grid.nx > 1 else 0.625
            )
            dlat = grid.dlat if hasattr(grid, 'dlat') else (
                grid.lat[1] - grid.lat[0] if grid.ny > 1 else 0.5
            )
            cell_areas = compute_cell_areas(grid.lat, dlon, dlat)  # shape (ny,)
            # data is kg/m2/s; * area_m2 -> kg/s; * 1e9 -> ug/s
            data = data * cell_areas[:, np.newaxis] * 1e9

        # Allocate to grid
        for j in range(grid.ny):
            for i in range(grid.nx):
                value = data[j, i]
                if value <= 0:
                    continue

                # Determine layer
                if source.elevated:
                    if source.layer_index is not None:
                        # Fixed-layer override (no plume rise)
                        k = min(source.layer_index, grid.nz - 1)
                    elif height_field is not None:
                        # Per-cell physical stack height, optionally with
                        # source-level scalar stack diam/temp/vel for ASME.
                        if needs_regrid:
                            height = _interpolate_point(
                                height_field, lon_nc, lat_nc,
                                grid.lon[i], grid.lat[j]
                            )
                        else:
                            height = height_field[j, i]
                        if height > 0:
                            k = find_injection_layer(
                                grid, i, j, float(height),
                                stack_diam=source.stack_diam,
                                stack_temp=source.stack_temp,
                                stack_vel=source.stack_vel,
                            )
                        else:
                            k = 0
                    elif source.stack_height > 0:
                        # Uniform scalar stack params — full ASME rise
                        # per cell using local met (stability, wind, T).
                        k = find_injection_layer(
                            grid, i, j, source.stack_height,
                            stack_diam=source.stack_diam,
                            stack_temp=source.stack_temp,
                            stack_vel=source.stack_vel,
                        )
                    else:
                        k = 0
                else:
                    k = 0

                vol = grid.volume[k, j, i]
                if vol <= 0:
                    continue

                emis_4d[orbit_idx, k, j, i] += value / vol

    ds.close()
    return emis_4d


# =============================================================================
# Helpers
# =============================================================================

def load_netcdf_source_per_bin(
    source: EmissionSource,
    grid,
    n_bins: int = 8,
    verbose: bool = False,
) -> np.ndarray:
    """Per-bin variant of ``load_netcdf_source``.

    The NetCDF must carry a ``bin`` dimension (configurable via
    ``source.bin_axis_name``) of length ``n_bins``.  The standard ``time``
    dimension still selects the month via ``source.time_index``.

    Returns
    -------
    ndarray, shape (n_bins, 6, nz, ny, nx)
        Emission rates in ug/m^3/s per UTC bin, indexed by ORBIT species.
    """
    import xarray as xr

    out = np.zeros(
        (n_bins, N_ORBIT_SPECIES, grid.nz, grid.ny, grid.nx), dtype=np.float64
    )
    ds = xr.open_dataset(source.path)

    bin_dim = source.bin_axis_name
    if bin_dim not in ds.dims:
        ds.close()
        raise ValueError(
            f"{source.path}: expected bin dimension '{bin_dim}' (set "
            f"EmissionSource.bin_axis_name to override); dims={tuple(ds.dims)}"
        )
    if ds.sizes[bin_dim] != n_bins:
        ds.close()
        raise ValueError(
            f"{source.path}: bin dim '{bin_dim}' has length "
            f"{ds.sizes[bin_dim]}, expected {n_bins}"
        )
    ds.close()

    # Reuse the per-month loader by selecting one bin at a time.  To avoid
    # re-opening the file 8 times we load it once here, then slice into
    # in-memory arrays for each bin.
    ds = xr.open_dataset(source.path)
    try:
        for tau in range(n_bins):
            ds_tau = ds.isel({bin_dim: tau})
            # Stage the bin slice as a temporary in-memory file via a
            # detour: write the source's per-bin variables into a synthetic
            # EmissionSource-compatible call by reusing the heavy lifting
            # already in load_netcdf_source.  We do this by passing the
            # already-sliced xr.Dataset through a private helper.
            out[tau] = _load_netcdf_from_dataset(
                ds_tau, source, grid, verbose=(verbose and tau == 0),
            )
    finally:
        ds.close()
    return out


def _load_netcdf_from_dataset(ds, source: EmissionSource, grid,
                               verbose: bool = False) -> np.ndarray:
    """Body of ``load_netcdf_source`` operating on an already-open Dataset.

    Used internally for the per-bin path so the file is opened once and
    the bin axis is sliced in-memory.
    """
    import xarray as xr  # noqa: F401  (kept symmetric with load_netcdf_source)

    emis_4d = np.zeros((N_ORBIT_SPECIES, grid.nz, grid.ny, grid.nx), dtype=np.float64)

    lon_var = _detect_coord_var(ds, ['lon', 'longitude', 'x', 'LON', 'LONGITUDE'])
    lat_var = _detect_coord_var(ds, ['lat', 'latitude', 'y', 'LAT', 'LATITUDE'])
    if lon_var is None or lat_var is None:
        return emis_4d

    lon_nc = ds[lon_var].values.copy()
    lat_nc = ds[lat_var].values.copy()
    if np.any(lon_nc > 180):
        lon_nc = np.where(lon_nc > 180, lon_nc - 360, lon_nc)
    lat_reversed = len(lat_nc) > 1 and lat_nc[0] > lat_nc[-1]
    if lat_reversed:
        lat_nc = lat_nc[::-1]

    if source.variable_mapping:
        species_vars = {spec: var for spec, var in source.variable_mapping.items()
                        if var in ds.data_vars}
    else:
        species_vars = _detect_netcdf_species(ds.data_vars)
    if not species_vars:
        return emis_4d

    is_flux_unit = source.units in ('kg/m2/s', 'kg/m2')
    if source.units == 'kg/m2':
        if source.averaging_period is None or source.averaging_period <= 0:
            raise ValueError(
                f"units='kg/m2' requires positive averaging_period, "
                f"got {source.averaging_period!r} for {source.path}"
            )
        unit_factor = 1.0 / (source.averaging_period * 86400.0)
    elif is_flux_unit:
        unit_factor = 1.0
    else:
        unit_factor = UNIT_CONVERSIONS.get(source.units, 1.0)

    needs_regrid = not _grids_match(lon_nc, lat_nc, grid.lon, grid.lat)

    height_field = None
    if source.elevated and source.height_variable:
        if source.height_variable in ds.data_vars:
            height_field = ds[source.height_variable].values
            if lat_reversed:
                height_field = height_field[::-1, :]

    for species_name, var_name in species_vars.items():
        species_lower = species_name.lower().replace('_', '').replace('-', '')
        legacy_idx = SPECIES_MAP.get(species_lower)
        if legacy_idx is None:
            continue
        orbit_idx = LEGACY9_TO_ORBIT.get(legacy_idx)
        if orbit_idx is None:
            continue

        data = ds[var_name].values.copy()
        if data.ndim > 2:
            if source.time_index is not None:
                data = data[source.time_index]
                while data.ndim > 2:
                    data = data[0]
            else:
                while data.ndim > 2:
                    data = data[0]
        if lat_reversed:
            data = data[::-1, :]
        nan_count = np.sum(np.isnan(data))
        if nan_count > 0:
            data = np.where(np.isnan(data), 0.0, data)
        data = data * unit_factor
        if species_lower in CONVERT_COLS and legacy_idx in ELEMENT_CONVERSION:
            data = data * ELEMENT_CONVERSION[legacy_idx]
        if needs_regrid:
            data = _regrid_emission(data, lon_nc, lat_nc, grid.lon, grid.lat)
        if is_flux_unit:
            dlon = grid.dlon if hasattr(grid, 'dlon') else (
                grid.lon[1] - grid.lon[0] if grid.nx > 1 else 0.625
            )
            dlat = grid.dlat if hasattr(grid, 'dlat') else (
                grid.lat[1] - grid.lat[0] if grid.ny > 1 else 0.5
            )
            cell_areas = compute_cell_areas(grid.lat, dlon, dlat)
            data = data * cell_areas[:, np.newaxis] * 1e9

        for j in range(grid.ny):
            for i in range(grid.nx):
                value = data[j, i]
                if value <= 0:
                    continue
                if source.elevated:
                    if source.layer_index is not None:
                        k = min(source.layer_index, grid.nz - 1)
                    elif height_field is not None:
                        if needs_regrid:
                            height = _interpolate_point(
                                height_field, lon_nc, lat_nc,
                                grid.lon[i], grid.lat[j]
                            )
                        else:
                            height = height_field[j, i]
                        if height > 0:
                            k = find_injection_layer(
                                grid, i, j, float(height),
                                stack_diam=source.stack_diam,
                                stack_temp=source.stack_temp,
                                stack_vel=source.stack_vel,
                            )
                        else:
                            k = 0
                    elif source.stack_height > 0:
                        k = find_injection_layer(
                            grid, i, j, source.stack_height,
                            stack_diam=source.stack_diam,
                            stack_temp=source.stack_temp,
                            stack_vel=source.stack_vel,
                        )
                    else:
                        k = 0
                else:
                    k = 0
                vol = grid.volume[k, j, i]
                if vol <= 0:
                    continue
                emis_4d[orbit_idx, k, j, i] += value / vol

    return emis_4d


def _detect_coord_var(ds, candidates: List[str]) -> Optional[str]:
    """Find coordinate variable from list of candidates."""
    for name in candidates:
        if name in ds.coords or name in ds.dims or name in ds.data_vars:
            return name
    return None


def _detect_netcdf_species(data_vars) -> Dict[str, str]:
    """Auto-detect species variables from NetCDF data variables."""
    detected = {}
    for var_name in data_vars:
        var_lower = var_name.lower().replace('_', '').replace('-', '')
        for pattern, spec_idx in NETCDF_SPECIES_PATTERNS.items():
            if pattern in var_lower:
                # Map back to species name
                for name, idx in SPECIES_MAP.items():
                    if idx == spec_idx:
                        detected[name] = var_name
                        break
                break
    return detected


def _grids_match(lon1: np.ndarray, lat1: np.ndarray,
                 lon2: np.ndarray, lat2: np.ndarray,
                 tol: float = 0.01) -> bool:
    """Check if two grids match within tolerance."""
    if len(lon1) != len(lon2) or len(lat1) != len(lat2):
        return False
    return (np.allclose(lon1, lon2, atol=tol) and
            np.allclose(lat1, lat2, atol=tol))


def _cell_edges(centres: np.ndarray) -> np.ndarray:
    """Cell edges from centres: midpoints inside, extrapolated at the ends."""
    c = np.asarray(centres, dtype=np.float64)
    if c.size < 2:
        raise ValueError("need at least two cell centres to infer edges")
    mid = 0.5 * (c[:-1] + c[1:])
    first = c[0] - (mid[0] - c[0])
    last = c[-1] + (c[-1] - mid[-1])
    return np.concatenate([[first], mid, [last]])


def _overlap_matrix(edges_dst: np.ndarray, edges_src: np.ndarray) -> np.ndarray:
    """(n_dst, n_src) overlap length between two sets of 1-D intervals."""
    lo = np.maximum(edges_dst[:-1, None], edges_src[None, :-1])
    hi = np.minimum(edges_dst[1:, None], edges_src[None, 1:])
    return np.clip(hi - lo, 0.0, None)


def _regrid_emission_conservative(data, lon_src, lat_src, lon_dst, lat_dst):
    """Area-weighted (mass-conservative) regridding between lat-lon grids.

    Bilinear interpolation answers "what is the field value at this
    point?"; an emission field needs "how much mass falls inside this
    cell?". The two agree only for fields smooth relative to both grids,
    and the 2026-08-02 audit measured the resulting domain-mass error at
    -21% to +25% across the production inventory, with per-cell errors up
    to 14x on fire emissions.

    Both grids are regular in longitude and latitude, so the overlap
    weights separate into a longitude factor and a latitude factor. Cell
    area on a sphere is R^2 * dlon * d(sin lat), so latitude weights are
    computed in sin-latitude space, which makes the scheme exact rather
    than second-order. R^2 cancels between the mass and the area.

    Cells of the destination grid not covered by the source receive only
    the mass that overlaps them (uncovered area contributes nothing),
    matching the previous fill_value=0.0 behaviour outside the source.
    """
    data = np.nan_to_num(np.asarray(data, dtype=np.float64), nan=0.0)
    lat_src = np.asarray(lat_src, dtype=np.float64)
    lon_src = np.asarray(lon_src, dtype=np.float64)
    lat_dst = np.asarray(lat_dst, dtype=np.float64)
    lon_dst = np.asarray(lon_dst, dtype=np.float64)

    # Work on ascending axes; restore the destination order at the end.
    if lat_src[0] > lat_src[-1]:
        lat_src, data = lat_src[::-1], data[::-1, :]
    if lon_src[0] > lon_src[-1]:
        lon_src, data = lon_src[::-1], data[:, ::-1]
    lat_flip = lat_dst[0] > lat_dst[-1]
    lon_flip = lon_dst[0] > lon_dst[-1]
    lat_dst_a = lat_dst[::-1] if lat_flip else lat_dst
    lon_dst_a = lon_dst[::-1] if lon_flip else lon_dst

    # Latitude weights in sin space (exact spherical cell area).
    sin_src = np.sin(np.deg2rad(np.clip(_cell_edges(lat_src), -90.0, 90.0)))
    sin_dst = np.sin(np.deg2rad(np.clip(_cell_edges(lat_dst_a), -90.0, 90.0)))
    w_lat = _overlap_matrix(sin_dst, sin_src)              # (nlat_dst, nlat_src)

    # Longitude weights in radians.
    lon_e_src = np.deg2rad(_cell_edges(lon_src))
    lon_e_dst = np.deg2rad(_cell_edges(lon_dst_a))
    w_lon = _overlap_matrix(lon_e_dst, lon_e_src)          # (nlon_dst, nlon_src)

    mass = w_lat @ data @ w_lon.T                          # R^2 factored out
    area = (np.diff(sin_dst)[:, None] * np.diff(lon_e_dst)[None, :])
    result = np.divide(mass, area, out=np.zeros_like(mass), where=area > 0)

    if lat_flip:
        result = result[::-1, :]
    if lon_flip:
        result = result[:, ::-1]
    return result


def _regrid_emission_bilinear(data, lon_src, lat_src, lon_dst, lat_dst):
    """Legacy bilinear point sampling. Not mass-conservative; retained so
    pre-2026-08-02 runs stay reproducible via ORBIT_REGRID=bilinear."""
    from scipy.interpolate import RegularGridInterpolator

    interpolator = RegularGridInterpolator(
        (lat_src, lon_src),
        data,
        method='linear',
        bounds_error=False,
        fill_value=0.0
    )

    lon_grid, lat_grid = np.meshgrid(lon_dst, lat_dst)
    points = np.column_stack([lat_grid.ravel(), lon_grid.ravel()])

    result = interpolator(points).reshape(len(lat_dst), len(lon_dst))
    return result


def _regrid_emission(data: np.ndarray, lon_src: np.ndarray, lat_src: np.ndarray,
                     lon_dst: np.ndarray, lat_dst: np.ndarray) -> np.ndarray:
    """Regrid an emission field onto the model grid.

    Conservative by default (ORBIT_REGRID=conservative); set
    ORBIT_REGRID=bilinear to restore the pre-2026-08-02 behaviour.
    """
    import os
    scheme = os.environ.get("ORBIT_REGRID", "conservative").lower()
    if scheme == "bilinear":
        return _regrid_emission_bilinear(data, lon_src, lat_src,
                                         lon_dst, lat_dst)
    if scheme != "conservative":
        raise ValueError(
            f"Unknown ORBIT_REGRID={scheme!r}; valid: conservative, bilinear")
    return _regrid_emission_conservative(data, lon_src, lat_src,
                                         lon_dst, lat_dst)


def _interpolate_point(data: np.ndarray, lon_src: np.ndarray, lat_src: np.ndarray,
                       lon_pt: float, lat_pt: float) -> float:
    """Interpolate a single point from gridded data."""
    from scipy.interpolate import RegularGridInterpolator

    interpolator = RegularGridInterpolator(
        (lat_src, lon_src),
        data,
        method='linear',
        bounds_error=False,
        fill_value=0.0
    )
    return float(interpolator((lat_pt, lon_pt)))
