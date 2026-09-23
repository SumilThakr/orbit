"""GADM admin-2 (district / subdistrict) aggregation for the marginal-
deaths driver.

Takes the per-cell adjoint output ∂J/∂e_(τ, species, cell) on the ORBIT
0.5°×0.625° grid and aggregates it to GADM v4.1 admin-2 polygons within
the South Asia bbox. The output is "deaths per 1000 kg of pollutant
emitted uniformly across the subdistrict, sustained year-round".

Implementation
--------------

1. Load GADM admin-2 polygons and clip them to the ORBIT grid (or a
   given bbox).
2. Intersect each polygon with every ORBIT grid cell it overlaps and
   measure the area of each piece in an equal-area projection, which
   gives exact per-(cell, gid) area fractions. Every polygon with
   positive area gets at least one entry, however small it is next to
   a cell.
3. For each subdistrict S: deaths_per_1000kg(p, τ, S) =
   (1000 / SECONDS_PER_YEAR) · area_weighted_mean(∂J/∂e_p,τ in S).

On the South Asian grid (about 2,000 admin-2 polygons and 4,600 surface
cells) the overlay takes a few seconds once the polygons are loaded.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr

from orbit.emissions.sources import ELEMENT_CONVERSION, SPECIES_MAP

SECONDS_PER_YEAR = 365.25 * 86400.0


def element_factor_for_species(species_key: str) -> float:
    """Return the netcdf loader's element-mass factor for a ORBIT adjoint
    species key (the strings stored in `adjoint_M*.nc`'s species axis).

    Maps the adjoint-output keys to ``orbit.emissions.sources.SPECIES_MAP``
    via case-insensitive name normalisation, then looks up
    ``ELEMENT_CONVERSION``. Returns 1.0 if there's no conversion (PM25,
    VBS, VOC pseudo-species — all already in compound mass terms by
    convention).

    Used by the aggregator so the "deaths per 1000 kg" output is per
    1000 kg of *compound* emitted (what the user puts into the netcdf)
    rather than per 1000 kg of *element* mass (what the solver sees
    internally after the loader's conversion).
    """
    norm = species_key.lower().replace("_", "").replace("-", "")
    # VOC pseudo-species (VOC_anthro/_bio/_bb) and explicit VOC variants
    # carry no element conversion at the netcdf loader (CONVERT_COLS lists
    # only nh3/sox/so2/nox/no2). PM25_primary normalises to "pm25primary"
    # which isn't in SPECIES_MAP — handle as PM25.
    if norm.startswith("voc"):
        return 1.0
    if norm.startswith("pm25"):
        return 1.0
    legacy_idx = SPECIES_MAP.get(norm)
    if legacy_idx is None:
        return 1.0
    return float(ELEMENT_CONVERSION.get(legacy_idx, 1.0))


@dataclass(frozen=True)
class SubdistrictLayer:
    """Mapping from ORBIT (lat, lon) cells to GADM admin-2 subdistricts.

    Fields stored as parallel arrays for cheap downstream bincount-style
    aggregation. The (cell_idx, gid_idx, fraction) triples encode a
    sparse coarse-grid-cell × subdistrict overlay (one row per
    (cell, gid) pairing with non-zero area fraction).
    """
    gid_list: np.ndarray            # (n_gid,) str — GADM GID_2 values
    gid_name: np.ndarray            # (n_gid,) str — GADM NAME_2 (display)
    gid_iso3: np.ndarray            # (n_gid,) str — GADM GID_0 (country)
    gid_area_km2: np.ndarray        # (n_gid,) float — total district area
    cell_idx: np.ndarray            # (n_entries,) int — flat (y*nx + x) into surface grid
    gid_idx: np.ndarray             # (n_entries,) int — position in gid_list
    area_km2: np.ndarray            # (n_entries,) float — (cell ∩ district) area
    cell_area_km2: np.ndarray       # (ny*nx,) float — full cell area (lat-dependent)


def _cell_area_km2(lat_centres: np.ndarray, dlat: float, dlon: float) -> np.ndarray:
    """Per-row cell area at 0.625° × 0.5° (or whatever the ORBIT spacing is).

    Returns a 1-D array of shape (ny,) — multiply by dlon (in degrees,
    converted via cos(lat)) for the full per-cell area. We compute
    once per row because lon-spacing is the same across rows.
    """
    # Earth radius (km).
    R = 6371.0
    # Cell area at lat = R² · dlon · (sin(lat_top) - sin(lat_bot)).
    lat_rad = np.deg2rad(lat_centres)
    dlat_rad = np.deg2rad(dlat)
    dlon_rad = np.deg2rad(dlon)
    return R * R * dlon_rad * (np.sin(lat_rad + dlat_rad / 2)
                               - np.sin(lat_rad - dlat_rad / 2))


def build_subdistrict_layer(
    *,
    gadm_gpkg: str,
    orbit_lat: np.ndarray,
    orbit_lon: np.ndarray,
    bbox: tuple[float, float, float, float] | None = None,   # (lon_min, lat_min, lon_max, lat_max)
    verbose: bool = True,
) -> SubdistrictLayer:
    """Build the per-cell × per-subdistrict overlay by exact intersection.

    Every admin-2 polygon is clipped to ``bbox`` (by default the outer
    edges of the ORBIT grid), dissolved by GID_2, and intersected with
    each grid cell it overlaps. The area of every (cell ∩ polygon) piece
    is measured in an equal-area projection and divided by the projected
    area of the cell, which gives the fraction of the cell that the
    polygon covers. A polygon with positive area therefore always gets at
    least one entry, however small it is next to a cell, and a polygon
    that lies inside one cell gets that cell alone.
    """
    import geopandas as gpd
    import shapely
    from shapely.geometry import box

    ny = orbit_lat.size
    nx = orbit_lon.size
    dlat = float(orbit_lat[1] - orbit_lat[0])
    dlon = float(orbit_lon[1] - orbit_lon[0])
    if bbox is None:
        bbox = (float(orbit_lon[0] - dlon / 2), float(orbit_lat[0] - dlat / 2),
                float(orbit_lon[-1] + dlon / 2), float(orbit_lat[-1] + dlat / 2))
    bbox = tuple(float(b) for b in bbox)
    bbox_geom = box(*bbox)

    if verbose:
        print("  Loading GADM admin-2 polygons...")
    gdf = gpd.read_file(gadm_gpkg, columns=["GID_0", "GID_2", "NAME_2", "geometry"],
                        bbox=bbox)
    gdf = gdf[gdf["GID_2"].notna() & (gdf["GID_2"] != "")].copy()
    # Repair invalid rings before any overlay; GADM v4.1 has a few.
    gdf["geometry"] = gdf.geometry.make_valid()
    gdf = gdf[gdf.intersects(bbox_geom)].copy()
    gdf["geometry"] = gdf["geometry"].intersection(bbox_geom)
    gdf = gdf[~gdf["geometry"].is_empty]
    if verbose:
        print(f"  {len(gdf)} polygon rows ({gdf['GID_2'].nunique()} unique GID_2) "
              f"in bbox (lon {bbox[0]}..{bbox[2]}, lat {bbox[1]}..{bbox[3]})")
    # Dissolve duplicate GID_2 entries — GADM v4.1 splits MultiPolygons
    # across multiple rows (islands, non-contiguous districts), and every
    # subdistrict must be one polygon with one gid_idx.
    if gdf["GID_2"].duplicated().any():
        gdf = gdf.dissolve(
            by="GID_2",
            aggfunc={"GID_0": "first", "NAME_2": "first"},
            as_index=False,
        )
        if verbose:
            print(f"  Dissolved → {len(gdf)} unique admin-2 polygons.")
    gdf = gdf.sort_values("GID_2", ignore_index=True)

    # Grid cells as boxes with flat index y * nx + x, in the polygons' CRS.
    lon2d, lat2d = np.meshgrid(orbit_lon, orbit_lat)
    cells = gpd.GeoSeries(
        shapely.box(lon2d.ravel() - dlon / 2, lat2d.ravel() - dlat / 2,
                    lon2d.ravel() + dlon / 2, lat2d.ravel() + dlat / 2),
        crs=gdf.crs,
    )

    # Candidate (cell, polygon) pairs from the spatial index, then the
    # exact intersection of each pair and its area in an equal-area
    # projection, as a fraction of the cell's projected area.
    cell_pos, gid_pos = gdf.sindex.query(cells.to_numpy(), predicate="intersects")
    pieces = shapely.intersection(cells.to_numpy()[cell_pos],
                                  gdf.geometry.to_numpy()[gid_pos])
    equal_area = "EPSG:6933"
    piece_area = gpd.GeoSeries(pieces, crs=gdf.crs).to_crs(equal_area).area.to_numpy()
    cell_area_proj = cells.to_crs(equal_area).area.to_numpy()
    frac = piece_area / cell_area_proj[cell_pos]
    keep = frac > 0.0
    cell_idx = cell_pos[keep].astype(np.int32)
    gid_idx = gid_pos[keep].astype(np.int32)
    frac = frac[keep].astype(np.float64)
    if verbose:
        covered = np.bincount(gid_idx, minlength=len(gdf)) > 0
        print(f"  {cell_idx.size} cell·gid pairs; {int(covered.sum())} of "
              f"{len(gdf)} polygons have positive area")

    # Cell areas (lat-dependent, lon-uniform within ORBIT).
    cell_row_area = _cell_area_km2(orbit_lat, dlat, dlon)   # (ny,)
    cell_area_2d = np.broadcast_to(
        cell_row_area[:, None], (ny, nx)
    ).ravel().astype(np.float64)
    area_per_entry = frac * cell_area_2d[cell_idx]

    # Total district areas (sum of (cell ∩ district) areas).
    gid_area = np.bincount(
        gid_idx, weights=area_per_entry, minlength=len(gdf)
    )

    return SubdistrictLayer(
        gid_list=gdf["GID_2"].to_numpy(),
        gid_name=gdf["NAME_2"].to_numpy(),
        gid_iso3=gdf["GID_0"].to_numpy(),
        gid_area_km2=gid_area,
        cell_idx=cell_idx,
        gid_idx=gid_idx,
        area_km2=area_per_entry,
        cell_area_km2=cell_area_2d,
    )


def aggregate_dJ_de_to_subdistricts(
    dJ_de_per_species_bin: np.ndarray,    # (n_species, N_BINS, ny, nx) — surface slab
    layer: SubdistrictLayer,
    surface_volume_m3: np.ndarray,        # (ny, nx) — required for unit conv
    species_keys: list[str] | None = None,   # per-species element factor lookup
) -> np.ndarray:
    """Convert ∂J/∂e per (species, bin, cell) → deaths per 1000 kg/yr
    per (species, subdistrict) under "1000 kg/yr sustained year-round,
    distributed uniformly per area within the subdistrict."

    Math derivation
    ---------------

    The ORBIT solver consumes emissions in µg/m³/s (per-cell-volume,
    after orbit/emissions/netcdf.py:247 ``emis_4d += value / vol``).
    So ∂J/∂e_τ_j from the adjoint is in (deaths/yr) per (µg/m³/s).

    A "1000 kg/yr sustained year-round" perturbation at cell j is a
    constant rate of 1000/SECONDS_PER_YEAR kg/s in every bin. In
    solver units that's (1000 / SECONDS_PER_YEAR) × 1e9 / vol_j
    = 1e12 / (SECONDS_PER_YEAR × vol_j) µg/m³/s per bin.

    Element-mass conversion. The netcdf loader applies an element_factor
    (NOx → N: 0.305; SO2 → S: 0.497; NH3 → N: 0.822; identity for
    PM25_primary and VOCs) to turn user-supplied compound-mass kg/m²/s
    into solver-internal element-mass kg/m²/s. So 1000 kg of compound
    NOx emitted produces only (1000 × 0.305) kg of N-element inside the
    solver — and the ∂J/∂e_τ from the adjoint is per µg of N-element.
    To report ``deaths_per_1000kg`` per 1000 kg of *compound* emitted
    (the natural policy unit, matching what the user actually puts into
    the emission netcdf), we multiply by element_factor at this step.

    The over-summed J = Σ_τ ⟨S, G c_τ⟩ relates to annual deaths as
    J = N_BINS × δDeaths_annual. So:

        δDeaths_annual(cell j sustained 1000 kg/yr compound)
            = (1/N_BINS) × Σ_τ ∂J/∂e_τ_j ×
              (element_factor × 1e12 / (SECONDS_PER_YEAR × vol_j))
            = (element_factor × 1e12 / (N_BINS × SECONDS_PER_YEAR × vol_j))
              · Σ_τ(∂J/∂e_τ_j)

    Aggregating to subdistrict S by uniform-per-area distribution:

        deaths_per_1000kg(S) = area-weighted mean over j∈S of
                               δDeaths_annual(cell j).

    Returns ndarray of shape (n_species, n_subdistrict). The per-bin
    dimension is collapsed: for "year-round sustained" semantics, the
    answer is a single number per (species, subdistrict). Per-bin
    diurnal-only-emission breakdown can be added as a separate
    diagnostic axis if needed.

    Parameters
    ----------
    species_keys
        Per-row species names (e.g. ``["PM25_primary", "NH3", "SO2",
        "NOx", "VOC_anthro", ...]``). Used to look up the element
        conversion factor per species so the output is uniformly per
        1000 kg of compound emitted. If None, no element conversion is
        applied (caller is responsible for the unit story — kept for
        backwards compat with tests that pass synthetic species).
    """
    n_species, n_bins, ny, nx = dJ_de_per_species_bin.shape
    if surface_volume_m3.shape != (ny, nx):
        raise ValueError(
            f"surface_volume_m3 shape {surface_volume_m3.shape} != "
            f"({ny}, {nx}) expected from dJ_de"
        )
    if species_keys is not None and len(species_keys) != n_species:
        raise ValueError(
            f"species_keys has {len(species_keys)} entries but dJ_de has "
            f"{n_species} species rows"
        )
    n_gid = layer.gid_list.size

    # Sum over bins → (n_species, ny, nx).
    dJ_summed_bins = dJ_de_per_species_bin.sum(axis=1)
    # Per-cell scale to "deaths per 1000 kg/yr sustained year-round".
    # 1e12 = 1000 kg/yr × 1e9 µg/kg ÷ 1 yr.
    with np.errstate(invalid="ignore", divide="ignore"):
        cell_scale = np.where(
            surface_volume_m3 > 0,
            1e12 / (n_bins * SECONDS_PER_YEAR * np.maximum(surface_volume_m3, 1e-30)),
            0.0,
        )  # (ny, nx)
    deaths_per_cell = dJ_summed_bins * cell_scale[None, :, :]   # (n_species, ny, nx)

    # Apply per-species element-mass factor so the output is per 1000 kg
    # of compound emitted (PM2.5_primary unchanged; NOx × 0.305; etc.).
    if species_keys is not None:
        factors = np.array(
            [element_factor_for_species(k) for k in species_keys],
            dtype=np.float64,
        )   # (n_species,)
        deaths_per_cell = deaths_per_cell * factors[:, None, None]

    deaths_per_cell_flat = deaths_per_cell.reshape(n_species, ny * nx)

    out = np.zeros((n_species, n_gid), dtype=np.float64)
    for s in range(n_species):
        weight = deaths_per_cell_flat[s, layer.cell_idx] * layer.area_km2
        num = np.bincount(layer.gid_idx, weights=weight, minlength=n_gid)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[s] = np.where(
                layer.gid_area_km2 > 0,
                num / np.maximum(layer.gid_area_km2, 1e-30),
                0.0,
            )
    return out


def write_deaths_per_1000kg_nc(
    out_path: str,
    *,
    deaths_per_1000kg: np.ndarray,   # (n_draws, n_species, n_gid)
    species_keys: list[str],
    layer: SubdistrictLayer,
    month: int,
    crf_mode: str,
    extra_attrs: dict | None = None,
) -> None:
    """Write the final per-(draw, species, subdistrict) NetCDF.

    Accepts either:
      - (n_draws, n_species, n_gid) for MC outputs, OR
      - (n_species, n_gid) for the deterministic single-draw layout
        (auto-promoted to a 1-draw cube).

    The bin axis is COLLAPSED in this output (year-round sustained
    semantics). The plan's full target shape (pollutant × month ×
    subdistrict × crf × cause × statistic) is built by stitching outputs
    across months and CRF modes in a downstream collator.
    """
    if deaths_per_1000kg.ndim == 2:
        deaths_per_1000kg = deaths_per_1000kg[np.newaxis, ...]
    n_draws, n_sp, n_gid = deaths_per_1000kg.shape
    ds = xr.Dataset(
        data_vars=dict(
            deaths_per_1000kg=(
                ("draw", "species", "subdistrict"),
                deaths_per_1000kg.astype(np.float32),
            ),
            subdistrict_gid=(("subdistrict",), layer.gid_list),
            subdistrict_name=(("subdistrict",), layer.gid_name),
            subdistrict_iso3=(("subdistrict",), layer.gid_iso3),
            subdistrict_area_km2=(("subdistrict",), layer.gid_area_km2.astype(np.float32)),
        ),
        coords=dict(
            draw=np.arange(n_draws),
            species=np.array(species_keys),
            subdistrict=np.arange(n_gid),
        ),
        attrs=dict(
            month=month,
            crf_mode=crf_mode,
            n_draws=int(n_draws),
            unit_convention=(
                "deaths per year per 1000 kg of COMPOUND pollutant emitted "
                "year-round (uniformly per area within the subdistrict, "
                "constant rate every diurnal bin). NOx is reported per "
                "1000 kg NO2-compound (not 1000 kg N-element); SO2 per "
                "1000 kg SO2-compound; NH3 per 1000 kg NH3-compound. The "
                "aggregator applies the netcdf-loader element_factor so "
                "values are directly comparable across species in the "
                "natural policy unit. See "
                "orbit.modes.subdistrict_aggregation.element_factor_for_species."
            ),
            gadm_version="v4.1 admin-2",
            description=(
                "Marginal deaths per kg from the periodic-orbit adjoint "
                "(orbit.modes.adjoint) against a deaths-"
                "gradient field. draw=0 is the deterministic mean; 1.. are "
                "MC samples over CRF + GBD-rate uncertainty. The bin axis "
                "is COLLAPSED (year-round sustained semantics; see "
                "aggregate_dJ_de_to_subdistricts docstring for the math)."
            ),
            **(extra_attrs or {}),
        ),
    )
    ds.deaths_per_1000kg.attrs.update(
        units="deaths year-1 / (1000 kg year-1)",
        long_name="marginal deaths per 1000 kg of pollutant emission",
    )
    encoding = {
        "deaths_per_1000kg": {"zlib": True, "complevel": 4, "_FillValue": None},
    }
    ds.to_netcdf(out_path, encoding=encoding)
