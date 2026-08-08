"""GADM admin-2 (district / subdistrict) aggregation for the marginal-
deaths driver.

Takes the per-cell adjoint output ∂J/∂e_(τ, species, cell) on the ORBIT
0.5°×0.625° grid and aggregates it to GADM v4.1 admin-2 polygons within
the South Asia bbox. The output is "deaths per 1000 kg of pollutant
emitted uniformly across the subdistrict, sustained year-round".

Implementation
--------------

1. Load GADM admin-2 polygons, filter to the ORBIT bbox.
2. Rasterize each polygon at sub-cell resolution (factor K=5 → 0.1° ×
   0.125° sub-cells of the ORBIT grid) and aggregate sub-cells to the
   coarse ORBIT grid with per-(cell, gid) area fractions.
3. For each subdistrict S: deaths_per_1000kg(p, τ, S) =
   (1000 / SECONDS_PER_YEAR) · area_weighted_mean(∂J/∂e_p,τ in S).

Sub-cell rasterisation (instead of whole-cell) buys us correct boundary
attribution at the cost of ~20× more polygon-vs-cell intersections; on
the SAS bbox (~1500 admin-2 × ~4600 surface cells × K²=25 sub-cells)
the cost is still <30 s.
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
    bbox: tuple[float, float, float, float],   # (lon_min, lat_min, lon_max, lat_max)
    sub_factor: int = 5,
    verbose: bool = True,
) -> SubdistrictLayer:
    """Build the per-cell × per-subdistrict overlay.

    Algorithm: rasterize each clipped admin-2 polygon at sub_factor×
    sub_factor sub-cell resolution; count sub-cells per (coarse-cell,
    gid); convert counts to per-(cell, gid) area fractions.
    """
    import geopandas as gpd
    from rasterio.features import rasterize
    from rasterio.transform import from_origin

    if verbose:
        print("  Loading GADM admin-2 polygons...")
    gdf = gpd.read_file(gadm_gpkg, columns=["GID_0", "GID_2", "NAME_2", "geometry"])
    gdf = gdf[gdf["GID_2"].notna() & (gdf["GID_2"] != "")]
    # Clip to the ORBIT bbox.
    from shapely.geometry import box
    bbox_geom = box(*bbox)
    gdf = gdf[gdf.intersects(bbox_geom)].copy()
    gdf["geometry"] = gdf["geometry"].intersection(bbox_geom)
    gdf = gdf[~gdf["geometry"].is_empty]
    if verbose:
        n_pre_dissolve = len(gdf)
        n_unique = gdf["GID_2"].nunique()
        print(f"  {n_pre_dissolve} polygon rows ({n_unique} unique GID_2) "
              f"in bbox (lon {bbox[0]}..{bbox[2]}, lat {bbox[1]}..{bbox[3]})")
    # Dissolve duplicate GID_2 entries — GADM v4.1 splits MultiPolygons
    # across multiple rows (islands, non-contiguous districts). Without
    # dissolving, each row is rasterised independently and the same
    # subdistrict gets multiple sparse entries with different gid_idx
    # values, breaking the area-weighted-mean math.
    if gdf["GID_2"].duplicated().any():
        gdf = gdf.dissolve(
            by="GID_2",
            aggfunc={"GID_0": "first", "NAME_2": "first"},
            as_index=False,
        )
        if verbose:
            print(f"  Dissolved → {len(gdf)} unique admin-2 polygons.")

    # Build the sub-cell raster: K * ny rows × K * nx cols.
    ny = orbit_lat.size
    nx = orbit_lon.size
    K = int(sub_factor)
    if K < 1:
        raise ValueError(f"sub_factor must be >= 1; got {K}")
    dlat = float(orbit_lat[1] - orbit_lat[0])
    dlon = float(orbit_lon[1] - orbit_lon[0])
    sub_dlat = dlat / K
    sub_dlon = dlon / K
    sub_ny = ny * K
    sub_nx = nx * K
    # rasterio expects "north-up" affine: y0 at top, dy negative.
    lat_max = float(orbit_lat[-1] + dlat / 2)
    lon_min = float(orbit_lon[0] - dlon / 2)
    transform = from_origin(lon_min, lat_max, sub_dlon, sub_dlat)

    # Burn shapes; cell value = position+1 in gdf (0 = sentinel-unassigned).
    # Area-descending order so small polygons survive in their fine sub-cells.
    # (Compute the area column explicitly — `key=lambda g: -g.area` on a
    # plain Series of geometry objects fails: that path goes through
    # pandas' generic Series, which doesn't expose .area; the GeoSeries
    # accessor only kicks in on the geometry column itself.)
    gdf = gdf.assign(_burn_area=gdf.geometry.area)
    gdf = gdf.sort_values("_burn_area", ascending=False, ignore_index=True)
    gdf = gdf.drop(columns=["_burn_area"])
    shapes = [(geom, idx + 1) for idx, geom in enumerate(gdf["geometry"])]
    raster = rasterize(
        shapes=shapes,
        out_shape=(sub_ny, sub_nx),
        transform=transform,
        fill=0,
        dtype=np.uint32,
    )
    if verbose:
        unique = np.unique(raster)
        print(f"  Sub-raster {sub_ny}×{sub_nx}, "
              f"{(raster > 0).sum() / raster.size * 100:.1f}% land coverage, "
              f"{len(unique) - 1} distinct admin-2 hits")

    # Aggregate to coarse cells. raster is "north-up" (top row = lat_max).
    # ORBIT lat array is ASCENDING (lat_min first). Flip rows so that
    # raster_ascending[0] = bottom (lat_min).
    raster_asc = raster[::-1, :]
    # Reshape (sub_ny, sub_nx) → (ny, K, nx, K) and count per (cell, gid).
    blocks = raster_asc.reshape(ny, K, nx, K)

    # For each coarse cell (y, x), iterate over its K×K sub-cells. Build
    # sparse (cell_flat, gid_idx, frac) entries.
    cell_idx_list = []
    gid_idx_list = []
    frac_list = []
    K_sq = float(K * K)
    for y in range(ny):
        for x in range(nx):
            block = blocks[y, :, x, :].ravel()
            block = block[block > 0]  # drop unassigned
            if block.size == 0:
                continue
            uniq, counts = np.unique(block, return_counts=True)
            cell_flat = y * nx + x
            for ug, c in zip(uniq, counts):
                cell_idx_list.append(cell_flat)
                gid_idx_list.append(int(ug) - 1)   # shape index back to 0-based
                frac_list.append(float(c) / K_sq)

    # Cell areas (lat-dependent, lon-uniform within ORBIT).
    cell_row_area = _cell_area_km2(orbit_lat, dlat, dlon)   # (ny,)
    cell_area_2d = np.broadcast_to(
        cell_row_area[:, None], (ny, nx)
    ).ravel().astype(np.float64)

    cell_idx = np.array(cell_idx_list, dtype=np.int32)
    gid_idx = np.array(gid_idx_list, dtype=np.int32)
    frac = np.array(frac_list, dtype=np.float64)
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
