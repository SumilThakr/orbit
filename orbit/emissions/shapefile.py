"""Shapefile/GeoPackage emission loading for ORBIT.

Loads shapefile emissions, adapted for
ORBIT's 6-species merged convention with element conversion applied on the
legacy indices before merging into ORBIT species.

Requires geopandas and shapely (optional dependencies).
Install with: pip install orbit[geo]
"""

import warnings
from typing import Dict, List, Tuple

import numpy as np

from orbit.emissions.sources import (
    CONVERT_COLS,
    ELEMENT_CONVERSION,
    N_ORBIT_SPECIES,
    LEGACY9_TO_ORBIT,
    SPECIES_MAP,
    UNIT_CONVERSIONS,
    EmissionSource,
)
from orbit.emissions.plumerise import find_injection_layer


def load_shapefile_source(source: EmissionSource, grid, verbose: bool = False) -> np.ndarray:
    """Load emissions from shapefile or geopackage into a 4D array.

    Supports point, polygon, and line geometries with spatial apportionment.

    Parameters
    ----------
    source : EmissionSource
        Source configuration
    grid : GridData
        Target grid (needs lon, lat, nz, ny, nx, volume, dlon, dlat)
    verbose : bool
        Print progress

    Returns
    -------
    ndarray, shape (6, nz, ny, nx)
        Emission rates in ug/m^3/s, indexed by ORBIT species
    """
    try:
        import geopandas as gpd
        from shapely.geometry import Point, box
        from shapely.strtree import STRtree
    except ImportError:
        raise ImportError(
            "geopandas and shapely are required for shapefile/geopackage loading. "
            "Install with: pip install orbit[geo]"
        )

    emis_4d = np.zeros((N_ORBIT_SPECIES, grid.nz, grid.ny, grid.nx), dtype=np.float64)

    gdf = gpd.read_file(source.path)
    if len(gdf) == 0:
        return emis_4d

    unit_factor = UNIT_CONVERSIONS.get(source.units, 1.0)

    # Auto-detect species columns
    detected_cols = _detect_species_columns(gdf.columns)

    if verbose and detected_cols:
        print(f"    Detected species columns: {list(detected_cols.keys())}")

    # Build grid cell polygons and spatial index
    grid_cells, strtree, cell_indices = _build_grid_cells(grid)

    # Grid cell dimensions
    dx = grid.dlon if hasattr(grid, 'dlon') else (
        grid.lon[1] - grid.lon[0] if grid.nx > 1 else 0.625
    )
    dy = grid.dlat if hasattr(grid, 'dlat') else (
        grid.lat[1] - grid.lat[0] if grid.ny > 1 else 0.5
    )
    x_origin = grid.lon[0] - dx / 2.0
    y_origin = grid.lat[0] - dy / 2.0

    n_sources = 0

    for idx_row, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue

        # Fix invalid geometries
        if not geom.is_valid:
            geom = geom.buffer(0)
            if geom.is_empty:
                continue

        # Get apportioned cells based on geometry type
        geom_type = geom.geom_type

        if geom_type == 'Point':
            apportioned = _apportion_point(geom, grid, x_origin, y_origin, dx, dy)
        elif geom_type in ('Polygon', 'MultiPolygon'):
            apportioned = _apportion_polygon(geom, grid_cells, strtree, cell_indices)
        elif geom_type in ('LineString', 'MultiLineString'):
            apportioned = _apportion_line(geom, grid_cells, strtree, cell_indices)
        else:
            if verbose:
                warnings.warn(f"Unknown geometry type {geom_type}, using centroid")
            centroid = geom.centroid
            apportioned = _apportion_point(
                Point(centroid.x, centroid.y), grid, x_origin, y_origin, dx, dy
            )

        if not apportioned:
            continue

        # Determine injection layer for elevated sources
        if source.elevated:
            height = float(row.get('height', 0) or 0)
            diam = float(row.get('diam', 0) or 0)
            temp = float(row.get('temp', 293) or 293)
            velocity = float(row.get('velocity', 0) or 0)
        else:
            height = 0

        # Allocate emissions to cells
        has_emissions = False
        for col, legacy_idx in detected_cols.items():
            value = row[col]
            if value is None or (isinstance(value, float) and np.isnan(value)):
                continue
            if value <= 0:
                continue

            has_emissions = True
            orbit_idx = LEGACY9_TO_ORBIT.get(legacy_idx)
            if orbit_idx is None:
                continue

            # Convert to ug/s
            rate_ug_s = float(value) * unit_factor

            # Apply element mass conversion if needed
            col_lower = col.lower().replace('_', '').replace('-', '')
            if col_lower in CONVERT_COLS and legacy_idx in ELEMENT_CONVERSION:
                rate_ug_s *= ELEMENT_CONVERSION[legacy_idx]

            # Apportion to cells
            for i, j, fraction in apportioned:
                if not (0 <= i < grid.nx and 0 <= j < grid.ny):
                    continue

                # Determine layer
                if height > 0:
                    k = find_injection_layer(grid, i, j, height,
                                             stack_diam=diam, stack_temp=temp,
                                             stack_vel=velocity)
                else:
                    k = 0

                vol = grid.volume[k, j, i]
                if vol <= 0:
                    continue

                emis_4d[orbit_idx, k, j, i] += (rate_ug_s * fraction) / vol

        if has_emissions:
            n_sources += 1

    if verbose:
        print(f"    Sources loaded: {n_sources}")

    return emis_4d


# =============================================================================
# Helpers (implemented here)
# =============================================================================

def _detect_species_columns(columns) -> Dict[str, int]:
    """Detect species columns from dataframe columns.

    Returns dict mapping column name -> the reference model species index.
    """
    detected = {}
    for col in columns:
        if col == 'geometry':
            continue
        col_lower = col.lower().replace('_', '').replace('-', '').replace('.', '')
        if col_lower in SPECIES_MAP:
            detected[col] = SPECIES_MAP[col_lower]
    return detected


def _build_grid_cells(grid) -> tuple:
    """Build grid cell polygons and spatial index.

    Returns (grid_cells, strtree, cell_indices) where cell_indices maps
    polygon index to (i, j).
    """
    from shapely.geometry import box
    from shapely.strtree import STRtree

    dx = grid.dlon if hasattr(grid, 'dlon') else (
        grid.lon[1] - grid.lon[0] if grid.nx > 1 else 0.625
    )
    dy = grid.dlat if hasattr(grid, 'dlat') else (
        grid.lat[1] - grid.lat[0] if grid.ny > 1 else 0.5
    )

    grid_cells = []
    cell_indices = {}

    for j in range(grid.ny):
        for i in range(grid.nx):
            lon_min = grid.lon[i] - dx / 2.0
            lon_max = grid.lon[i] + dx / 2.0
            lat_min = grid.lat[j] - dy / 2.0
            lat_max = grid.lat[j] + dy / 2.0

            cell = box(lon_min, lat_min, lon_max, lat_max)
            cell_idx = len(grid_cells)
            grid_cells.append(cell)
            cell_indices[cell_idx] = (i, j)

    strtree = STRtree(grid_cells)
    return grid_cells, strtree, cell_indices


def _apportion_point(point, grid, x_origin: float, y_origin: float,
                     dx: float, dy: float) -> List[Tuple[int, int, float]]:
    """Apportion point to single grid cell (fraction = 1.0)."""
    lon, lat = point.x, point.y
    i = int((lon - x_origin) / dx)
    j = int((lat - y_origin) / dy)

    if 0 <= i < grid.nx and 0 <= j < grid.ny:
        return [(i, j, 1.0)]
    return []


def _apportion_polygon(geom, grid_cells, strtree,
                        cell_indices) -> List[Tuple[int, int, float]]:
    """Apportion polygon emissions by area intersection.

    Each cell receives fraction = intersection_area / total_polygon_area.
    """
    from shapely.geometry import MultiPolygon

    if geom.is_empty:
        return []

    total_area = geom.area
    if total_area <= 0:
        return []

    if isinstance(geom, MultiPolygon):
        geom = geom.buffer(0)

    candidate_indices = strtree.query(geom)

    apportioned = []
    for cell_idx in candidate_indices:
        cell = grid_cells[cell_idx]
        intersection = geom.intersection(cell)

        if intersection.is_empty:
            continue

        intersection_area = intersection.area
        if intersection_area <= 0:
            continue

        fraction = intersection_area / total_area
        i, j = cell_indices[cell_idx]
        apportioned.append((i, j, fraction))

    return apportioned


def _apportion_line(geom, grid_cells, strtree,
                    cell_indices) -> List[Tuple[int, int, float]]:
    """Apportion line emissions by length intersection.

    Each cell receives fraction = clipped_segment_length / total_line_length.
    """
    if geom.is_empty:
        return []

    total_length = geom.length
    if total_length <= 0:
        return []

    candidate_indices = strtree.query(geom)

    apportioned = []
    for cell_idx in candidate_indices:
        cell = grid_cells[cell_idx]
        intersection = geom.intersection(cell)

        if intersection.is_empty:
            continue

        if intersection.geom_type == 'GeometryCollection':
            intersection_length = sum(
                g.length for g in intersection.geoms
                if g.geom_type in ('LineString', 'MultiLineString')
            )
        else:
            intersection_length = intersection.length

        if intersection_length <= 0:
            continue

        fraction = intersection_length / total_length
        i, j = cell_indices[cell_idx]
        apportioned.append((i, j, fraction))

    return apportioned
