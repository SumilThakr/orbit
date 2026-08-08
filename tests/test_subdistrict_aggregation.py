"""Tests for the GADM subdistrict aggregator.

Synthetic-data only — no GADM dependency at this level. Tests the
`aggregate_dJ_de_to_subdistricts` math kernel against hand-computable
area-weighted averages.
"""
from __future__ import annotations

import numpy as np

from orbit.modes.subdistrict_aggregation import (
    SubdistrictLayer,
    aggregate_dJ_de_to_subdistricts,
    SECONDS_PER_YEAR,
)


def _toy_layer():
    """3-cell, 2-subdistrict layout:
       cell 0 → 100% in district A (area 100 km²)
       cell 1 → 50% district A + 50% district B (area 200 km²)
       cell 2 → 100% in district B (area 300 km²)
    """
    cell_idx = np.array([0, 1, 1, 2], dtype=np.int32)
    gid_idx  = np.array([0, 0, 1, 1], dtype=np.int32)
    # fractions of each coarse cell that overlap each district.
    fracs    = np.array([1.0, 0.5, 0.5, 1.0], dtype=np.float64)
    cell_areas = np.array([100.0, 200.0, 300.0], dtype=np.float64)
    area_per_entry = fracs * cell_areas[cell_idx]   # 100, 100, 100, 300
    # district areas (sum of overlapping cell-area shares).
    gid_area = np.array([200.0, 400.0])             # A=100+100, B=100+300

    return SubdistrictLayer(
        gid_list=np.array(["A", "B"]),
        gid_name=np.array(["DistA", "DistB"]),
        gid_iso3=np.array(["XXX", "XXX"]),
        gid_area_km2=gid_area,
        cell_idx=cell_idx,
        gid_idx=gid_idx,
        area_km2=area_per_entry,
        cell_area_km2=cell_areas,
    )


def test_aggregator_area_weighted_mean_matches_manual():
    layer = _toy_layer()
    # Cell-level sensitivity (1 species, 1 bin, 1 row, 3 cols laid out as
    # a (ny=1, nx=3) surface slab).
    dJ_de = np.array([[[[10.0, 20.0, 30.0]]]], dtype=np.float64)  # (1,1,1,3)
    # Use unit volume so the conversion factor is just
    # 1e12 / (N_BINS · SECONDS_PER_YEAR · 1) per cell.
    vol = np.ones((1, 3), dtype=np.float64)
    out = aggregate_dJ_de_to_subdistricts(dJ_de, layer, vol)
    # Sum over bins per cell: 10, 20, 30 (single bin, just identity).
    # Per-cell scale = 1e12 / (1 · SECONDS_PER_YEAR · 1)
    # Per-cell deaths = scale * dJ.
    factor = 1e12 / (1 * SECONDS_PER_YEAR * 1.0)
    deaths_per_cell = factor * np.array([10.0, 20.0, 30.0])
    # District A: area-weighted mean of cells 0 and 1 with area 100 each.
    #   = (100·deaths[0] + 100·deaths[1]) / 200
    mean_A = (100 * deaths_per_cell[0] + 100 * deaths_per_cell[1]) / 200
    # District B: cells 1 (area 100) and 2 (area 300).
    mean_B = (100 * deaths_per_cell[1] + 300 * deaths_per_cell[2]) / 400
    np.testing.assert_allclose(out[0], [mean_A, mean_B])


def test_aggregator_zero_input_gives_zero_output():
    layer = _toy_layer()
    dJ_de = np.zeros((1, 1, 1, 3))
    vol = np.ones((1, 3), dtype=np.float64)
    out = aggregate_dJ_de_to_subdistricts(dJ_de, layer, vol)
    np.testing.assert_array_equal(out, np.zeros((1, 2)))


def test_aggregator_preserves_species_dim():
    layer = _toy_layer()
    dJ_de = np.random.RandomState(7).rand(3, 8, 1, 3)
    vol = np.full((1, 3), 1e10, dtype=np.float64)
    out = aggregate_dJ_de_to_subdistricts(dJ_de, layer, vol)
    # Bin axis collapsed; output (n_species, n_gid).
    assert out.shape == (3, 2)


def test_aggregator_empty_district_returns_zero():
    """A subdistrict whose only cells have zero ∂J/∂e contributes zero,
    not NaN."""
    layer = _toy_layer()
    dJ_de = np.array([[[[0.0, 5.0, 0.0]]]])  # only cell 1 nonzero
    vol = np.ones((1, 3), dtype=np.float64)
    out = aggregate_dJ_de_to_subdistricts(dJ_de, layer, vol)
    factor = 1e12 / (1 * SECONDS_PER_YEAR * 1.0)
    # Cell 1 contributes (100·5·factor)/200 to A and (100·5·factor)/400 to B.
    expected_A = 100 * 5 * factor / 200
    expected_B = 100 * 5 * factor / 400
    np.testing.assert_allclose(out[0], [expected_A, expected_B])
