"""Compare adjoint vs forward-marginal δDeaths for a localized perturbation.

Reads:
  - adjoint NetCDF (from compute_marginal_deaths.py)
  - marginal NPZ from the forward verification sbatch
  - the perturbation NetCDF used in the forward run (to compute the
    effective emission rate at the test cell)

Computes two δDeaths predictions for the same emission perturbation:

  δDeaths_fwd = Σ_k S_orbit_k · ⟨ mean over bins (surface δPM25) ⟩_k

  δDeaths_adj = (1/N_BINS) · Σ_τ ∂J/∂e[species, τ, j_cell] · e_rate_kg_s

where e_rate_kg_s = NetCDF magnitude × cell area in m². The (1/N_BINS)
factor reverses the J = Σ_τ ⟨S, G c_τ⟩ over-summation that the adjoint
solver carries (J is N_BINS × the true annual-mean deaths).

A ratio adjoint/forward of 1.00 ± 0.01 means the math + units are right.
A ratio of N_BINS=8 means we forgot a division. Other ratios → bug.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import xarray as xr

# Match orbit.core.orbit.N_BINS.
N_BINS = 8


def _cell_area_m2(lat_centres: np.ndarray, dlat_deg: float, dlon_deg: float) -> np.ndarray:
    """Per-row cell area in m² for an EPSG:4326 lat-lon raster."""
    R_m = 6_371_000.0
    lat_rad = np.deg2rad(lat_centres)
    dlat_rad = np.deg2rad(dlat_deg)
    dlon_rad = np.deg2rad(dlon_deg)
    return R_m * R_m * dlon_rad * (np.sin(lat_rad + dlat_rad / 2)
                                    - np.sin(lat_rad - dlat_rad / 2))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--adjoint", type=Path, required=True,
                    help="adjoint_M01.nc from compute_marginal_deaths.py")
    ap.add_argument("--marginal", type=Path, required=True,
                    help="marginal_M01_verify.npz from the forward sbatch")
    ap.add_argument("--perturbation", type=Path, required=True,
                    help="The perturbation NetCDF the forward run was given.")
    ap.add_argument("--species", default="NOx",
                    help="Perturbed species (must match the variable name in "
                         "the perturbation NetCDF and the adjoint output).")
    ap.add_argument("--lat-idx", type=int, required=True,
                    help="0-based lat index of the perturbation cell.")
    ap.add_argument("--lon-idx", type=int, required=True,
                    help="0-based lon index of the perturbation cell.")
    args = ap.parse_args(argv)

    # 1. Load adjoint output.
    ds_adj = xr.open_dataset(args.adjoint)
    dJ_de = (
        ds_adj.dJ_de.sel(species=args.species).isel(draw=0).values
    )   # (n_bins, ny, nx)
    if "draw" in ds_adj.S_orbit.dims:
        S_orbit = ds_adj.S_orbit.isel(draw=0).values
    else:
        S_orbit = ds_adj.S_orbit.values   # (y, x) legacy layout
    if "surface_volume_m3" not in ds_adj.data_vars:
        raise SystemExit(
            "Adjoint NetCDF lacks `surface_volume_m3` — re-run the adjoint "
            "with the unit-fix patch (orbit/modes/adjoint.py)."
        )
    surface_volume_m3 = ds_adj["surface_volume_m3"].values
    lat = ds_adj.lat.values
    lon = ds_adj.lon.values
    ds_adj.close()

    ny, nx = S_orbit.shape
    vol_j = float(surface_volume_m3[args.lat_idx, args.lon_idx])

    # 2. Perturbation magnitude in BOTH unit systems:
    #    - kg/s at the cell (for reporting)
    #    - µg/m³/s in solver units (for adjoint multiplication)
    ds_pert = xr.open_dataset(args.perturbation)
    pert_value = float(
        ds_pert[args.species].values[0, args.lat_idx, args.lon_idx]
    )   # kg/m²/s
    ds_pert.close()

    dlat = float(abs(lat[1] - lat[0]))
    dlon = float(abs(lon[1] - lon[0]))
    cell_row_areas = _cell_area_m2(lat, dlat, dlon)
    cell_area = float(cell_row_areas[args.lat_idx])
    e_rate_kg_s = pert_value * cell_area
    # Element-mass conversion: the netcdf loader applies it for NOx/NH3/
    # SO2 (compound mass on disk → element mass in the solver). For NOx
    # that's 0.3045 (N / NO2-equivalent). For NH3, 0.8224 (N / NH3).
    # For SO2, 0.5005 (S / SO2). PM25_primary and VBS bins have no
    # conversion. Match what orbit.emissions.netcdf.ELEMENT_CONVERSION
    # does in load_netcdf_source.
    from orbit.emissions.netcdf import (
        ELEMENT_CONVERSION, SPECIES_MAP,
    )
    species_lower = args.species.lower().replace("_", "").replace("-", "")
    legacy_idx = SPECIES_MAP.get(species_lower)
    element_factor = ELEMENT_CONVERSION.get(legacy_idx, 1.0)
    # Solver consumes µg/m³/s, with element-mass conversion ALREADY applied
    # by the netcdf loader: e_solver = pert × element × area × 1e9 / vol.
    e_solver_ug_m3_s = (
        pert_value * element_factor * cell_area * 1e9 / vol_j
    )
    print(f"Perturbation cell: lat={float(lat[args.lat_idx]):.3f}, "
          f"lon={float(lon[args.lon_idx]):.3f}")
    print(f"  cell area:                {cell_area:.3e} m²")
    print(f"  surface vol:              {vol_j:.3e} m³ "
          f"(layer thickness ~{vol_j/cell_area:.1f} m)")
    print(f"  pert value:               {pert_value:.3e} kg/m²/s ({args.species})")
    print(f"  element factor:           {element_factor:.4f} "
          f"({'identity' if element_factor == 1.0 else 'compound→element mass'})")
    print(f"  e_rate_kg_s:              {e_rate_kg_s:.3e}")
    print(f"  e_solver (µg/m³/s):       {e_solver_ug_m3_s:.3e}")

    # 3. Adjoint prediction (CORRECTED unit conversion).
    djde_at_j = dJ_de[:, args.lat_idx, args.lon_idx]
    sum_djde = float(djde_at_j.sum())
    # δJ = Σ_τ ∂J/∂e_τ × e_solver  (same e per bin, sustained)
    # δDeaths = δJ / N_BINS  (reverses the over-summation in J = Σ_τ ⟨S, G c_τ⟩)
    delta_deaths_adj = sum_djde * e_solver_ug_m3_s / N_BINS
    print(f"  ∂J/∂e sum over bins:      {sum_djde:.4e}")
    print(f"  δDeaths_adj:              {delta_deaths_adj:.4e}")

    # 4. Forward prediction (linearised deaths response).
    marg = np.load(args.marginal, allow_pickle=False)
    # delta_pm25_orbit shape: (8, nz, ny, nx). Surface (z=0), annual mean over bins.
    delta_pm25_surf_annual = marg["delta_pm25_orbit"][:, 0].mean(axis=0)  # (ny, nx)
    if delta_pm25_surf_annual.shape != (ny, nx):
        raise SystemExit(
            f"Shape mismatch: marginal δPM25 surface {delta_pm25_surf_annual.shape} "
            f"vs adjoint grid ({ny}, {nx})"
        )
    delta_deaths_fwd = float(np.sum(S_orbit * delta_pm25_surf_annual))
    print(f"  δPM25 max:                       {float(delta_pm25_surf_annual.max()):.3e} µg/m³")
    print(f"  δDeaths_fwd  (= S · δPM25):      {delta_deaths_fwd:.4e}")

    # 5. Report.
    if delta_deaths_fwd != 0:
        ratio = delta_deaths_adj / delta_deaths_fwd
    else:
        ratio = float("nan")
    print()
    print(f"=== RATIO adjoint / forward = {ratio:.4f} ===")
    print("Expected ≈ 1.00 (linearisation should be tight at this perturbation)")
    if abs(ratio - 1.0) < 0.05:
        print("✅ MATCH within 5% — adjoint math + units verified.")
    elif abs(ratio - 8.0) < 0.5:
        print("⚠️  ~8× off: the J = Σ_τ ⟨S, G c_τ⟩ over-summation needs an "
              "extra /N_BINS at the aggregator or adjoint receptor step.")
    elif abs(ratio - 1.0 / 8.0) < 0.02:
        print("⚠️  ~1/8× off: an over-division by N_BINS slipped in.")
    else:
        print("❌ Unexpected ratio — check growth Jacobian, K^T propagation, "
              "or operator-consistency between forward and adjoint.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
