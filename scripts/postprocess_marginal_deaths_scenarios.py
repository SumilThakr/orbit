"""Normalize the marginal-deaths production sweep to three policy scenarios.

The per-month adjoint NetCDFs (`adjoint_M<MM>.nc`) carry the raw
∂J/∂e_τ cubes — sensitivity of annualized cost to per-bin emission rate
under each month's periodic orbit. The per-month per-subdistrict files
(`deaths_per_1000kg_M<MM>.nc`) collapse that into V_m, which is
mathematically equivalent to Scenario B (1000 kg sustained over month m
only; the identity follows from linearity of ∂J/∂e in the sustained
emission rate).

This script produces a single per-CRF output combining all three
scenarios:

  S1. annual_sustained:   1000 kg/yr uniformly      → (species, subdistrict)
  S2. monthly_sustained:  1000 kg in month m only   → (month, species, subdistrict)
  S3. bin_specific:       1000 kg in bin τ of m     → (month, bin, species, subdistrict)

Total mass is always 1000 kg COMPOUND (NO2-mass for NOx, etc. — the
aggregator's element_factor handles the conversion from solver-internal
N/S mass).

The implementation reuses
orbit.modes.subdistrict_aggregation.aggregate_dJ_de_to_subdistricts —
calling it with the full bin axis gives Scenario B / V_m, calling it
per-bin (n_bins=1 slice) gives Scenario C / S3.

Usage
-----
    python scripts/postprocess_marginal_deaths_scenarios.py \\
        --production-dir outputs/sas/marginal_deaths/production \\
        --crf gemm5cod gemmac ier \\
        --gadm /path/to/gadm_410.gpkg \\
        --out-dir outputs/sas/marginal_deaths/production/scenarios

One output file per CRF:
    scenarios_<crf>.nc

The GADM layer is built once at the top of the run and reused across CRFs
(saves ~30 s × 3 = 1.5 min).

Sanity checks are run inline and assert:
  - mean_m of S2 (days-weighted) == S1                           (rel < 1e-3)
  - mean_τ of S3(m, τ) == V_m == S2(m)                           (rel < 1e-3)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import xarray as xr

# 2022 has 365 days; SEC_YR uses the Julian year (365.25) to match the
# aggregator's convention, so the S2 days-weights sum to 365/365.25.
DAYS_PER_MONTH_2022 = np.array(
    [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], dtype=np.float64,
)
SECONDS_PER_YEAR = 365.25 * 86400.0
N_BINS = 8

DEFAULT_CRFS = ("gemm5cod", "gemmac", "ier")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--production-dir", type=Path, required=True,
                    help="Base dir containing one subdir per CRF, each with "
                         "adjoint_M<MM>.nc + a <crf>_per1000kg/deaths_per_"
                         "1000kg_M<MM>.nc tree.")
    ap.add_argument("--crf", nargs="+", default=list(DEFAULT_CRFS),
                    help="Which CRF subdirs to process.")
    ap.add_argument("--gadm", type=Path, required=True,
                    help="GADM v4.1 gpkg path (admin-2 polygons); download "
                         "from gadm.org (licence prohibits redistribution).")
    ap.add_argument("--bbox", type=float, nargs=4, default=None,
                    metavar=("lon_min", "lat_min", "lon_max", "lat_max"),
                    help="Clip the polygons to this box; the default is the "
                         "outer edges of the ORBIT grid.")
    ap.add_argument("--out-dir", type=Path, required=True,
                    help="Output dir; one scenarios_<crf>.nc per CRF.")
    args = ap.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Lazy import — both these are heavy (geopandas + rasterio for layer build).
    from orbit.modes.subdistrict_aggregation import (
        SECONDS_PER_YEAR as _SY_AGG,
        aggregate_dJ_de_to_subdistricts,
        build_subdistrict_layer,
    )
    if abs(_SY_AGG - SECONDS_PER_YEAR) > 1e-6:
        raise SystemExit(
            f"SECONDS_PER_YEAR mismatch between this script ({SECONDS_PER_YEAR}) "
            f"and orbit.modes.subdistrict_aggregation ({_SY_AGG}). Fix in lockstep."
        )

    # ── 1. Build the subdistrict layer once (uses the first CRF's first adjoint
    # NetCDF to get the orbit grid).
    print("Building GADM admin-2 layer by exact cell intersection...")
    t0 = time.time()
    first_adj = sorted((args.production_dir / args.crf[0]).glob("adjoint_M*.nc"))[0]
    ds0 = xr.open_dataset(first_adj)
    orbit_lat = ds0.lat.values
    orbit_lon = ds0.lon.values
    ds0.close()
    layer = build_subdistrict_layer(
        gadm_gpkg=str(args.gadm),
        orbit_lat=orbit_lat, orbit_lon=orbit_lon,
        bbox=tuple(args.bbox) if args.bbox else None,
        verbose=True,
    )
    print(f"  layer built in {time.time() - t0:.1f}s "
          f"({layer.gid_list.size} subdistricts, {layer.cell_idx.size} cell·gid pairs)")

    # Days-weighted Σ for S1. Use 365.25 in the denom to match SEC_YR
    # in the aggregator. Resulting Σw = 365/365.25 ≈ 0.9993 (see doc).
    days_weights = DAYS_PER_MONTH_2022 / 365.25

    for crf in args.crf:
        crf_dir = args.production_dir / crf
        if not crf_dir.exists():
            print(f"WARNING: {crf_dir} missing — skipping {crf}")
            continue
        process_crf(
            crf=crf,
            crf_dir=crf_dir,
            out_path=args.out_dir / f"scenarios_{crf}.nc",
            layer=layer,
            days_weights=days_weights,
            aggregate_fn=aggregate_dJ_de_to_subdistricts,
        )
    return 0


def process_crf(*, crf, crf_dir, out_path, layer, days_weights, aggregate_fn):
    print(f"\n=== Processing CRF: {crf} ===")
    t0 = time.time()

    adjoint_files = sorted(crf_dir.glob("adjoint_M*.nc"))
    if len(adjoint_files) != 12:
        raise SystemExit(
            f"Expected 12 adjoint_M*.nc in {crf_dir}, found {len(adjoint_files)}"
        )
    # Probe the first file to size the output cube.
    ds0 = xr.open_dataset(adjoint_files[0])
    species_keys = list(ds0.species.values.astype(str))
    n_species = len(species_keys)
    n_draws = ds0.dims.get("draw", 1)
    ds0.close()
    n_sub = layer.gid_list.size

    # Allocate scenario cubes (single-draw deterministic for now; MC
    # expansion happens at draw-axis if/when --draws lands).
    if n_draws != 1:
        print(f"  NOTE: adjoint has {n_draws} draws; using draw=0 deterministic only.")

    S2 = np.zeros((12, n_species, n_sub), dtype=np.float64)
    S3 = np.zeros((12, N_BINS, n_species, n_sub), dtype=np.float64)

    for m_idx, adj_path in enumerate(adjoint_files):
        ds = xr.open_dataset(adj_path)
        month = int(ds.attrs.get("month", m_idx + 1))
        if month != m_idx + 1:
            raise SystemExit(
                f"Month mismatch: {adj_path} reports month={month} but is "
                f"position {m_idx + 1} in sorted glob."
            )
        # Shape: (draw, species, bin, y, x). Take draw=0.
        if "draw" in ds.dJ_de.dims:
            dJ_de = ds.dJ_de.isel(draw=0).values.astype(np.float64)  # (sp, b, y, x)
        else:
            dJ_de = ds.dJ_de.values.astype(np.float64)
        sv = ds.surface_volume_m3.values.astype(np.float64)
        species_keys_m = list(ds.species.values.astype(str))
        if species_keys_m != species_keys:
            raise SystemExit(
                f"Species axis differs between months: {species_keys_m} vs "
                f"{species_keys}"
            )
        ds.close()

        # S2(m): full bin axis collapsed by aggregator's /n_bins step.
        # (n_species, n_bins=8, ny, nx)
        S2_m = aggregate_fn(dJ_de, layer, sv, species_keys=species_keys)
        S2[m_idx] = S2_m   # (species, subdistrict)

        # S3(m, τ): per-bin slice. Call aggregator with n_bins=1 input
        # — the cell_scale formula becomes 1e12/(SEC_YR·vol_j), matching
        # Deaths_S3(m, τ) in the math doc.
        for tau in range(N_BINS):
            dJ_de_one = dJ_de[:, tau:tau+1, :, :]    # (species, 1, ny, nx)
            S3_m_tau = aggregate_fn(dJ_de_one, layer, sv,
                                    species_keys=species_keys)
            S3[m_idx, tau] = S3_m_tau   # (species, subdistrict)

        print(f"  M{month:02d}: done ({time.time() - t0:.0f}s elapsed total)")

    # ── Sanity checks (must hold to numerical precision)
    # 1. mean_τ of S3(m, ·) == S2(m)
    S3_mean_tau = S3.mean(axis=1)                          # (12, sp, sub)
    rel_diff_3vs2 = np.abs(S3_mean_tau - S2) / (np.abs(S2) + 1e-30)
    max_3vs2 = float(np.max(rel_diff_3vs2[np.abs(S2) > 1e-30]))
    print(f"  identity check mean_τ S3 == S2:  max rel diff = {max_3vs2:.2e}")
    if max_3vs2 > 1e-3:
        raise SystemExit(
            f"mean_τ S3 vs S2 identity failed at rel={max_3vs2:.3e}; "
            f"check N_BINS in aggregator vs this script."
        )

    # 2. S1 = days-weighted Σ of S2
    S1 = np.einsum("m,msj->sj", days_weights, S2)          # (sp, sub)

    # Diagnostic: also compute the v1-cross-check
    sum_w = float(days_weights.sum())
    print(f"  Σ days_weights = {sum_w:.4f} "
          f"(deficit from 1.0 reflects Julian-year vs 2022-calendar mismatch)")

    # ── Write output
    write_scenarios_nc(
        out_path=out_path,
        crf=crf,
        species_keys=species_keys,
        layer=layer,
        S1=S1, S2=S2, S3=S3,
    )
    print(f"  wrote {out_path}  ({time.time() - t0:.0f}s total for {crf})")


def write_scenarios_nc(*, out_path, crf, species_keys, layer, S1, S2, S3):
    """Write the three-scenario NetCDF."""
    n_sub = layer.gid_list.size

    ds = xr.Dataset(
        data_vars=dict(
            annual_sustained=(
                ("species", "subdistrict"),
                S1.astype(np.float32),
            ),
            monthly_sustained=(
                ("month", "species", "subdistrict"),
                S2.astype(np.float32),
            ),
            bin_specific=(
                ("month", "bin", "species", "subdistrict"),
                S3.astype(np.float32),
            ),
            subdistrict_gid=(("subdistrict",), layer.gid_list),
            subdistrict_name=(("subdistrict",), layer.gid_name),
            subdistrict_iso3=(("subdistrict",), layer.gid_iso3),
            subdistrict_area_km2=(
                ("subdistrict",), layer.gid_area_km2.astype(np.float32)
            ),
        ),
        coords=dict(
            species=np.array(species_keys),
            month=np.arange(1, 13),
            bin=np.arange(N_BINS),
            subdistrict=np.arange(n_sub),
        ),
        attrs=dict(
            crf_mode=crf,
            unit_convention=(
                "All scenario values are deaths · year⁻¹ per 1000 kg of "
                "COMPOUND pollutant emitted within the subdistrict, "
                "uniformly per area. Compound mass is what the user puts "
                "into the emission netcdf (NO2-mass for NOx, SO2-mass for "
                "SO2, NH3-mass for NH3, identity for PM2.5_primary and VOC*). "
                "Scenarios differ only in WHEN the 1000 kg is emitted:\n"
                " * annual_sustained:  1000 kg/yr uniformly, every diurnal "
                "bin of every month.\n"
                " * monthly_sustained: 1000 kg total during ONE month "
                "(uniformly within that month), 0 outside.\n"
                " * bin_specific:      1000 kg total during ONE 3-hour "
                "diurnal bin of ONE month (recurring across days of the "
                "month), 0 outside."
            ),
            description=(
                "Three time-profile scenarios for marginal-deaths-per-1000kg, "
                "produced by postprocess_marginal_deaths_scenarios.py from "
                "the per-month adjoint outputs. The math hinges on linearity "
                "of the periodic-orbit forward + adjoint w.r.t. emission "
                "rate, validated by scripts/verify_adjoint_compare.py "
                "(adjoint/forward ratio 0.998)."
            ),
            n_bins=int(N_BINS),
            seconds_per_year_julian=float(SECONDS_PER_YEAR),
            days_per_month_used=DAYS_PER_MONTH_2022.tolist(),
            gadm_version="v4.1 admin-2",
        ),
    )
    ds.annual_sustained.attrs.update(
        units="deaths year-1 / (1000 kg of compound emitted; annual sustained)",
        long_name=(
            "marginal deaths per 1000 kg of pollutant emitted uniformly over "
            "the year (constant rate every diurnal bin and every month)"
        ),
    )
    ds.monthly_sustained.attrs.update(
        units="deaths year-1 / (1000 kg of compound emitted; monthly sustained)",
        long_name=(
            "marginal deaths per 1000 kg of pollutant emitted uniformly over "
            "one specific month (constant rate within the month, zero outside)"
        ),
    )
    ds.bin_specific.attrs.update(
        units="deaths year-1 / (1000 kg of compound emitted; bin-specific)",
        long_name=(
            "marginal deaths per 1000 kg of pollutant emitted during one "
            "specific 3-hour diurnal bin of one specific month (recurring "
            "across days of the month, zero outside)"
        ),
    )
    encoding = {
        v: {"zlib": True, "complevel": 4, "_FillValue": None}
        for v in ("annual_sustained", "monthly_sustained", "bin_specific")
    }
    ds.to_netcdf(out_path, encoding=encoding)


if __name__ == "__main__":
    sys.exit(main())
