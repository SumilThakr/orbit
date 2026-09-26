#!/usr/bin/env python3
"""Write the subdistrict marginal damages of a scenarios NetCDF as CSV.

One CSV per scenarios_<crf>.nc: a row per (subdistrict, species) with the GADM
identifier, name, country, area and the annual-sustained damage, plus the
twelve monthly-sustained damages. Units follow the NetCDF's
``unit_convention`` attribute (deaths per year per 1000 kg emitted per year,
as written by postprocess_marginal_deaths_scenarios.py). This is the CSV
form of the results deposit the README refers to.

Usage:
    python scripts/export_subdistrict_damages_csv.py <scenarios_dir> [--out-dir DIR]
"""
import argparse
import csv
import glob
import os

import numpy as np
import xarray as xr


def export(path, out_dir):
    ds = xr.open_dataset(path)
    crf = str(ds.attrs.get("crf_mode", os.path.basename(path)))
    out = os.path.join(out_dir, os.path.basename(path).replace(".nc", ".csv"))
    species = [str(s) for s in ds["species"].values]
    months = [int(m) for m in ds["month"].values]
    annual = ds["annual_sustained"].values          # (species, subdistrict)
    monthly = ds["monthly_sustained"].values        # (month, species, subdistrict)
    gid = ds["subdistrict_gid"].values
    name = ds["subdistrict_name"].values
    iso3 = ds["subdistrict_iso3"].values
    area = ds["subdistrict_area_km2"].values
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["# " + str(ds.attrs.get("unit_convention", "")).replace("\n", " ")])
        w.writerow(["# crf_mode=" + crf + "; source=" + os.path.basename(path)])
        w.writerow(["gid_2", "name", "iso3", "area_km2", "species", "annual_sustained"]
                   + [f"monthly_sustained_M{m:02d}" for m in months])
        for j in range(gid.size):
            for k, sp in enumerate(species):
                w.writerow([str(gid[j]), str(name[j]), str(iso3[j]), f"{float(area[j]):.3f}", sp,
                            f"{float(annual[k, j]):.6e}"]
                           + [f"{float(monthly[mi, k, j]):.6e}" for mi in range(len(months))])
    ds.close()
    return out, gid.size * len(species)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("scenarios_dir")
    ap.add_argument("--out-dir", default=None, help="default: the scenarios directory")
    a = ap.parse_args()
    out_dir = a.out_dir or a.scenarios_dir
    os.makedirs(out_dir, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(a.scenarios_dir, "scenarios_*.nc")))
    if not paths:
        raise SystemExit(f"no scenarios_*.nc under {a.scenarios_dir}")
    for p in paths:
        out, n = export(p, out_dir)
        print(f"{out}: {n} rows")


if __name__ == "__main__":
    main()
