#!/usr/bin/env python3
"""Measure how far the transport operator is from conserving air mass.

For one month of preprocessed grids this script assembles each transport
block (horizontal convection-diffusion, vertical advection, vertical
diffusion, CMFMC convection) on every diurnal bin and reports three things.

1. Row sums of each block and of their sum, per layer, over laterally
   interior cells. For a uniform mixing ratio the transport block changes cell
   i at the rate -rowsum_i, so a non-zero row sum is a spurious source (< 0)
   or sink (> 0) of mixing ratio. Rates are printed per day.
2. The dP-weighted column sums of the vertical block at the surface, which
   measure any flux through the ground, and the rate at which the layer-0
   mid-level omega_plus would remove surface-layer mass if it were applied at
   the ground (the leak closed on 2026-09-25), against dry and wet deposition
   for a uniform field.
3. The implied air density rho = dP / (g Dz): its range at the surface, the
   ratio of surface density to each layer's density, and the surface density
   at a few sites relative to Delhi. These bound the effect of reading the
   transported variable as a concentration at local density.

Usage:
    python scripts/check_air_mass_continuity.py \
        --grid-dir $ORBIT_DATA/inputs/grids_2022 --month 1

Nothing is written; the report goes to stdout. Takes about a minute.
"""
import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orbit.core.grid_data import load_grid  # noqa: E402
from orbit.core.indexing import CellIndexer  # noqa: E402
from orbit.core.convdiff import assemble_horizontal_convdiff  # noqa: E402
from orbit.core.advection import assemble_vertical_advection  # noqa: E402
from orbit.core.mixing import assemble_vertical_diffusion  # noqa: E402
from orbit.core.convection import assemble_cmfmc_transport  # noqa: E402

GRAVITY = 9.80665
DAY = 86400.0

SITES = {
    "Delhi": (28.6, 77.2), "Kolkata": (22.6, 88.4), "Lucknow": (26.8, 80.9),
    "Dehradun": (30.3, 78.0), "Shimla": (31.1, 77.2), "Kathmandu": (27.7, 85.3),
    "Lhasa": (29.65, 91.1), "Leh": (34.2, 77.6), "Thimphu": (27.5, 89.6),
    "Srinagar": (34.1, 74.8), "Bengaluru": (13.0, 77.6),
}


def row_sums(T, shape):
    return np.asarray(T.sum(axis=1)).ravel().reshape(shape)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--grid-dir", required=True, help="directory of sas_<year>_M<MM>_B<bb>.nc files")
    ap.add_argument("--month", type=int, default=1)
    ap.add_argument("--year-tag", default="2022")
    ap.add_argument("--constants", default=None, help="MERRA2 constants NetCDF (land mask); optional")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(
        args.grid_dir, f"sas_{args.year_tag}_M{args.month:02d}_B*.nc")))
    if not paths:
        sys.exit(f"no grid files for month {args.month} under {args.grid_dir}")

    acc = {}
    for p in paths:
        g = load_grid(p, args.constants)
        nz, ny, nx = g.nz, g.ny, g.nx
        idx = CellIndexer(nz, ny, nx)
        blocks = {
            "horizontal": assemble_horizontal_convdiff(g, idx),
            "vertical advection": assemble_vertical_advection(g, idx),
            "vertical diffusion": assemble_vertical_diffusion(g, idx),
            "convection": assemble_cmfmc_transport(g, idx),
        }
        blocks["total"] = sum(blocks.values())
        for name, T in blocks.items():
            acc.setdefault(name, []).append(row_sums(T, (nz, ny, nx)) * DAY)

        area = (g.dx[None, :, None] * g.dy) * np.ones((nz, ny, nx))
        W = (area * g.dP).ravel()
        WT_v = (W @ blocks["vertical advection"]).reshape(nz, ny, nx)
        acc.setdefault("ground flux", []).append(WT_v[0] / area[0])          # Pa/s leaving through the ground
        acc.setdefault("leak if open", []).append(g.omega_plus[0] / g.dP[0] * DAY)
        acc.setdefault("dry", []).append(g.particle_dry_dep[0] / g.Dz[0] * DAY)
        acc.setdefault("wet0", []).append(g.particle_wet_dep[0] * DAY)
        acc.setdefault("rho", []).append(g.dP / (GRAVITY * np.where(g.Dz > 0, g.Dz, 1.0)))
        acc.setdefault("Psurf", []).append(g.Psurf)
        acc.setdefault("Dz", []).append(g.Dz)
    lat, lon = g.lat, g.lon
    inner = (slice(None), slice(1, -1), slice(1, -1))

    print(f"{len(paths)} bins, month {args.month}, grid {nz} x {ny} x {nx}")
    print("\n1. Row sums per day over laterally interior cells, all bins pooled.")
    print("   Positive = net air divergence (spurious sink of a uniform mixing ratio).")
    for name in ["horizontal", "vertical advection", "vertical diffusion", "convection", "total"]:
        R = np.stack(acc[name])[(slice(None),) + inner]
        print(f"\n   {name}")
        print(f"   {'layer':>5} {'median|r|':>10} {'p90|r|':>10} {'p99|r|':>10} {'max|r|':>10} {'mean r':>10}")
        for k in range(nz):
            r = R[:, k].ravel()
            a = np.abs(r)
            print(f"   {k:5d} {np.median(a):10.4f} {np.percentile(a, 90):10.4f} "
                  f"{np.percentile(a, 99):10.4f} {a.max():10.4f} {r.mean():10.4f}")

    print("\n2. Flux through the ground.")
    gf = np.stack(acc["ground flux"])
    print(f"   max |W^T T_vadv| at the surface / area = {np.abs(gf).max():.3e} Pa/s "
          "(zero when the ground is closed)")
    leak = np.stack(acc["leak if open"])[:, 1:-1, 1:-1]
    dry = np.stack(acc["dry"])[:, 1:-1, 1:-1]
    wet0 = np.stack(acc["wet0"])[:, 1:-1, 1:-1]
    print("   Surface-layer loss rates per day for a uniform field (interior cells):")
    print(f"   {'':>28} {'median':>8} {'p90':>8} {'max':>8} {'domain':>8}")
    for name, v in [("omega_plus[0]/dP[0] (if open)", leak), ("dry deposition", dry), ("wet deposition, layer 0", wet0)]:
        print(f"   {name:>28} {np.median(v):8.3f} {np.percentile(v, 90):8.3f} {v.max():8.3f} {v.mean():8.3f}")

    print("\n3. Implied air density rho = dP/(g Dz), bin mean.")
    rho = np.stack(acc["rho"]).mean(axis=0)
    Ps = np.stack(acc["Psurf"]).mean(axis=0)
    r0 = rho[0]
    print(f"   surface rho: min {r0.min():.3f} p5 {np.percentile(r0, 5):.3f} median {np.median(r0):.3f} "
          f"p95 {np.percentile(r0, 95):.3f} max {r0.max():.3f} kg/m3")
    print("   rho[0]/rho[k], domain mean, by layer:")
    print("   " + " ".join(f"{(r0 / rho[k]).mean():.3f}" for k in range(nz)))
    jD = np.argmin(np.abs(lat - SITES["Delhi"][0])); iD = np.argmin(np.abs(lon - SITES["Delhi"][1]))
    print(f"   {'site':>10} {'Psurf hPa':>10} {'rho0':>7} {'rho0/Delhi':>11}")
    for name, (la, lo) in SITES.items():
        if not (lat.min() <= la <= lat.max() and lon.min() <= lo <= lon.max()):
            continue
        j = np.argmin(np.abs(lat - la)); i = np.argmin(np.abs(lon - lo))
        print(f"   {name:>10} {Ps[j, i] / 100:10.0f} {r0[j, i]:7.3f} {r0[j, i] / r0[jD, iD]:11.3f}")


if __name__ == "__main__":
    main()
