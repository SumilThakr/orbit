"""Do the domain-share and spatial-isolation screens flag the same cells?

The concern this answers: an emission cap is only defensible if the cells it
removes are genuinely implausible, not just the top of a legitimate tail. If
two screens built on unrelated criteria -- "holds too much of the domain" vs
"looks nothing like its neighbourhood" -- independently select the same
cells, that is evidence about the data rather than about either threshold.

Prints, per CEDS file, every cell flagged by either screen, with both scores,
plus the full isolation distribution so the gap between real sources and
artefacts can be seen rather than asserted.

Usage:
    python scripts/compare_emission_screens.py --emission-dir DIR [--top 12]
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import xarray as xr

from orbit.emissions.netcdf import isolation_ratio

SHARE_THRESHOLD = 0.05
ISOLATION_THRESHOLD = 50.0
MIN_SHARE = 0.005


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--emission-dir", required=True)
    ap.add_argument("--pattern", default="ceds_*anthro*.nc")
    ap.add_argument("--top", type=int, default=12,
                    help="rows of the isolation ranking to print per file")
    ap.add_argument("--radius", type=int, default=2)
    a = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(a.emission_dir, a.pattern)))
    if not paths:
        raise SystemExit(f"no files matching {a.pattern} in {a.emission_dir}")

    agree = disagree_share_only = disagree_iso_only = 0

    for path in paths:
        ds = xr.open_dataset(path)
        var = [v for v in ds.data_vars if ds[v].ndim >= 2][0]
        da = ds[var]
        while da.ndim > 2:
            da = da.isel({da.dims[0]: 0})
        data = np.nan_to_num(da.values.astype(np.float64), nan=0.0)
        data[data < 0] = 0.0
        lat = ds.get("lat"); lon = ds.get("lon")
        lat = lat.values if lat is not None else None
        lon = lon.values if lon is not None else None

        total = data.sum()
        if total <= 0:
            ds.close(); continue
        share = data / total
        iso = isolation_ratio(data, a.radius)

        by_share = share > SHARE_THRESHOLD
        by_iso = (iso > ISOLATION_THRESHOLD) & (share > MIN_SHARE)
        flagged = by_share | by_iso

        print(f"\n=== {os.path.basename(path)}  ({var}) ===")
        print(f"    domain total {total:.4e}   nonzero cells "
              f"{int((data > 0).sum())}")

        if flagged.any():
            print(f"    {'cell':>16}{'share%':>10}{'isolation':>12}"
                  f"{'share-scr':>11}{'iso-scr':>9}")
            for j, i in zip(*np.nonzero(flagged)):
                where = (f"{lat[j]:.2f}N {lon[i]:.2f}E"
                         if lat is not None else f"({j},{i})")
                s, iv = by_share[j, i], by_iso[j, i]
                if s and iv:
                    agree += 1
                elif s:
                    disagree_share_only += 1
                else:
                    disagree_iso_only += 1
                isod = "inf" if not np.isfinite(iso[j, i]) else f"{iso[j, i]:.0f}"
                print(f"    {where:>16}{share[j, i] * 100:>10.2f}{isod:>12}"
                      f"{'YES' if s else '-':>11}{'YES' if iv else '-':>9}")
        else:
            print("    nothing flagged by either screen")

        # the distribution, so the gap is visible rather than asserted
        finite = iso[(data > 0) & np.isfinite(iso)]
        if finite.size:
            ranked = np.sort(finite)[::-1][:a.top]
            print(f"    top-{a.top} isolation among non-zero cells: "
                  + " ".join(f"{v:.0f}" for v in ranked))
        ds.close()

    print(f"\n{'=' * 60}")
    print(f"cells flagged by BOTH screens      : {agree}")
    print(f"cells flagged by domain-share only : {disagree_share_only}")
    print(f"cells flagged by isolation only    : {disagree_iso_only}")
    print("\nAgreement between two unrelated criteria is the evidence that\n"
          "these are artefacts rather than the upper tail of a real\n"
          "distribution. Disagreement is where judgement is actually needed.")


if __name__ == "__main__":
    main()
