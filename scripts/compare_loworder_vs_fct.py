#!/usr/bin/env python
"""Gate check: how far do marginal damages move when the operator changes?

Before committing to a regeneration, run one month low-order against the
existing FCT forward field and measure how far the damages actually move,
so the switch can be classified as presentational or numerical.

Compares two adjoint_M<MM>.nc written by scripts/compute_marginal_deaths.py
-- one with --horizontal-fct and one without -- and reports, per species:

  * the sign audit. The low-order transport block is an M-matrix, so an
    inert primary species (PM25_primary, POA) MUST have zero negative
    cells. NH3 and SO2 legitimately keep a few: ISORROPIA substitution
    chemistry (NH4NO3 <-> (NH4)2SO4) genuinely lets extra emission reduce
    PM in a handful of cells. That distinction is the whole point of the
    audit, so it is reported per species rather than pooled.
  * the domain total of dJ/de, which is what aggregate damage tables move
    with.
  * the distribution of the per-cell ratio low/FCT, weighted by |dJ/de|
    under FCT so that cells carrying real damage dominate. The six-cell FD
    panel prices the expected offset at 3-29 % low; a weighted median far
    outside that band is worth stopping for.

Usage:
    python scripts/compare_loworder_vs_fct.py \
        --fct  orbit_out/marginal_deaths/production/gemm5cod/adjoint_M01.nc \
        --low  orbit_out/marginal_deaths/production_loworder/gemm5cod/adjoint_M01.nc
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import netCDF4 as nc

INERT = {"PM25_primary", "POA"}


def load(path: Path):
    ds = nc.Dataset(path)
    v = ds.variables["dJ_de"]
    a = np.asarray(v[:], dtype=np.float64)
    ax = v.dimensions.index("species")
    names = [s if isinstance(s, str) else bytes(s).decode().strip()
             for s in ds.variables["species"][:]]
    fct = int(ds.getncattr("horizontal_fct")) if "horizontal_fct" in ds.ncattrs() else -1
    ds.close()
    # sum over draw and bin -> (species, y, x); bins are additive in emission
    a = np.moveaxis(a, ax, 0)
    return names, a.reshape(a.shape[0], -1, *a.shape[-2:]).sum(axis=1), fct


def wq(v, w, qs):
    o = np.argsort(v)
    v, w = v[o], w[o]
    c = np.cumsum(w)
    return [float(v[np.searchsorted(c, q * c[-1])]) for q in qs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fct", type=Path, required=True)
    ap.add_argument("--low", type=Path, required=True)
    ap.add_argument("--floor-quantile", type=float, default=0.90,
                    help="ratios are summarised over cells above this "
                         "quantile of |dJ/de| under FCT (default 0.90), so "
                         "near-zero cells cannot dominate the spread")
    a = ap.parse_args()

    nf, F, fct_flag = load(a.fct)
    nl, L, low_flag = load(a.low)
    if nf != nl:
        raise SystemExit(f"species mismatch:\n  {nf}\n  {nl}")
    print(f"FCT file  horizontal_fct={fct_flag}   {a.fct}")
    print(f"low file  horizontal_fct={low_flag}   {a.low}")
    if fct_flag != 1 or low_flag != 0:
        print("  !! attribute check FAILED -- these are not one FCT and one "
              "low-order field")

    print(f"\n{'species':<14}{'neg FCT':>9}{'neg low':>9}"
          f"{'totalFCT':>12}{'totallow':>12}{'tot ratio':>10}"
          f"{'  wtd ratio p10/p50/p90 (top decile cells)':>10}")
    verdict = []
    for i, s in enumerate(nf):
        f, l = F[i].ravel(), L[i].ravel()
        m = np.isfinite(f) & np.isfinite(l)
        f, l = f[m], l[m]
        negf, negl = int((f < 0).sum()), int((l < 0).sum())
        tf, tl = f.sum(), l.sum()
        thr = np.quantile(np.abs(f), a.floor_quantile)
        sel = (np.abs(f) >= thr) & (f > 0)
        if sel.sum() > 10:
            r = l[sel] / f[sel]
            p10, p50, p90 = wq(r, np.abs(f[sel]), [0.1, 0.5, 0.9])
            rs = f"  {p10:6.3f} /{p50:6.3f} /{p90:6.3f}"
        else:
            p50, rs = float("nan"), "        (too few cells)"
        print(f"{s:<14}{negf:>9,}{negl:>9,}{tf:>12.4g}{tl:>12.4g}"
              f"{tl/tf if tf else float('nan'):>10.4f}{rs}")
        verdict.append((s, negl, p50))

    print("\nacceptance checks (stated in the headline-decision note, not "
          "after the fact)")
    ok = True
    for s, negl, _ in verdict:
        if s in INERT:
            good = negl == 0
            ok &= good
            print(f"  [{'PASS' if good else 'FAIL'}] {s}: low-order negatives "
                  f"= {negl} (must be 0; the operator is an M-matrix)")
    meds = [p for s, _, p in verdict if s in INERT and np.isfinite(p)]
    if meds:
        m = float(np.mean(meds))
        good = 0.60 <= m <= 1.05
        ok &= good
        print(f"  [{'PASS' if good else 'FAIL'}] inert-species weighted median "
              f"ratio = {m:.3f}; the note prices low-order at 3-29 % low, so "
              f"0.60-1.05 is the expected band")
    print(f"\n{'GATE PASSES' if ok else 'GATE FAILS -- stop and investigate'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
