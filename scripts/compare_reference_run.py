#!/usr/bin/env python
"""Compare a solve against a reference and decide whether it reproduces it.

A naive verdict (the global max of cellwise relative differences) would
fail every faithful reproduction: the ISORROPIA finite-difference
cross-partials (f_*_d*_3d) contain isolated cells straddling a phase
boundary, where a last-bit input change flips the one-sided difference and
produces an O(1) cellwise relative difference at a vanishing absolute
magnitude. The measured harmless signature is at most 0.03% of cells above
1e-6 of array scale with a scaled RMS below 5e-6, while every state array
agrees to 1e-13 of scale.

This script therefore normalizes by array scale (the 99.9th percentile of
|reference|, so a few huge cells cannot deflate everything) and applies:

- state arrays (concentrations, PM2.5, deposition, budgets):
    REPRODUCED iff max scaled diff <= 1e-6.
- FD-derivative arrays (f_*_d*_3d, *_marg_asym_3d):
    REPRODUCED iff the fraction of cells with scaled diff > 1e-6 is
    <= 2e-3 AND scaled RMS <= 3e-5. Isolated boundary flips pass; any
    systematic change (different LUT, physics, emissions) fails the RMS
    bound by orders of magnitude.

Arrays present only in the reference fail the check (a missing output is
a regression). Arrays only in the test are reported but do not fail
(newer code adds outputs, e.g. the deposition maps).

Usage:
    python compare_reference_run.py TEST.npz REFERENCE.npz
"""

import fnmatch
import sys

import numpy as np

# Cellwise "this cell materially differs" threshold, in units of array scale.
CELL_TOL = 1e-6
# State arrays: no cell may materially differ.
# FD-derivative arrays: bounds on the flip population and the field-wide RMS.
FD_FRAC_TOL = 2e-3
FD_RMS_TOL = 3e-5

# ISORROPIA finite-difference cross-partials and their asymmetry
# diagnostics: the arrays with known benign near-discontinuity flips.
FD_PATTERNS = ("f_*_d*_3d", "*_marg_asym_3d")

HEADLINE_KEYS = ("pm25_surf", "pm25_surface", "pm25_mean", "total_pm25_surf")


def _is_fd_array(name):
    return any(fnmatch.fnmatch(name, p) for p in FD_PATTERNS)


def _scaled_stats(a, b):
    """(max, RMS, exceedance fraction) of |a-b| in units of array scale."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    d = np.abs(a - b)
    finite = np.isfinite(d)
    if not finite.any():
        return np.nan, np.nan, np.nan, 0.0
    d = d[finite]
    scale = max(float(np.percentile(np.abs(b[finite]), 99.9)), 1e-300)
    s = d / scale
    return float(s.max()), float(np.sqrt((s ** 2).mean())), \
        float((s > CELL_TOL).mean()), scale


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    test_path, ref_path = sys.argv[1], sys.argv[2]
    t = np.load(test_path, allow_pickle=True)
    r = np.load(ref_path, allow_pickle=True)

    print(f"test:      {test_path}")
    print(f"reference: {ref_path}")
    print()

    # ── headline: surface PM2.5 ───────────────────────────────────────────
    for k in HEADLINE_KEYS:
        if k in t.files and k in r.files:
            mx, rms, frac, scale = _scaled_stats(t[k], r[k])
            d = np.abs(np.asarray(t[k], float) - np.asarray(r[k], float))
            print(f"Surface PM2.5  ({k}, shape {np.shape(t[k])})")
            print(f"  test mean        {np.nanmean(t[k]):.6f} ug/m3")
            print(f"  reference mean   {np.nanmean(r[k]):.6f} ug/m3")
            print(f"  max |diff|       {np.nanmax(d):.3e} ug/m3")
            print(f"  max scaled diff  {mx:.3e}  (scale {scale:.3e})")
            print()
            break

    shared = sorted(set(t.files) & set(r.files))
    only_t = sorted(set(t.files) - set(r.files))
    only_r = sorted(set(r.files) - set(t.files))

    failures = []
    fd_rows = []
    skipped = []
    for k in shared:
        try:
            a, b = t[k], r[k]
        except Exception:
            skipped.append(k)
            continue
        if a.dtype.kind not in "fiu" or b.dtype.kind not in "fiu":
            skipped.append(k)
            continue
        if a.shape != b.shape:
            failures.append((k, f"SHAPE MISMATCH {a.shape} vs {b.shape}"))
            continue
        mx, rms, frac, scale = _scaled_stats(a, b)
        if not np.isfinite(mx):
            continue
        if _is_fd_array(k):
            if frac > FD_FRAC_TOL or rms > FD_RMS_TOL:
                failures.append(
                    (k, f"FD-derivative: frac>{CELL_TOL:g} = {frac:.2e} "
                        f"(tol {FD_FRAC_TOL:g}), scaled RMS = {rms:.2e} "
                        f"(tol {FD_RMS_TOL:g})"))
            elif frac > 0 or mx > CELL_TOL:
                fd_rows.append((k, mx, rms, frac))
        elif mx > CELL_TOL:
            failures.append(
                (k, f"max scaled diff {mx:.3e} > {CELL_TOL:g} "
                    f"(scale {scale:.3e})"))
    for k in only_r:
        failures.append((k, "missing from test output"))

    print(f"Compared {len(shared)} shared arrays "
          f"({len(skipped)} non-numeric skipped)")
    if only_t:
        print(f"  only in test (informational): {only_t}")
    print()

    if fd_rows:
        print("FD-derivative arrays with benign near-discontinuity flips "
              "(within tolerance):")
        for k, mx, rms, frac in fd_rows:
            print(f"  {k:<40s} max scaled {mx:.2e}  RMS {rms:.2e}  "
                  f"frac {frac:.2e}")
        print()

    if failures:
        print("FAILURES:")
        for k, why in failures:
            print(f"  {k:<40s} {why}")
        print()
        print("RESULT: DIFFERS")
        sys.exit(1)
    print("RESULT: REPRODUCED")
    sys.exit(0)


if __name__ == "__main__":
    main()
