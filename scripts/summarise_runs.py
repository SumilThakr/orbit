"""Summarise a set of ORBIT runs: composition, OA, and satellite skill.

One table per question, so an overnight sweep can be read in one screen
rather than by opening a dozen NPZs.

Usage:
    python scripts/summarise_runs.py TAG[:LABEL] TAG[:LABEL] ... [--month 1]
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

OUT = os.environ.get("ORBIT_OUT_ROOT", "orbit_out")
NZ, NY, NX = 15, 71, 65
PAIRS = os.path.expanduser(
    "~/orbit_data/deposit_B/satellite_comparison/satellite_pm25_pairs.npz")


def load(tag, month):
    p = os.path.join(OUT, tag, f"orbit_M{month:02d}.npz")
    return np.load(p) if os.path.exists(p) else None


def surf(z, key):
    a = z[key]
    return a[0] if a.ndim == 3 else a


def skill(tag, month):
    """Pearson r2 / NMB / NME via the canonical eval script.

    Delegated rather than reimplemented: regenerate_satellite_eval.py owns
    the pairing geometry, the coverage floors and the observation-stats
    conventions. Duplicating that here would be a second source of truth
    that could silently drift.
    """
    import re
    import subprocess
    import tempfile

    script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "regenerate_satellite_eval.py")
    if not os.path.exists(script):
        return None
    with tempfile.TemporaryDirectory() as td:
        try:
            out = subprocess.run(
                [sys.executable, script, os.path.join(OUT, tag), td,
                 "--field", "pm25_mean"],
                capture_output=True, text=True, timeout=600).stdout
        except Exception:
            return None
    block = out.split("MONTHLY POOLED")[-1].split("PER-MONTH")[0]
    got = {}
    for key, name in (("n", "n"), ("r2", "r2"), ("nmb_pct", "nmb"),
                      ("nme_pct", "nme")):
        m = re.search(rf"^\s*{key}\s+(\S+)", block, re.M)
        if m:
            got[name] = float(m.group(1))
    return got if len(got) == 4 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="TAG or TAG:LABEL")
    ap.add_argument("--month", type=int, default=1)
    a = ap.parse_args()

    from orbit.core.orbit import _SPECIES_NAMES
    IDX = {n: i for i, n in enumerate(_SPECIES_NAMES)}

    entries = []
    for spec in a.runs:
        tag, _, label = spec.partition(":")
        z = load(tag, a.month)
        if z is None:
            print(f"  (missing: {tag} M{a.month:02d})", file=sys.stderr)
            continue
        entries.append((label or tag, z, tag))

    if not entries:
        raise SystemExit("no runs found")

    def sp(z, name):
        return z["c_mean"][IDX[name]].reshape(NZ, NY, NX)[0]

    print(f"\nmonth {a.month:02d}   domain surface means (ug/m3)")
    print(f"{'run':<20}{'PM2.5':>9}{'POA':>8}{'SOA':>8}{'OA':>8}"
          f"{'M_OA':>9}{'othPri':>9}")
    for label, z, _tag in entries:
        poa, soa = sp(z, "POA"), z["soa_mean"]
        print(f"{label:<20}{surf(z,'pm25_mean').mean():>9.3f}"
              f"{poa.mean():>8.3f}{soa.mean():>8.3f}{(poa+soa).mean():>8.3f}"
              f"{z['M_OA_mean'].mean():>9.3f}{sp(z,'PM2.5').mean():>9.3f}")

    # polluted decile, defined once from the first run so all share a mask
    ref = surf(entries[0][1], "pm25_mean")
    mask = ref >= np.percentile(ref, 90)
    print(f"\ntop-decile polluted cells (n={int(mask.sum())}, "
          f"mask from '{entries[0][0]}')")
    print(f"{'run':<20}{'PM2.5':>9}{'POA':>8}{'SOA':>8}{'OA':>8}")
    for label, z, _tag in entries:
        poa, soa = sp(z, "POA"), z["soa_mean"]
        print(f"{label:<20}{surf(z,'pm25_mean')[mask].mean():>9.2f}"
              f"{poa[mask].mean():>8.2f}{soa[mask].mean():>8.2f}"
              f"{(poa+soa)[mask].mean():>8.2f}")

    print(f"\nsatellite skill, month {a.month:02d}")
    print(f"{'run':<20}{'n':>7}{'R2':>9}{'NMB%':>9}{'NME%':>9}")
    for label, z, _tag in entries:
        s = skill(_tag, a.month)
        if s is None:
            print(f"{label:<20}{'(pairs unavailable)':>34}")
        else:
            print(f"{label:<20}{s['n']:>7}{s['r2']:>9.3f}"
                  f"{s['nmb']:>9.1f}{s['nme']:>9.1f}")

    # pairwise deltas against the first run
    base_label, base, _ = entries[0]
    if len(entries) > 1:
        print(f"\ndelta vs '{base_label}' (domain surface PM2.5)")
        b = surf(base, "pm25_mean")
        for label, z, _tag in entries[1:]:
            d = surf(z, "pm25_mean") - b
            print(f"{label:<20} mean {d.mean():+8.4f}  "
                  f"({100*d.mean()/b.mean():+6.2f}%)   "
                  f"polluted {d[mask].mean():+8.3f}  "
                  f"max|d| {np.abs(d).max():.3f}")


if __name__ == "__main__":
    main()
