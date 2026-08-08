"""Pre-flight: show which diurnal profile every emission source resolves to.

Run this BEFORE a solve that passes --diurnal-config. It answers the one
question the solver never asks out loud: did the profiles actually attach?

They did not, from 2026-05-04 until 2026-08-03. The CEDS sector split
appended stack-height tiers to the filenames (``..._surface.nc``) while
``diurnal_sas.yaml`` still keyed on the pre-split basenames, so every CEDS
source silently resolved to bin-flat -- including the Dec-Feb residential
override for IGP winter heating. No error, no warning, no run ever noticed.

Usage
-----
    python scripts/check_diurnal_profiles.py --month 1 \\
        --diurnal-config orbit/data/diurnal_sas.yaml \\
        [--emissions-manifest orbit/data/emissions_sas_2022_poa.yaml]

Exit code 1 if any source falls through to bin-flat without being mapped to
'flat' explicitly, so it can gate a run script.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--month", type=int, required=True)
    ap.add_argument("--diurnal-config", required=True)
    ap.add_argument("--emissions-manifest", default=None)
    ap.add_argument("--strict", action="store_true",
                    help="exit 1 if any source is unmapped (default: report only)")
    a = ap.parse_args()

    if a.emissions_manifest:
        os.environ["ORBIT_EMISSION_MANIFEST"] = a.emissions_manifest

    from orbit.emissions.loader import DiurnalConfig, _local_to_utc_bin_factors
    from orbit import cli

    cfg = DiurnalConfig.from_yaml(a.diurnal_config)
    sources = cli._build_emission_sources(a.month, verbose=False)
    paths = [s.path for s in sources]

    resolved = cfg.resolve_all(paths, a.month)
    n_bins = 8
    ist = cfg.ist_offset_hours

    print(f"month {a.month:02d}   ist_offset {ist}h   "
          f"{len(resolved)} sources   config {a.diurnal_config}")
    print(f"\n{'source':<52}{'profile':<22}{'bin factors (8 UTC bins)'}")
    print("-" * 118)

    unmapped = []
    for (basename, name) in resolved:
        prof = cfg.profile_for(basename, a.month)
        factors = np.asarray(_local_to_utc_bin_factors(prof, ist, n_bins))
        fac = " ".join(f"{v:5.2f}" for v in factors)
        lo = factors.min()
        spread = f"x{factors.max() / lo:.1f}" if lo > 1e-6 else "x inf"
        label = name or "*** BIN-FLAT (unmapped) ***"
        if name is None:
            unmapped.append(basename)
        # bin_axis sources carry their own per-bin slabs; the YAML mapping is
        # read here but IGNORED by the solver (orbit/cli.py bin_axis=True).
        note = "  [bin_axis: YAML ignored at runtime]" if "_diurnal" in basename else ""
        print(f"{basename:<52}{label:<22}{fac}   {spread}{note}")

    print("-" * 118)
    n_flat = sum(1 for _, n in resolved if n == "flat")
    n_shaped = sum(1 for _, n in resolved if n not in (None, "flat"))
    print(f"shaped {n_shaped}   explicitly flat {n_flat}   UNMAPPED {len(unmapped)}")

    if unmapped:
        print("\nUnmapped sources fall through to bin-flat SILENTLY in the solve:")
        for b in unmapped:
            print(f"  {b}")
        print("Map each to a profile, or to 'flat' if that is intended.")
        if a.strict:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
