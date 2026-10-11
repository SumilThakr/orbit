"""Biomass-burning SOA yields: Tsimpidi (2010) Table 2 weighted by GFED5.1 shares.

Reproduces the ``biomass_burning_{high,low}_nox`` rows of
``orbit/emissions/vbs_yields.py``. The seven GFED5.1 species that are summed
into ``gfed5_voc_bb_2022_monthly.nc`` are weighted by their 2022 emitted mass
over the ORBIT domain (cell centres 60.0-100.0E, 4.0-39.0N) and mapped to
Tsimpidi's SAPRC99 lumped species:

    C6H6 (benzene)      -> ARO1       C7H8 (toluene)        -> ARO1
    C8H10 (xylenes)     -> ARO2       Higher_Alkenes        -> mean(OLE1, OLE2)
    Higher_Alkanes      -> mean(ALK4, ALK5)                 (ORBIT assumptions)
    C5H8 (isoprene)     -> ISOP       C10H16 (monoterpenes) -> TERP

Usage:
    python scripts/bb_voc_yields.py [--gfed GFED5.1_monthly_2022.nc]
"""
from __future__ import annotations

import argparse
import os

import numpy as np

# Tsimpidi et al. (2010) Table 2: yields at C* = 1, 10, 100, 1000 µg/m³.
T2_HIGH = {
    "ALK4": (0.000, 0.038, 0.000, 0.000), "ALK5": (0.000, 0.150, 0.000, 0.000),
    "OLE1": (0.001, 0.005, 0.038, 0.150), "OLE2": (0.003, 0.026, 0.083, 0.270),
    "ARO1": (0.003, 0.165, 0.300, 0.435), "ARO2": (0.002, 0.195, 0.300, 0.435),
    "ISOP": (0.001, 0.023, 0.015, 0.000), "TERP": (0.012, 0.122, 0.201, 0.500),
}
T2_LOW = {
    "ALK4": (0.000, 0.075, 0.000, 0.000), "ALK5": (0.000, 0.300, 0.000, 0.000),
    "OLE1": (0.005, 0.009, 0.060, 0.225), "OLE2": (0.023, 0.044, 0.129, 0.375),
    "ARO1": (0.075, 0.225, 0.375, 0.525), "ARO2": (0.075, 0.300, 0.375, 0.525),
    "ISOP": (0.009, 0.030, 0.015, 0.000), "TERP": (0.107, 0.092, 0.359, 0.600),
}
GFED_TO_TSIMPIDI = {
    "C6H6": ["ARO1"], "C7H8": ["ARO1"], "C8H10": ["ARO2"],
    "Higher_Alkenes": ["OLE1", "OLE2"], "Higher_Alkanes": ["ALK4", "ALK5"],
    "C5H8": ["ISOP"], "C10H16": ["TERP"],
}
# GFED5.1 monthly file for 2022 (van der Werf et al., 2025); set ORBIT_GFED5_FILE or
# pass --gfed. The test suite checks the table against the recorded 2022 shares.
DEFAULT_GFED = os.environ.get("ORBIT_GFED5_FILE", "GFED5.1_monthly_2022.nc")
LON_MIN, LON_MAX, LAT_MIN, LAT_MAX = 60.0, 100.0, 4.0, 39.0


def species_shares(path: str) -> dict:
    import xarray as xr
    ds = xr.open_dataset(path)
    ds = ds.sel(lon=slice(LON_MIN, LON_MAX), lat=slice(LAT_MIN, LAT_MAX))
    if ds.sizes.get("lat", 1) == 0:          # descending latitude axis
        ds = xr.open_dataset(path).sel(lon=slice(LON_MIN, LON_MAX), lat=slice(LAT_MAX, LAT_MIN))
    area = np.cos(np.deg2rad(ds.lat))
    tot = {}
    for sp in GFED_TO_TSIMPIDI:
        da = ds[sp].fillna(0.0)
        tot[sp] = float((da * area).sum())    # relative mass, units cancel in the shares
    s = sum(tot.values())
    return {k: v / s for k, v in tot.items()}


def weighted_rows(shares: dict):
    hi = np.zeros(4); lo = np.zeros(4)
    for sp, lumped in GFED_TO_TSIMPIDI.items():
        yh = np.mean([T2_HIGH[l] for l in lumped], axis=0)
        yl = np.mean([T2_LOW[l] for l in lumped], axis=0)
        hi += shares[sp] * yh
        lo += shares[sp] * yl
    # ORBIT's five bins: nothing at C* = 0.1 from Tsimpidi's four-product set
    return np.concatenate([[0.0], hi]), np.concatenate([[0.0], lo])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gfed", default=DEFAULT_GFED)
    a = ap.parse_args()
    shares = species_shares(a.gfed)
    print("GFED5.1 2022 mass shares over the ORBIT domain:")
    for k, v in shares.items():
        print(f"  {k:<16}{v:.3f}  -> {GFED_TO_TSIMPIDI[k]}")
    hi, lo = weighted_rows(shares)
    print("biomass_burning_high_nox:", np.round(hi, 3))
    print("biomass_burning_low_nox: ", np.round(lo, 3))
    from orbit.emissions.vbs_yields import VBS_PARENT_YIELDS
    for key, row in (("biomass_burning_high_nox", hi), ("biomass_burning_low_nox", lo)):
        d = np.abs(VBS_PARENT_YIELDS[key] - row).max()
        print(f"  {key}: max |table - recomputed| = {d:.4f}")


if __name__ == "__main__":
    main()
