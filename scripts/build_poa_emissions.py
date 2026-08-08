"""Split CEDS anthropogenic primary PM2.5 into POA and other primary.

ORBIT's ``ceds_pm25_anthro_*`` files are not a separate CEDS product: they
are built as ``BC + 1.8 x OC`` from the CEDS CMIP sector-resolved files
(see the ``history`` attribute on any of them).  The split back into its
two components is therefore *exact*, with no reconciliation and no new
free parameter:

    POA           = POM_OC_RATIO x OC     (the absorbing organic aerosol)
    other primary = BC                    (chemically inert, non-absorbing)

Separating them is what lets ``M_OA`` include POA in the Pankow
partitioning, gives the IVOC source a basis, and makes modelled organic
aerosol comparable to observations.

Source data: ``HEMCO/CEDS/v2025-04/{year}/{OC,BC}-em-anthro_CMIP_CEDS_{year}.nc``
from ``s3://geos-chem/`` (public HTTPS, no credentials).

Usage
-----
    python scripts/build_poa_emissions.py --ceds-dir DIR --out-dir DIR \
        [--year 2022] [--om-oc 1.8]
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import xarray as xr

# Tier -> CEDS sector names, matching 02_process_ceds.py's sector indices
# (0 agr, 1 ene, 2 ind, 3 tra, 4 rco, 5 slv, 6 wst, 7 shp) and the stack
# heights in the production manifest.
TIER_SECTORS = {
    "surface": ["agr", "tra", "rco", "slv", "shp"],   # sectors 0,3,4,5,7
    "low":     ["wst"],                                # 6  — 30 m
    "medium":  ["ind"],                                # 2  — 50 m
    "high":    ["ene"],                                # 1  — 220 m
}

# SAS subset box, matching the shipped files exactly.
LON_MIN, LON_MAX = 58.0, 100.0
LAT_MIN, LAT_MAX = 4.0, 39.0

DEFAULT_OM_OC = 1.8   # POM_OC_RATIO in 02_process_ceds.py; Malm et al. (2011)


def _tier_sum(ds: xr.Dataset, prefix: str, sectors: list[str]) -> xr.DataArray:
    """Sum the per-sector variables making up one tier."""
    missing = [s for s in sectors if f"{prefix}_{s}" not in ds]
    if missing:
        raise KeyError(f"{prefix}: missing sector variables {missing}; "
                       f"have {[v for v in ds.data_vars]}")
    out = sum(ds[f"{prefix}_{s}"] for s in sectors)
    return out


def _subset(da: xr.DataArray) -> xr.DataArray:
    return da.sel(lon=slice(LON_MIN, LON_MAX), lat=slice(LAT_MIN, LAT_MAX))


def build(ceds_dir: str, out_dir: str, year: int, om_oc: float) -> None:
    oc_path = os.path.join(ceds_dir, f"OC-em-anthro_CMIP_CEDS_{year}.nc")
    bc_path = os.path.join(ceds_dir, f"BC-em-anthro_CMIP_CEDS_{year}.nc")
    oc_ds, bc_ds = xr.open_dataset(oc_path), xr.open_dataset(bc_path)
    os.makedirs(out_dir, exist_ok=True)

    print(f"OM/OC = {om_oc}   POA = {om_oc} x OC,   other primary = BC")
    print(f"{'tier':<10}{'POA kg/s':>14}{'other kg/s':>14}{'POA frac':>10}")

    tot_poa = tot_oth = 0.0
    for tier, sectors in TIER_SECTORS.items():
        oc = _subset(_tier_sum(oc_ds, "OC", sectors))
        bc = _subset(_tier_sum(bc_ds, "BC", sectors))
        poa = om_oc * oc
        # The other-primary variable is named pm25_anthro so ORBIT's species
        # auto-detection maps it to the PrimaryPM25 slot: it *is* primary
        # PM2.5, just with the absorbing organic fraction carved out.
        for name, da, kind in (("poa", poa, "POA (POM)"),
                               ("bcoth", bc, "other primary (BC)")):
            da = da.rename("poa_anthro" if name == "poa" else "pm25_anthro")
            da.attrs = {
                "long_name": f"Primary anthropogenic {kind}, {tier} tier",
                "units": "kg m-2 s-1",
                "formula": (f"{om_oc} * OC_em_anthro" if name == "poa"
                            else "BC_em_anthro")
                           + f" (CEDS sectors {sectors})",
                "POM_to_OC_ratio": float(om_oc),
                "tier": tier,
                "ceds_sectors": str(sectors),
                "note": ("Exact split of the ceds_pm25_anthro tiers, which "
                         "are themselves BC + 1.8*OC; POA + other == PM2.5"),
            }
            ds = da.to_dataset()
            ds.attrs = {
                "title": (f"CEDS anthropogenic {kind} for {year} — tier={tier}"),
                "source": "CEDS v_2025_04_18 (source_id: CEDS-CMIP-2025-04-18)",
                "references": ("Hoesly et al. (2018), doi:10.5194/gmd-11-369-2018; "
                               "Malm et al. (2011) for the POM/OC ratio"),
                "history": (f"Processed by scripts/build_poa_emissions.py: "
                            f"tier-restricted sector sum, {year} selection, "
                            + (f"{om_oc}*OC" if name == "poa" else "BC")),
                "subset_region": "sas",
            }
            path = os.path.join(
                out_dir, f"ceds_{name}_anthro_{year}_monthly_{tier}.nc")
            ds.to_netcdf(path)

        p, o = float(poa.sum()), float(bc.sum())
        tot_poa += p
        tot_oth += o
        print(f"{tier:<10}{p:>14.4e}{o:>14.4e}{p/(p+o):>10.3f}")

    print(f"{'ALL':<10}{tot_poa:>14.4e}{tot_oth:>14.4e}"
          f"{tot_poa/(tot_poa+tot_oth):>10.3f}")
    oc_ds.close()
    bc_ds.close()


def verify(orig_dir: str, out_dir: str, year: int) -> bool:
    """POA + other must reproduce the shipped PM2.5 tiers exactly."""
    print("\nverification: POA + other vs shipped ceds_pm25_anthro tiers")
    ok = True
    for tier in TIER_SECTORS:
        ref = xr.open_dataset(
            os.path.join(orig_dir, f"ceds_pm25_anthro_{year}_monthly_{tier}.nc"))
        poa = xr.open_dataset(
            os.path.join(out_dir, f"ceds_poa_anthro_{year}_monthly_{tier}.nc"))
        oth = xr.open_dataset(
            os.path.join(out_dir, f"ceds_bcoth_anthro_{year}_monthly_{tier}.nc"))
        a = ref["pm25_anthro"].values
        b = poa["poa_anthro"].values + oth["pm25_anthro"].values
        rel = np.abs(a - b).max() / max(np.abs(a).max(), 1e-30)
        flag = "OK" if rel < 1e-10 else "MISMATCH"
        if rel >= 1e-10:
            ok = False
        print(f"  {tier:<10} max rel diff {rel:.3e}   {flag}")
        for d in (ref, poa, oth):
            d.close()
    print("VERDICT:", "exact split" if ok else "DOES NOT reproduce")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ceds-dir", required=True,
                    help="directory holding {OC,BC}-em-anthro_CMIP_CEDS_YYYY.nc")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--year", type=int, default=2022)
    ap.add_argument("--om-oc", type=float, default=DEFAULT_OM_OC)
    ap.add_argument("--verify-against", default=None,
                    help="directory of shipped ceds_pm25_anthro tiers")
    a = ap.parse_args()
    build(a.ceds_dir, a.out_dir, a.year, a.om_oc)
    if a.verify_against:
        if not verify(a.verify_against, a.out_dir, a.year):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
