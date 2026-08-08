"""One-month end-to-end check of the marginal-deaths deliverable.

The production postprocessor (postprocess_marginal_deaths_scenarios.py)
requires all 12 adjoint months before it will run. This script drives the
same two library calls on a single month so the GADM path, the unit
conversion and the NetCDF writer can be proven before a full year of
adjoint solves is committed.

Usage:
    python scripts/derisk_subdistrict_january.py ADJOINT_NC GADM_GPKG OUT_NC
"""

import sys

import numpy as np
import xarray as xr

from orbit.modes.subdistrict_aggregation import (
    aggregate_dJ_de_to_subdistricts,
    build_subdistrict_layer,
    write_deaths_per_1000kg_nc,
)

BBOX = (58.0, 4.0, 100.0, 39.0)   # matches the postprocessor default


def main() -> int:
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(2)
    adj_path, gadm, out_path = sys.argv[1], sys.argv[2], sys.argv[3]

    ds = xr.open_dataset(adj_path)
    month = int(ds.attrs["month"])
    crf = str(ds.attrs["crf_mode"])
    species_keys = [str(s) for s in ds["species"].values]
    print(f"adjoint: month={month} crf={crf} species={species_keys}")

    layer = build_subdistrict_layer(
        gadm_gpkg=gadm,
        orbit_lat=ds.lat.values,
        orbit_lon=ds.lon.values,
        bbox=BBOX,
        sub_factor=5,
        verbose=True,
    )
    print(f"layer: {layer.gid_list.size} subdistricts, "
          f"{layer.cell_idx.size} cell-gid pairs")

    # (draw, species, bin, y, x) -> single deterministic draw
    dJ_de = np.asarray(ds["dJ_de"].values, dtype=np.float64)[0]
    surface_volume = np.asarray(ds["surface_volume_m3"].values, dtype=np.float64)

    deaths = aggregate_dJ_de_to_subdistricts(
        dJ_de, layer, surface_volume, species_keys=species_keys,
    )
    print(f"deaths_per_1000kg: shape={deaths.shape} "
          f"finite={np.isfinite(deaths).all()}")
    for i, k in enumerate(species_keys):
        v = deaths[i]
        print(f"  {k:<14} min={v.min():.4g} median={np.median(v):.4g} "
              f"max={v.max():.4g}")
        top = np.argsort(v)[::-1][:5]
        names = [f"{layer.gid_name[j]} ({layer.gid_iso3[j]})" for j in top]
        print("    top-5 subdistricts: "
              + ", ".join(f"{n} {v[j]:.3g}" for n, j in zip(names, top)))

    write_deaths_per_1000kg_nc(
        out_path,
        deaths_per_1000kg=deaths,
        species_keys=species_keys,
        layer=layer,
        month=month,
        crf_mode=crf,
        extra_attrs={"provenance": f"derisk single-month run from {adj_path}"},
    )
    print(f"wrote {out_path}")
    ds.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
