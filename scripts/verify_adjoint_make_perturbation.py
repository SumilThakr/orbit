"""Build a one-cell additive-emission NetCDF for the adjoint verification.

Writes a small NetCDF with shape (bin=8, lat=ny, lon=nx) carrying a
single-cell pulse of `magnitude` kg/m^2/s at (lat=lat_target, lon=lon_target),
zero everywhere else, identical across all 8 diurnal bins. Used as a
`--add-emissions` input to scripts/run_orbit.py --mode marginal.

The perturbation NetCDF uses the SAS preproc grid (71 lat × 65 lon at
0.5°×0.625°), matching the ORBIT grid the forward marginal solver
operates on.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import xarray as xr


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--orbit-npz", type=Path, required=True,
                    help="Production orbit NPZ to copy the lat/lon grid from.")
    ap.add_argument("--species", default="NOx",
                    help="Pollutant species name (sets the data-variable name).")
    ap.add_argument("--lat-idx", type=int, required=True,
                    help="0-based latitude index of the perturbation cell.")
    ap.add_argument("--lon-idx", type=int, required=True,
                    help="0-based longitude index of the perturbation cell.")
    ap.add_argument("--magnitude", type=float, default=1e-9,
                    help="Per-cell emission flux in kg/m^2/s. Default 1e-9 "
                         "gives ~1 kg/s effective rate after multiplying by "
                         "a 0.5° cell area (~3.4e9 m^2 in SAS).")
    ap.add_argument("--out", type=Path, required=True,
                    help="Output NetCDF path.")
    args = ap.parse_args(argv)

    grid = np.load(args.orbit_npz, allow_pickle=False)
    lat = grid["lat"]
    lon = grid["lon"]
    ny, nx = lat.size, lon.size
    n_bins = 8

    field = np.zeros((n_bins, ny, nx), dtype=np.float64)
    if not (0 <= args.lat_idx < ny):
        raise SystemExit(f"lat-idx {args.lat_idx} out of range [0, {ny})")
    if not (0 <= args.lon_idx < nx):
        raise SystemExit(f"lon-idx {args.lon_idx} out of range [0, {nx})")
    field[:, args.lat_idx, args.lon_idx] = args.magnitude

    ds = xr.Dataset(
        data_vars={args.species: (("bin", "lat", "lon"), field)},
        coords=dict(
            bin=np.arange(n_bins, dtype=np.int32),
            lat=lat.astype(np.float64),
            lon=lon.astype(np.float64),
        ),
        attrs=dict(
            description=(
                f"Adjoint-verification one-cell pulse: {args.species} at "
                f"lat={float(lat[args.lat_idx]):.3f}, "
                f"lon={float(lon[args.lon_idx]):.3f}, "
                f"magnitude={args.magnitude} kg/m^2/s, all 8 bins identical."
            ),
            units="kg/m2/s",
        ),
    )
    ds[args.species].attrs["units"] = "kg/m2/s"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(args.out)
    print(f"Wrote {args.out}: ({n_bins}, {ny}, {nx})  "
          f"cell ({args.lat_idx},{args.lon_idx}) = "
          f"lat={float(lat[args.lat_idx]):.3f} lon={float(lon[args.lon_idx]):.3f}, "
          f"magnitude={args.magnitude} kg/m2/s.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
