"""Is daytime vertical mixing too weak?

Compares the grid's Kzz against boundary-layer similarity expectations:
  surface layer (neutral/near-neutral):  K = kappa u* z
  convective mixed layer:                K ~ 0.1 w* h  (peak, mid-BL)
w* is reconstructed from the BL depth and a plausible surface heat flux
range; the point is the order of magnitude, not a precise value.
"""
import os
import sys

import numpy as np

SITES = [(28.5, 77.5, "Delhi"), (22.5, 88.12, "Kolkata"), (21.5, 83.12, "Korba")]


def main():
    from orbit.cli import _preproc_path
    from orbit.core.grid_data import load_grid

    first_path = _preproc_path(1, 1)
    if first_path.startswith("/path/to/data") or not os.path.exists(first_path):
        sys.exit(
            "input grids not found:\n"
            f"  {first_path}\n"
            "Set ORBIT_PREPROC_DIR (and ORBIT_CONSTANTS) to the unpacked data "
            "archive; see the README's data section."
        )

    g1 = load_grid(first_path, os.environ.get("ORBIT_CONSTANTS"))
    lat, lon = np.asarray(g1.lat), np.asarray(g1.lon)

    print("January, layer-0 Kzz and PBL depth by diurnal bin\n")
    print(f"{'bin':>4}{'Delhi Kzz':>11}{'Pblh':>8}{'Kolkata':>10}{'Pblh':>8}{'Korba':>9}{'Pblh':>8}")
    grids = {}
    for b in range(1, 9):
        g = load_grid(_preproc_path(1, b), os.environ.get("ORBIT_CONSTANTS"))
        grids[b] = g
        kz, pb = np.asarray(g.Kzz), np.asarray(g.Pblh)
        row = f"{b:>4}"
        for la, lo, _ in SITES:
            j, i = int(np.argmin(abs(lat - la))), int(np.argmin(abs(lon - lo)))
            p = pb[j, i] if pb.ndim == 2 else pb[0, j, i]
            row += f"{kz[0, j, i]:>11.2f}{p:>8.0f}" if _ == "Delhi" else f"{kz[0, j, i]:>10.2f}{p:>8.0f}"
        print(row)

    # vertical profile in the most convective bin vs the most stable
    print("\nKzz vertical profile at Delhi (m2/s)")
    edges = np.concatenate([[0.0], np.cumsum(np.asarray(g1.Dz)[:, 40, 28])])
    j, i = int(np.argmin(abs(lat - 28.5))), int(np.argmin(abs(lon - 77.5)))
    print(f"{'z_top':>8}" + "".join(f"{'b'+str(b):>8}" for b in range(1, 9)))
    for k in range(8):
        print(f"{edges[k+1]:>8.0f}" + "".join(f"{np.asarray(grids[b].Kzz)[k,j,i]:>8.2f}"
                                              for b in range(1, 9)))

    # similarity expectations
    print("\nExpectation checks at Delhi, most convective bin (bin 3)")
    g3 = grids[3]
    pb = np.asarray(g3.Pblh); h = float(pb[j, i] if pb.ndim == 2 else pb[0, j, i])
    u = float(np.asarray(g3.WindSpeed)[0, j, i])
    kz0 = float(np.asarray(g3.Kzz)[0, j, i])
    z = edges[1]
    ustar = 0.04 * u + 0.05          # crude: u*/U ~ 0.04-0.10 over land
    print(f"  PBL depth h = {h:.0f} m,  U = {u:.2f} m/s,  u* ~ {ustar:.2f} m/s")
    print(f"  surface-layer K = kappa u* z = {0.4*ustar*z:.1f} m2/s   (model {kz0:.2f})")
    for wstar in (0.5, 1.0, 1.5):
        print(f"  convective K ~ 0.1 w* h with w*={wstar} m/s, h={h:.0f}:"
              f" {0.1*wstar*h:>7.1f} m2/s")
    kmax = max(np.asarray(g3.Kzz)[k, j, i] for k in range(10))
    print(f"  model column max Kzz (bin 3) = {kmax:.2f} m2/s")

    # mixing timescale surface -> 1 km
    print("\nTime to mix 0-1 km, tau ~ h^2 / K")
    for b in (1, 3):
        kcol = np.asarray(grids[b].Kzz)[:8, j, i].mean()
        print(f"  bin {b}: mean K(0-1km) = {kcol:6.2f} m2/s -> tau = {1e6/kcol/3600:8.1f} h")


if __name__ == "__main__":
    main()
