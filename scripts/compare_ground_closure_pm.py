#!/usr/bin/env python3
"""Paired primary-PM2.5 periodic solves with and without the ground leak.

Solves the periodic orbit for primary PM2.5 (transport plus deposition, no
chemistry, bin-flat emissions from the active manifest) for one month under
up to four transport operators, and compares the bin-mean surface field:

  closed      the production operator: interface omega_edge where the grids
              carry it, ground closed, state a concentration at local density
  leak        the operator before the fix: the layer-0 mid-level omega_plus is
              reinstated as a downward loss through the ground, exactly as
              assemble_vertical_advection applied it
  mixing      closed without the density transform: the transported variable
              proportional to a mixing ratio, the reading before 2026-09-26
  cellcentred closed with the cell-centred omega read at the bottom faces and
              the top closed, the vertical flux before 2026-09-26 (only on
              grids that carry omega_edge)
  diagnosed   closed, with the bin-mean omega replaced by the value diagnosed
              from the horizontal face-flux divergence (rows of the advective
              blocks then sum to zero); unsplit, so compare it with 'unsplit',
              which is 'closed' with the same unsplit treatment of the
              released omega; not in production

The paired ratios are the result; the absolute fields are not the production
configuration (no diurnal profiles, no horizontal FCT, no POA split).

Usage:
    ORBIT_PREPROC_DIR=$ORBIT_DATA/inputs/grids_2022 \
    ORBIT_EMISSION_DIR=$ORBIT_DATA/emissions/sas_finn \
    ORBIT_CONSTANTS=$ORBIT_DATA/inputs/MERRA2.20150101.CN.05x0625.nc4 \
    python scripts/compare_ground_closure_pm.py --month 1 \
        --variants closed leak density [--out fields.npz] \
        [--emissions-manifest orbit/data/emissions_sas_2022.yaml]

Cost: 8 sparse LU factorisations per variant, about 90 s each on a
workstation, plus a few seconds of GMRES.
"""
import argparse
import glob
import os
import sys
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

GRAVITY = 9.80665
SITES = {
    "Delhi": (28.6, 77.2), "Kolkata": (22.6, 88.4), "Lucknow": (26.8, 80.9),
    "Patna": (25.6, 85.1), "Dehradun": (30.3, 78.0), "Shimla": (31.1, 77.2),
    "Kathmandu": (27.7, 85.3), "Lhasa": (29.65, 91.1), "Leh": (34.2, 77.6),
    "Srinagar": (34.1, 74.8), "Bengaluru": (13.0, 77.6), "Mumbai": (19.1, 72.9),
}


class _LU:
    def __init__(self, A):
        self.lu = spla.splu(A.tocsc())
        self.shape = A.shape

    def solve(self, b):
        return self.lu.solve(b)


def _stats(r):
    return (f"median {np.median(r):.3f}  p10 {np.percentile(r, 10):.3f}  "
            f"p90 {np.percentile(r, 90):.3f}  mean {r.mean():.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--month", type=int, default=1)
    ap.add_argument("--variants", nargs="+", default=["closed", "leak", "mixing", "cellcentred"],
                    choices=["closed", "leak", "mixing", "cellcentred", "unsplit", "diagnosed"])
    ap.add_argument("--out", default=None, help="save the bin-mean 3-D fields per variant to this NPZ")
    ap.add_argument("--emissions-manifest", default=None,
                    help="emission manifest YAML; default is the shipped POA-split manifest, whose "
                         "files must be under ORBIT_EMISSION_DIR")
    args = ap.parse_args()

    import orbit.cli as cli
    cli._MANIFEST_PATH_OVERRIDE = args.emissions_manifest
    from orbit.cli import _build_emission_sources, _preproc_path, CONSTANTS
    from orbit.core.grid_data import load_grid
    from orbit.core.indexing import CellIndexer
    from orbit.core.operator import assemble_transport_block
    from orbit.core.convdiff import assemble_horizontal_convdiff
    from orbit.core.deposition import assemble_deposition, IDX_PM25
    from orbit.core.orbit import solve_orbit_one_species, DTAU, N_BINS
    from orbit.emissions.loader import load_emissions_diurnal
    from orbit.emissions.sources import N_ORBIT_SPECIES

    paths = [_preproc_path(args.month, b) for b in range(1, N_BINS + 1)]
    grids = [load_grid(p, CONSTANTS) for p in paths]
    g0 = grids[0]
    nz, ny, nx = g0.nz, g0.ny, g0.nx
    N = nz * ny * nx
    idx = CellIndexer(nz, ny, nx)

    sources = _build_emission_sources(args.month)
    e = load_emissions_diurnal(sources, g0, idx, diurnal_cfg=None, month=args.month, n_bins=N_BINS)
    slot = 1  # PrimaryPM2.5 in the loader's 6-slot layout
    e_list = [e[t, slot * N:(slot + 1) * N] for t in range(N_BINS)]
    print(f"primary PM2.5 emissions: {(e_list[0] > 0).sum()} emitting cells")

    I_N = sp.identity(N, format="csc")
    lus = {v: [] for v in args.variants}
    t0 = time.time()
    for g in grids:
        D = assemble_deposition(g, idx, IDX_PM25)
        T_closed = assemble_transport_block(g, idx)
        rho = (g.dP / (GRAVITY * np.where(g.Dz > 0, g.Dz, 1.0))).ravel()
        ops = {}
        if "closed" in lus:
            ops["closed"] = T_closed
        if "leak" in lus:
            leak = np.zeros((nz, ny, nx))
            leak[0] = np.where(g.dP[0] > 0, g.omega_plus[0] / np.where(g.dP[0] > 0, g.dP[0], 1.0), 0.0)
            ops["leak"] = T_closed + sp.diags(leak.ravel())
        if "mixing" in lus:
            # The pre-2026-09-26 reading: the untransformed block acting on
            # concentrations (state proportional to a mixing ratio).
            g.concentration_state = False
            ops["mixing"] = assemble_transport_block(g, idx)
            g.concentration_state = True
        if "cellcentred" in lus and g.has_interface_omega:
            # The pre-2026-09-26 vertical flux: the cell-centred omega read
            # at the bottom faces, ground and top closed.
            keep = (g.omega_edge, g.omega_edge_plus, g.omega_edge_minus)
            g.omega_edge = g.omega_edge_plus = g.omega_edge_minus = np.array([])
            ops["cellcentred"] = assemble_transport_block(g, idx)
            g.omega_edge, g.omega_edge_plus, g.omega_edge_minus = keep
        if "unsplit" in lus or "diagnosed" in lus:
            om_plus, om_minus = g.omega_plus.copy(), g.omega_minus.copy()
            edge = (g.omega_edge, g.omega_edge_plus, g.omega_edge_minus)
            g.omega_edge = g.omega_edge_plus = g.omega_edge_minus = np.array([])
            if "unsplit" in lus:
                g.omega_plus = np.maximum(g.omega, 0.0)
                g.omega_minus = np.maximum(-g.omega, 0.0)
                ops["unsplit"] = assemble_transport_block(g, idx)
            if "diagnosed" in lus:
                g.concentration_state = False
                div = np.asarray(assemble_horizontal_convdiff(g, idx).sum(axis=1)).ravel().reshape(nz, ny, nx) * g.dP
                g.concentration_state = True
                om_d = np.zeros_like(g.omega)
                for k in range(nz - 1):
                    om_d[k + 1] = om_d[k] + div[k]
                g.omega_plus = np.maximum(om_d, 0.0)
                g.omega_minus = np.maximum(-om_d, 0.0)
                ops["diagnosed"] = assemble_transport_block(g, idx)
            g.omega_plus, g.omega_minus = om_plus, om_minus
            g.omega_edge, g.omega_edge_plus, g.omega_edge_minus = edge
        for name, T in ops.items():
            lus[name].append(_LU(I_N + (T + D) * DTAU))
    print(f"factorised {len(args.variants) * N_BINS} matrices in {time.time() - t0:.0f} s")

    fields = {}
    for name, lu8 in lus.items():
        orb, info = solve_orbit_one_species(lu8, e_list, tol=1e-7, maxiter=300)
        fields[name] = np.mean(orb[:N_BINS], axis=0).reshape(nz, ny, nx)
        print(f"{name:>10}: gmres {info['gmres_iters']} iters, residual {info['residual']:.1e}, "
              f"surface domain mean {fields[name][0].mean():.3f} ug/m3")
    if args.out:
        np.savez(args.out, lat=g0.lat, lon=g0.lon, **fields)

    land = g0.is_land.astype(bool)
    Ps = np.mean([g.Psurf for g in grids], axis=0)
    plains = (Ps > 95000) & land
    inner = np.zeros((ny, nx), bool)
    inner[1:-1, 1:-1] = True
    pairs = [(a, b) for a, b in [("closed", "leak"), ("closed", "mixing"), ("closed", "cellcentred"),
                                 ("diagnosed", "unsplit"), ("unsplit", "closed")]
             if a in fields and b in fields]
    for a, b in pairs:
        A, B = fields[a][0], fields[b][0]
        m = (B > 1) & inner
        print(f"\n=== surface {a} / {b}, bin mean, cells with {b} > 1 ug/m3")
        print(f"   all such cells : {_stats((A / B)[m])}")
        print(f"   plains > 950 hPa: {_stats((A / B)[m & plains])}")
        print(f"   other land      : {_stats((A / B)[m & land & ~plains])}")
        print(f"   land-mean ratio : {A[land].mean() / B[land].mean():.3f}")
    lat, lon = g0.lat, g0.lon
    names = list(fields)
    print(f"\n{'site':>10} {'Ps hPa':>7} " + " ".join(f"{n:>10}" for n in names))
    for s, (la, lo) in SITES.items():
        if not (lat.min() <= la <= lat.max() and lon.min() <= lo <= lon.max()):
            continue
        j = np.argmin(np.abs(lat - la)); i = np.argmin(np.abs(lon - lo))
        print(f"{s:>10} {Ps[j, i] / 100:7.0f} " + " ".join(f"{fields[n][0][j, i]:10.2f}" for n in names))


if __name__ == "__main__":
    main()
