#!/usr/bin/env python
"""Finite-difference audit of negative adjoint sensitivities (Jan 2022).

Run D (outer-GMRES FCT adjoint, 2026-08-03) established that the negative
PM25_primary dJ_de entries are CONVERGED features of the frozen-limiter
FCT adjoint, not iteration error. The frozen van Leer/Zalesak limiter
makes the linearized operator an unlimited second-order scheme — linear,
non-monotone (Godunov) — so negatives may be linearization artifacts. Or
the true nonlinear scheme's limiter response may genuinely produce them.
Only a nonlinear finite difference can tell the two apart.

Three subcommands:

  make-sources  Pick probe cells from the adjoint file (top-K most
                negative bin-summed dJ_de + a strongly positive control),
                write one single-cell NetCDF emission source per probe on
                the model grid (variable `pm25`, kg/m2/s), plus
                probes.json describing them.
  analyze       After the zero-out runs: for each probe, compute
                  dJ_nonlin = sum(S * delta_pm25_mean) / dq
                from the zero-out dNPZ, and the adjoint's prediction
                  dJ_adj = sum(dJ_de * delta_e) / dq
                with delta_e built by the SAME loader path the forward
                uses (orbit.modes.perturbation.build_delta_emissions).
                Report signs and ratios.

The probe magnitude q is max(PROBE_FRAC x baseline cell rate, PROBE_FLOOR)
so it works in low-emission cells (where the negatives live) without
leaving the quasi-linear regime in polluted ones.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

PROBE_FRAC = 0.2            # of the cell's baseline primary-PM2.5 rate
PROBE_FLOOR = 5.0e-12       # kg/m2/s; ~10% of a typical rural CEDS cell
N_NEG_PROBES = 3


def _load_adjoint(path):
    import netCDF4 as nc
    ds = nc.Dataset(path)
    dj = np.asarray(ds["dJ_de"][0, 0])          # (8, 71, 65)
    S = np.asarray(ds["S_orbit"][0])            # (71, 65)
    lat = np.asarray(ds["lat"][:])
    lon = np.asarray(ds["lon"][:])
    ds.close()
    return dj, S, lat, lon


def cmd_make_sources(args):
    import netCDF4 as nc
    dj, S, lat, lon = _load_adjoint(args.adjoint)
    dj_sum = dj.sum(axis=0)                      # bin-summed sensitivity

    # Baseline primary-PM2.5 rate per cell, for probe sizing. Sum the
    # pm25-channel variables of the CEDS bcoth files (regridded crudely by
    # nearest neighbour — sizing only, not physics).
    base_rate = np.zeros_like(dj_sum)
    import glob
    for f in glob.glob(os.path.join(args.emission_dir, "ceds_bcoth_*_monthly*.nc")):
        d = nc.Dataset(f)
        var = [v for v in d.variables if v not in ("lat", "lon", "time", "bin")]
        elat = np.asarray(d["lat"][:]); elon = np.asarray(d["lon"][:])
        for v in var:
            arr = np.asarray(d[v][args.month - 1])   # (elat, elon)
            iy = np.abs(elat[None, :] - lat[:, None]).argmin(axis=1)
            ix = np.abs(elon[None, :] - lon[:, None]).argmin(axis=1)
            base_rate += arr[np.ix_(iy, ix)]
        d.close()

    probes = []
    if getattr(args, "cells", None):
        # Named panel: one probe per requested location, nearest cell wins.
        for entry in args.cells.split(";"):
            entry = entry.strip()
            if not entry:
                continue
            name, coords = entry.split(":", 1)
            plat, plon = (float(x) for x in coords.split(","))
            j = int(np.abs(lat - plat).argmin())
            i = int(np.abs(lon - plon).argmin())
            probes.append((name.strip(), j, i))
    else:
        flat = dj_sum.ravel()
        order_neg = np.argsort(flat)              # ascending: most negative first
        for k in range(N_NEG_PROBES):
            j, i = np.unravel_index(order_neg[k], dj_sum.shape)
            probes.append(("neg%d" % (k + 1), int(j), int(i)))
        j, i = np.unravel_index(int(np.argmax(dj_sum)), dj_sum.shape)
        probes.append(("poscontrol", int(j), int(i)))

    probe_frac = float(getattr(args, "probe_frac", PROBE_FRAC))
    _pf = getattr(args, "probe_floor", None)
    # NB `or` would swallow an explicit 0.0 (falsy) and restore the
    # default floor — exactly the silent rescale this flag exists to stop.
    probe_floor = float(PROBE_FLOOR if _pf is None else _pf)
    os.makedirs(args.out_dir, exist_ok=True)
    meta = []
    for name, j, i in probes:
        q = float(max(probe_frac * base_rate[j, i], probe_floor))
        path = os.path.join(args.out_dir, f"probe_{name}_j{j}_i{i}.nc")
        with nc.Dataset(path, "w") as d:
            d.createDimension("lat", len(lat))
            d.createDimension("lon", len(lon))
            d.createDimension("time", 12)
            vlat = d.createVariable("lat", "f8", ("lat",))
            vlat[:] = lat; vlat.units = "degrees_north"
            vlon = d.createVariable("lon", "f8", ("lon",))
            vlon[:] = lon; vlon.units = "degrees_east"
            vt = d.createVariable("time", "f8", ("time",))
            vt[:] = np.arange(12) * 30.4; vt.units = "days since 2022-01-01"
            vp = d.createVariable("pm25", "f8", ("time", "lat", "lon"))
            arr = np.zeros((12, len(lat), len(lon)))
            arr[:, j, i] = q
            vp[:] = arr; vp.units = "kg m-2 s-1"
            d.description = (
                f"FD probe {name}: single-cell primary PM2.5 source at "
                f"(j={j}, i={i}) = ({lat[j]:.2f}N, {lon[i]:.2f}E), q={q:.3e}"
            )
        meta.append(dict(
            name=name, j=j, i=i, lat=float(lat[j]), lon=float(lon[i]),
            q_kg_m2_s=q, probe_frac=probe_frac, source_nc=path,
            dj_sum=float(dj_sum[j, i]), S_here=float(S[j, i]),
            base_rate=float(base_rate[j, i]),
        ))
        print(f"{name:11s} (j={j:2d}, i={i:2d}) {lat[j]:6.2f}N {lon[i]:7.2f}E  "
              f"dJ_sum={dj_sum[j, i]:+.3e}  S={S[j, i]:.3e}  "
              f"base={base_rate[j, i]:.2e}  q={q:.2e}")
    with open(os.path.join(args.out_dir, "probes.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nWrote {len(meta)} probes to {args.out_dir}")

    # PROBE_FLOOR clamps q from below, which silently turns a nominal 1%
    # step into whatever the floor happens to be. A step-size study that
    # does not check this reports the wrong abscissa.
    clamped = [m["name"] for m in meta
               if probe_frac * m["base_rate"] < probe_floor]
    if clamped:
        print(f"WARNING: PROBE_FLOOR ({probe_floor:.1e}) binds at "
              f"{len(clamped)}/{len(meta)} probes: {', '.join(clamped)}. "
              f"Their effective step is NOT {probe_frac:.0%} of baseline. "
              f"Exclude them from any step-size study, or lower the floor.")


def cmd_analyze(args):
    with open(os.path.join(args.probe_dir, "probes.json")) as f:
        probes = json.load(f)
    dj, S, lat, lon = _load_adjoint(args.adjoint)

    # delta_e in solver layout for the adjoint prediction, via the same
    # loader path the perturbed forward used.
    from orbit.core.indexing import CellIndexer
    from orbit.core.grid_data import load_grid
    from orbit.modes.perturbation import Perturbation, build_delta_emissions
    from orbit.emissions.sources import EmissionSource

    # Grid/indexer come from bin 1 of the month's preproc (any bin works —
    # delta_e only needs geometry + Dz).
    bin_path = os.path.join(
        args.preproc_base, f"sas_2022_M{args.month:02d}_B01.nc")
    grid = load_grid(bin_path, args.constants)
    indexer = CellIndexer(grid.nz, grid.ny, grid.nx)
    from orbit.core.deposition import IDX_PM25
    N = indexer.N
    nz, ny, nx = indexer.nz, indexer.ny, indexer.nx

    # The zero-out emission-assembly path differs systematically from the
    # production forward's (max ~3.7 ug/m3 in the GFED BB region, byte-
    # identical across probes — see 2026-08-03_fct_adjoint_investigation.md).
    # A "null" probe (q=1e-20, numerically nothing) measures exactly that
    # offset; subtracting its delta isolates each probe's true response.
    from orbit.modes.perturbation import unify_sign

    def _surface_delta(name):
        path = os.path.join(
            args.zeroout_dir, name, f"zeroout_M{args.month:02d}.npz")
        if not os.path.exists(path):
            return None
        d = unify_sign(path, target="perturbation_response")
        arr = d["delta_pm25_mean"]
        return arr[0] if arr.ndim == 3 else arr        # surface level

    null_delta = _surface_delta("null")
    if null_delta is None:
        print("WARNING: no null run found — reporting raw deltas, which "
              "include the systematic assembly offset. Do not trust signs.")
    else:
        # Detection floor for verdicts: the S-weighted null response IS
        # the total systematic + numerical error of the whole chain (on
        # the fixed path it measured ~1e-9 deaths, 2026-08-03).
        floor = float(np.abs(S * null_delta).sum())
        print(f"noise floor (Sum|S*null|): {floor:.3e} deaths — verdicts "
              f"below ~10x this are 'indistinguishable from zero'")

    # Two-gate design (decision 2026-08-03):
    #   Gate 1 (bookkeeping): linear-marginal control ratio vs the adjoint
    #     must be in [0.95, 1.05] — tests units/probe assembly/S-weighting
    #     through two independent code paths. Passed at 0.9999.
    #   Gate 2 (per-verdict): the neg-probe nonlinear responses must clear
    #     the null-run noise floor.
    # The original single gate (nonlinear/adjoint in [0.8, 1.25]) conflated
    # bookkeeping with linearization fidelity; the latter FAILS for a real,
    # measured reason (frozen limiter over-predicts a 20%-step response by
    # ~1.7x at the control) and is reported as a finding, not a defect.
    if getattr(args, "control_marginal", None):
        m = np.load(args.control_marginal, allow_pickle=False)
        dm = m["delta_pm25_orbit"][:, 0].mean(axis=0)
        dm = dm if dm.shape == S.shape else dm[0]
        dJ_marg = float((S * dm).sum())
        print(f"GATE 1 control (linear marginal): dJ_marg = {dJ_marg:+.4e}")

    print(f"{'probe':11s} {'dJ_adj':>12s} {'dJ_nonlin':>12s} "
          f"{'ratio':>7s}  verdict")
    for p in probes:
        if p["name"] == "null":
            continue
        pert = Perturbation(add_sources=[EmissionSource(
            path=p["source_nc"], format="netcdf", units="kg/m2/s",
            variable_mapping={"pm25": "pm25"}, time_index=None,
        )])
        delta_e = build_delta_emissions(
            pert, [], grid, indexer, args.month)     # (8, N_SPECIES*N)
        de_pm = delta_e[:, IDX_PM25 * N:(IDX_PM25 + 1) * N]
        de_surf = de_pm.reshape(8, nz, ny, nx)[:, 0]     # emissions are surface
        # /8: J = Σ_τ ⟨S, G c_τ⟩ over-counts the annual mean by N_BINS for a
        # sustained emission (same division as verify_adjoint_compare.py).
        dJ_adj = float((dj * de_surf).sum()) / 8.0

        d_pm25 = _surface_delta(p["name"])               # (ny, nx), + = increase
        if d_pm25 is None:
            print(f"{p['name']:11s} {dJ_adj:+12.4e} {'MISSING':>12s}")
            continue
        if null_delta is not None:
            d_pm25 = d_pm25 - null_delta
        dJ_nl = float((S * d_pm25).sum())

        ratio = dJ_nl / dJ_adj if abs(dJ_adj) > 1e-300 else np.nan
        if p["name"].startswith("neg"):
            if null_delta is not None and abs(dJ_nl) < 10.0 * float(
                    np.abs(S * null_delta).sum()):
                verdict = "INDISTINGUISHABLE FROM ZERO (below noise floor)"
            elif dJ_nl < 0:
                verdict = "NEGATIVE IS REAL (nonlinear agrees, attenuated)"
            else:
                verdict = "ARTIFACT (nonlinear is non-negative)"
        else:
            verdict = ("control: nonlinear/linear step response "
                       "(NOT a pass/fail gate — see §7.1)")
            if getattr(args, "control_marginal", None):
                g1 = dJ_marg / dJ_adj if abs(dJ_adj) > 1e-300 else np.nan
                ok = "PASS" if 0.95 < g1 < 1.05 else "FAIL"
                print(f"GATE 1 (bookkeeping) marginal/adjoint = {g1:.4f}  "
                      f"[0.95, 1.05] {ok}")
        print(f"{p['name']:11s} {dJ_adj:+12.4e} {dJ_nl:+12.4e} "
              f"{ratio:7.3f}  {verdict}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make-sources")
    m.add_argument("--adjoint", required=True)
    m.add_argument("--emission-dir", required=True)
    m.add_argument("--out-dir", required=True)
    m.add_argument("--month", type=int, default=1)
    m.add_argument("--probe-frac", type=float, default=PROBE_FRAC,
                   help=f"Probe size as a fraction of the cell's baseline "
                        f"primary-PM2.5 rate (default {PROBE_FRAC}). Varying "
                        f"this is how the step-size study is run.")
    m.add_argument("--probe-floor", type=float, default=None,
                   help=f"Lower clamp on q (default {PROBE_FLOOR:.1e} kg/m2/s). "
                        f"Set to 0 for a genuinely uniform --probe-frac step: "
                        f"the default floor silently rescales small-baseline "
                        f"cells and mislabels a step-size study's abscissa. "
                        f"Safe to lower far — the measured S-weighted "
                        f"differencing noise floor is ~5e-9 deaths against "
                        f"probe responses of O(1) deaths.")
    m.add_argument("--cells", default=None,
                   help="Explicit probe cells as "
                        "'name:lat,lon;name:lat,lon;...' instead of the "
                        "top-K most-negative selection. Nearest grid cell "
                        "wins. Used for the named-city panel.")
    a = sub.add_parser("analyze")
    a.add_argument("--adjoint", required=True)
    a.add_argument("--probe-dir", required=True)
    a.add_argument("--zeroout-dir", required=True)
    a.add_argument("--preproc-base", required=True)
    a.add_argument("--constants", required=True)
    a.add_argument("--month", type=int, default=1)
    a.add_argument("--control-marginal", default=None,
                   help="marginal_M01.npz from a LINEAR marginal run of the "
                        "poscontrol probe. Enables Gate 1 (bookkeeping): "
                        "marginal/adjoint must be in [0.95, 1.05].")
    args = ap.parse_args()
    if args.cmd == "make-sources":
        cmd_make_sources(args)
    else:
        cmd_analyze(args)


if __name__ == "__main__":
    sys.exit(main())
