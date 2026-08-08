#!/usr/bin/env python
"""Regenerate the satellite comparison against a given ORBIT run.

Re-samples the model side at the SAME cell-months an existing satellite
pairs file defines, so the satellite observations, the pairing geometry,
the pixel counts, and the coverage floors are all inherited unchanged:
only the model changes. It cannot re-do the pairing itself (that needs
the daily V01FL files, which are not in the evaluation deposit); it does
not touch the CPCB or SPARTAN products, whose observations are
restricted.

Stats: Pearson r^2, OLS slope/intercept, NMB/NME as percentages of the
observed sum, and the weighted variants using n_pixels.

Usage:
    python scripts/regenerate_satellite_eval.py ORBIT_DIR OUT_DIR \\
        [--field pm25_mean_baseline] [--pairs PATH]
"""

import argparse
import os

import numpy as np
from scipy import stats as sps

DEFAULT_PAIRS = os.path.expanduser(
    "~/orbit_data/deposit_B/satellite_comparison/satellite_pm25_pairs.npz")
MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
MIN_MONTHS_ANNUAL = 9      # matches the shipped product's annual filter


def linfit_stats(obs, model, weights=None):
    """Unweighted or weighted fit stats, per _obs_common conventions."""
    obs = np.asarray(obs, float)
    model = np.asarray(model, float)
    keep = np.isfinite(obs) & np.isfinite(model)
    if weights is not None:
        w = np.asarray(weights, float)
        keep &= np.isfinite(w) & (w > 0)
    obs, model = obs[keep], model[keep]
    n = obs.size
    if n < 2:
        return {"n": n, "r2": np.nan, "slope": np.nan, "intercept": np.nan,
                "nmb_pct": np.nan, "nme_pct": np.nan, "mb": np.nan,
                "rmse": np.nan, "obs_mean": np.nan, "model_mean": np.nan}
    bias = model - obs
    if weights is None:
        s, b, r, _, _ = sps.linregress(obs, model)
        return {"n": int(n), "r2": float(r ** 2), "slope": float(s),
                "intercept": float(b),
                "nmb_pct": float(bias.sum() / obs.sum() * 100),
                "nme_pct": float(np.abs(bias).sum() / obs.sum() * 100),
                "mb": float(bias.mean()),
                "rmse": float(np.sqrt(np.mean(bias ** 2))),
                "obs_mean": float(obs.mean()),
                "model_mean": float(model.mean())}
    w = np.asarray(weights, float)[keep]
    sw = w.sum()
    obs_bar, mod_bar = (w * obs).sum() / sw, (w * model).sum() / sw
    cov = (w * (obs - obs_bar) * (model - mod_bar)).sum() / sw
    var_o = (w * (obs - obs_bar) ** 2).sum() / sw
    var_m = (w * (model - mod_bar) ** 2).sum() / sw
    r = cov / np.sqrt(var_o * var_m) if var_o > 0 and var_m > 0 else np.nan
    slope, intercept = np.polyfit(obs, model, 1, w=w)
    return {"n": int(n), "r2": float(r ** 2) if np.isfinite(r) else np.nan,
            "slope": float(slope), "intercept": float(intercept),
            "nmb_pct": float((w * bias).sum() / (w * obs).sum() * 100),
            "nme_pct": float((w * np.abs(bias)).sum() / (w * obs).sum() * 100),
            "mb": float((w * bias).sum() / sw),
            "rmse": float(np.sqrt((w * bias ** 2).sum() / sw)),
            "obs_mean": float(obs_bar), "model_mean": float(mod_bar)}


def sample_model(orbit_dir, month, field, lats, lons):
    """Model value at each (lat, lon) for one month, or None if absent."""
    path = os.path.join(orbit_dir, f"orbit_M{month:02d}.npz")
    if not os.path.exists(path):
        return None
    d = np.load(path)
    glat, glon = d["lat"], d["lon"]
    grid = d[field][0]
    iy = np.abs(glat[None, :] - lats[:, None]).argmin(axis=1)
    ix = np.abs(glon[None, :] - lons[:, None]).argmin(axis=1)
    # The pair lat/lons are cell centres of this same grid; anything
    # further than a fraction of a cell means the grids disagree.
    dlat = float(np.abs(np.diff(glat)).min())
    dlon = float(np.abs(np.diff(glon)).min())
    if (np.abs(glat[iy] - lats).max() > 0.1 * dlat
            or np.abs(glon[ix] - lons).max() > 0.1 * dlon):
        raise SystemExit(f"month {month}: pair coordinates do not land on "
                         f"the model grid; wrong domain or resolution?")
    return grid[iy, ix]


def block(title, obs, model, weights):
    lines = [title, "-" * 80,
             f"{'metric':>14s} {'UNW':>25s} {'WTD':>25s}"]
    unw = linfit_stats(obs, model)
    wtd = linfit_stats(obs, model, weights)
    for key in ("n", "r2", "slope", "intercept", "nmb_pct", "nme_pct",
                "mb", "rmse", "obs_mean", "model_mean"):
        lines.append(f"{key:>14s} {str(unw[key]):>25s} {str(wtd[key]):>25s}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("orbit_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--field", default="pm25_mean_baseline",
                    help="model field to score (default matches the "
                         "shipped product's convention)")
    ap.add_argument("--pairs", default=DEFAULT_PAIRS)
    args = ap.parse_args()

    src = np.load(args.pairs)
    obs_all, lat_all, lon_all = src["obs"], src["lat"], src["lon"]
    month_all, npix_all = src["month"], src["n_pixels"]

    model_all = np.full(obs_all.shape, np.nan, dtype=np.float64)
    have = []
    for m in range(1, 13):
        sel = month_all == m
        if not sel.any():
            continue
        vals = sample_model(args.orbit_dir, m, args.field,
                            lat_all[sel].astype(float),
                            lon_all[sel].astype(float))
        if vals is None:
            print(f"  month {m:02d}: no output, skipped")
            continue
        model_all[sel] = vals
        have.append(m)
    if not have:
        raise SystemExit(f"no monthly outputs found in {args.orbit_dir}")
    print(f"  months sampled: {have}")

    ok = np.isfinite(model_all)
    obs, model = obs_all[ok].astype(float), model_all[ok]
    month, npix = month_all[ok], npix_all[ok].astype(float)
    lat, lon = lat_all[ok], lon_all[ok]

    os.makedirs(args.out_dir, exist_ok=True)
    out_npz = os.path.join(args.out_dir, "satellite_pm25_pairs.npz")
    np.savez_compressed(out_npz, obs=obs.astype(np.float32),
                        model=model.astype(np.float32),
                        lat=lat, lon=lon, month=month,
                        n_pixels=npix.astype(np.int32))

    parts = [
        "ORBIT vs satellite-derived PM2.5 (V01FL India 2022, total PM2.5)",
        "=" * 80,
        "Regenerated by scripts/regenerate_satellite_eval.py. Observations,",
        "pairing geometry, pixel counts and coverage floors are inherited",
        "verbatim from the shipped deposit_B pairs; only the model side is",
        "re-sampled.",
        "",
        f"ORBIT dir:    {os.path.abspath(args.orbit_dir)}",
        f"Model field:  {args.field}",
        f"Months:       {have}",
        f"Paired cell-month points: {obs.size}",
        "",
        "  UNW = each cell-month counts once",
        "  WTD = each cell-month weighted by n_pixels",
        "",
    ]

    # Annual: per cell, requires >= MIN_MONTHS_ANNUAL valid months.
    cells = {}
    for o, mo, la, lo, w in zip(obs, model, lat, lon, npix):
        cells.setdefault((round(float(la), 4), round(float(lo), 4)),
                         []).append((o, mo, w))
    a_obs, a_mod, a_w = [], [], []
    for recs in cells.values():
        if len(recs) >= MIN_MONTHS_ANNUAL:
            arr = np.array(recs, dtype=float)
            a_obs.append(arr[:, 0].mean())
            a_mod.append(arr[:, 1].mean())
            a_w.append(arr[:, 2].mean())
    if a_obs:
        parts.append(block(f"ANNUAL (per cell, >= {MIN_MONTHS_ANNUAL} valid "
                           f"months)", np.array(a_obs), np.array(a_mod),
                           np.array(a_w)))
        parts.append(f"\nN cells: {len(a_obs)}\n")
    else:
        parts.append(f"ANNUAL: skipped ({len(have)} months available, "
                     f"{MIN_MONTHS_ANNUAL} required)\n")

    parts.append(block("MONTHLY POOLED", obs, model, npix))
    parts.append("")
    parts.append("Per-month statistics")
    parts.append("-" * 80)
    parts.append(f"{'mo':>5s} {'wt':>3s} {'n':>6s} {'R2':>6s} {'slope':>7s} "
                 f"{'icpt':>7s} {'NMB%':>7s} {'NME%':>7s} {'obs_mu':>8s} "
                 f"{'mod_mu':>8s}")
    for m in have:
        sel = month == m
        for tag, w in (("UNW", None), ("WTD", npix[sel])):
            s = linfit_stats(obs[sel], model[sel], w)
            parts.append(
                f"  M{m:02d} {tag:>3s} {s['n']:>6d} {s['r2']:>6.2f} "
                f"{s['slope']:>7.2f} {s['intercept']:>7.1f} "
                f"{s['nmb_pct']:>7.0f} {s['nme_pct']:>7.0f} "
                f"{s['obs_mean']:>8.1f} {s['model_mean']:>8.1f}")

    text = "\n".join(parts) + "\n"
    out_txt = os.path.join(args.out_dir, "satellite_pm25_stats.txt")
    with open(out_txt, "w") as fh:
        fh.write(text)
    print(text)
    print(f"wrote {out_npz}\nwrote {out_txt}")


if __name__ == "__main__":
    main()
