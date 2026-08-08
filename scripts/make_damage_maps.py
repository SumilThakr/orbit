#!/usr/bin/env python
"""Gridded marginal-damage maps from the production adjoint sweep.

The scenarios NetCDF carries subdistrict aggregates; this plots the
underlying field at the model's own resolution, which is where the
spatial structure actually lives.

Unit convention (matches scripts/verify_adjoint_compare.py):
    e_solver [ug/m3/s] = rate_kg_s * 1e9 / surface_volume_m3
    deaths/yr          = sum_tau dJ_de * e_solver / N_BINS
for a sustained 1000 kg/yr emitted into that cell's surface layer. So
every map is "deaths per year per 1000 kg/yr emitted here".

  M1  annual damage by species (multi-panel)
  M2  seasonal cycle of primary-PM damage (12 monthly panels)
  M3  CRF spread: max-min across the three response functions, relative
  M4  where the negatives are, by species

Usage:
    python scripts/make_damage_maps.py --production-dir ... --out-dir ...
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

N_BINS = 8
SEC_PER_YR = 365.25 * 86400.0
SURFACE = "#fcfcfb"
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8a8a85"
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
# Sequential = one hue, light->dark (never a rainbow).
CMAP_SEQ = "YlGnBu"
CMAP_DIV = "RdBu_r"


def _style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "font.size": 9,
        "axes.titlesize": 10, "axes.labelsize": 9,
        "axes.edgecolor": INK3, "axes.linewidth": 0.8,
        "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.labelcolor": INK,
        "axes.grid": False, "legend.frameon": False,
    })


def damage_per_1000kg(ds, species):
    """(ny, nx) deaths/yr per 1000 kg/yr sustained emission in the cell."""
    dj = ds.dJ_de.sel(species=species).isel(draw=0).values      # (bin, y, x)
    vol = ds["surface_volume_m3"].values                        # (y, x)
    rate_kg_s = 1000.0 / SEC_PER_YR
    e_solver = rate_kg_s * 1e9 / vol                            # ug/m3/s
    return dj.sum(axis=0) * e_solver / N_BINS


def _open(prod, crf, month):
    import xarray as xr
    p = Path(prod) / crf / f"adjoint_M{month:02d}.nc"
    return xr.open_dataset(p) if p.exists() else None


def _operator_label(prod, crf):
    """'FCT' or 'low-order', read from the adjoint files' horizontal_fct attr."""
    import xarray as xr
    for m in range(1, 13):
        p = Path(prod) / crf / f"adjoint_M{m:02d}.nc"
        if p.exists():
            with xr.open_dataset(p) as ds:
                return "FCT" if int(ds.attrs.get("horizontal_fct", 1)) else "low-order"
    return "unknown"




def _map(ax, lon, lat, field, title, norm=None, cmap=CMAP_SEQ):
    m = ax.pcolormesh(lon, lat, field, shading="auto", cmap=cmap,
                      norm=norm, rasterized=True)
    ax.set_title(title, loc="left", fontsize=9, pad=4)
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(labelsize=7)
    return m


def m1_species(prod, out_dir, crf, species_list):
    """Annual mean damage per species."""
    fields, lat, lon = {}, None, None
    for sp in species_list:
        acc, n = None, 0
        for mo in range(1, 13):
            ds = _open(prod, crf, mo)
            if ds is None:
                continue
            if sp not in [str(s) for s in ds.species.values]:
                ds.close(); continue
            f = damage_per_1000kg(ds, sp)
            acc = f if acc is None else acc + f
            n += 1
            lat, lon = ds.lat.values, ds.lon.values
            ds.close()
        if n:
            fields[sp] = acc / n
    if not fields:
        print("  M1: no data — skipped"); return
    keys = [k for k in species_list if k in fields]
    ncol = min(3, len(keys)); nrow = int(np.ceil(len(keys) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 3.3 * nrow),
                             squeeze=False)
    pos = np.concatenate([f[f > 0] for f in fields.values()])
    vmin, vmax = np.percentile(pos, 2), np.percentile(pos, 99.5)
    norm = LogNorm(vmin=max(vmin, vmax * 1e-4), vmax=vmax)
    for i in range(nrow * ncol):
        ax = axes[i // ncol][i % ncol]
        if i >= len(keys):
            ax.axis("off"); continue
        k = keys[i]
        mesh = _map(ax, lon, lat, np.where(fields[k] > 0, fields[k], np.nan),
                    k, norm=norm)
    cb = fig.colorbar(mesh, ax=axes, shrink=0.7, pad=0.02, extend="both")
    cb.set_label("deaths · yr⁻¹ per 1000 kg · yr⁻¹ emitted", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    fig.suptitle(f"Marginal damages by species, annual mean — {crf}",
                 x=0.005, ha="left", fontsize=11)
    p = Path(out_dir) / f"M1_damage_by_species_{crf}.png"
    fig.savefig(p, dpi=170, bbox_inches="tight"); plt.close(fig)
    print(f"  wrote {p}")


def m2_seasonal(prod, out_dir, crf, species="PM25_primary"):
    fields, lat, lon = {}, None, None
    for mo in range(1, 13):
        ds = _open(prod, crf, mo)
        if ds is None:
            continue
        fields[mo] = damage_per_1000kg(ds, species)
        lat, lon = ds.lat.values, ds.lon.values
        ds.close()
    if not fields:
        print("  M2: no data — skipped"); return
    pos = np.concatenate([f[f > 0] for f in fields.values()])
    norm = LogNorm(vmin=np.percentile(pos, 2), vmax=np.percentile(pos, 99.5))
    fig, axes = plt.subplots(3, 4, figsize=(14.5, 9.5), squeeze=False)
    for i in range(12):
        ax = axes[i // 4][i % 4]
        mo = i + 1
        if mo not in fields:
            ax.axis("off"); continue
        f = fields[mo]
        mesh = _map(ax, lon, lat, np.where(f > 0, f, np.nan), MONTHS[i],
                    norm=norm)
        med = np.nanmedian(f[f > 0])
        ax.annotate(f"median {med:.2e}", xy=(0.03, 0.04),
                    xycoords="axes fraction", fontsize=7, color=INK2)
    cb = fig.colorbar(mesh, ax=axes, shrink=0.6, pad=0.02, extend="both")
    cb.set_label("deaths · yr⁻¹ per 1000 kg · yr⁻¹", fontsize=8)
    fig.suptitle(f"Seasonal cycle of {species} marginal damage — {crf}"
                 "   (each month's periodic orbit, annualized)",
                 x=0.005, ha="left", fontsize=11)
    p = Path(out_dir) / f"M2_seasonal_{species}_{crf}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  wrote {p}")


def m3_crf_spread(prod, out_dir, crfs, species="PM25_primary"):
    per = {}
    lat = lon = None
    for crf in crfs:
        acc, n = None, 0
        for mo in range(1, 13):
            ds = _open(prod, crf, mo)
            if ds is None:
                continue
            f = damage_per_1000kg(ds, species)
            acc = f if acc is None else acc + f
            n += 1; lat, lon = ds.lat.values, ds.lon.values
            ds.close()
        if n:
            per[crf] = acc / n
    if len(per) < 2:
        print("  M3: need >=2 CRFs — skipped"); return
    stack = np.stack(list(per.values()))
    mean = stack.mean(axis=0)
    spread = (stack.max(axis=0) - stack.min(axis=0)) / np.where(
        np.abs(mean) > 0, np.abs(mean), np.nan)
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    mesh = _map(ax, lon, lat, np.where(mean > 0, 100 * spread, np.nan),
                f"CRF spread in {species} damage  (max−min)/mean",
                cmap=CMAP_SEQ)
    cb = fig.colorbar(mesh, ax=ax, shrink=0.8, pad=0.02)
    cb.set_label("% of the 3-CRF mean", fontsize=8)
    med = np.nanmedian(100 * spread[mean > 0])
    ax.annotate(f"median spread {med:.0f}% of the mean\n"
                f"CRFs: {', '.join(per)}",
                xy=(0.03, 0.04), xycoords="axes fraction",
                fontsize=8, color=INK2)
    fig.tight_layout()
    p = Path(out_dir) / f"M3_crf_spread_{species}.png"
    fig.savefig(p, dpi=180); plt.close(fig)
    print(f"  wrote {p}")


def m4_negatives(prod, out_dir, crf, species_list):
    """Where each species goes negative — the linearization gauge, mapped."""
    rows = []
    lat = lon = None
    for sp in species_list:
        cnt = None
        for mo in range(1, 13):
            ds = _open(prod, crf, mo)
            if ds is None:
                continue
            if sp not in [str(s) for s in ds.species.values]:
                ds.close(); continue
            neg = (damage_per_1000kg(ds, sp) < 0).astype(np.float32)
            cnt = neg if cnt is None else cnt + neg
            lat, lon = ds.lat.values, ds.lon.values
            ds.close()
        if cnt is not None:
            rows.append((sp, cnt))
    if not rows:
        print("  M4: no data — skipped"); return
    ncol = min(3, len(rows)); nrow = int(np.ceil(len(rows) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 3.3 * nrow),
                             squeeze=False)
    for i in range(nrow * ncol):
        ax = axes[i // ncol][i % ncol]
        if i >= len(rows):
            ax.axis("off"); continue
        sp, cnt = rows[i]
        mesh = _map(ax, lon, lat, np.where(cnt > 0, cnt, np.nan),
                    f"{sp}  ({100*(cnt>0).mean():.1f}% of cells)",
                    cmap="OrRd")
        mesh.set_clim(1, 12)
    cb = fig.colorbar(mesh, ax=axes, shrink=0.7, pad=0.02)
    cb.set_label("months with negative damage (of 12)", fontsize=8)
    # The operator is read from the data rather than asserted: the same figure
    # is produced from both the FCT and the low-order sweeps, and mislabelling
    # which one is on the page is precisely the error this regeneration exists
    # to fix. Under low-order the transport block is an M-matrix, so anything
    # still negative is chemistry (ISORROPIA substitution), not linearisation.
    _op = _operator_label(prod, crf)
    _note = ("see investigation doc §6–7: real but exaggerated, "
             "0.08% of damage-weighted magnitude"
             if _op == "FCT" else
             "monotone operator: residual negatives are ISORROPIA "
             "substitution chemistry, not transport")
    fig.suptitle(f"Where the linearization goes negative — {crf}, "
                 f"{_op} operator   ({_note})",
                 x=0.005, ha="left", fontsize=10)
    p = Path(out_dir) / f"M4_negatives_{crf}.png"
    fig.savefig(p, dpi=170, bbox_inches="tight"); plt.close(fig)
    print(f"  wrote {p}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--production-dir", required=True)
    ap.add_argument("--out-dir", default="docs/figures/production")
    ap.add_argument("--crf", nargs="+", default=["gemm5cod", "gemmac", "ier"])
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    _style()
    main_sp = ["PM25_primary", "POA", "NH3", "SO2", "NOx", "VOC_anthro"]
    primary = args.crf[0]
    print(f"Writing maps to {args.out_dir}/")
    m1_species(args.production_dir, args.out_dir, primary, main_sp)
    m2_seasonal(args.production_dir, args.out_dir, primary)
    m3_crf_spread(args.production_dir, args.out_dir, args.crf)
    m4_negatives(args.production_dir, args.out_dir, primary,
                 ["PM25_primary", "POA", "NH3", "SO2", "NOx", "VOC_anthro"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
