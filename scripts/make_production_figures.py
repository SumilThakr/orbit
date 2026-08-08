#!/usr/bin/env python
"""QC report + figures for the production marginal-damages sweep.

Runs after `postprocess_marginal_deaths_scenarios.py`. Two jobs:

  1. A QC report (always, to stdout and qc_report.txt) covering what
     actually landed: which months/CRFs exist, and — because the sweep
     is deliberately iteration-limited (investigation doc §8.3a) — the
     per-month `fct_converged` / `fct_deferred_iters` attributes. The
     report states the residual situation rather than implying
     convergence.
  2. Figures, each skipped with a printed reason if its inputs are
     missing, so a partial sweep still produces everything it can.

     P1  deaths per 1000 kg by species (annual_sustained), ranked
     P2  seasonality: monthly_sustained by month, small multiples
     P3  diurnal: bin_specific by 3-hour bin, small multiples
     P4  CRF sensitivity: the three concentration-response functions

Design notes: P1–P3 encode magnitude, not identity, so they use one hue
(never a cycled categorical ramp); species are separated by facet, not
color. Only P4 is genuinely categorical (3 CRFs) and uses validated
categorical slots.

Usage:
    python scripts/make_production_figures.py \
        --production-dir /path/to/marginal_deaths/production \
        --out-dir docs/figures/production
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

C_BLUE, C_ORANGE, C_AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8a8a85"
SURFACE = "#fcfcfb"
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10,
        "axes.edgecolor": INK3, "axes.linewidth": 0.8,
        "xtick.color": INK2, "ytick.color": INK2,
        "axes.labelcolor": INK, "text.color": INK,
        "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6,
        "axes.axisbelow": True, "legend.frameon": False,
        "axes.spines.top": False, "axes.spines.right": False,
    })


def qc_report(prod_dir, out_dir, crfs):
    """What landed, and how converged is it really?"""
    import xarray as xr
    lines = ["=== PRODUCTION SWEEP QC ===", ""]
    for crf in crfs:
        d = Path(prod_dir) / crf
        files = sorted(d.glob("adjoint_M*.nc"))
        lines.append(f"[{crf}] {len(files)}/12 adjoint months present")
        if not files:
            lines.append("   (nothing to report)")
            lines.append("")
            continue
        conv, iters, species = [], [], None
        for f in files:
            try:
                ds = xr.open_dataset(f)
                conv.append(int(ds.attrs.get("fct_converged", -1)))
                iters.append(int(ds.attrs.get("fct_deferred_iters", -1)))
                if species is None:
                    species = [str(s) for s in ds.species.values]
                ds.close()
            except Exception as e:
                lines.append(f"   ! {f.name}: unreadable ({e})")
        if conv:
            n_conv = sum(1 for c in conv if c == 1)
            lines.append(f"   species: {species}")
            lines.append(f"   fct_converged: {n_conv}/{len(conv)} months "
                         f"reached the 2e-3 tolerance")
            lines.append(f"   fct_deferred_iters: min={min(iters)} "
                         f"max={max(iters)}")
            # The ACHIEVED inf-norm residual is the accuracy-relevant
            # number and is not stored in the NetCDF; recover it from the
            # run logs so the report states measurement, not intent.
            res = {}
            for lg in sorted((d / "logs").glob("M*_*.out")):
                mm = lg.name.split("_")[0]
                for ln in lg.read_text(errors="ignore").splitlines():
                    if "max rel" in ln and "residual" in ln:
                        try:
                            res[mm] = float(ln.split("=")[-1].strip())
                        except ValueError:
                            pass
            if res:
                worst = max(res, key=res.get)
                lines.append(
                    f"   achieved ∞-norm residual on μ: "
                    f"min={min(res.values()):.2e} median="
                    f"{float(np.median(list(res.values()))):.2e} "
                    f"max={max(res.values()):.2e} (worst {worst})")
                hi = {k: v for k, v in res.items() if v > 2.5e-2}
                if hi:
                    lines.append(
                        "   HIGH-RESIDUAL MONTHS (>2.5e-2, beyond the regime "
                        f"§6 probed): {', '.join(f'{k}={v:.1e}' for k, v in sorted(hi.items()))}")
            if n_conv < len(conv):
                lines.append(
                    "   NOTE: iteration-limited BY DESIGN "
                    "(ORBIT_ADJOINT_FCT_MAX_ITERS=12; investigation doc "
                    "§8.3a). The 2-norm GMRES residual is a well-behaved "
                    "0.6-6e-3, but the accuracy-relevant ∞-norm is larger "
                    "(above). §6 measured that halving the residual moves "
                    "the answer 0.3% in the ~1.2e-2 regime; months well "
                    "above that are NOT covered by that evidence — see the "
                    "M11 convergence check (job 14852386). Report the "
                    "residual; do not claim convergence.")
        lines.append("")
    scen = sorted((Path(prod_dir) / "scenarios").glob("scenarios_*.nc"))
    lines.append(f"scenarios written: {[p.name for p in scen]}")
    txt = "\n".join(lines)
    print(txt)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "qc_report.txt").write_text(txt + "\n")


def _load(prod_dir, crf):
    import xarray as xr
    p = Path(prod_dir) / "scenarios" / f"scenarios_{crf}.nc"
    return xr.open_dataset(p) if p.exists() else None


def fig_p1(prod_dir, out_dir, crf):
    ds = _load(prod_dir, crf)
    if ds is None:
        print(f"  P1: scenarios_{crf}.nc missing — skipped"); return
    sp = [str(s) for s in ds.species.values]
    # Population-relevant central tendency across subdistricts.
    med = np.nanmedian(ds.annual_sustained.values, axis=1)
    order = np.argsort(med)
    fig, ax = plt.subplots(figsize=(7.2, 0.45 * len(sp) + 2.0))
    y = np.arange(len(sp))
    ax.barh(y, med[order], color=C_BLUE, zorder=3, height=0.62)
    ax.set_yticks(y); ax.set_yticklabels([sp[i] for i in order], fontsize=9)
    ax.set_xlabel("deaths · yr⁻¹ per 1000 kg emitted (median subdistrict)")
    ax.set_title(f"Marginal damages by species — {crf}", loc="left", pad=8)
    for yi, v in zip(y, med[order]):
        ax.annotate(f"{v:.3g}", xy=(v, yi), xytext=(4, 0),
                    textcoords="offset points", va="center",
                    fontsize=8, color=INK2)
    ax.margins(x=0.16)
    fig.tight_layout()
    p = Path(out_dir) / f"P1_species_ranking_{crf}.png"
    fig.savefig(p, dpi=200); plt.close(fig); ds.close()
    print(f"  wrote {p}")


def _small_multiples(values, sp, xlabels, xlabel, title, path, rotate=False):
    """values: (n_species, n_x). One hue; species separated by facet."""
    n = len(sp)
    ncol = min(4, n); nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.1 * ncol, 2.3 * nrow),
                             squeeze=False)
    x = np.arange(values.shape[1])
    for i in range(nrow * ncol):
        ax = axes[i // ncol][i % ncol]
        if i >= n:
            ax.axis("off"); continue
        ax.bar(x, values[i], color=C_BLUE, zorder=3, width=0.7)
        ax.set_title(sp[i], loc="left", fontsize=9, pad=4)
        ax.set_xticks(x)
        ax.set_xticklabels(xlabels, fontsize=6.5,
                           rotation=90 if rotate else 0)
        ax.tick_params(axis="y", labelsize=7)
        if i % ncol == 0:
            ax.set_ylabel("deaths / 1000 kg", fontsize=8)
    fig.suptitle(title, x=0.005, ha="left", fontsize=11)
    fig.supxlabel(xlabel, fontsize=9, color=INK2)
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    fig.savefig(path, dpi=200); plt.close(fig)
    print(f"  wrote {path}")


def fig_p2(prod_dir, out_dir, crf):
    ds = _load(prod_dir, crf)
    if ds is None:
        print(f"  P2: scenarios_{crf}.nc missing — skipped"); return
    sp = [str(s) for s in ds.species.values]
    v = np.nanmedian(ds.monthly_sustained.values, axis=2).T   # (species, month)
    _small_multiples(v, sp, MONTHS, "month the 1000 kg is emitted",
                     f"Seasonality of marginal damages — {crf}",
                     Path(out_dir) / f"P2_seasonality_{crf}.png", rotate=True)
    ds.close()


def fig_p3(prod_dir, out_dir, crf):
    ds = _load(prod_dir, crf)
    if ds is None:
        print(f"  P3: scenarios_{crf}.nc missing — skipped"); return
    sp = [str(s) for s in ds.species.values]
    # mean over months, median over subdistricts -> (species, bin)
    v = np.nanmedian(np.nanmean(ds.bin_specific.values, axis=0), axis=2).T
    labels = [f"{3*i:02d}h" for i in range(v.shape[1])]
    _small_multiples(v, sp, labels, "3-hour diurnal bin (IST) of emission",
                     f"Diurnal timing of marginal damages — {crf}",
                     Path(out_dir) / f"P3_diurnal_{crf}.png")
    ds.close()


def fig_p4(prod_dir, out_dir, crfs):
    got = [(c, _load(prod_dir, c)) for c in crfs]
    got = [(c, d) for c, d in got if d is not None]
    if len(got) < 2:
        print("  P4: need >=2 CRFs — skipped"); return
    sp = [str(s) for s in got[0][1].species.values]
    colors = [C_BLUE, C_ORANGE, C_AQUA]
    x = np.arange(len(sp)); w = 0.8 / len(got)
    fig, ax = plt.subplots(figsize=(1.05 * len(sp) + 3.2, 4.3))
    for k, (crf, ds) in enumerate(got):
        med = np.nanmedian(ds.annual_sustained.values, axis=1)
        ax.bar(x + (k - (len(got) - 1) / 2) * w, med, w,
               color=colors[k % len(colors)], zorder=3)
        ax.annotate(crf, xy=(0.0, 1.06 + 0.055 * k), xycoords="axes fraction",
                    color=colors[k % len(colors)], fontsize=9,
                    fontweight="bold")
        ds.close()
    ax.set_xticks(x); ax.set_xticklabels(sp, fontsize=8, rotation=20,
                                         ha="right")
    ax.set_ylabel("deaths · yr⁻¹ per 1000 kg (median subdistrict)")
    ax.set_title("Concentration-response function sensitivity",
                 loc="left", pad=8 + 12 * len(got))
    fig.tight_layout()
    p = Path(out_dir) / "P4_crf_sensitivity.png"
    fig.savefig(p, dpi=200); plt.close(fig)
    print(f"  wrote {p}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--production-dir", required=True)
    ap.add_argument("--out-dir", default="docs/figures/production")
    ap.add_argument("--crf", nargs="+", default=["gemm5cod", "gemmac", "ier"])
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    _style()
    qc_report(args.production_dir, args.out_dir, args.crf)
    print(f"\nWriting figures to {args.out_dir}/")
    for crf in args.crf:
        for fn in (fig_p1, fig_p2, fig_p3):
            try:
                fn(args.production_dir, args.out_dir, crf)
            except Exception as e:
                print(f"  {fn.__name__}({crf}) failed: {e}")
    try:
        fig_p4(args.production_dir, args.out_dir, args.crf)
    except Exception as e:
        print(f"  fig_p4 failed: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
