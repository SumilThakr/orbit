#!/usr/bin/env python
"""Figures for the FCT adjoint investigation (2026-08-03).

Four figures, paper/talk grade:

  F1  the measurement   — FCT outer-iteration residual vs iteration for
                          Anderson / Picard / GMRES, log-y, with the
                          (lambda-1)/lambda saturation line. Parsed live
                          from the Slurm logs of runs A/B/C/D.
  F2  the mechanism     — where the negative dJ_de cells sit relative to
                          the deaths gradient S. Map + a scatter that
                          makes the "negatives live where S ~ 0" claim
                          checkable rather than asserted.
  F3  the audit         — per-probe linear prediction vs nonlinear truth.
  F4  the panel         — six-cell truth vs both linearizations, and the
                          r spread. This is the figure that kills the
                          bracket hypothesis; it carries the deliverable
                          uncertainty statement.

Numbers come from scripts/fct_results_data.json (measured values with
job provenance); F1 and F2 read the run artifacts directly.

Usage:
    python scripts/make_fct_figures.py --out-dir docs/figures
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Categorical palette, validated colorblind-safe (dataviz validator,
# light surface: all checks PASS; worst adjacent CVD dE 9.1 protan).
# Contrast WARN on slots 3/4 is discharged by direct labels everywhere.
C_BLUE, C_ORANGE, C_AQUA, C_YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8a8a85"
SURFACE = "#fcfcfb"

DATAROOT = os.environ.get("ORBIT_DATA_ROOT", ".")


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


def _first_log(d):
    p = Path(DATAROOT) / "orbit_out/fct_stall_diag" / d / "logs"
    outs = sorted(p.glob("*.out")) if p.exists() else []
    return outs[0] if outs else None


def _parse_residuals(path, gmres=False):
    """Pull the outer-iteration residual trace out of a run log."""
    if path is None or not Path(path).exists():
        return []
    txt = Path(path).read_text(errors="ignore")
    pat = (r"outer GMRES iter \d+: rel resid = ([0-9.eE+-]+)" if gmres
           else r"deferred iter \d+: max rel .* change = ([0-9.eE+-]+)")
    return [float(x) for x in re.findall(pat, txt)]


def fig1(out_dir):
    """F1: the whole solver story in one panel."""
    series = [
        ("Anderson(5)", _parse_residuals(_first_log("A_anderson_k1e10")), C_BLUE, False),
        ("Picard", _parse_residuals(_first_log("C_picard_k1e6")), C_ORANGE, False),
        ("outer GMRES", _parse_residuals(_first_log("D_gmres_fix"), gmres=True), C_AQUA, False),
    ]
    series = [s for s in series if len(s[1]) > 1]
    if not series:
        print("  F1: no residual traces found — skipped")
        return
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for name, vals, color, _ in series:
        it = np.arange(1, len(vals) + 1)
        ax.semilogy(it, vals, color=color, lw=2, solid_capstyle="round")
        # direct label at the trace end (discharges the contrast WARN)
        ax.annotate(name, xy=(it[-1], vals[-1]), xytext=(4, 0),
                    textcoords="offset points", color=color, fontsize=9,
                    va="center", fontweight="bold")

    ax.axhline(0.2475, color=INK3, lw=1, ls=(0, (4, 3)))
    ax.annotate(r"Picard saturates at $(\lambda-1)/\lambda = 0.2475$"
                "\n" r"$\Rightarrow\ \lambda \approx +1.33$: $K$ is not a contraction",
                xy=(0.05, 0.62), xycoords="axes fraction",
                color=INK2, fontsize=8, linespacing=1.4)
    ax.axhline(2.0e-3, color=INK3, lw=1, ls=":")
    ax.annotate("target tolerance 2e-3", xy=(0.05, 2.0e-3),
                xycoords=("axes fraction", "data"), xytext=(0, -13),
                textcoords="offset points", color=INK2, fontsize=8)

    ax.set_xlabel("outer iteration (one inner adjoint solve each)")
    ax.set_ylabel(r"relative residual in $\mu$  ($\infty$-norm)")
    ax.set_title("Deferred-correction iteration cannot converge; the linear solve does",
                 loc="left", pad=8)
    n_max = max(len(s[1]) for s in series)
    ax.set_xlim(0, n_max * 1.22)          # headroom for the direct labels
    fig.tight_layout()
    p = Path(out_dir) / "F1_fct_residuals.png"
    fig.savefig(p, dpi=200); plt.close(fig)
    print(f"  wrote {p}")


def fig2(out_dir):
    """F2: negatives vs the deaths gradient — map + scatter."""
    import xarray as xr
    adj = Path(DATAROOT) / "orbit_out/fct_stall_diag/D_gmres_fix/adjoint_M01.nc"
    if not adj.exists():
        print("  F2: run-D adjoint missing — skipped")
        return
    ds = xr.open_dataset(adj)
    dj = ds.dJ_de.sel(species="PM25_primary").isel(draw=0).values.sum(axis=0)
    S = (ds.S_orbit.isel(draw=0).values if "draw" in ds.S_orbit.dims
         else ds.S_orbit.values)
    lat, lon = ds.lat.values, ds.lon.values
    ds.close()

    neg = dj < 0
    pop = S > 0
    f_all = 100 * neg.mean()
    f_pop = 100 * (neg & pop).sum() / pop.sum()
    f_mag = 100 * (np.abs(dj[neg & pop]).sum() / np.abs(dj[(~neg) & pop]).sum())

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(10.6, 4.4), gridspec_kw={"width_ratios": [1.5, 1]})

    # Left: where the negatives are, over the population-weighted gradient.
    Sp = np.where(S > 0, S, np.nan)
    ax1.pcolormesh(lon, lat, np.log10(Sp), cmap="Greys", shading="auto",
                   alpha=0.85, rasterized=True)
    yy, xx = np.where(neg)
    ax1.scatter(lon[xx], lat[yy], s=5, color=C_ORANGE, linewidths=0)
    ax1.set_xlabel("longitude"); ax1.set_ylabel("latitude")
    ax1.set_title("Negatives occupy the unpopulated ocean and periphery",
                  loc="left", pad=8)
    ax1.annotate("grey = deaths gradient $S$ (darker = more exposed people)",
                 xy=(0.02, -0.155), xycoords="axes fraction",
                 color=INK2, fontsize=8)
    ax1.set_aspect("equal", adjustable="box")
    ax1.grid(False)

    # Right: three hero numbers. The composition IS the finding — forcing
    # it into a chart with three different denominators would obscure it.
    ax2.axis("off")
    ax2.set_title("What they cost the damages estimate", loc="left", pad=8)
    tiles = [
        (f"{f_all:.1f}%", "of all grid cells carry a negative",
         "the raw, alarming-looking number", C_ORANGE),
        (f"{f_pop:.1f}%", "of POPULATED cells ($S>0$)",
         f"{100*(S[neg]==0).mean():.0f}% of negatives sit where $S=0$ exactly", C_ORANGE),
        (f"{f_mag:.2f}%", "of total $|\\partial J/\\partial e|$ over populated cells",
         "what they actually contribute to damages", C_BLUE),
    ]
    for i, (big, mid, sub, col) in enumerate(tiles):
        y = 0.88 - i * 0.33
        ax2.annotate(big, xy=(0.0, y), xycoords="axes fraction",
                     fontsize=26, color=col, fontweight="bold", va="top")
        ax2.annotate(mid, xy=(0.0, y - 0.115), xycoords="axes fraction",
                     fontsize=9, color=INK, va="top")
        ax2.annotate(sub, xy=(0.0, y - 0.175), xycoords="axes fraction",
                     fontsize=8, color=INK2, va="top")
    fig.tight_layout()
    p = Path(out_dir) / "F2_negatives_vs_gradient.png"
    fig.savefig(p, dpi=200); plt.close(fig)
    print(f"  wrote {p}")


def fig3(out_dir, data):
    """F3: FD audit — linear prediction vs nonlinear truth per probe."""
    probes = data["fd_audit_v2"]["probes"]
    labels = [p["label"] for p in probes]
    adj = np.array([p["dJ_adj"] for p in probes])
    nl = np.array([p["dJ_nonlin"] for p in probes])
    x = np.arange(len(probes)); w = 0.38

    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(9.6, 4.2), gridspec_kw={"width_ratios": [3, 1.25]})

    for ax, sel, ttl in ((axL, slice(0, 3), "The three negative probes"),
                         (axR, slice(3, 4), "Positive control")):
        xs = x[sel] - x[sel][0]
        ax.bar(xs - w/2, adj[sel], w, color=C_BLUE, zorder=3)
        ax.bar(xs + w/2, nl[sel], w, color=C_ORANGE, zorder=3)
        ax.axhline(0, color=INK, lw=1)
        ax.set_xticks(xs); ax.set_xticklabels(labels[sel], fontsize=8)
        ax.set_title(ttl, loc="left", pad=30)
        for xi, a, n in zip(xs, adj[sel], nl[sel]):
            for xo, v in ((-w/2, a), (w/2, n)):
                ax.annotate(f"{v:+.2f}", xy=(xi + xo, v),
                            xytext=(0, 4 if v >= 0 else -12),
                            textcoords="offset points", ha="center",
                            fontsize=7.5, color=INK2)
        lo, hi = ax.get_ylim()                   # headroom so labels clear
        ax.set_ylim(lo - 0.16 * (hi - lo), hi + 0.24 * (hi - lo))
    axL.set_ylabel("annualized deaths from the probe")
    # Series identity above the plot area, where no bar can reach it.
    axL.annotate("frozen-limiter linear prediction", xy=(0.0, 1.025),
                 xycoords="axes fraction", color=C_BLUE, fontsize=9,
                 fontweight="bold")
    axL.annotate("nonlinear model (truth)", xy=(0.45, 1.025),
                 xycoords="axes fraction", color=C_ORANGE, fontsize=9,
                 fontweight="bold")
    fig.suptitle("Finite-difference audit: the negatives are real but exaggerated",
                 x=0.005, ha="left", fontsize=11)
    # Note anchored to the FIGURE: axes-relative placement collided with
    # the two-line tick labels.
    fig.text(0.005, 0.035,
             "neg1, neg3: nonlinear agrees on the sign, attenuated ×12 and ×2.3    |    "
             "neg2: the linearization manufactures a negative (true response is positive)",
             color=INK2, fontsize=8)
    fig.tight_layout(rect=(0, 0.09, 1, 0.93))
    p = Path(out_dir) / "F3_fd_audit.png"
    fig.savefig(p, dpi=200); plt.close(fig)
    print(f"  wrote {p}")


def fig4(out_dir, data):
    """F4: the six-cell panel — the bracket fails; the mean is unbiased."""
    cells = data["panel"]["cells"]
    s = data["panel"]["summary"]
    names = [c["name"] for c in cells]
    truth = np.array([c["truth"] for c in cells])
    froz = np.array([c["frozen_fct"] for c in cells])
    low = np.array([c["low_order"] for c in cells])
    x = np.arange(len(cells)); w = 0.27

    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(11.0, 4.4), gridspec_kw={"width_ratios": [1.55, 1]})

    axL.bar(x - w, low, w, color=C_AQUA, zorder=3)
    axL.bar(x, truth, w, color=INK2, zorder=3)
    axL.bar(x + w, froz, w, color=C_BLUE, zorder=3)
    axL.set_xticks(x); axL.set_xticklabels(names, fontsize=8.5)
    axL.set_ylabel("annualized deaths (q = 1% probe)")
    axL.set_title("Truth vs both linearizations, six cells", loc="left", pad=8)
    axL.set_ylim(0, max(truth.max(), froz.max(), low.max()) * 1.30)
    # Series identity above the plot area, clear of the tallest bar.
    for xf, lab, col in ((0.0, "low-order", C_AQUA),
                         (0.20, "nonlinear truth", INK2),
                         (0.50, "frozen-limiter", C_BLUE)):
        axL.annotate(lab, xy=(xf, 1.10), xycoords="axes fraction",
                     color=col, fontsize=9, fontweight="bold")
    # mark the cells where truth escapes both operators
    for xi, c, t in zip(x, cells, truth):
        if not c["bracket"]:
            axL.annotate("✗", xy=(xi, t), xytext=(0, 11),
                         textcoords="offset points", ha="center",
                         color=C_ORANGE, fontsize=12, fontweight="bold")
    axL.annotate("✗ = truth exceeds BOTH operators (no bracket)",
                 xy=(0.02, 0.955), xycoords="axes fraction",
                 color=C_ORANGE, fontsize=8.5)

    # Right: the ratio that carries the uncertainty statement.
    r_hi = truth / froz
    order = np.argsort(r_hi)
    yy = np.arange(len(cells))
    axR.hlines(yy, 1.0, r_hi[order], color=INK3, lw=1)
    axR.scatter(r_hi[order], yy, s=46, color=C_BLUE, zorder=3)
    axR.axvline(1.0, color=INK, lw=1.2)
    axR.axvline(s["r_hi_mean"], color=C_ORANGE, lw=1.5, ls=(0, (4, 3)))
    axR.set_yticks(yy)
    axR.set_yticklabels([names[i] for i in order], fontsize=8.5)
    axR.set_xlabel("truth / frozen-limiter prediction")
    axR.set_title("Cell error is two-signed", loc="left", pad=8)
    # Labels sit on the far side of each lollipop from the x=1 anchor, so
    # they can never collide with the unity or panel-mean rules.
    for yi, v in zip(yy, r_hi[order]):
        away = 9 if v >= 1.0 else -9
        axR.annotate(f"{v:.2f}", xy=(v, yi), xytext=(away, -3),
                     textcoords="offset points",
                     ha="left" if v >= 1.0 else "right",
                     fontsize=8, color=INK2)
    axR.set_ylim(-1.35, len(cells) - 0.35)
    axR.margins(x=0.16)
    axR.annotate(f"panel mean {s['r_hi_mean']:.2f}", xy=(s["r_hi_mean"], 0.90),
                 xycoords=("data", "axes fraction"), xytext=(6, 0),
                 textcoords="offset points",
                 color=C_ORANGE, fontsize=8.5, ha="left", fontweight="bold")
    axR.annotate("under-predicts →", xy=(0.98, 0.02), xycoords="axes fraction",
                 ha="right", color=INK2, fontsize=8)
    axR.annotate("← over-predicts", xy=(0.02, 0.02), xycoords="axes fraction",
                 color=INK2, fontsize=8)

    fig.suptitle(
        "No two-sided bracket: single-cell per-ton error spans "
        f"{s['r_hi_min']:.2f}–{s['r_hi_max']:.2f}, but the panel mean is unbiased "
        f"({s['r_hi_mean']:.2f} ± {s['r_hi_sem']:.2f}, n={s['n']})",
        x=0.005, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    p = Path(out_dir) / "F4_panel_bracket.png"
    fig.savefig(p, dpi=200); plt.close(fig)
    print(f"  wrote {p}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", default="docs/figures")
    ap.add_argument("--data", default="scripts/fct_results_data.json")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    data = json.loads(Path(args.data).read_text())
    _style()
    print(f"Writing figures to {args.out_dir}/")
    fig1(args.out_dir)
    fig2(args.out_dir)
    fig3(args.out_dir, data)
    fig4(args.out_dir, data)
    return 0


if __name__ == "__main__":
    sys.exit(main())
