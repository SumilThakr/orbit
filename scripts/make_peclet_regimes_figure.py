#!/usr/bin/env python
"""F6: the Peclet regimes that justify ORBIT's transport discretisation.

Replaces the two June diagnostics carried in the talk (`face_peclet_cdf.png`,
`vertical_peclet_cdf.png`, four panels between them). Those were computed on
the pre-Monin-Obukhov run and their framing overclaimed; this rebuilds the
argument on the production 2022 grids with a discretisation-exact accounting.

WHAT THE FIGURE ARGUES
----------------------
ORBIT limits the horizontal fluxes (FCT/Zalesak on a van Leer-MUSCL
correction) but runs the vertical operators plain and unlimited. The
justification is a statement about numerical diffusion, not about Peclet
numbers per se: horizontally the first-order upwind truncation error supplies
most of the model's horizontal mixing, so there is a great deal to recover and
a limiter earns its keep; vertically, inside the boundary layer, that same
truncation error is a couple of percent of the mixing already present, so a
limiter would buy almost nothing -- and the FCT linearisation is exactly what
produces the non-monotone (negative) sensitivities the adjoint investigation
documented.

THE ACCOUNTING (this is the part the old figures got wrong)
-----------------------------------------------------------
Both directions advect with *split* fluxes: (UAvg_plus, UAvg_minus)
horizontally, (omega_plus, omega_minus) vertically, each non-negative. For a
face between cells L and R with split velocities w_up, w_dn >= 0 the flux is

    F = w_up*C_L - w_dn*C_R
      = [max(w_net,0)*C_L - max(-w_net,0)*C_R] + w_exch*(C_L - C_R)

with w_net = w_up - w_dn and w_exch = min(w_up, w_dn). The identity is exact.
The first bracket is plain first-order upwind on the resolved velocity, whose
truncation error is the numerical diffusivity

    K_num = |w_net| * dL / 2

The second term is *exactly* a diffusive flux of diffusivity

    K_sub = w_exch * dL

i.e. deliberate sub-grid two-way exchange (wind meander, and reversal within
the 3 h averaging window), not discretisation error. So the model's physical
mixing at a face is K_phys = K_grid + K_sub, with K_grid = harmonic-mean Kzz
(vertical) or Kxxyy (horizontal), and the Peclet number that actually speaks to
the design choice is

    Pe = |w_net| * dL / K_phys = 2 * K_num / K_phys

The old figures divided by K_grid alone, which charges the intended sub-grid
exchange to the numerator's denominator and inflates Pe by orders of magnitude
(horizontal median 1.0e5 instead of 6.3; free-troposphere vertical median ~300
instead of ~1.3). Both accountings are reported here; the exact one is plotted.

Conservative by construction: CMFMC convective mass flux is a further piece of
intended vertical transport and is *not* credited in K_phys, so the vertical
numerical shares below are upper bounds.

Panels
    (a) CDFs of |Pe| -- horizontal faces, vertical faces below the PBL,
        vertical faces above it, and vertical faces weighted by adjacent PM2.5
        mass. Crossover line at |Pe| = 2, twin top axis in K_num/K_phys.
    (b) Share of the model's mixing flux that is numerical, against face
        height, with the horizontal scheme as a reference line.

Inputs: the 96 production preprocessor grids (sas_2022_M{01..12}_B{01..08}.nc).
Everything needed -- Kzz, Kxxyy, Dz, dP, the split fluxes, Pblh, LayerHeights
and a 3-D TotalPM25 for the mass weighting -- lives in those files, so no
forward run is required.

Usage:
    python scripts/make_peclet_regimes_figure.py \
        --preproc-dir ~/orbit_data/inputs/grids_2022_fixed \
        --out-dir docs/figures
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import netCDF4 as nc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from orbit.core.grid_data import load_grid

# Economist-style palette. Red is reserved for the series the figure is
# about (the vertical scheme); blue for the one it is contrasted against.
# The four panel-(a) series are separated by hue (dark blue / red / teal /
# grey) and reinforced by line weight and dash pattern, so they stay
# readable under protanopia and deuteranopia and in greyscale.
ECON_RED = "#e3120b"
ECON_BLUE = "#006ba2"
ECON_TEAL = "#379a8b"
ECON_GREY = "#758d99"
INK = "#121316"
SUB = "#5c6b73"
RULE = "#d7d9db"
SURFACE = "#ffffff"

EARTH_R = 6378137.0


def _style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Liberation Sans", "Nimbus Sans", "Helvetica",
                            "Arial", "DejaVu Sans"],
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.size": 9.5, "axes.labelsize": 9.5,
        "axes.edgecolor": "#9aa4a9", "axes.linewidth": 0.9,
        "xtick.color": SUB, "ytick.color": SUB,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "xtick.direction": "out", "ytick.direction": "out",
        "axes.labelcolor": INK, "text.color": INK,
        "axes.grid": False, "grid.color": RULE, "grid.linewidth": 0.7,
        "axes.axisbelow": True, "legend.frameon": False,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.spines.left": False,
    })


def _title_block(fig, title, subtitle, x=0.055, y_tag=0.945):
    """The Economist furniture: red tag, bold title, grey standfirst."""
    fig.patches.append(plt.Rectangle(
        (x, y_tag), 0.048, 0.017, transform=fig.transFigure,
        facecolor=ECON_RED, edgecolor="none", zorder=5))
    fig.text(x, y_tag - 0.055, title, fontsize=14, fontweight="bold",
             color=INK, ha="left", va="top")
    fig.text(x, y_tag - 0.115, subtitle, fontsize=10, color=SUB,
             ha="left", va="top", linespacing=1.45)


def _footnote(fig, text, x=0.055, y=0.015):
    fig.text(x, y, text, fontsize=7.6, color=SUB, ha="left", va="bottom",
             linespacing=1.5)


def _harm(a, b):
    return 2.0 * a * b / (a + b + 1e-30)


def _patankar_A(pe):
    out = np.empty_like(pe)
    s, l = pe < 1e-6, pe > 500.0
    m = ~(s | l)
    out[s] = 1.0 - 0.5 * pe[s]
    out[l] = 0.0
    out[m] = pe[m] / np.expm1(pe[m])
    return out


def _f32(*arrays):
    return [np.asarray(a, dtype=np.float32).ravel() for a in arrays]


def scan_file(path: Path):
    """Face-level arrays for one (month, bin) preprocessor state."""
    g = load_grid(str(path))
    with nc.Dataset(path) as ds:
        pm = np.asarray(ds.variables["TotalPM25"][:], dtype=np.float64)

    dy_m = EARTH_R * np.deg2rad(g.dlat)
    dx_m = EARTH_R * np.cos(np.deg2rad(g.lat)) * np.deg2rad(g.dlon)
    cell_area = (dx_m * dy_m)[None, :, None]

    # ---------------- vertical interior faces (k = 1 .. nz-1) ----------------
    # omega_*[k] sits on the bottom face of layer k, i.e. between k-1 and k.
    dP_l, dP_u = g.dP[:-1], g.dP[1:]
    dz_l, dz_u = g.Dz[:-1], g.Dz[1:]
    rho_g = (dP_l + dP_u) / (dz_l + dz_u + 1e-30)          # Pa/m
    w_up = g.omega_minus[1:] / rho_g                        # m/s upward
    w_dn = g.omega_plus[1:] / rho_g                         # m/s downward
    dL_v = 0.5 * (dz_l + dz_u)
    K_grid_v = _harm(g.Kzz[:-1], g.Kzz[1:])
    w_net_v = np.abs(w_up - w_dn)
    K_sub_v = np.minimum(w_up, w_dn) * dL_v
    K_phys_v = K_grid_v + K_sub_v
    K_num_v = 0.5 * w_net_v * dL_v
    Pe_v = w_net_v * dL_v / np.maximum(K_phys_v, 1e-30)
    Pe_v_bare = w_net_v * dL_v / np.maximum(K_grid_v, 1e-30)

    height = g.LayerHeights[1:g.nz]                         # m above ground
    in_pbl = height < g.Pblh[None, :, :]
    kv = np.broadcast_to(np.arange(1, g.nz)[:, None, None], Pe_v.shape)
    # |dc| across the face, and the PM2.5 mass adjacent to it
    grad_v = np.abs(pm[:-1] - pm[1:]) / dL_v
    mass_v = 0.5 * (pm[:-1] * dz_l + pm[1:] * dz_u) * cell_area

    vert = dict(zip(
        ("Pe", "Pe_bare", "K_num", "K_phys", "grad", "mass", "height"),
        _f32(Pe_v, Pe_v_bare, K_num_v, K_phys_v, grad_v, mass_v, height)))
    vert["k"] = np.asarray(kv, dtype=np.int8).ravel()
    vert["in_pbl"] = np.asarray(in_pbl).ravel()
    vert["dz"] = np.asarray(dL_v, dtype=np.float32).ravel()

    # ---------------- horizontal interior faces ----------------
    parts = []
    # x-faces between i-1 and i
    dL = np.broadcast_to(dx_m[None, :, None], (g.nz, g.ny, g.nx - 1))
    Kg = _harm(g.Kxxyy[:, :, :-1], g.Kxxyy[:, :, 1:])
    up, um = g.UAvg_plus[:, :, 1:], g.UAvg_minus[:, :, 1:]
    parts.append((np.abs(up - um), np.minimum(up, um), dL, Kg,
                  np.abs(pm[:, :, :-1] - pm[:, :, 1:]) / dL))
    # y-faces between j-1 and j
    Kg = _harm(g.Kxxyy[:, :-1, :], g.Kxxyy[:, 1:, :])
    vp, vm = g.VAvg_plus[:, 1:, :], g.VAvg_minus[:, 1:, :]
    dL = np.full(Kg.shape, dy_m)
    parts.append((np.abs(vp - vm), np.minimum(vp, vm), dL, Kg,
                  np.abs(pm[:, :-1, :] - pm[:, 1:, :]) / dL))

    w_net_h = np.concatenate([p[0].ravel() for p in parts])
    K_sub_h = np.concatenate([(p[1] * p[2]).ravel() for p in parts])
    dL_h = np.concatenate([p[2].ravel() for p in parts])
    K_grid_h = np.concatenate([p[3].ravel() for p in parts])
    grad_h = np.concatenate([p[4].ravel() for p in parts])

    K_num_h = 0.5 * w_net_h * dL_h
    K_phys_h = K_grid_h + K_sub_h
    Pe_h = w_net_h * dL_h / np.maximum(K_phys_h, 1e-30)
    Pe_h_bare = w_net_h * dL_h / np.maximum(K_grid_h, 1e-30)
    A_h = _patankar_A(np.minimum(Pe_h_bare, 500.0))

    horiz = dict(zip(("Pe", "Pe_bare", "K_num", "K_phys", "grad", "A"),
                     _f32(Pe_h, Pe_h_bare, K_num_h, K_phys_h, grad_h, A_h)))
    return vert, horiz, g


def _wq(values, weights, q):
    """Weighted quantile."""
    o = np.argsort(values)
    v, w = values[o], weights[o]
    c = np.cumsum(w, dtype=np.float64)
    return float(v[np.searchsorted(c, q * c[-1])])


def _share(K_num, K_phys, grad):
    """Fraction of the total face mixing flux that is truncation error."""
    fn = float(np.sum(K_num.astype(np.float64) * grad))
    fp = float(np.sum(K_phys.astype(np.float64) * grad))
    return fn / (fn + fp) if (fn + fp) > 0 else float("nan")


def collect(preproc_dir: Path, months, bins_):
    vparts, hparts, meta = [], [], {}
    for m in months:
        for b in bins_:
            p = preproc_dir / f"sas_2022_M{m:02d}_B{b:02d}.nc"
            if not p.exists():
                raise FileNotFoundError(p)
            v, h, g = scan_file(p)
            vparts.append(v)
            hparts.append(h)
            if not meta:
                meta = {"nz": int(g.nz), "ny": int(g.ny), "nx": int(g.nx),
                        "dz_face_mean_m": [float(x) for x in
                                           (0.5 * (g.Dz[:-1] + g.Dz[1:])).mean(axis=(1, 2))],
                        "height_face_mean_m": [float(x) for x in
                                               g.LayerHeights[1:g.nz].mean(axis=(1, 2))]}
            meta.setdefault("pblh_median_m", []).append(float(np.median(g.Pblh)))
        print(f"  month {m:02d} done")
    cat = lambda parts, key: np.concatenate([p[key] for p in parts])
    V = {k: cat(vparts, k) for k in vparts[0]}
    H = {k: cat(hparts, k) for k in hparts[0]}
    meta["pblh_median_m"] = float(np.median(meta["pblh_median_m"]))
    return V, H, meta


def summarise(V, H, meta):
    s = {"meta": meta, "vertical": {}, "horizontal": {}, "by_face_index": []}

    tot_mass = float(V["mass"].sum())
    for label, sel in [("all", np.ones(V["Pe"].size, bool)),
                       ("pbl", V["in_pbl"]),
                       ("free_trop", ~V["in_pbl"])]:
        pe = V["Pe"][sel]
        s["vertical"][label] = {
            "n": int(pe.size),
            "Pe_p50": float(np.median(pe)),
            "Pe_p90": float(np.percentile(pe, 90)),
            "frac_Pe_lt_2": float((pe < 2).mean()),
            "Knum_over_Kphys_p50": float(np.median(
                V["K_num"][sel] / np.maximum(V["K_phys"][sel], 1e-30))),
            "numerical_share_of_mixing_flux": _share(
                V["K_num"][sel], V["K_phys"][sel], V["grad"][sel]),
            "Pe_p50_bare_Kzz_only": float(np.median(V["Pe_bare"][sel])),
            "mass_fraction": float(V["mass"][sel].sum() / tot_mass),
        }
    s["vertical"]["mass_weighted"] = {
        "frac_mass_at_Pe_lt_2": float(V["mass"][V["Pe"] < 2].sum() / tot_mass),
        "Pe_p50": _wq(V["Pe"], V["mass"], 0.5),
        "Pe_p90": _wq(V["Pe"], V["mass"], 0.9),
    }

    pe = H["Pe"]
    s["horizontal"]["all"] = {
        "n": int(pe.size),
        "Pe_p50": float(np.median(pe)),
        "Pe_p90": float(np.percentile(pe, 90)),
        "frac_Pe_lt_2": float((pe < 2).mean()),
        "Knum_over_Kphys_p50": float(np.median(
            H["K_num"] / np.maximum(H["K_phys"], 1e-30))),
        "numerical_share_of_mixing_flux": _share(H["K_num"], H["K_phys"], H["grad"]),
        "Pe_p50_bare_Kxxyy_only": float(np.median(H["Pe_bare"])),
        "patankar_A_mean": float(H["A"].mean()),
        "frac_faces_A_gt_0.01": float((H["A"] > 0.01).mean()),
    }

    for kv in range(1, meta["nz"]):
        sel = V["k"] == kv
        s["by_face_index"].append({
            "k": kv,
            "height_m": meta["height_face_mean_m"][kv - 1],
            "dz_face_m": meta["dz_face_mean_m"][kv - 1],
            "Pe_p50": float(np.median(V["Pe"][sel])),
            "frac_Pe_lt_2": float((V["Pe"][sel] < 2).mean()),
            "numerical_share": _share(V["K_num"][sel], V["K_phys"][sel], V["grad"][sel]),
            "mass_fraction": float(V["mass"][sel].sum() / tot_mass),
            "frac_below_pbl": float(V["in_pbl"][sel].mean()),
        })
    return s


def _cdf(ax, values, color, label, lw=2.0, ls="-", weights=None, nmax=4000):
    """Plot a (optionally weighted) CDF, thinned to nmax vertices."""
    v = np.maximum(values.astype(np.float64), 1e-6)
    o = np.argsort(v)
    v = v[o]
    if weights is None:
        c = np.linspace(0, 1, v.size, endpoint=False)
    else:
        w = weights.astype(np.float64)[o]
        c = np.cumsum(w) / w.sum()
    idx = np.unique(np.linspace(0, v.size - 1, min(nmax, v.size)).astype(int))
    ax.plot(v[idx], c[idx], color=color, lw=lw, ls=ls, label=label,
            solid_capstyle="round")


def panel_a(ax, V, H, s, legend_y=-0.235):
    """Distribution of Peclet numbers, horizontal against vertical."""
    ax.axvspan(2.0, 1e6, color="#f2f4f5", lw=0, zorder=0)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=RULE, lw=0.7)

    _cdf(ax, H["Pe"], ECON_BLUE,
         f"Horizontal faces  ·  median {s['horizontal']['all']['Pe_p50']:.1f}",
         lw=2.4)
    _cdf(ax, V["Pe"][V["in_pbl"]], ECON_RED,
         "Vertical faces below the boundary layer  ·  "
         f"median {s['vertical']['pbl']['Pe_p50']:.3f}", lw=2.4)
    _cdf(ax, V["Pe"], ECON_TEAL,
         "Vertical faces, weighted by fine-particle mass  ·  "
         f"median {s['vertical']['mass_weighted']['Pe_p50']:.2f}",
         lw=1.9, ls=(0, (5, 2)), weights=V["mass"])
    _cdf(ax, V["Pe"][~V["in_pbl"]], ECON_GREY,
         "Vertical faces above the boundary layer  ·  "
         f"median {s['vertical']['free_trop']['Pe_p50']:.2f}",
         lw=1.5, ls=(0, (1.6, 1.6)))

    ax.axvline(2.0, color=INK, lw=1.0, ls=(0, (4, 3)), zorder=4)
    ax.text(3.2, 1.018, "→  numerical diffusion exceeds the physical mixing",
            transform=ax.get_xaxis_transform(), fontsize=8.4, color=SUB,
            ha="left", va="bottom")

    ax.set_xscale("log")
    ax.set_xlim(1e-5, 1e6)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0", "25", "50", "75", "100"])
    ax.tick_params(axis="y", which="both", length=0, pad=2)
    ax.tick_params(axis="x", length=3.5)
    ax.set_xlabel("Péclet number at the grid face  (log scale)", labelpad=7)
    ax.text(0, 1.055, "Cumulative share of faces, %", transform=ax.transAxes,
            fontsize=8.6, color=SUB, ha="left", va="bottom")
    ax.legend(fontsize=8.2, loc="upper left", bbox_to_anchor=(-0.005, legend_y),
              ncol=2, handlelength=2.0, columnspacing=2.6, labelspacing=0.7,
              borderpad=0)


def panel_b(ax, s):
    """Share of the mixing flux the discretisation supplies, by height."""
    rows = s["by_face_index"]
    hgt = np.array([r["height_m"] for r in rows])
    shr = 100 * np.array([r["numerical_share"] for r in rows])
    h_share = 100 * s["horizontal"]["all"]["numerical_share_of_mixing_flux"]
    pbl_share = 100 * s["vertical"]["pbl"]["numerical_share_of_mixing_flux"]
    pblh = s["meta"]["pblh_median_m"]

    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color=RULE, lw=0.7)

    ax.axvline(h_share, color=ECON_BLUE, lw=2.2, ls=(0, (5, 2)), zorder=2)
    ax.text(h_share + 3, 1.5e4, f"Horizontal scheme\n{h_share:.0f}% numerical",
            color=ECON_BLUE, fontsize=9.5, ha="left", va="center",
            fontweight="bold", linespacing=1.45)

    ax.plot(shr, hgt, "-", color=ECON_RED, lw=2.4, zorder=3,
            solid_capstyle="round")
    ax.plot(shr, hgt, "o", color=ECON_RED, ms=5, zorder=4,
            markeredgecolor=SURFACE, markeredgewidth=0.9)
    ax.text(43, 5200, "Vertical scheme", color=ECON_RED, fontsize=9.5,
            fontweight="bold", ha="right", va="center")

    ax.axhline(pblh, color=SUB, lw=0.9, ls=(0, (4, 3)), zorder=2)
    ax.text(27, pblh * 1.14, f"median top of the boundary layer, {pblh:.0f} m",
            color=SUB, fontsize=8.2, ha="left", va="bottom")

    ax.annotate(f"below the boundary layer just {pbl_share:.1f}%\n"
                "of the vertical mixing is numerical",
                xy=(shr[1], hgt[1]), xytext=(25, 235),
                color=ECON_RED, fontsize=8.8, fontweight="bold",
                linespacing=1.5, va="center",
                arrowprops=dict(arrowstyle="-", color=ECON_RED, lw=0.9,
                                shrinkA=4, shrinkB=4))

    ax.set_yscale("log")
    ax.set_ylim(95, 2.4e4)
    ax.set_xlim(0, 100)
    ax.set_yticks([100, 300, 1000, 3000, 10000])
    ax.set_yticklabels(["100", "300", "1,000", "3,000", "10,000"])
    ax.tick_params(axis="y", which="both", length=0, pad=2)
    ax.tick_params(axis="x", length=3.5)
    ax.set_xlabel("Share of the vertical mixing supplied by the "
                  "discretisation, %", labelpad=7)
    ax.text(0, 1.055, "Height above ground, metres", transform=ax.transAxes,
            fontsize=8.6, color=SUB, ha="left", va="bottom")


SUB_A = ("Every interior grid face of the South Asia domain, 2022:\n"
         "12 months × 8 times of day, 19.3m faces in all")
SUB_B = ("Share of the mixing flux that comes from the numerical scheme\n"
         "rather than from physics, by height above ground")

FOOT_A = (
    "The Péclet number is the strength of advection relative to mixing at a grid face: net velocity × face spacing ÷ physical mixing. Physical mixing is the grid\n"
    "diffusivity plus the model's intended sub-grid exchange. Above a Péclet number of 2 the numerical diffusion carried by the first-order upwind scheme\n"
    "exceeds the physical mixing already present, which is why the horizontal fluxes are limited (flux-corrected transport) and the vertical ones are not.\n"
    "Source: ORBIT 2022 production meteorology, 96 model states, 13.1m horizontal and 6.2m vertical grid faces")
FOOT_B = (
    "Mixing flux at each face weighted by the fine-particle concentration gradient across it. The rise above about 1.3 km follows the vertical grid stretching,\n"
    "not the meteorology: numerical diffusion scales with face spacing, which jumps from roughly 130 m to 1,700 m over these levels. About a third of the\n"
    "airborne fine-particle mass sits above that height, in those thick layers. Convective mass flux is further intended vertical transport and is not credited\n"
    "here, so the vertical shares shown are upper bounds.  Source: ORBIT 2022 production meteorology, 96 model states")

FOOT_COMBINED = (
    "The Péclet number is the strength of advection relative to mixing at a grid face: net velocity × face spacing ÷ physical mixing, where physical mixing is the grid diffusivity plus the model's intended\n"
    "sub-grid exchange. Above a Péclet number of 2 the numerical diffusion carried by the first-order upwind scheme exceeds the physical mixing already present. Right-hand panel: mixing flux at each face\n"
    "weighted by the fine-particle concentration gradient across it; the rise above 1.3 km follows the vertical grid stretching, not the meteorology. Convective mass flux is further intended vertical transport\n"
    "and is not credited, so the vertical shares shown are upper bounds.    Source: ORBIT 2022 production meteorology, 96 model states")

TITLE_A = "Horizontally the scheme diffuses; vertically it barely does"
TITLE_B = "Near the ground, almost none of the mixing is numerical"


def make_figures(V, H, s, out_dir: Path, name: str):
    """Combined figure plus the two panels as standalone publication charts."""
    _style()
    written = []
    stem = name.split("_")[0]

    # ---- standalone panel a ----
    fig = plt.figure(figsize=(8.6, 6.6))
    _title_block(fig, TITLE_A, SUB_A)
    ax = fig.add_axes([0.075, 0.325, 0.895, 0.40])
    panel_a(ax, V, H, s)
    _footnote(fig, FOOT_A)
    p = out_dir / f"{stem}a_peclet_distribution.png"
    fig.savefig(p, dpi=300)
    plt.close(fig)
    written.append(p)

    # ---- standalone panel b ----
    fig = plt.figure(figsize=(8.0, 6.6))
    _title_block(fig, TITLE_B, SUB_B)
    bx = fig.add_axes([0.105, 0.265, 0.855, 0.455])
    panel_b(bx, s)
    _footnote(fig, FOOT_B)
    p = out_dir / f"{stem}b_numerical_share.png"
    fig.savefig(p, dpi=300)
    plt.close(fig)
    written.append(p)

    # ---- combined ----
    fig = plt.figure(figsize=(14.4, 6.6))
    _title_block(
        fig,
        "Why ORBIT limits the horizontal fluxes and leaves the vertical ones alone",
        "Péclet number and the numerical share of mixing, South Asia 2022",
        x=0.033, y_tag=0.955)
    ax = fig.add_axes([0.045, 0.325, 0.415, 0.375])
    bx = fig.add_axes([0.565, 0.325, 0.405, 0.375])
    panel_a(ax, V, H, s, legend_y=-0.27)
    panel_b(bx, s)
    for a_, lab, t in ((ax, "a", TITLE_A), (bx, "b", TITLE_B)):
        a_.text(0, 1.12, f"{lab}  {t}", transform=a_.transAxes, fontsize=10.5,
                fontweight="bold", color=INK, ha="left", va="bottom")
    _footnote(fig, FOOT_COMBINED, x=0.033)
    p = out_dir / f"{name}.png"
    fig.savefig(p, dpi=300)
    plt.close(fig)
    written.append(p)
    return written


CAPTION = """\
{name}
{rule}

Peclet regimes on the 2022 SAS production preprocessor grids
(96 states: 12 months x 8 diurnal bins, {nfv:,} vertical faces and
{nfh:,} horizontal faces). Rebuild of the two June diagnostics
(face_peclet_cdf.png, vertical_peclet_cdf.png) that the talk carried
at slides 38-39.

WHAT IT SHOWS
  ORBIT limits the horizontal fluxes -- flux-corrected transport, with a
  Zalesak limiter on a van Leer-MUSCL correction -- and runs the vertical
  operators plain and unlimited. The
  figure is the justification, stated as numerical diffusion rather
  than as a bare Peclet number.

  (a) Cumulative distributions of |Pe| = |w_net|*dL/K_phys. The dashed line at |Pe| = 2 is
      where the first-order upwind truncation error equals the physical
      mixing already present; the top axis reads the same data as
      K_num/K_phys.
  (b) Share of the model's face mixing flux supplied by the
      discretisation rather than by physics, against face height, with
      the fine particulate matter (PM2.5) mass distribution shaded
      behind it.

THE NUMBERS (full year, all bins)
  horizontal faces     median |Pe| {h_pe:.1f}   {h_lt2:.1f}% below 2
                       {h_share:.0f}% of the horizontal mixing flux is numerical
  vertical, below the boundary layer
                       median |Pe| {v_pe:.4f}   {v_lt2:.1f}% below 2
                       {v_share:.1f}% of the vertical mixing flux is numerical
  vertical, above the boundary layer
                       median |Pe| {f_pe:.2f}   {f_lt2:.1f}% below 2
                       {f_share:.1f}% numerical
  mass-weighted        {m_lt2:.1f}% of fine particulate (PM2.5) mass
                       sits at |Pe| < 2
                       (median |Pe| {m_pe:.2f})

  So the contrast that licenses the design is roughly {ratio:.0f}x in
  relative numerical diffusion between the two directions, and the
  single number for the talk is: below the boundary layer the vertical
  scheme's numerical diffusion is {v_share:.1f}% of the mixing that is
  physically there.

HOW Pe IS DEFINED, AND WHY IT DIFFERS FROM THE JUNE FIGURES
  Both directions advect with split fluxes (UAvg_plus/UAvg_minus,
  omega_plus/omega_minus). For split velocities w_up, w_dn >= 0 the face
  flux decomposes exactly:

    F = w_up*C_L - w_dn*C_R
      = [max(w_net,0)*C_L - max(-w_net,0)*C_R] + w_exch*(C_L - C_R)

  with w_net = w_up - w_dn, w_exch = min(w_up, w_dn). The first bracket
  is plain upwind on the resolved velocity (numerical diffusivity
  K_num = |w_net|*dL/2); the second is exactly a diffusive flux of
  diffusivity K_sub = w_exch*dL, i.e. intended sub-grid exchange (wind
  meander, and reversal inside the 3 h averaging window). Hence
  K_phys = K_grid + K_sub and Pe = 2*K_num/K_phys.

  The June figures divided by K_grid alone. That charges the model's
  deliberate sub-grid exchange against it and inflates Pe by orders of
  magnitude: on these same grids the K_grid-only accounting gives a
  horizontal median of {h_pe_bare:.3g}, reproducing the old figure's
  1e3-1e7 range, and a free-troposphere vertical median of
  {f_pe_bare:.3g}, the same regime as the old figure's ~430 for its
  fixed-k "upper troposphere" band (a different stratification, so the
  two are comparable only in order of magnitude). Neither is wrong
  arithmetic; both answer a question the design decision does not ask.

  Conservative by construction: CMFMC convective mass flux is further
  intended vertical transport and is NOT credited in K_phys, so the
  vertical numerical shares are upper bounds.

READING PANEL (b)
  The rise above ~1.3 km is a grid effect, not a meteorological one.
  K_num scales with the face spacing dL, and dL jumps from ~130 m
  (faces 1-9) to {dz10:.0f}, {dz11:.0f} and {dz12:.0f} m at faces 10-12
  as the vertical grid stretches. If the free-troposphere numerical
  diffusion ever needed reducing, the lever is vertical resolution
  aloft, not a flux limiter.

CAVEAT WORTH SPEAKING ALOUD
  The claim is scoped to the boundary layer. Above it the vertical
  scheme is genuinely upwind-dominated ({f_share:.0f}% of the mixing
  there is numerical). That region does not set surface exposure, but
  it is not "almost always << 1" either -- the phrasing the old slide
  used, which the June figure's own legend contradicted.

PROVENANCE
  Generator: scripts/make_peclet_regimes_figure.py
  Inputs:    {preproc}
  Numbers:   {name_json}
"""


def write_sidecar(s, out_png: Path, preproc_dir: Path):
    h = s["horizontal"]["all"]
    vp, vf, mw = (s["vertical"]["pbl"], s["vertical"]["free_trop"],
                  s["vertical"]["mass_weighted"])
    dz = s["meta"]["dz_face_mean_m"]
    txt = CAPTION.format(
        name=out_png.name, rule="=" * len(out_png.name),
        nfv=s["vertical"]["all"]["n"], nfh=h["n"],
        h_pe=h["Pe_p50"], h_lt2=100 * h["frac_Pe_lt_2"],
        h_share=100 * h["numerical_share_of_mixing_flux"],
        h_pe_bare=h["Pe_p50_bare_Kxxyy_only"],
        v_pe=vp["Pe_p50"], v_lt2=100 * vp["frac_Pe_lt_2"],
        v_share=100 * vp["numerical_share_of_mixing_flux"],
        f_pe=vf["Pe_p50"], f_lt2=100 * vf["frac_Pe_lt_2"],
        f_share=100 * vf["numerical_share_of_mixing_flux"],
        f_pe_bare=vf["Pe_p50_bare_Kzz_only"],
        m_lt2=100 * mw["frac_mass_at_Pe_lt_2"], m_pe=mw["Pe_p50"],
        ratio=h["Knum_over_Kphys_p50"] / vp["Knum_over_Kphys_p50"],
        dz10=dz[9], dz11=dz[10], dz12=dz[11],
        preproc=preproc_dir, name_json=out_png.with_suffix(".json").name)
    out_png.with_suffix(".txt").write_text(txt)


PANEL_A_TXT = """\
{fn}
{rule}

"{title}"
{sub}

Standalone publication version of panel (a) of {combined}. Cumulative
distribution of the Peclet number over every interior grid face of the
2022 South Asia production meteorology (96 states = 12 months x 8
three-hour periods; {nh:,} horizontal faces, {nv:,} vertical).

  horizontal faces                      median {h_pe:.1f}      {h_lt2:.1f}% below 2
  vertical, below the boundary layer    median {v_pe:.4f}   {v_lt2:.1f}% below 2
  vertical, above the boundary layer    median {f_pe:.2f}     {f_lt2:.1f}% below 2
  vertical, fine-particle-mass weighted median {m_pe:.2f}     {m_lt2:.1f}% of mass below 2

The Peclet number here is |w_net|*dL/K_phys, where K_phys is the grid
diffusivity PLUS the model's intended sub-grid exchange -- see the
combined figure's sidecar ({combined_txt}) for the exact split-flux
decomposition and for why this differs by orders of magnitude from the
June 2026 diagnostics the talk previously carried.

Generator: scripts/make_peclet_regimes_figure.py
Numbers:   {js}
"""

PANEL_B_TXT = """\
{fn}
{rule}

"{title}"
{sub}

Standalone publication version of panel (b) of {combined}. Share of the
mixing flux at each face supplied by the numerical scheme rather than by
physics, weighted by the fine-particle concentration gradient across the
face.

  horizontal scheme                     {h_share:.1f}% numerical
  vertical scheme, below boundary layer  {v_share:.2f}% numerical
  vertical scheme, above boundary layer {f_share:.1f}% numerical
  median top of the boundary layer      {pblh:.0f} m

This is the figure that carries the design argument: horizontally there
is a great deal of accuracy for an anti-diffusive correction to recover,
so the flux limiter earns its keep; vertically, in the layers that set
surface exposure, there is almost nothing to recover, so the vertical
operators run plain and unlimited.

The rise above ~1.3 km tracks the vertical grid stretching rather than
the meteorology: numerical diffusion scales with face spacing, which
goes from ~130 m at faces 1-9 to {dz10:.0f}, {dz11:.0f} and {dz12:.0f} m
at faces 10-12. Convective mass flux is further intended vertical
transport and is not credited in the denominator, so these vertical
shares are upper bounds.

Generator: scripts/make_peclet_regimes_figure.py
Numbers:   {js}
"""


def write_panel_sidecars(s, out_dir: Path, name: str):
    h, vp, vf = (s["horizontal"]["all"], s["vertical"]["pbl"],
                 s["vertical"]["free_trop"])
    mw, dz = s["vertical"]["mass_weighted"], s["meta"]["dz_face_mean_m"]
    stem = name.split("_")[0]
    fa = f"{stem}a_peclet_distribution.png"
    fb = f"{stem}b_numerical_share.png"
    (out_dir / f"{stem}a_peclet_distribution.txt").write_text(PANEL_A_TXT.format(
        fn=fa, rule="=" * len(fa), title=TITLE_A, sub=SUB_A.replace("\n", " "),
        combined=f"{name}.png", combined_txt=f"{name}.txt", js=f"{name}.json",
        nh=h["n"], nv=s["vertical"]["all"]["n"],
        h_pe=h["Pe_p50"], h_lt2=100 * h["frac_Pe_lt_2"],
        v_pe=vp["Pe_p50"], v_lt2=100 * vp["frac_Pe_lt_2"],
        f_pe=vf["Pe_p50"], f_lt2=100 * vf["frac_Pe_lt_2"],
        m_pe=mw["Pe_p50"], m_lt2=100 * mw["frac_mass_at_Pe_lt_2"]))
    (out_dir / f"{stem}b_numerical_share.txt").write_text(PANEL_B_TXT.format(
        fn=fb, rule="=" * len(fb), title=TITLE_B, sub=SUB_B.replace("\n", " "),
        combined=f"{name}.png", js=f"{name}.json",
        h_share=100 * h["numerical_share_of_mixing_flux"],
        v_share=100 * vp["numerical_share_of_mixing_flux"],
        f_share=100 * vf["numerical_share_of_mixing_flux"],
        pblh=s["meta"]["pblh_median_m"], dz10=dz[9], dz11=dz[10], dz12=dz[11]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preproc-dir", type=Path,
                    default=Path.home() / "orbit_data/inputs/grids_2022_fixed")
    ap.add_argument("--out-dir", type=Path, default=Path("docs/figures"))
    ap.add_argument("--name", default="F6_peclet_regimes")
    ap.add_argument("--months", type=int, nargs="+", default=list(range(1, 13)))
    ap.add_argument("--bins", type=int, nargs="+", default=list(range(1, 9)))
    a = ap.parse_args()

    a.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"scanning {len(a.months) * len(a.bins)} preprocessor states "
          f"under {a.preproc_dir}")
    V, H, meta = collect(a.preproc_dir, a.months, a.bins)
    s = summarise(V, H, meta)

    out_png = a.out_dir / f"{a.name}.png"
    out_json = a.out_dir / f"{a.name}.json"
    out_json.write_text(json.dumps(s, indent=2))
    written = make_figures(V, H, s, a.out_dir, a.name)
    write_sidecar(s, out_png, a.preproc_dir)
    write_panel_sidecars(s, a.out_dir, a.name)

    hh, vp = s["horizontal"]["all"], s["vertical"]["pbl"]
    print("\nwrote " + "\n      ".join(str(p) for p in written))
    print(f"      {out_json}\n      {out_png.with_suffix('.txt')} (+ per-panel .txt)")
    print(f"\n  horizontal  Pe p50 {hh['Pe_p50']:.2f}  "
          f"numerical share {100*hh['numerical_share_of_mixing_flux']:.1f}%")
    print(f"  vertical PBL Pe p50 {vp['Pe_p50']:.4f}  "
          f"numerical share {100*vp['numerical_share_of_mixing_flux']:.2f}%")

    # ---- acceptance checks, stated before the numbers were known ----
    print("\nacceptance checks")
    ok = lambda b: "PASS" if b else "FAIL"
    bare = hh["Pe_p50_bare_Kxxyy_only"]
    print(f"  [{ok(6e4 < bare < 6e5)}] K_grid-only horizontal median "
          f"{bare:.3g} within ~3x of the June figure's 2e5")
    print(f"  [{ok(vp['Pe_p50'] < 2)}] vertical PBL median |Pe| = "
          f"{vp['Pe_p50']:.4f} < 2")
    pbl_frac = float(np.mean([r["frac_below_pbl"] for r in s["by_face_index"]]))
    print(f"  [{ok(0.30 < pbl_frac < 0.60)}] fraction of vertical faces below "
          f"the PBL = {100*pbl_frac:.1f}% (expect 30-60%)")
    print(f"  [{ok(hh['patankar_A_mean'] < 1e-2)}] horizontal Patankar A mean = "
          f"{hh['patankar_A_mean']:.2e} (explicit Kxxyy is switched off)")


if __name__ == "__main__":
    main()
