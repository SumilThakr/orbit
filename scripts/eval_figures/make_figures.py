"""Generate three of the six ORBIT-summary figures from production NPZs:

  02_speciated_annual_maps.png  — annual speciated surface PM2.5 (six panels)
  04_monthly_pm25_maps.png      — 12-panel monthly surface PM2.5
  05_january_diurnal_maps.png   — 8-panel January diurnal cycle

All three use cartopy for country outlines / coastlines and put mean values
in parentheses on the panel-title line (not below it).
"""
# Renders the ORBIT-summary map figures from production NPZs; paths and the
# reported field are environment-overridable, the numerics are fixed.
#
# FIELD: the original hardcoded "iso_pm25_mean". ORBIT_FIG_FIELD lets the
# same script emit the settled reporting field (pm25_mean) instead; the
# figure is otherwise byte-for-byte the same code path.
import os as _os
import sys as _sys
FIELD = _os.environ.get("ORBIT_FIG_FIELD", "iso_pm25_mean")

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
except ImportError:
    _sys.exit("cartopy is required for the map figures: "
              "conda install -c conda-forge cartopy")

# Paths are environment-overridable; numerics untouched.
ROOT = Path(_os.environ.get(
    "ORBIT_FORWARD_DIR", "orbit_out/final_2022"))
GROWN = ROOT
# The old summary contrasted two runs: "grown_gc" and a non-grown
# "dry". The 2022 campaign produced ONE production configuration, so
# DRY points at the same run. Only figure 02 (speciated annual maps)
# reads it, and it reads species fields (c_mean) rather than the
# grown PM2.5 total, so the panel is well-defined either way -- but
# if the grown/dry contrast mattered scientifically, that needs a
# second run and a decision, not a path edit.
DRY = ROOT
OUT = Path(_os.environ.get(
    "ORBIT_EVAL_FIG_OUT", "docs/figures/eval"))
OUT.mkdir(parents=True, exist_ok=True)

# MW conversions
N_TO_NH4 = 18.03851 / 14.0067
N_TO_NO3 = 62.00501 / 14.0067
S_TO_SO4 = 96.0632 / 32.0655

# Species indices (orbit/core/deposition.py)
IDX_PM25 = 1
IDX_TOTAL_NH = 2
IDX_PSO4 = 5
IDX_TOTAL_NO3 = 6

month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
cmap = plt.get_cmap("YlOrRd")
PROJ = ccrs.PlateCarree()


def load_month(mm, grown=True):
    src = GROWN if grown else DRY
    return np.load(src / f"orbit_M{mm:02d}.npz")


def add_geo(ax, lon, lat):
    """Country outlines + coastlines on a cartopy ax."""
    ax.add_feature(cfeature.BORDERS, linewidth=0.4, edgecolor="0.25")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.4)
    ax.add_feature(cfeature.OCEAN, facecolor="lightblue", alpha=0.3)
    ax.set_extent([float(lon.min()), float(lon.max()),
                   float(lat.min()), 36.0], crs=PROJ)
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.4)
    gl.top_labels = False
    gl.right_labels = False


print("loading 12 months...")
months_grown = [load_month(m, grown=True) for m in range(1, 13)]
months_dry = [load_month(m, grown=False) for m in range(1, 13)]
lat = months_dry[0]["lat"]
lon = months_dry[0]["lon"]
NY = lat.size
NX = lon.size
SURF = NY * NX
LON, LAT = np.meshgrid(lon, lat)


# ---------------------------------------------------------------------------
# FIGURE 04: 12-panel monthly PM2.5 concentration maps (grown, surface)

print("figure: 12-panel monthly maps")
fig, axes = plt.subplots(3, 4, figsize=(16, 11), constrained_layout=True,
                         subplot_kw={"projection": PROJ})
all_pm = np.stack([d[FIELD][0] for d in months_grown])
vmax = float(np.percentile(all_pm[np.isfinite(all_pm)], 98))
norm = mpl.colors.Normalize(vmin=0, vmax=vmax)
for m, ax in enumerate(axes.flat):
    pm = months_grown[m][FIELD][0]
    im = ax.pcolormesh(LON, LAT, pm, cmap=cmap, norm=norm, shading="nearest",
                       transform=PROJ)
    add_geo(ax, lon, lat)
    ax.set_title(f"{month_names[m]} ({pm.mean():.1f} µg m$^{{-3}}$)",
                 fontsize=11)
cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
cbar.set_label("PM$_{2.5}$ (µg m$^{-3}$, 35% RH)")
fig.suptitle("Monthly-average surface PM$_{2.5}$ concentrations predicted "
             "by ORBIT (2022)", fontsize=14, y=1.02)
fig.savefig(OUT / "05_monthly_pm25_maps.png", dpi=120, bbox_inches='tight')
plt.close(fig)
print("  ->", OUT / "05_monthly_pm25_maps.png")


# ---------------------------------------------------------------------------
# FIGURE 05: 8-panel January diurnal cycle (grown, surface)

print("figure: January diurnal")
m01 = months_grown[0]
diurnal = m01["iso_pm25_orbit"][:, 0]   # (8, NY, NX) surface
fig, axes = plt.subplots(2, 4, figsize=(16, 7.5), constrained_layout=True,
                         subplot_kw={"projection": PROJ})
utc_hours = [1.5, 4.5, 7.5, 10.5, 13.5, 16.5, 19.5, 22.5]
ist_hours = [(h + 5.5) % 24 for h in utc_hours]


def hhmm(h):
    H = int(h) % 24
    M = int(round((h - int(h)) * 60))
    return f"{H:02d}:{M:02d}"


norm = mpl.colors.Normalize(vmin=0, vmax=float(np.percentile(diurnal, 99)))
for b, ax in enumerate(axes.flat):
    im = ax.pcolormesh(LON, LAT, diurnal[b], cmap=cmap, norm=norm,
                       shading="nearest", transform=PROJ)
    add_geo(ax, lon, lat)
    ax.set_title(f"UTC {utc_hours[b]:.1f}h  /  IST {hhmm(ist_hours[b])} "
                 f"({diurnal[b].mean():.1f} µg m$^{{-3}}$)",
                 fontsize=10)
cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
cbar.set_label("PM$_{2.5}$ (µg m$^{-3}$, 35% RH)")
fig.suptitle("January 2022 diurnal cycle of surface PM$_{2.5}$ predicted "
             "by ORBIT (8 × 3-h UTC bins)", fontsize=14, y=1.02)
fig.savefig(OUT / "06_january_diurnal_maps.png", dpi=120, bbox_inches='tight')
plt.close(fig)
print("  ->", OUT / "06_january_diurnal_maps.png")


# ---------------------------------------------------------------------------
# FIGURE 02: Speciated annual maps (dry, surface)

def surf_species(d, idx):
    return d["c_mean"][idx, :SURF].reshape(NY, NX)


print("figure: speciated annual maps")
primary_yr = np.mean([surf_species(d, IDX_PM25) for d in months_dry], axis=0)
so4_yr = np.mean([surf_species(d, IDX_PSO4) * S_TO_SO4 for d in months_dry],
                 axis=0)


def nh4_for_month(d):
    f = d["iso_f_nh4_mean"][0]
    c = surf_species(d, IDX_TOTAL_NH)
    return f * c * N_TO_NH4
nh4_yr = np.mean([nh4_for_month(d) for d in months_dry], axis=0)


def no3_for_month(d):
    f = d["iso_f_no3_mean"][0]
    c = surf_species(d, IDX_TOTAL_NO3)
    return f * c * N_TO_NO3
no3_yr = np.mean([no3_for_month(d) for d in months_dry], axis=0)

soa_yr = np.mean([d["soa_mean"] for d in months_dry], axis=0)
total_yr = primary_yr + nh4_yr + no3_yr + so4_yr + soa_yr

species_maps = [
    ("Primary PM$_{2.5}$",  primary_yr),
    ("NH$_4^+$",            nh4_yr),
    ("NO$_3^-$",            no3_yr),
    ("SO$_4^{2-}$",         so4_yr),
    ("SOA",                 soa_yr),
    ("Total PM$_{2.5}$",    total_yr),
]
# ONE shared viridis scale across all six panels. Previously each panel was
# normalised to its own 99th percentile, so NH4+ (which reaches ~5 ug m-3)
# and Primary PM2.5 (~37) rendered with identical colour ranges and the
# species could not be compared by eye -- the panels looked equally intense
# when they differ by nearly an order of magnitude. The shared scale is set
# from the 99th percentile of Total PM2.5, the natural maximum of the set,
# and `extend="max"` flags the few cells above it. Each panel's own mean and
# max stay in its title so the compressed panels are still quantified.
# ONE shared LINEAR scale across all six panels. Previously each panel was
# normalised to its own 99th percentile, so NH4+ (which reaches ~5 ug m-3) and
# Primary PM2.5 (~37) rendered with identical colour ranges: the panels looked
# equally intense when they differ by nearly an order of magnitude, and the
# species could not be compared by eye. With a shared linear norm, equal colour
# means equal concentration in every panel.
#
# cividis rather than viridis, chosen by rendering the candidates rather than by
# metric. Under a linear 0-40 scale the four secondary species (NH4+ 5.5, SOA
# 6.3, SO4 8.0, NO3- 9.2 at their 99th percentiles) occupy only the bottom
# eighth of the bar. viridis puts near-black purple there and they read as flat;
# cividis starts at a mid-lightness blue, so the same slice keeps usable
# contrast and the Indo-Gangetic plume stays visible in NH4+. cividis is also
# designed for deuteranopia, so this is an accessibility improvement on viridis,
# not a regression. (inferno scored best on a CIELAB path-length metric and
# rendered WORST -- its low end is near-black. Do not re-pick this by metric.)
#
# vmax = 40 ug m-3, which is Total PM2.5's 99th percentile (40.07) rounded. It
# is deliberately NOT lowered to give the secondary species more bar: at vmax 20
# Total clips 17% of cells and at 25 it clips 9.5%, which destroys the gradient
# across the Indo-Gangetic plain -- the most important feature in the figure.
# At vmax 40 nothing clips beyond 1%.
SPEC_VMAX = 40.0
spec_cmap = plt.get_cmap("cividis")
spec_norm = mpl.colors.Normalize(vmin=0.0, vmax=SPEC_VMAX)

fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True,
                         subplot_kw={"projection": PROJ})
for (title, arr), ax in zip(species_maps, axes.flat):
    im = ax.pcolormesh(LON, LAT, arr, cmap=spec_cmap, norm=spec_norm,
                       shading="nearest", transform=PROJ)
    add_geo(ax, lon, lat)
    ax.set_title(f"{title}  (mean {arr.mean():.1f}, max {np.nanmax(arr):.1f} "
                 f"µg m$^{{-3}}$)", fontsize=11)
cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.030, pad=0.015,
                    extend="max")
cbar.set_label("µg m$^{-3}$  — shared linear scale, identical in all six panels")
fig.suptitle("Annual-average speciated surface PM$_{2.5}$ concentrations "
             "predicted by ORBIT (2022)", fontsize=14)
fig.savefig(OUT / "02_speciated_annual_maps.png", dpi=120, bbox_inches='tight')
plt.close(fig)
print("  ->", OUT / "02_speciated_annual_maps.png")

print("done.")
