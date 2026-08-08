"""Shared plotting primitives for marginal/zero-out NPZ visualisation.

Three thin helpers — all the decisions (axes, colorbar range, contours,
cell labels) live here so the per-mode plot scripts stay short and
declarative.

Cartopy is preferred for the SAS overview map; if it isn't importable,
``surface_map`` falls back to plain matplotlib with lat/lon axes.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np


# Diagnostic cells reused across plot scripts; lifted from
# scripts/run_orbit.py so orbit/plotting/ is self-contained.
NAMED_CELLS: List[Tuple[str, int, int]] = [
    # (label, j, i) — nominal lat/lon in comment
    ("Delhi",      49, 28),   # 28.6N, 77.5E — polluted urban, IGP
    ("Kanpur",     45, 32),   # 26.5N, 80.0E — IGP, downwind Delhi
    ("Kolkata",    37, 45),   # 22.5N, 88.1E — coastal, east IGP
    ("Arabian-Sea", 22,  8),  # 15.0N, 65.0E — low-NOx reference
    ("Andaman",    16, 52),   # 12.0N, 92.5E — low-NOx reference
]


# UTC hour at the centre of each 3-hour bin (forward solver convention).
# Bin i covers [3i, 3(i+1)) hours UTC; the centre is 3i + 1.5h.
BIN_HOURS_UTC: List[float] = [1.5 + 3 * i for i in range(8)]


def _diverging_limits(data: np.ndarray, percentile: float = 99.0) -> Tuple[float, float]:
    """Symmetric colour limits anchored at zero.

    Uses the chosen percentile of |data| to clip outliers; falls back
    to data.max if data is all-zero or all-NaN.
    """
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return -1e-12, 1e-12
    abs_finite = np.abs(finite)
    if percentile is None or percentile >= 100:
        v = float(abs_finite.max())
    else:
        v = float(np.percentile(abs_finite, percentile))
    if v == 0.0:
        v = 1e-12
    return -v, v


def surface_map(
    field_2d: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    ax=None,
    title: str = "",
    cbar_label: str = "",
    cmap: str = "RdBu_r",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    diverging: bool = True,
    cells: Optional[Sequence[Tuple[str, int, int]]] = NAMED_CELLS,
    use_cartopy: bool = True,
):
    """Render a (ny, nx) surface field over the SAS grid.

    Parameters
    ----------
    field_2d : (ny, nx) array.
    lon, lat : 1-D coords.
    ax : optional matplotlib Axes (or GeoAxes). Created if None.
    diverging : if True and vmin/vmax not specified, sets symmetric
        limits at the 99th percentile of |field|.
    cells : iterable of (label, j, i) to overplot. Set None to disable.
    use_cartopy : if True and cartopy is importable, uses GeoAxes with
        coastlines/borders. Falls back to plain axes otherwise.

    Returns the matplotlib Axes used.
    """
    import matplotlib.pyplot as plt

    if vmin is None or vmax is None:
        if diverging:
            vmin, vmax = _diverging_limits(field_2d)
        else:
            finite = field_2d[np.isfinite(field_2d)]
            vmin = float(finite.min()) if finite.size else 0.0
            vmax = float(finite.max()) if finite.size else 1.0

    cartopy_ok = False
    if use_cartopy and ax is None:
        try:
            import cartopy.crs as ccrs
            import cartopy.feature as cfeat
            fig = plt.gcf() if plt.get_fignums() else plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
            ax.set_extent([float(lon.min()), float(lon.max()),
                           float(lat.min()), float(lat.max())],
                          crs=ccrs.PlateCarree())
            ax.add_feature(cfeat.COASTLINE, linewidth=0.6, edgecolor="black")
            ax.add_feature(cfeat.BORDERS, linewidth=0.4, edgecolor="gray")
            cartopy_ok = True
        except Exception:
            ax = None  # fall through to plain axes

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    LON, LAT = np.meshgrid(lon, lat)
    pcm = ax.pcolormesh(
        LON, LAT, field_2d,
        cmap=cmap, vmin=vmin, vmax=vmax, shading="auto",
    )
    if cells:
        for label, j, i in cells:
            if 0 <= j < lat.size and 0 <= i < lon.size:
                ax.plot(lon[i], lat[j], marker="o", markersize=4,
                        markerfacecolor="none", markeredgecolor="black",
                        markeredgewidth=1.0)
                ax.annotate(
                    label, xy=(lon[i], lat[j]),
                    xytext=(4, 4), textcoords="offset points",
                    fontsize=7, color="black",
                )

    if title:
        ax.set_title(title, fontsize=11)
    if not cartopy_ok:
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_aspect("equal", adjustable="datalim")
    cbar = plt.colorbar(pcm, ax=ax, shrink=0.8, pad=0.02)
    if cbar_label:
        cbar.set_label(cbar_label, fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    return ax


def diurnal_trace(
    field_orbit: np.ndarray,
    cell_idx: Tuple[int, int],
    ax=None,
    label: str = "",
    surface_level: int = 0,
):
    """Plot the diurnal trace of a (8, nz, ny, nx) field at one cell.

    field_orbit : (8, nz, ny, nx) — typically delta_pm25_orbit.
    cell_idx : (j, i) into ny, nx.
    surface_level : level index to extract (default 0 = surface).

    Returns the Axes.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 3.5))

    j, i = cell_idx
    if field_orbit.ndim != 4:
        raise ValueError(
            f"diurnal_trace expects a (8, nz, ny, nx) field, "
            f"got shape {field_orbit.shape}"
        )
    series = field_orbit[:, surface_level, j, i]
    ax.plot(BIN_HOURS_UTC, series, marker="o", label=label or None)
    ax.axhline(0.0, color="gray", linewidth=0.5, linestyle=":")
    ax.set_xlabel("UTC hour (bin centre)")
    ax.set_xticks(BIN_HOURS_UTC)
    ax.set_xticklabels([f"{h:.0f}" for h in BIN_HOURS_UTC])
    return ax


def named_cell_table(
    field_3d: np.ndarray,
    cells: Sequence[Tuple[str, int, int]] = NAMED_CELLS,
    surface_level: int = 0,
    units: str = "",
    fmt: str = "{:+.3e}",
) -> str:
    """Build a markdown-formatted table of field values at named cells.

    field_3d : (nz, ny, nx) — typically delta_pm25_mean.
    Returns a multi-line string suitable for printing or appending to
    a caption.
    """
    if field_3d.ndim != 3:
        raise ValueError(
            f"named_cell_table expects a (nz, ny, nx) field, "
            f"got shape {field_3d.shape}"
        )
    header_unit = f" ({units})" if units else ""
    lines = [
        f"| Cell | Value{header_unit} |",
        "|---|---|",
    ]
    for label, j, i in cells:
        if 0 <= j < field_3d.shape[1] and 0 <= i < field_3d.shape[2]:
            val = float(field_3d[surface_level, j, i])
            lines.append(f"| {label} | {fmt.format(val)} |")
        else:
            lines.append(f"| {label} | (out of grid) |")
    return "\n".join(lines)
