"""Deposition flux maps as a model output.

Deposition enters the operator as a diagonal sink, so the deposited flux in
each cell is recoverable exactly from the converged orbit -- no extra solve and
no operator instrumentation. This module turns the rates and the periodic-orbit
concentrations into per-species dry and wet deposition maps.

Definitions
-----------
Dry deposition acts on the surface layer only, at velocity ``vd``:

    dry flux = vd * c[0]                       [ug m-2 s-1]

(the operator applies ``vd / Dz[0]`` per unit volume; multiplying back by
``Dz[0]`` gives the per-area flux, so the layer thickness cancels).

Wet scavenging acts throughout the column at rate ``wd(z)``:

    wet flux = sum_z wd[z] * c[z] * Dz[z]      [ug m-2 s-1]

Both are averaged over the eight bins of the periodic orbit, so the result is
the orbit-mean flux -- the deposition the modelled average day delivers.

Not included: the SOA photolytic sink. It is assembled into the same diagonal
for numerical reasons but is a chemical loss, not deposition, and counting it
here would overstate deposited mass. (It is zero in the default configuration
anyway, since ``ORBIT_VBS_A_PHOTO`` defaults to 0.)

Units and mass basis
--------------------
Fluxes are ``ug m-2 s-1``. Useful conversions::

    x 31.5576   -> g m-2 yr-1
    x 315.576   -> kg ha-1 yr-1
    x 31557.6   -> kg km-2 yr-1

The inorganic species carry **element mass**, matching the solver's internal
convention: ``TotalNH`` and ``TotalNO3`` are kg of N, ``SO2`` and ``pSO4`` are
kg of S. That is the basis critical-load and ecosystem work already uses
(kg N ha-1 yr-1), so it is reported as-is rather than converted to compound
mass. ``PrimaryPM2.5`` and the VBS bins are already aerosol/organic mass.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from orbit.core.deposition import (
    N_SPECIES,
    _get_dry_dep_velocity_2d,
    _get_wet_dep_rate_3d,
)

#: Multiply a ``ug m-2 s-1`` flux by this to get ``kg ha-1 yr-1``.
UG_M2_S_TO_KG_HA_YR = 315.576

#: Multiply a ``ug m-2 s-1`` flux by this to get ``g m-2 yr-1``.
UG_M2_S_TO_G_M2_YR = 31.5576


def compute_deposition_maps(orbits, grids, indexer) -> Dict[str, np.ndarray]:
    """Orbit-mean dry and wet deposition flux maps.

    Parameters
    ----------
    orbits : dict[int, list[ndarray]]
        Per-species periodic orbit: ``orbits[s][step]`` is the flat (N,)
        concentration at step ``0..N_BINS``. Steps ``1..N_BINS`` are the
        bin states and are the ones averaged, matching how ``c_mean`` is
        formed elsewhere.
    grids : list[GridData]
        Per-bin grids, length ``N_BINS``. Each carries that bin's
        meteorology, so deposition velocities and scavenging rates vary
        across the orbit.
    indexer : CellIndexer
        Supplies the (nz, ny, nx) shape.

    Returns
    -------
    dict with keys ``dry`` and ``wet``, each ``(N_SPECIES, ny, nx)`` float32
    in ``ug m-2 s-1``, plus ``total`` (their sum).
    """
    n_bins = len(grids)
    if n_bins == 0:
        raise ValueError("compute_deposition_maps needs at least one bin grid")

    nz, ny, nx = grids[0].nz, grids[0].ny, grids[0].nx

    dry = np.zeros((N_SPECIES, ny, nx), dtype=np.float64)
    wet = np.zeros((N_SPECIES, ny, nx), dtype=np.float64)

    for s in range(N_SPECIES):
        if s not in orbits:
            continue
        for tau in range(n_bins):
            grid = grids[tau]
            # orbits[s] holds steps 0..n_bins; step tau+1 is bin tau's state,
            # consistent with the c_mean convention (mean of c_1..c_8).
            c = np.asarray(orbits[s][tau + 1], dtype=np.float64)
            c3 = c.reshape((nz, ny, nx))

            # Dry: surface layer only. vd * c[0] already has per-area units.
            vd = _get_dry_dep_velocity_2d(grid, s)
            dry[s] += vd * c3[0]

            # Wet: column integral of rate * concentration * layer thickness.
            wd = _get_wet_dep_rate_3d(grid, s)
            wet[s] += np.einsum("zyx,zyx,zyx->yx", wd, c3, grid.Dz)

    dry /= n_bins
    wet /= n_bins

    # Deposition is a loss, so fluxes should be non-negative. Small negatives
    # can appear where the solve leaves slightly negative concentrations;
    # clip so downstream maps and totals are physical, and report how much
    # was clipped rather than hiding it.
    neg_dry = float(dry[dry < 0].sum())
    neg_wet = float(wet[wet < 0].sum())
    np.clip(dry, 0.0, None, out=dry)
    np.clip(wet, 0.0, None, out=wet)

    return {
        "dry": dry.astype(np.float32),
        "wet": wet.astype(np.float32),
        "total": (dry + wet).astype(np.float32),
        "clipped_negative_ug_m2_s": np.float32(neg_dry + neg_wet),
    }


def summarise(maps: Dict[str, np.ndarray], species_names: List[str]) -> str:
    """A short human-readable deposition table for the run log."""
    dry, wet = maps["dry"], maps["wet"]
    lines = ["  species        dry            wet          total   "
             "(kg ha-1 yr-1, domain mean)"]
    order = np.argsort(-(dry + wet).mean(axis=(1, 2)))
    for s in order:
        d = dry[s].mean() * UG_M2_S_TO_KG_HA_YR
        w = wet[s].mean() * UG_M2_S_TO_KG_HA_YR
        if d + w <= 0:
            continue
        name = species_names[s] if s < len(species_names) else f"idx{s}"
        lines.append(f"  {name:<12s} {d:>10.4f}   {w:>12.4f}   {d + w:>10.4f}")
    return "\n".join(lines)
