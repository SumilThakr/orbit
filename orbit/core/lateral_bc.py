"""Lateral boundary inflow (Dirichlet-type inflow source on regional domains).

ORBIT's horizontal transport has an open lateral boundary: edge cells lose mass
through boundary faces where the wind blows outward
(``convdiff.assemble_horizontal_convdiff``, x faces from line 204, y faces from
line 318; mirrored in ``operator.compute_lateral_boundary_loss``), but nothing
enters through faces where the wind blows inward. Air entering the domain
therefore carries zero concentration.

This module adds the missing inflow as a solution-independent source on the
orbit right-hand side, the same channel as the O3 production term and the FCT
correction (``extra_rhs_per_bin_per_species``):

    s_bc[cell] = r_in[cell] * c_bc[cell]

``r_in`` is the first-order upwind inflow rate through the cell's boundary
faces, built exactly as ``compute_lateral_boundary_loss`` builds the outflow
rate with the wind direction reversed:

- west face of i = 0:    UAvg_plus[:, :, 0] / dx   (split)   or max(U, 0) / dx
- east face of i = nx-1: the true east outer face (grid_data.edge_face), / dx
  (from 1.0.4 always the input's last staggered face; earlier the cell's west
  face stood in)
- south face of j = 0:   VAvg_plus[:, 0, :] / dy * dx_face/dx_cell (split) ...
- north face of j = ny-1: VAvg_minus[:, -1, :] / dy * dx_face/dx_cell (split) ...

No terrain ratio is applied (there is no outside cell thickness; ratio 1).
Diffusive inflow and the model top are not included.

``c_bc`` is the boundary concentration in the species' own ORBIT tracer units
(the same units as the orbit state), per 3-h bin. Only cells with r_in > 0 use
it. Because the term does not depend on the solution, the operator, its LU
factors and the adjoint are unchanged; marginal (linearised) solves must not
include it (a boundary perturbation of zero; ``modes/marginal.py`` calls
``solve_orbit_all_species`` directly and never sees this term).

File format (one per month), NPZ:
    lat (ny,), lon (nx,), nz (scalar), species (n_s,) str,
    c_bc (n_s, 8, nz, ny, nx) float32   tracer units, value per bin
Species names are ``cli.SPECIES_NAMES`` entries. Missing species get no inflow.

Enabled in ``cli.run_forward_month`` by ``ORBIT_LATERAL_BC`` = a path with
``{month:02d}`` (e.g. ``/x/bc_M{month:02d}.npz``). Off by default.
"""

import os

import numpy as np

from orbit.core.grid_data import EARTH_RADIUS, DEG_TO_RAD, edge_face

ENV_VAR = "ORBIT_LATERAL_BC"


def compute_lateral_boundary_inflow(grid) -> np.ndarray:
    """Inflow rate r_in >= 0 [1/s] per cell, shape (N,); zero off the boundary."""
    nz, ny, nx = grid.nz, grid.ny, grid.nx
    use_split = grid.has_split_fluxes
    dx_3d = np.broadcast_to(grid.dx[np.newaxis, :, np.newaxis], (nz, ny, nx)).copy()
    valid = dx_3d > 0
    safe_dx = np.where(valid, dx_3d, 1.0)
    rate = np.zeros((nz, ny, nx), dtype=np.float64)

    if not grid.periodic_lon:
        if use_split:
            u_in_w = grid.UAvg_plus[:, :, 0]
            u_in_e = edge_face(grid, "U_minus")
        else:
            u_in_w = np.maximum(grid.UAvg[:, :, 0], 0.0)
            u_in_e = np.maximum(-edge_face(grid, "U"), 0.0)
        rate[:, :, 0] += np.where(valid[:, :, 0] & (u_in_w > 0),
                                  u_in_w / safe_dx[:, :, 0], 0.0)
        rate[:, :, -1] += np.where(valid[:, :, -1] & (u_in_e > 0),
                                   u_in_e / safe_dx[:, :, -1], 0.0)

    dy = grid.dy if grid.dy > 0 else 1.0
    lat_face_s = grid.lat[0] - grid.dlat / 2.0
    lat_face_n = grid.lat[-1] + grid.dlat / 2.0
    dxf = lambda la: EARTH_RADIUS * np.cos(la * DEG_TO_RAD) * grid.dlon * DEG_TO_RAD  # noqa: E731
    tr_s = dxf(lat_face_s) / (grid.dx[0] if grid.dx[0] > 0 else 1.0)
    tr_n = dxf(lat_face_n) / (grid.dx[-1] if grid.dx[-1] > 0 else 1.0)
    if use_split:
        v_in_s = grid.VAvg_plus[:, 0, :]
        v_in_n = edge_face(grid, "V_minus")
    else:
        v_in_s = np.maximum(grid.VAvg[:, 0, :], 0.0)
        v_in_n = np.maximum(-edge_face(grid, "V"), 0.0)
    rate[:, 0, :] += np.where(valid[:, 0, :] & (v_in_s > 0), v_in_s / dy * tr_s, 0.0)
    rate[:, -1, :] += np.where(valid[:, -1, :] & (v_in_n > 0), v_in_n / dy * tr_n, 0.0)
    return rate.ravel()


class LateralBCSource(dict):
    """{species index: [r_in * c_bc per bin]} (the RHS source), with
    ``c_bc`` = {species index: [c_bc per bin, flat (N,)]} kept for the
    horizontal FCT, whose first interior face at an inflow edge needs the
    outside concentration as its far-upwind value (1.0.4 item 19)."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.c_bc = {}


def load_lateral_bc(path, grids, species_names, n_bins=8):
    """Read a boundary file and return {species_idx: [s_bc per bin (N,)]}.

    ``grids`` is the list of per-bin GridData of the run. Checks that the
    file's lat/lon/nz match the run grid (loud failure otherwise) and that
    concentrations are finite and non-negative.
    """
    g0 = grids[0]
    d = np.load(path, allow_pickle=False)
    lat, lon, nz = d["lat"], d["lon"], int(d["nz"])
    if (nz != g0.nz or lat.shape != (g0.ny,) or lon.shape != (g0.nx,)
            or not np.allclose(lat, g0.lat, atol=1e-6)
            or not np.allclose(lon, g0.lon, atol=1e-6)):
        raise ValueError(
            f"lateral BC grid mismatch in {path}: file nz={nz} "
            f"lat {lat[0]:.3f}..{lat[-1]:.3f} ({lat.size}) lon {lon[0]:.3f}..{lon[-1]:.3f} "
            f"({lon.size}); run nz={g0.nz} lat {g0.lat[0]:.3f}..{g0.lat[-1]:.3f} "
            f"({g0.ny}) lon {g0.lon[0]:.3f}..{g0.lon[-1]:.3f} ({g0.nx})")
    c_bc = d["c_bc"]
    names = [str(s) for s in d["species"]]
    if c_bc.shape != (len(names), n_bins, g0.nz, g0.ny, g0.nx):
        raise ValueError(f"c_bc shape {c_bc.shape} != ({len(names)}, {n_bins}, "
                         f"{g0.nz}, {g0.ny}, {g0.nx})")
    if not np.all(np.isfinite(c_bc)) or np.any(c_bc < 0):
        raise ValueError(f"c_bc in {path} has non-finite or negative values")
    unknown = sorted(set(names) - set(species_names))
    if unknown:
        raise ValueError(f"unknown species in {path}: {unknown}")
    rates = [compute_lateral_boundary_inflow(grids[t]) for t in range(n_bins)]
    out = LateralBCSource()
    for i, name in enumerate(names):
        s = species_names.index(name)
        cb = [c_bc[i, t].astype(np.float64).ravel() for t in range(n_bins)]
        out[s] = [rates[t] * cb[t] for t in range(n_bins)]
        out.c_bc[s] = cb
    return out


def from_env(month, grids, species_names, n_bins=8, verbose=True):
    """Boundary source dict for ``month`` if ``ORBIT_LATERAL_BC`` is set, else None."""
    tmpl = os.environ.get(ENV_VAR, "").strip()
    if not tmpl:
        return None
    path = tmpl.format(month=month)
    src = load_lateral_bc(path, grids, species_names, n_bins=n_bins)
    if verbose:
        tot = {species_names[s]: float(sum(v.sum() for v in vecs) / n_bins)
               for s, vecs in src.items()}
        print(f"  [lateral BC] {path}: inflow source (bin-mean sum over cells, "
              f"tracer units/s) " + ", ".join(f"{k}={v:.3e}" for k, v in tot.items()))
    return src


def merge_into(extra_rhs, src, n_bins=8):
    """Add ``src`` into an extra-RHS dict (creating it if None); returns the dict."""
    if src is None:
        return extra_rhs
    if extra_rhs is None:
        extra_rhs = {}
    for s, vecs in src.items():
        if s in extra_rhs:
            extra_rhs[s] = [extra_rhs[s][t] + vecs[t] for t in range(n_bins)]
        else:
            extra_rhs[s] = list(vecs)
    return extra_rhs
