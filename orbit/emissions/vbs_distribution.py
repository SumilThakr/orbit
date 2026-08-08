"""VOC → 1D-VBS bin distribution — shared by the forward solver and the
marginal/zero-out perturbation builders.

Single source of truth for the VOC→SoA emission routing: the NOx-regime
fraction, the 5-bin solver indices, the biogenic monoterpene/isoprene split,
and the per-source VOC→VBS yield distribution. Both
``scripts/run_orbit.py`` (forward) and ``orbit/modes/perturbation.py``
(marginal/zero-out δe) import from here, so a VOC perturbation routes mass to
the VBS bins exactly as the forward does — no divergence between the model and
its source-receptor linearisation.

Extracted verbatim from run_orbit.py (2026-05-20) when adding VOC→VBS
support to the perturbation builder.
"""
from __future__ import annotations
import os

import numpy as np

from orbit.core.deposition import N_SPECIES, IDX_VBS_BINS


# CAMS biogenic VOC monoterpene/isoprene split for SAS (Sindelarova 2014 SI
# tropical default 15/85). Override with ORBIT_BIO_VOC_SPLIT="monoterpene:X;isoprene:Y".
_BIO_VOC_SPLIT_DEFAULT = {"monoterpene": 0.15, "isoprene": 0.85}


def parse_bio_voc_split(spec):
    """Parse an ORBIT_BIO_VOC_SPLIT spec ("monoterpene:X;isoprene:Y") into
    a {class: fraction} dict layered over the SAS-tropical 15/85 default.
    Malformed entries are skipped rather than fatal."""
    out = dict(_BIO_VOC_SPLIT_DEFAULT)
    if not spec:
        return out
    for entry in spec.split(";"):
        entry = entry.strip()
        if not entry or ":" not in entry:
            continue
        cls, frac = entry.split(":", 1)
        try:
            out[cls.strip()] = float(frac.strip())
        except ValueError:
            continue
    return out


# Parsed once from the environment, shared by forward and perturbation paths.
BIO_VOC_SPLIT = parse_bio_voc_split(os.environ.get("ORBIT_BIO_VOC_SPLIT", ""))


def compute_nox_regime(grid):
    """Return F_HIGH_NOX(ny, nx) from archive [NO2]/[OH].

    Smooth sigmoid centred at log10([NO2]/[OH]) = 1.5 with 0.5-decade
    transition width. Falls back to F = 0.5 (regime-neutral) if either
    archive field is absent.
    """
    no2 = getattr(grid, "archive_NO2", np.array([]))
    oh = getattr(grid, "archive_OH", np.array([]))
    if no2.size == 0 or oh.size == 0:
        return np.full((grid.ny, grid.nx), 0.5, dtype=np.float64)
    no2_surf = no2[0]
    oh_surf = np.maximum(oh[0], 1.0)
    ratio = no2_surf / oh_surf
    log_ratio = np.log10(np.maximum(ratio, 1e-12))
    return 1.0 / (1.0 + np.exp(-(log_ratio - 1.5) / 0.5))


def bin_indices_in_solver_layout():
    """Return the 5-tuple of solver-layout indices for VBS bins, in
    low-to-high C* order matching VBS_PARENT_YIELDS rows."""
    return IDX_VBS_BINS


def distribute_voc_to_vbs_bins(
    voc_source, grid, indexer, diurnal_cfg, month, n_bins,
    F_HIGH_NOX, verbose=False, bio_voc_split=None,
):
    """Load a single VOC source and distribute its mass across the 5 VBS
    bins per parent-class yields (with cell-dependent NOx-regime
    interpolation when parent class is 'anthro' or 'bio_voc').

    Returns
    -------
    contrib_per_bin : ndarray (n_bins, N_SPECIES * N)
        Solver-layout (13-species) sparse contribution from this source.
    voc_mass_total : float
        Diagnostic — total VOC mass (sum over time, space) for logging.
    """
    from orbit.emissions.vbs_yields import VBS_PARENT_YIELDS, NOX_REGIME_PAIRS
    from orbit.emissions.netcdf import load_netcdf_source, load_netcdf_source_per_bin
    if bio_voc_split is None:
        bio_voc_split = BIO_VOC_SPLIT
    N = indexer.N
    ny, nx = grid.ny, grid.nx

    pcls = voc_source.voc_parent_class
    if pcls is None:
        return np.zeros((n_bins, N_SPECIES * N), dtype=np.float64), 0.0

    # Per-cell yields tuple (low-to-high C*) given the regime mix for
    # this parent class.
    if pcls == "bio_voc":
        # CAMS biogenic VOC: lumped monoterpene + isoprene file. Split
        # by ORBIT_BIO_VOC_SPLIT (default SAS-tropical 15/85 from
        # Sindelarova 2014 SI), then combine the per-class yields.
        y_mono = VBS_PARENT_YIELDS["bio_monoterpene"]
        y_iso  = VBS_PARENT_YIELDS["bio_isoprene"]
        f_mono = bio_voc_split.get("monoterpene", 0.15)
        f_iso  = bio_voc_split.get("isoprene",    0.85)
        y_combined = f_mono * y_mono + f_iso * y_iso        # (5,)
        yields_eff = np.broadcast_to(
            y_combined[:, None, None], (5, ny, nx)
        ).copy()
    elif pcls in NOX_REGIME_PAIRS:
        hi_key, lo_key = NOX_REGIME_PAIRS[pcls]
        y_hi = VBS_PARENT_YIELDS[hi_key]   # (5,)
        y_lo = VBS_PARENT_YIELDS[lo_key]
        # Per-cell effective yields (5, ny, nx)
        F = F_HIGH_NOX[None, :, :]   # (1, ny, nx)
        yields_eff = (F * y_hi[:, None, None] +
                      (1.0 - F) * y_lo[:, None, None])     # (5, ny, nx)
    elif pcls in VBS_PARENT_YIELDS:
        yields_eff = np.broadcast_to(
            VBS_PARENT_YIELDS[pcls][:, None, None], (5, ny, nx)
        ).copy()
    else:
        raise ValueError(
            f"Unknown VBS parent class '{pcls}' for source {voc_source.path}; "
            f"valid: {list(VBS_PARENT_YIELDS) + list(NOX_REGIME_PAIRS)} + 'bio_voc'"
        )

    # Load per-source (slot 0 = raw VOC mass on the loader's 6-species axis).
    if voc_source.bin_axis:
        slabs = load_netcdf_source_per_bin(
            voc_source, grid, n_bins, verbose,
        )  # (n_bins, 6, nz, ny, nx)
        voc_mass_per_bin_3d = slabs[:, 0]   # (n_bins, nz, ny, nx)
    else:
        monthly = load_netcdf_source(voc_source, grid, verbose)  # (6, nz, ny, nx)
        voc_mass_3d = monthly[0]             # (nz, ny, nx)
        # Diurnal factors per-bin
        if diurnal_cfg is None:
            factors = np.ones(n_bins, dtype=np.float64)
        else:
            from orbit.emissions.loader import _factors_for_source
            factors = _factors_for_source(voc_source, diurnal_cfg, month, n_bins)
        voc_mass_per_bin_3d = factors[:, None, None, None] * voc_mass_3d[None]

    # Apply yields and place into the 13-species solver layout. Slot 0
    # of the solver = IDX_VBS_C100 (5th VBS bin index in 0..4 order, since
    # IDX_VBS_BINS = (C01, C1, C10, C100, C1000)).
    bin_solver_idx = bin_indices_in_solver_layout()  # (5-tuple of solver indices)
    contrib = np.zeros((n_bins, N_SPECIES * N), dtype=np.float64)
    voc_mass_total = 0.0
    for tau in range(n_bins):
        for i_bin, solver_s in enumerate(bin_solver_idx):
            bin_3d = yields_eff[i_bin, None] * voc_mass_per_bin_3d[tau]   # (nz, ny, nx)
            contrib[tau, solver_s * N:(solver_s + 1) * N] += bin_3d.ravel()
        voc_mass_total += float(voc_mass_per_bin_3d[tau].sum())

    return contrib, voc_mass_total
