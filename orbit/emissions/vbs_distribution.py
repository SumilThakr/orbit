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


# CAMS biogenic VOC monoterpene/isoprene mass split. Sindelarova et al.
# (2014, ACP 14, 9317) Table 1: global isoprene 594 and monoterpenes
# 95 Tg/yr, so 0.14 / 0.86 (1.0.2 used 0.15 / 0.85, attributed to the
# paper's SI). Override with ORBIT_BIO_VOC_SPLIT="monoterpene:X;isoprene:Y".
_BIO_VOC_SPLIT_DEFAULT = {"monoterpene": 0.14, "isoprene": 0.86}


def parse_bio_voc_split(spec):
    """Parse an ORBIT_BIO_VOC_SPLIT spec ("monoterpene:X;isoprene:Y") into
    a {class: fraction} dict layered over the Sindelarova Table 1 14/86 default.
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


# Generic peroxy-radical rate constants, MCM (Jenkin et al. 1997; Saunders et
# al. 2003): KRO2NO = 2.7e-12 exp(360/T), KRO2HO2 = 2.91e-13 exp(1300/T),
# both in cm3 molec-1 s-1.
K_RO2_NO_A, K_RO2_NO_B = 2.7e-12, 360.0
K_RO2_HO2_A, K_RO2_HO2_B = 2.91e-13, 1300.0
K_BOLTZ = 1.380649e-23          # J/K
# HO2/OH ratio used to estimate HO2 from the archived OH. Boundary-layer
# measurements put HO2/OH at roughly 10 to 100, lower in NO-rich air
# (Stone, Whalley and Heard 2012, Chem. Soc. Rev. 41, 6348). The GEOS-Chem
# archive ORBIT reads carries OH, NO and NO2 but not HO2. The value 50 is an
# ORBIT assumption (1.0.3); replacing it with an archived HO2 field is a
# 1.0.4 candidate.
HO2_TO_OH_RATIO = float(os.environ.get("ORBIT_HO2_TO_OH_RATIO", "50"))


def compute_nox_regime(grid):
    """Return F_HIGH_NOX(ny, nx): the fraction of RO2 that reacts with NO.

    beta = k_NO [NO] / (k_NO [NO] + k_HO2 [HO2]), the branching ratio that
    Lane et al. (2008, Atmos. Environ. 42, 7439), Henze et al. (2008, ACP
    8, 2405) and Pye et al. (2010, ACP 10, 11261) use to interpolate between
    high-NOx (RO2 + NO) and low-NOx (RO2 + HO2) SOA yields. Surface layer
    of this bin's grid.

    Units: archive_NO is in ppbv and is converted to molec cm-3 with the
    layer pressure and temperature; archive_OH is in molec cm-3 and HO2 is
    taken as HO2_TO_OH_RATIO x OH (see above). Through 1.0.2 this function
    compared log10(NO2 [ppbv] / OH [molec cm-3]) with a threshold of 1.5,
    which mixed units and put every cell in the low-NOx regime
    (docs/notes/1.0.3_fixes/model_issues_2026-10-09.md, issue 3).

    Falls back to F = 0.5 (regime-neutral) if either archive field is absent.
    """
    no = getattr(grid, "archive_NO", np.array([]))
    oh = getattr(grid, "archive_OH", np.array([]))
    if no.size == 0 or oh.size == 0:
        return np.full((grid.ny, grid.nx), 0.5, dtype=np.float64)
    T = getattr(grid, "Temperature", np.array([]))
    P = getattr(grid, "Pressure", np.array([]))
    T_surf = T[0] if T.size else np.full((grid.ny, grid.nx), 298.0)
    P_surf = P[0] if P.size else np.full((grid.ny, grid.nx), 101325.0)
    n_air = P_surf / (K_BOLTZ * T_surf) * 1e-6            # molec cm-3
    no_molec = np.maximum(no[0], 0.0) * 1e-9 * n_air
    ho2_molec = np.maximum(oh[0], 0.0) * HO2_TO_OH_RATIO
    k_no = K_RO2_NO_A * np.exp(K_RO2_NO_B / T_surf)
    k_ho2 = K_RO2_HO2_A * np.exp(K_RO2_HO2_B / T_surf)
    num = k_no * no_molec
    den = num + k_ho2 * ho2_molec
    with np.errstate(invalid="ignore", divide="ignore"):
        beta = np.where(den > 0.0, num / den, 0.5)
    return np.clip(beta, 0.0, 1.0)


def compute_nox_regime_per_bin(grids):
    """F_HIGH_NOX for each bin grid, stacked to (n_bins, ny, nx)."""
    return np.stack([compute_nox_regime(g) for g in grids], axis=0)


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
    from orbit.emissions.vbs_yields import NOX_REGIME_PAIRS, regime_yields
    from orbit.emissions.netcdf import load_netcdf_source, load_netcdf_source_per_bin
    if bio_voc_split is None:
        bio_voc_split = BIO_VOC_SPLIT
    N = indexer.N
    ny, nx = grid.ny, grid.nx

    pcls = voc_source.voc_parent_class
    if pcls is None:
        return np.zeros((n_bins, N_SPECIES * N), dtype=np.float64), 0.0

    # Per-cell, per-bin effective yields (n_bins, 5, ny, nx). Every class
    # interpolates between its high- and low-NOx sets with the RO2 + NO
    # branching ratio F (1.0.3; through 1.0.2 only the anthropogenic class
    # did). F_HIGH_NOX is (ny, nx) for one map shared by every bin, or
    # (n_bins, ny, nx) for a per-bin map.
    F = np.asarray(F_HIGH_NOX, dtype=np.float64)
    if F.ndim == 2:
        F = np.broadcast_to(F[None], (n_bins, ny, nx))
    if pcls == "bio_voc":
        # CAMS biogenic VOC: lumped monoterpene + isoprene file. Split
        # by ORBIT_BIO_VOC_SPLIT (default 14/86 from Sindelarova 2014
        # Table 1), then combine the per-class yields.
        f_mono = bio_voc_split.get("monoterpene", 0.14)
        f_iso  = bio_voc_split.get("isoprene",    0.86)
        yields_eff = (f_mono * regime_yields("bio_monoterpene", F)
                      + f_iso * regime_yields("bio_isoprene", F))    # (n_bins, 5, ny, nx)
    elif pcls in NOX_REGIME_PAIRS:
        yields_eff = regime_yields(pcls, F)                           # (n_bins, 5, ny, nx)
    else:
        raise ValueError(
            f"Unknown VBS parent class '{pcls}' for source {voc_source.path}; "
            f"valid: {list(NOX_REGIME_PAIRS)} + 'bio_voc'"
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
        y_tau = yields_eff[tau]                                   # (5, ny, nx)
        for i_bin, solver_s in enumerate(bin_solver_idx):
            bin_3d = y_tau[i_bin, None] * voc_mass_per_bin_3d[tau]   # (nz, ny, nx)
            contrib[tau, solver_s * N:(solver_s + 1) * N] += bin_3d.ravel()
        voc_mass_total += float(voc_mass_per_bin_3d[tau].sum())

    return contrib, voc_mass_total
