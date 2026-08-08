"""Transposed growth Jacobian for the marginal-deaths adjoint pipeline.

The forward PM2.5 growth law (`orbit.modes.marginal._compute_delta_pm25`)
maps the 13-species element-mass state c to compound-mass surface PM2.5:

    δPM25_surf_τ[k] = δc_PM[k]
                    + (1/D_τ[k]) · Σ_i F_p_τ[i, k] · δc_VBS_BIN_i[k]
                    + p_nh_τ[k] · N_TO_NH4 · δc_TotalNH[k]
                    + S_TO_SO4         · δc_pSO4[k]
                    + p_no3_τ[k] · N_TO_NO3 · δc_TotalNO3[k]

where k ranges over surface cells (z=0). The growth Jacobian G_τ has
shape (N_surface, N_SPECIES · N_total_cells); the only nonzero entries
sit in the surface block of the contributing-species columns.

For the adjoint solve we never form G_τ. We just need its action on a
surface deaths-gradient field S_τ:

    receptor_τ = G_τ^T S_τ  ∈ R^{N_SPECIES · N_total_cells}

which is what :func:`apply_growth_transpose` returns. It is the
per-bin RHS forcing the adjoint orbit consumes (the R_τ in
`λ_τ = R_τ + P_τ^T λ_{τ+1}`).
"""
from __future__ import annotations

import numpy as np

from orbit.core.constants import N_TO_NH4, N_TO_NO3, S_TO_SO4
from orbit.core.deposition import (
    IDX_PM25, IDX_POA, IDX_TOTAL_NH, IDX_PSO4, IDX_TOTAL_NO3, IDX_VBS_BINS,
    N_SPECIES,
)
from orbit.modes.marginal import C_STAR_VALS, N_BINS


def _compute_D_per_bin(M_OA_flat, baseline_vbs_c_bin_flat, c_star):
    """Pankow partitioning-feedback denominator at one bin.

    D[k] = 1 − Σ_k_bin C_k · C*_k / (M_OA + C*_k)²

    Mirrors the form in orbit.modes.marginal._compute_delta_pm25
    (lines 720–727); clipped to [0.1, 1.0] to avoid blowups in
    pathological cells.
    """
    M_safe = np.maximum(M_OA_flat, 1.0e-6)
    D = np.ones_like(M_safe)
    for k_bin in range(len(c_star)):
        C_k = baseline_vbs_c_bin_flat[k_bin]
        D -= C_k * c_star[k_bin] / (M_safe + c_star[k_bin]) ** 2
    return np.clip(D, 0.1, 1.0)


def apply_growth_transpose(
    S_surface_per_bin: np.ndarray,
    grids,
    indexer,
    baseline_vbs_c: np.ndarray | None,
    *,
    p_nh4_surface_per_bin: np.ndarray | None = None,
    p_no3_surface_per_bin: np.ndarray | None = None,
) -> np.ndarray:
    """Map surface deaths gradient S (per bin) to its element-mass receptor.

    Parameters
    ----------
    S_surface_per_bin : ndarray, shape (N_BINS, n_lat, n_lon)
        Surface deaths-gradient field per bin (deaths · year⁻¹ · μg⁻¹ · m³ · cell⁻¹).
        Equal across bins for an annual-mean death gradient.
    grids : list of per-bin Grid objects
        Same objects the forward solver uses; carry F_p_vbs, M_OA_3d.
    indexer : CellIndexer
        Provides nz/ny/nx and the cell flattening convention.
    baseline_vbs_c : ndarray, shape (N_BINS, 5, nz, ny, nx) or None
        Baseline per-bin per-VBS-bin element-mass concentrations.
        If None, falls back to F_p only (no 1/D feedback) — matches
        the forward solver's fallback when M_OA isn't available.
    p_nh4_surface_per_bin, p_no3_surface_per_bin : (N_BINS, ny, nx) or None
        Per-bin surface ISORROPIA partitioning fractions (the
        thermodynamic f_NH4 = NH4/(NH4+NH3), f_NO3 = NO3/(NO3+HNO3)).
        These MUST come from the production orbit's ISORROPIA closure
        (iso_f_nh4_marg_3d / iso_f_no3_marg_3d preferred; iso_f_nh4_mean
        broadcast as a bin-mean approximation is acceptable). The
        legacy bin-file NHPartitioning/NOPartitioning fields are
        gas-particle ratios that don't capture the equilibrium nitrate
        split — passing those gives wrong NH/NO3 receptor contributions.
        If None, the function looks at grids[tau].NHPartitioning /
        NO3Partitioning as a fallback and warns if they're empty.

    Returns
    -------
    receptor : ndarray, shape (N_BINS, N_SPECIES, N)
        Per-bin adjoint receptor in flattened-state representation.
        Only surface cells (z=0) and the PM-contributing species
        (PM25, TotalNH, pSO4, TotalNO3, the 5 VBS bins) carry nonzeros.
    """
    nz = indexer.nz
    ny = indexer.ny
    nx = indexer.nx
    N = nz * ny * nx
    if S_surface_per_bin.shape != (N_BINS, ny, nx):
        raise ValueError(
            f"S_surface_per_bin must be (N_BINS={N_BINS}, ny={ny}, nx={nx}); "
            f"got {S_surface_per_bin.shape}"
        )

    receptor = np.zeros((N_BINS, N_SPECIES, N), dtype=np.float64)
    c_star = np.asarray(C_STAR_VALS, dtype=np.float64)

    # 3D index layout: indexer flattens as (z, y, x) → z * ny * nx + y * nx + x.
    # Surface block is the FIRST (ny * nx) entries (z=0).
    surface_size = ny * nx

    for tau in range(N_BINS):
        g = grids[tau]
        S_flat = S_surface_per_bin[tau].ravel()      # (ny*nx,)
        # Choose source of partitioning: explicit per-bin arg → fall back
        # to grid attribute → fail loudly if neither populated.
        if p_nh4_surface_per_bin is not None:
            p_nh = p_nh4_surface_per_bin[tau].ravel()
        else:
            p_nh = g.NHPartitioning.ravel()
            if p_nh.size == 0:
                raise ValueError(
                    "Empty NHPartitioning on grid and no p_nh4_surface_per_bin "
                    "passed; cannot compute NH receptor. Pass the production "
                    "iso_f_nh4 field from the orbit NPZ."
                )
        if p_no3_surface_per_bin is not None:
            p_no3 = p_no3_surface_per_bin[tau].ravel()
        else:
            p_no3 = g.NO3Partitioning.ravel()
            if p_no3.size == 0:
                raise ValueError(
                    "Empty NO3Partitioning on grid and no p_no3_surface_per_bin "
                    "passed; cannot compute NO3 receptor. Pass the production "
                    "iso_f_no3 field from the orbit NPZ."
                )

        # The receptor at species c, cell k (z=0 surface) is the coefficient
        # of c in the growth law, multiplied by S[k]:
        #
        #   δPM25_surf = ∂/∂(δc_PM)         · 1                        · δc_PM
        #              + ∂/∂(δc_VBS_i)      · F_p_i / D                · δc_VBS_i
        #              + ∂/∂(δc_TotalNH)    · p_nh · N_TO_NH4          · δc_TotalNH
        #              + ∂/∂(δc_pSO4)       · S_TO_SO4                 · δc_pSO4
        #              + ∂/∂(δc_TotalNO3)   · p_no3 · N_TO_NO3         · δc_TotalNO3
        #
        # All baseline coefficients are evaluated at the 3D state but the
        # receptor only lights up the SURFACE entries (k_surface = y*nx+x).
        # Off-surface entries stay zero — deaths are surface-only.

        # PM25 primary: coefficient 1.
        receptor[tau, IDX_PM25, :surface_size] = S_flat

        # Pankow partitioning-feedback denominator, shared by the POA and
        # VBS receptor terms. Hoisted above both so POA gets the same 1/D
        # the VBS bins get (see the POA block below). Ones = no baseline
        # VBS state, hence no feedback.
        F_p = getattr(g, "F_p_vbs", None)
        M_OA = getattr(g, "M_OA_3d", None)
        have_vbs = F_p is not None and F_p.shape[0] == len(IDX_VBS_BINS)
        if have_vbs and M_OA is not None and baseline_vbs_c is not None:
            M_OA_surface = M_OA.reshape(nz, ny, nx)[0].ravel()      # (ny*nx,)
            baseline_surface = baseline_vbs_c[tau, :, 0, :, :].reshape(
                len(IDX_VBS_BINS), -1
            )
            D = _compute_D_per_bin(M_OA_surface, baseline_surface, c_star)
        else:
            D = np.ones(surface_size, dtype=np.float64)

        # POA: coefficient 1/D — the SAME amplification the VBS bins carry.
        # POA is primary in transport but not inert thermodynamically: it is
        # part of M_OA, so adding POA raises the absorbing mass and pulls
        # semi-volatiles into the particle phase. See the derivation in
        # marginal._compute_delta_pm25; this must stay in lockstep with it or
        # adjoint/marginal duality breaks. Without any POA term at all the
        # receptor is identically zero and ∂J/∂e_POA comes back as a silent
        # zero for the 84.7% of CEDS anthropogenic primary mass POA carries.
        # Guarded for pre-split 13-species baselines.
        if N_SPECIES > IDX_POA:
            receptor[tau, IDX_POA, :surface_size] = S_flat / D

        # TotalNH: coefficient p_nh · N_TO_NH4. Surface slice. p_nh may
        # have arrived as a full 3D (`g.NHPartitioning`, size N) or as
        # surface-only (size surface_size via explicit kwarg).
        if p_nh.size == N:
            p_nh_surf = p_nh[:surface_size]
        elif p_nh.size == surface_size:
            p_nh_surf = p_nh
        else:
            raise ValueError(
                f"p_nh size {p_nh.size} must be N={N} (3D) or "
                f"surface_size={surface_size} (surface-only)."
            )
        receptor[tau, IDX_TOTAL_NH, :surface_size] = (
            S_flat * p_nh_surf * N_TO_NH4
        )

        # pSO4: coefficient S_TO_SO4.
        receptor[tau, IDX_PSO4, :surface_size] = S_flat * S_TO_SO4

        # TotalNO3: coefficient p_no3 · N_TO_NO3.
        if p_no3.size == N:
            p_no3_surf = p_no3[:surface_size]
        elif p_no3.size == surface_size:
            p_no3_surf = p_no3
        else:
            raise ValueError(
                f"p_no3 size {p_no3.size} must be N={N} (3D) or "
                f"surface_size={surface_size} (surface-only)."
            )
        receptor[tau, IDX_TOTAL_NO3, :surface_size] = (
            S_flat * p_no3_surf * N_TO_NO3
        )

        # VBS bins: coefficient F_p_i / D. D was computed above, shared
        # with the POA term.
        if have_vbs:
            for i, sp_idx in enumerate(IDX_VBS_BINS):
                # F_p shape: (5, nz, ny, nx). Surface = [i, 0, :, :].
                F_p_surface = F_p[i, 0].ravel()
                receptor[tau, sp_idx, :surface_size] = (
                    S_flat * F_p_surface / D
                )

    return receptor
