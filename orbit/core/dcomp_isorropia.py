"""ISORROPIA closure on ORBIT's transported TotalNH / TotalNO3.

For Phase 3d the outer iteration needs two partitioning fractions
(f_NH4, f_NO3) consistent with ORBIT's own TotalNH / TotalNO3 state,
not GEOS-Chem's archived concentrations.  These come from the 7D
ISORROPIA LUT.

Two flavours of each fraction are produced:

- **Equilibrium** (``*PartitioningEq``): the direct LUT f at the current
  total.  Used for the mass-extraction step (PM2.5 = primary + SOA +
  f_NH4·totalNH·MW + f_NO3·totalNO3·MW + ...).
- **Marginal** (``*Partitioning``): ``d(f·total) / d(total)`` evaluated
  at the current total by numerical differentiation.  Used for the
  deposition operator: the loss rate of a marginal addition of
  TotalNO3 or TotalNH is blended by this number.

Both are populated on the ``GridData`` so the existing deposition code
(``_get_no3_partitioning`` / ``_get_nh_partitioning``) picks them up
without a signature change.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from orbit.core.grid_data import GridData
from orbit.core.deposition import (
    IDX_TOTAL_NH, IDX_TOTAL_NO3, IDX_PSO4,
)


# Fountoukis & Nenes 2007: crustal Ca fraction of fine dust
_DUST_TO_CA = 0.036
# Seinfeld & Pandis Table 9.7: Na fraction of fine sea salt
_SEASALT_TO_NA = 0.306

# Perturbation for numerical differentiation (fractional).  1% is large
# enough that LUT interpolation noise stays well below the signal, small
# enough that the finite-difference approximation to the derivative is
# local.
_MARGINAL_DELTA = 0.01


def _query_partitioning(
    lut,
    totalSO4_flat,
    totalNH_flat,
    totalNO3_flat,
    Ca_flat,
    Na_flat,
    T_flat,
    RH_flat,
):
    """Return (f_nh4, f_no3) flat arrays on the LUT."""
    f_nh4, f_no3, _awater, _ph = lut.query(
        totalSO4_flat, totalNH_flat, totalNO3_flat,
        Ca_flat, Na_flat, T_flat, RH_flat,
    )
    return f_nh4, f_no3


def _concentration_for_bin(orbits, idx, tau):
    """End-of-bin concentration for species ``idx`` at bin ``tau``."""
    arr = np.asarray(orbits[idx][tau + 1], dtype=np.float64)
    return np.maximum(arr, 0.0)


def partitioning_per_bin(
    grids: List[GridData],
    orbits,
    lut,
    include_cross_partials: bool = True,
) -> Dict[int, Dict[str, np.ndarray]]:
    """Query the LUT at each bin's concentrations and compute the full
    ISORROPIA Jacobian block for use by iso-coupled marginal mode.

    Returns a dict {bin: {...fields...}} where each field is a
    (nz, ny, nx) array. Always includes:

      f_nh_eq, f_no3_eq    : direct LUT partitionings
      f_nh_marg, f_no3_marg: diagonal ∂(f·c)/∂c via symmetric ±δ FD

    With ``include_cross_partials=True`` (Phase 6a-rev, default per the
    iso-coupled-marginal addendum), also includes the 4 cross-coupling
    partials:

      f_nh_dno3  = ∂(f_nh4 · NH) / ∂NO3   |_{NH, SO4 fixed}
      f_no3_dnh  = ∂(f_no3 · NO3) / ∂NH   |_{NO3, SO4 fixed}
      f_nh_dso4  = ∂(f_nh4 · NH) / ∂SO4   |_{NH, NO3 fixed}
      f_no3_dso4 = ∂(f_no3 · NO3) / ∂SO4  |_{NH, NO3 fixed}

    These are the linearised NH4↔NO3↔SO4 ISORROPIA cross-couplings
    that iso-coupled marginal mode uses to capture ammonium/nitrate
    competition without re-running the LUT.

    Plus a per-cell asymmetry diagnostic (Phase 6a-rev / tipping-points
    follow-up paper):

      f_nh_marg_asym, f_no3_marg_asym : |f(+δ) + f(-δ) - 2·f(0)|
                                        / max(|f(+δ)| + |f(-δ)|, 1e-12)
                                        per cell. Cells with large
                                        asymmetry sit near regime
                                        thresholds (basin boundary).

    SO4 row of the 3×3 block is not computed: the LUT does not carry
    f_so4 (only f_nh4 + f_no3 are stored), and pSO4 is treated as
    ~100% particle in ORBIT. Computing the SO4 row would require
    regenerating the LUT.

    Symmetric FD (±δ averaged) reduces sensitivity near regime
    thresholds and makes the asymmetry diagnostic available. Cost:
    each diagonal probe is now 2 LUT queries (vs. 1 one-sided). Cross
    partials use the +δ branch of the symmetric probes (no extra
    queries beyond what the diagonals already required, except for
    SO4 which is one extra ±δ pair). Total ≈ 8 LUT queries per cell
    per bin (vs. 3 one-sided pre-rev), ~10-20% more wall.
    """
    out: Dict[int, Dict[str, np.ndarray]] = {}
    for tau, grid in enumerate(grids):
        nz, ny, nx = grid.nz, grid.ny, grid.nx
        N = nz * ny * nx

        # ORBIT's element-mass element totals (ug/m3).
        totalNH = _concentration_for_bin(orbits, IDX_TOTAL_NH, tau).reshape(N)
        totalNO3 = _concentration_for_bin(orbits, IDX_TOTAL_NO3, tau).reshape(N)
        totalSO4 = _concentration_for_bin(orbits, IDX_PSO4, tau).reshape(N)

        # Meteorology + crustal/seasalt
        T_flat = grid.Temperature.ravel()
        RH_flat = grid.RH.ravel() / 100.0  # percent -> fraction
        if grid.dust_fine.size == N:
            Ca_flat = grid.dust_fine.ravel() * _DUST_TO_CA
        else:
            Ca_flat = np.zeros(N)
        if grid.sea_salt_fine.size == N:
            Na_flat = grid.sea_salt_fine.ravel() * _SEASALT_TO_NA
        else:
            Na_flat = np.zeros(N)

        # Baseline (zero-perturbation) LUT query.
        f_nh_eq, f_no3_eq = _query_partitioning(
            lut, totalSO4, totalNH, totalNO3, Ca_flat, Na_flat, T_flat, RH_flat,
        )
        base_part_nh  = f_nh_eq  * totalNH
        base_part_no3 = f_no3_eq * totalNO3

        # ── Diagonal probes ────────────────────────────────────────
        # Symmetric ±δ FD per the iso-coupled-marginal addendum:
        # average the +δ and -δ derivatives and emit an asymmetry
        # diagnostic. Cells with |f(+δ) + f(-δ) - 2·f(0)| substantial
        # are near regime thresholds (basin-flip candidates).

        d = _MARGINAL_DELTA

        # NH probe (±δ on totalNH)
        NH_plus  = totalNH * (1.0 + d) + 1e-12
        NH_minus = np.maximum(totalNH * (1.0 - d), 0.0)
        f_nh_p, f_no3_at_nh_p = _query_partitioning(
            lut, totalSO4, NH_plus,  totalNO3, Ca_flat, Na_flat, T_flat, RH_flat,
        )
        f_nh_m, f_no3_at_nh_m = _query_partitioning(
            lut, totalSO4, NH_minus, totalNO3, Ca_flat, Na_flat, T_flat, RH_flat,
        )

        dNH_plus  = NH_plus  - totalNH
        dNH_minus = totalNH  - NH_minus
        # ∂(f_nh · NH)/∂NH from +δ and -δ branches
        deriv_nh_plus  = np.where(
            dNH_plus  > 0, (f_nh_p * NH_plus  - base_part_nh) / np.maximum(dNH_plus,  1e-30), f_nh_eq,
        )
        deriv_nh_minus = np.where(
            dNH_minus > 0, (base_part_nh - f_nh_m * NH_minus) / np.maximum(dNH_minus, 1e-30), f_nh_eq,
        )
        f_nh_marg = np.clip(0.5 * (deriv_nh_plus + deriv_nh_minus), 0.0, 1.0)
        f_nh_marg_asym = (
            np.abs(f_nh_p + f_nh_m - 2.0 * f_nh_eq)
            / np.maximum(np.abs(f_nh_p) + np.abs(f_nh_m), 1e-12)
        )

        # NO3 probe (±δ on totalNO3)
        NO3_plus  = totalNO3 * (1.0 + d) + 1e-12
        NO3_minus = np.maximum(totalNO3 * (1.0 - d), 0.0)
        f_nh_at_no3_p, f_no3_p = _query_partitioning(
            lut, totalSO4, totalNH, NO3_plus,  Ca_flat, Na_flat, T_flat, RH_flat,
        )
        f_nh_at_no3_m, f_no3_m = _query_partitioning(
            lut, totalSO4, totalNH, NO3_minus, Ca_flat, Na_flat, T_flat, RH_flat,
        )
        dNO3_plus  = NO3_plus  - totalNO3
        dNO3_minus = totalNO3  - NO3_minus
        deriv_no3_plus  = np.where(
            dNO3_plus  > 0, (f_no3_p * NO3_plus  - base_part_no3) / np.maximum(dNO3_plus,  1e-30), f_no3_eq,
        )
        deriv_no3_minus = np.where(
            dNO3_minus > 0, (base_part_no3 - f_no3_m * NO3_minus) / np.maximum(dNO3_minus, 1e-30), f_no3_eq,
        )
        f_no3_marg = np.clip(0.5 * (deriv_no3_plus + deriv_no3_minus), 0.0, 1.0)
        f_no3_marg_asym = (
            np.abs(f_no3_p + f_no3_m - 2.0 * f_no3_eq)
            / np.maximum(np.abs(f_no3_p) + np.abs(f_no3_m), 1e-12)
        )

        bin_out = {
            "f_nh_eq":          f_nh_eq.reshape((nz, ny, nx)),
            "f_nh_marg":        f_nh_marg.reshape((nz, ny, nx)),
            "f_no3_eq":         f_no3_eq.reshape((nz, ny, nx)),
            "f_no3_marg":       f_no3_marg.reshape((nz, ny, nx)),
            "f_nh_marg_asym":   f_nh_marg_asym.reshape((nz, ny, nx)),
            "f_no3_marg_asym":  f_no3_marg_asym.reshape((nz, ny, nx)),
        }

        if include_cross_partials:
            # MARGINAL cross-derivatives — clipped, symmetric-±δ-averaged
            # (corrected 2026-05-01).
            #
            # The deposition operator (forward + marg) consumes the
            # CLIPPED diagonal f_marg (clip 0..1 above, line 177/200).
            # The cross-derivative ∂f_marg/∂c_other entering the iso
            # K-block must therefore be the derivative of the SAME
            # clipped f_marg, otherwise the K-block predicts a forcing
            # the operator cannot exhibit. At sulfate-saturated cells
            # (Delhi in winter, much of the IGP), the unclipped f_marg
            # would go negative; clipping pins the operator's value at
            # 0; the unclipped derivative is non-zero but the clipped
            # one is exactly 0 — the empirical 50% marg/zo magnitude
            # gap was driven by this mismatch.
            #
            # Approach: at each c_other ± δ slice, compute f_marg_self
            # via the same symmetric-±δ-averaged + clipped scheme used
            # by the diagonal probe; then centered-FD over c_other to
            # get ∂f_marg/∂c_other.
            #
            # Cost: 2 extra LUT queries per cell-bin (the central
            # (NH=c, NO3=c, SO4±δ) point shared between the NH-SO4 and
            # NO3-SO4 planes). The (NH=c, NO3±δ, SO4=c) central points
            # are reused from the diagonal probe.

            SO4_plus  = totalSO4 * (1.0 + d) + 1e-12
            SO4_minus = np.maximum(totalSO4 * (1.0 - d), 0.0)
            d_NH = NH_plus - NH_minus    # 2δ in NH
            d_NO3 = NO3_plus - NO3_minus
            d_SO4 = SO4_plus - SO4_minus
            safe_d_NH  = np.maximum(d_NH,  1e-30)
            safe_d_NO3 = np.maximum(d_NO3, 1e-30)
            safe_d_SO4 = np.maximum(d_SO4, 1e-30)
            safe_dNH_p = np.maximum(dNH_plus,  1e-30)
            safe_dNH_m = np.maximum(dNH_minus, 1e-30)
            safe_dNO3_p = np.maximum(dNO3_plus,  1e-30)
            safe_dNO3_m = np.maximum(dNO3_minus, 1e-30)

            # ─── NH-NO3 plane: 4 corners (no extra queries beyond v3) ───
            f_eq_pp_nh, f_eq_pp_no3 = _query_partitioning(
                lut, totalSO4, NH_plus,  NO3_plus,  Ca_flat, Na_flat, T_flat, RH_flat,
            )
            f_eq_pm_nh, f_eq_pm_no3 = _query_partitioning(
                lut, totalSO4, NH_plus,  NO3_minus, Ca_flat, Na_flat, T_flat, RH_flat,
            )
            f_eq_mp_nh, f_eq_mp_no3 = _query_partitioning(
                lut, totalSO4, NH_minus, NO3_plus,  Ca_flat, Na_flat, T_flat, RH_flat,
            )
            f_eq_mm_nh, f_eq_mm_no3 = _query_partitioning(
                lut, totalSO4, NH_minus, NO3_minus, Ca_flat, Na_flat, T_flat, RH_flat,
            )
            # Central NH at NO3 ± δ (reused from diagonal probe):
            #   f_nh_at_no3_p, f_nh_at_no3_m, f_no3_at_nh_p, f_no3_at_nh_m

            # f_marg_NH at NO3+δ (clipped, symmetric±δ avg over c_NH at NO3+δ slice).
            base_part_nh_at_no3_p = f_nh_at_no3_p * totalNH
            deriv_nh_p_at_no3_p = np.where(
                dNH_plus > 0,
                (f_eq_pp_nh * NH_plus - base_part_nh_at_no3_p) / safe_dNH_p,
                f_nh_at_no3_p,
            )
            deriv_nh_m_at_no3_p = np.where(
                dNH_minus > 0,
                (base_part_nh_at_no3_p - f_eq_mp_nh * NH_minus) / safe_dNH_m,
                f_nh_at_no3_p,
            )
            f_marg_nh_at_no3_plus = np.clip(
                0.5 * (deriv_nh_p_at_no3_p + deriv_nh_m_at_no3_p), 0.0, 1.0,
            )
            # f_marg_NH at NO3-δ
            base_part_nh_at_no3_m = f_nh_at_no3_m * totalNH
            deriv_nh_p_at_no3_m = np.where(
                dNH_plus > 0,
                (f_eq_pm_nh * NH_plus - base_part_nh_at_no3_m) / safe_dNH_p,
                f_nh_at_no3_m,
            )
            deriv_nh_m_at_no3_m = np.where(
                dNH_minus > 0,
                (base_part_nh_at_no3_m - f_eq_mm_nh * NH_minus) / safe_dNH_m,
                f_nh_at_no3_m,
            )
            f_marg_nh_at_no3_minus = np.clip(
                0.5 * (deriv_nh_p_at_no3_m + deriv_nh_m_at_no3_m), 0.0, 1.0,
            )
            df_marg_nh_dno3 = (f_marg_nh_at_no3_plus - f_marg_nh_at_no3_minus) / safe_d_NO3
            f_nh_dno3 = totalNH * df_marg_nh_dno3

            # f_marg_NO3 at NH+δ (clipped, symmetric±δ avg over c_NO3 at NH+δ slice).
            base_part_no3_at_nh_p = f_no3_at_nh_p * totalNO3
            deriv_no3_p_at_nh_p = np.where(
                dNO3_plus > 0,
                (f_eq_pp_no3 * NO3_plus - base_part_no3_at_nh_p) / safe_dNO3_p,
                f_no3_at_nh_p,
            )
            deriv_no3_m_at_nh_p = np.where(
                dNO3_minus > 0,
                (base_part_no3_at_nh_p - f_eq_pm_no3 * NO3_minus) / safe_dNO3_m,
                f_no3_at_nh_p,
            )
            f_marg_no3_at_nh_plus = np.clip(
                0.5 * (deriv_no3_p_at_nh_p + deriv_no3_m_at_nh_p), 0.0, 1.0,
            )
            # f_marg_NO3 at NH-δ
            base_part_no3_at_nh_m = f_no3_at_nh_m * totalNO3
            deriv_no3_p_at_nh_m = np.where(
                dNO3_plus > 0,
                (f_eq_mp_no3 * NO3_plus - base_part_no3_at_nh_m) / safe_dNO3_p,
                f_no3_at_nh_m,
            )
            deriv_no3_m_at_nh_m = np.where(
                dNO3_minus > 0,
                (base_part_no3_at_nh_m - f_eq_mm_no3 * NO3_minus) / safe_dNO3_m,
                f_no3_at_nh_m,
            )
            f_marg_no3_at_nh_minus = np.clip(
                0.5 * (deriv_no3_p_at_nh_m + deriv_no3_m_at_nh_m), 0.0, 1.0,
            )
            df_marg_no3_dnh = (f_marg_no3_at_nh_plus - f_marg_no3_at_nh_minus) / safe_d_NH
            f_no3_dnh = totalNO3 * df_marg_no3_dnh

            # ─── NH-SO4 plane (NH±δ, NO3=c, SO4±δ corners) ────────────
            f_eq_pp_nh_so4, _ = _query_partitioning(
                lut, SO4_plus,  NH_plus,  totalNO3, Ca_flat, Na_flat, T_flat, RH_flat,
            )
            f_eq_pm_nh_so4, _ = _query_partitioning(
                lut, SO4_minus, NH_plus,  totalNO3, Ca_flat, Na_flat, T_flat, RH_flat,
            )
            f_eq_mp_nh_so4, _ = _query_partitioning(
                lut, SO4_plus,  NH_minus, totalNO3, Ca_flat, Na_flat, T_flat, RH_flat,
            )
            f_eq_mm_nh_so4, _ = _query_partitioning(
                lut, SO4_minus, NH_minus, totalNO3, Ca_flat, Na_flat, T_flat, RH_flat,
            )

            # ─── NO3-SO4 plane (NH=c, NO3±δ, SO4±δ corners) ───────────
            _, f_eq_pp_no3_so4 = _query_partitioning(
                lut, SO4_plus,  totalNH, NO3_plus,  Ca_flat, Na_flat, T_flat, RH_flat,
            )
            _, f_eq_pm_no3_so4 = _query_partitioning(
                lut, SO4_minus, totalNH, NO3_plus,  Ca_flat, Na_flat, T_flat, RH_flat,
            )
            _, f_eq_mp_no3_so4 = _query_partitioning(
                lut, SO4_plus,  totalNH, NO3_minus, Ca_flat, Na_flat, T_flat, RH_flat,
            )
            _, f_eq_mm_no3_so4 = _query_partitioning(
                lut, SO4_minus, totalNH, NO3_minus, Ca_flat, Na_flat, T_flat, RH_flat,
            )

            # NEW (the +2 cost beyond v3): central (NH=c, NO3=c, SO4±δ)
            # — needed by the clipped f_marg_self at the SO4±δ slice.
            # Each query returns both f_nh (used by NH-SO4 plane) and
            # f_no3 (used by NO3-SO4 plane).
            f_eq_c_nh_at_so4_p, f_eq_c_no3_at_so4_p = _query_partitioning(
                lut, SO4_plus,  totalNH, totalNO3, Ca_flat, Na_flat, T_flat, RH_flat,
            )
            f_eq_c_nh_at_so4_m, f_eq_c_no3_at_so4_m = _query_partitioning(
                lut, SO4_minus, totalNH, totalNO3, Ca_flat, Na_flat, T_flat, RH_flat,
            )

            # f_marg_NH at SO4+δ (clipped, symmetric±δ avg over c_NH).
            base_part_nh_at_so4_p = f_eq_c_nh_at_so4_p * totalNH
            deriv_nh_p_at_so4_p = np.where(
                dNH_plus > 0,
                (f_eq_pp_nh_so4 * NH_plus - base_part_nh_at_so4_p) / safe_dNH_p,
                f_eq_c_nh_at_so4_p,
            )
            deriv_nh_m_at_so4_p = np.where(
                dNH_minus > 0,
                (base_part_nh_at_so4_p - f_eq_mp_nh_so4 * NH_minus) / safe_dNH_m,
                f_eq_c_nh_at_so4_p,
            )
            f_marg_nh_at_so4_plus = np.clip(
                0.5 * (deriv_nh_p_at_so4_p + deriv_nh_m_at_so4_p), 0.0, 1.0,
            )
            # f_marg_NH at SO4-δ
            base_part_nh_at_so4_m = f_eq_c_nh_at_so4_m * totalNH
            deriv_nh_p_at_so4_m = np.where(
                dNH_plus > 0,
                (f_eq_pm_nh_so4 * NH_plus - base_part_nh_at_so4_m) / safe_dNH_p,
                f_eq_c_nh_at_so4_m,
            )
            deriv_nh_m_at_so4_m = np.where(
                dNH_minus > 0,
                (base_part_nh_at_so4_m - f_eq_mm_nh_so4 * NH_minus) / safe_dNH_m,
                f_eq_c_nh_at_so4_m,
            )
            f_marg_nh_at_so4_minus = np.clip(
                0.5 * (deriv_nh_p_at_so4_m + deriv_nh_m_at_so4_m), 0.0, 1.0,
            )
            df_marg_nh_dso4 = (f_marg_nh_at_so4_plus - f_marg_nh_at_so4_minus) / safe_d_SO4
            f_nh_dso4 = totalNH * df_marg_nh_dso4

            # f_marg_NO3 at SO4+δ (clipped, symmetric±δ avg over c_NO3).
            base_part_no3_at_so4_p = f_eq_c_no3_at_so4_p * totalNO3
            deriv_no3_p_at_so4_p = np.where(
                dNO3_plus > 0,
                (f_eq_pp_no3_so4 * NO3_plus - base_part_no3_at_so4_p) / safe_dNO3_p,
                f_eq_c_no3_at_so4_p,
            )
            deriv_no3_m_at_so4_p = np.where(
                dNO3_minus > 0,
                (base_part_no3_at_so4_p - f_eq_mp_no3_so4 * NO3_minus) / safe_dNO3_m,
                f_eq_c_no3_at_so4_p,
            )
            f_marg_no3_at_so4_plus = np.clip(
                0.5 * (deriv_no3_p_at_so4_p + deriv_no3_m_at_so4_p), 0.0, 1.0,
            )
            # f_marg_NO3 at SO4-δ
            base_part_no3_at_so4_m = f_eq_c_no3_at_so4_m * totalNO3
            deriv_no3_p_at_so4_m = np.where(
                dNO3_plus > 0,
                (f_eq_pm_no3_so4 * NO3_plus - base_part_no3_at_so4_m) / safe_dNO3_p,
                f_eq_c_no3_at_so4_m,
            )
            deriv_no3_m_at_so4_m = np.where(
                dNO3_minus > 0,
                (base_part_no3_at_so4_m - f_eq_mm_no3_so4 * NO3_minus) / safe_dNO3_m,
                f_eq_c_no3_at_so4_m,
            )
            f_marg_no3_at_so4_minus = np.clip(
                0.5 * (deriv_no3_p_at_so4_m + deriv_no3_m_at_so4_m), 0.0, 1.0,
            )
            df_marg_no3_dso4 = (f_marg_no3_at_so4_plus - f_marg_no3_at_so4_minus) / safe_d_SO4
            f_no3_dso4 = totalNO3 * df_marg_no3_dso4

            bin_out["f_nh_dno3"]  = f_nh_dno3.reshape((nz, ny, nx))
            bin_out["f_no3_dnh"]  = f_no3_dnh.reshape((nz, ny, nx))
            bin_out["f_nh_dso4"]  = f_nh_dso4.reshape((nz, ny, nx))
            bin_out["f_no3_dso4"] = f_no3_dso4.reshape((nz, ny, nx))

        out[tau] = bin_out
    return out


def update_grid_partitioning(
    grids: List[GridData],
    orbits,
    lut,
    alpha: float = 0.5,
    prev_per_bin=None,
    include_cross_partials: bool = True,
) -> Dict[int, Dict[str, np.ndarray]]:
    """Query LUT and update per-bin ``GridData`` partitioning fields.

    Mutates each grid: sets ``NHPartitioning`` (marginal), ``NHPartitioningEq``,
    ``NO3Partitioning`` (marginal), ``NO3PartitioningEq``.  The operator
    (deposition) consumes the marginal form; mass extraction consumes
    equilibrium.

    Parameters
    ----------
    grids : list of 8 GridData
    orbits : {species_idx: [c_0, ..., c_8]}
    lut : IsorropiaLUT
    alpha : float
        Under-relaxation: blend the new fraction with the prior iteration's.
        Use 1.0 to take the new value directly (no damping).
    prev_per_bin : dict or None
        Previous-iteration per-bin dict (same shape as the return of
        ``partitioning_per_bin``) — used for the under-relaxation blend.
        If None or empty, no blending is applied.

    Returns
    -------
    per_bin : dict {bin: {"f_nh_eq", ...}}
        Blended (post-relaxation) partitioning for this iteration.  Feed
        back on the next call as ``prev_per_bin``.
    """
    new_per_bin = partitioning_per_bin(
        grids, orbits, lut,
        include_cross_partials=include_cross_partials,
    )

    blended_per_bin = {}
    for tau, grid in enumerate(grids):
        new = new_per_bin[tau]
        if prev_per_bin and tau in prev_per_bin and 0.0 < alpha < 1.0:
            prev = prev_per_bin[tau]
            blended = {
                k: alpha * new[k] + (1.0 - alpha) * prev[k]
                for k in new.keys()
            }
        else:
            blended = new
        blended_per_bin[tau] = blended

        # Mutate the grid so downstream (deposition, mass extraction)
        # picks up the new fractions without any wiring change.
        grid.NHPartitioning    = blended["f_nh_marg"]
        grid.NHPartitioningEq  = blended["f_nh_eq"]
        grid.NO3Partitioning   = blended["f_no3_marg"]
        grid.NO3PartitioningEq = blended["f_no3_eq"]

    return blended_per_bin


def partitioning_summary(per_bin: Dict[int, Dict[str, np.ndarray]]):
    """Print a short per-bin surface-mean summary of (NH, NO3) partitioning."""
    for tau in sorted(per_bin.keys()):
        d = per_bin[tau]
        print(f"    Bin {tau + 1}: "
              f"f_NH4 eq={d['f_nh_eq'][0].mean():.3f} marg={d['f_nh_marg'][0].mean():.3f}  "
              f"f_NO3 eq={d['f_no3_eq'][0].mean():.3f} marg={d['f_no3_marg'][0].mean():.3f}")
