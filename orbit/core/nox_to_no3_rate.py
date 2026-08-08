"""Archive-driven NOx -> TotalNO3 first-order rate for the
``feat/orbit-isorropia-archive`` branch.

All oxidant inputs come from the GC 3h archive via
``grid.archive_{OH,NO,NO2,NO3rad}`` — no diagnostic OH, no PSS iterate.
Two channels are superposed into a single per-cell rate ``k_tot`` (s^-1)
consumed by ``operator.assemble_species_operators`` as ``nox_to_no3_rate``:

* **Daytime (OH + NO2)**: pseudo-first-order on NOx::

      k_day = k_{OH+NO2}(T, M) · [OH] · f_NO2
              where f_NO2 = [NO2] / ([NO] + [NO2])

* **Nighttime (N2O5 hydrolysis)**: N2O5 diagnosed from NO2 + NO3-radical
  thermal equilibrium, then hydrolysed on fine aerosol with a constant
  uptake coefficient::

      [N2O5] = K_eq(T) · [NO2] · [NO3_rad]
      k_het  = γ_N2O5 · v_th/4 · A_fine
      k_night = 2 · k_het · [N2O5] / [NOx]

  The factor of 2 accounts for N2O5 hydrolysis yielding two HNO3. The
  daytime ``[NO3_rad]`` in the archive is typically a numerical floor
  (photolysed away), so ``k_night`` is naturally small when the sun is
  up and does not double-count with ``k_day``.

Fine-mode aerosol surface area is computed inline from existing bin file
fields (dust_fine, sea_salt_fine, pS, pNH, pNO) under a monodisperse-sphere
approximation with an effective radius of 0.15 μm and a bulk density of
1.7 g/cm3 — standard continental-polluted accumulation-mode assumption.
"""

from __future__ import annotations

import numpy as np

from orbit.core.grid_data import GridData
from orbit.core.constants import N_TO_NH4, N_TO_NO3, S_TO_SO4
from orbit.core.oxidants import (
    air_density,
    k_OH_NO2,
    Keq_N2O5,
    mean_speed_N2O5,
)

# Accumulation-mode aerosol assumption (see module docstring).
_FINE_RHO_KGM3 = 1700.0
_FINE_REFF_M   = 0.15e-6

# JPL/IUPAC recommended N2O5 uptake coefficient on mixed sulfate-ammonium
# nitrate aerosol, aqueous conditions. Real γ varies 0.005-0.05 with RH and
# composition; a constant 0.02 is a conventional v1 choice.
GAMMA_N2O5 = 0.02


def _ppbv_to_molcm3(c_ppbv: np.ndarray, M_molcm3: np.ndarray) -> np.ndarray:
    """Convert mixing ratio in ppbv to number density in molec/cm3.

    c_ppbv · 1e-9 · M = molec/cm3, where M is total air number density.
    """
    return np.asarray(c_ppbv) * 1e-9 * M_molcm3


def _fine_aerosol_surface_area(grid: GridData) -> np.ndarray:
    """Fine-mode aerosol surface area (m2 / m3) from bin file fields.

    SA = 3 · m / (ρ · r_eff) for monodisperse spheres. Inputs are ug/m3 of
    element or compound mass; converted to kg/m3 before the formula. SO4,
    NH4, NO3 element-to-compound conversions use the standard element-to-
    compound mass ratios already defined in ``orbit.core.constants``.
    """
    nz, ny, nx = grid.nz, grid.ny, grid.nx
    shape = (nz, ny, nx)
    fine_ugm3 = np.zeros(shape, dtype=np.float64)

    if grid.pS.size > 0:
        fine_ugm3 = fine_ugm3 + grid.pS * S_TO_SO4          # S mass -> SO4 compound mass
    if grid.pNH.size > 0:
        fine_ugm3 = fine_ugm3 + grid.pNH * N_TO_NH4          # N mass -> NH4 compound mass
    if grid.pNO.size > 0:
        fine_ugm3 = fine_ugm3 + grid.pNO * N_TO_NO3          # N mass -> NO3 compound mass
    if grid.dust_fine.size > 0:
        fine_ugm3 = fine_ugm3 + grid.dust_fine
    if grid.sea_salt_fine.size > 0:
        fine_ugm3 = fine_ugm3 + grid.sea_salt_fine

    mass_kgm3 = fine_ugm3 * 1e-9                              # ug/m3 -> kg/m3
    sa_per_kg = 3.0 / (_FINE_RHO_KGM3 * _FINE_REFF_M)         # m2/kg
    return mass_kgm3 * sa_per_kg                              # m2/m3


def build_nox_to_no3_rate(
    grid: GridData,
    disable_night: bool = False,
) -> np.ndarray:
    """Assemble the per-cell NOx -> TotalNO3 first-order rate (s^-1).

    Returns ``None`` when the archive oxidant fields are not populated
    (e.g. old bin files without the Step-1 preprocessor rerun). Callers
    should fall back to ``nox_to_no3_rate=None`` in that case, which
    leaves the coupling inactive.

    Parameters
    ----------
    grid : GridData
        Must carry ``archive_OH`` (molec/cm3) and ``archive_NO``,
        ``archive_NO2``, ``archive_NO3rad`` (ppbv), plus ``Temperature``
        and ``Pressure``.
    disable_night : bool
        If True, zero the N2O5 hydrolysis channel (ablation).
    """
    if grid.archive_OH.size == 0 or grid.archive_NO2.size == 0:
        return None

    T = grid.Temperature
    P = grid.Pressure
    M = air_density(T, P)                                    # molec/cm3

    # Daytime OH + NO2 channel.
    OH = grid.archive_OH                                      # molec/cm3 (native)
    NO_molcm3  = _ppbv_to_molcm3(grid.archive_NO,  M) if grid.archive_NO.size  > 0 else np.zeros_like(OH)
    NO2_molcm3 = _ppbv_to_molcm3(grid.archive_NO2, M)
    NOx = NO_molcm3 + NO2_molcm3
    safe_NOx = np.where(NOx > 0, NOx, 1.0)
    f_NO2 = np.where(NOx > 0, NO2_molcm3 / safe_NOx, 1.0)    # night -> all NO2

    k_day = k_OH_NO2(T, M) * OH * f_NO2                       # s^-1

    # Nighttime N2O5 hydrolysis channel.
    k_night = np.zeros_like(k_day)
    if not disable_night and grid.archive_NO3rad.size > 0:
        NO3rad_molcm3 = _ppbv_to_molcm3(grid.archive_NO3rad, M)
        N2O5 = Keq_N2O5(T) * NO2_molcm3 * NO3rad_molcm3       # molec/cm3 (thermal eq)

        A_fine_m2m3 = _fine_aerosol_surface_area(grid)        # m2/m3
        A_fine_cm2cm3 = A_fine_m2m3 * 1e-2                    # m2/m3 -> cm2/cm3
        v_th_cms = mean_speed_N2O5(T)                         # cm/s
        k_het = 0.25 * GAMMA_N2O5 * v_th_cms * A_fine_cm2cm3  # s^-1 (N2O5 loss)

        # k_night is the pseudo-first-order loss on NOx: each N2O5 hydrolysis
        # removes 2 NOx equivalents (goes to 2 HNO3).
        k_night = np.where(
            NOx > 0,
            2.0 * k_het * N2O5 / safe_NOx,
            0.0,
        )

    return k_day + k_night
