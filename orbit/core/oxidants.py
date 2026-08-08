"""Cell-local per-bin diagnostic oxidants for DCOMP chemistry.

Computes [OH], [HO2], the NO/NO2 photostationary-state split, and nighttime
[NO3]/[N2O5] from precursor concentrations, met state, and photolysis rates.
All quantities are diagnostic — computed by steady-state algebra at each
cell, not transported. This is justified by their short lifetimes
(OH ~1 s, HO2 ~seconds, NO3 ~minutes) relative to bin duration (3 h).

Inputs come from:
  - Orbit solve      : c_NOx, c_SO2 (N- and S-element mass ug/m3)
  - HEMCO climatology: CO, CH4 (compound mass ug/m3, monthly mean)
  - Preprocessor     : speciated VOCs (ISOP/MTPA/MTPO/LIMO/BENZ/TOLU/XYLE/NAP),
                       T, RH, pressure, aerosol surface area
  - Photolysis module: j_NO2, j_O1D per bin

Outputs (per bin, per cell, on the ORBIT grid):
  - OH_molcm3, HO2_molcm3 : molec/cm3
  - NO_molcm3, NO2_molcm3 : molec/cm3 (PSS split of NOx)
  - f_NO2                 : NO2 fraction of NOx (dimensionless)
  - NO3_molcm3, N2O5_molcm3 : molec/cm3 (nighttime)
  - k_n2o5_het            : N2O5 heterogeneous hydrolysis rate, s^-1

The implicit OH-HO2 coupling is closed by 3-5 fixed-point iterations
from a high-NOx limit initial guess; SAS winter urban regime is firmly
high-NOx so convergence is fast. For remote clean conditions (low NOx),
the iteration takes a few more steps but remains stable.

References
----------
Rate constants: NASA JPL Publication 19-5, Burkholder et al. (2020).
OH steady state: standard formulation, e.g. Jacob (1999) "Introduction
   to Atmospheric Chemistry".
N2O5 hydrolysis: Bertram and Thornton (2009) for gamma parameterisation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

# Physical constants
AVOGADRO = 6.02214076e23         # molec/mol
KB = 1.380649e-23                # J/K Boltzmann
R_GAS = 8.31446                  # J/mol/K

# Molecular weights (g/mol)
MW_N  = 14.007
MW_S  = 32.065
MW_O3 = 48.0
MW_CO = 28.01
MW_CH4 = 16.04
MW_SO2 = 64.07
MW_NO = 30.01
MW_NO2 = 46.01
MW_OH = 17.01   # Used by the prescribed-OH ablation when converting
                # HEMCO monthly-mean OH from ug/m3 compound mass to molec/cm3.

# Molecular weights for speciated VOCs (g/mol) used to convert compound
# mass concentrations to molec/cm3 for rate evaluation.
MW_VOC = {
    "ISOP": 68.12, "MTPA": 136.23, "MTPO": 136.23, "LIMO": 136.23,
    "BENZ": 78.11, "TOLU": 92.14,  "XYLE": 106.16, "NAP":  128.17,
}

# VOC+OH rate constants at 298 K (cm3/molec/s), JPL/IUPAC recommended,
# surface-tropospheric conditions. Order-of-magnitude values; a fuller
# treatment would include Arrhenius temperature dependence per species.
# Sources: Atkinson et al. 2006 (IUPAC), JPL 19-5.
K_VOC_OH_298 = {
    "ISOP": 1.0e-10,   # isoprene
    "MTPA": 5.3e-11,   # alpha-pinene
    "MTPO": 8.7e-11,   # other monoterpene (avg)
    "LIMO": 1.7e-10,   # limonene
    "BENZ": 1.2e-12,   # benzene
    "TOLU": 5.6e-12,   # toluene
    "XYLE": 1.4e-11,   # xylene (mixed)
    "NAP":  2.3e-11,   # naphthalene
}

# RO2 yield per OH+VOC reaction (approx; first-generation products).
Y_RO2 = {
    "ISOP": 1.0, "MTPA": 1.0, "MTPO": 1.0, "LIMO": 1.0,
    "BENZ": 0.5, "TOLU": 0.7, "XYLE": 0.8, "NAP": 0.6,
}


# -----------------------------------------------------------------------------
# Rate constants
# -----------------------------------------------------------------------------

def _arrhenius(A: float, EaR: float, T: np.ndarray) -> np.ndarray:
    """Arrhenius form: k = A * exp(-EaR / T), with EaR = Ea/R in Kelvin."""
    return A * np.exp(-EaR / T)


def k_NO_O3(T):
    """NO + O3 -> NO2 + O2. JPL: 3.0e-12 * exp(-1500/T)."""
    return 3.0e-12 * np.exp(-1500.0 / T)


def k_HO2_NO(T):
    """HO2 + NO -> OH + NO2. JPL: 3.3e-12 * exp(270/T)."""
    return 3.3e-12 * np.exp(270.0 / T)


def k_HO2_HO2(T, M):
    """HO2 + HO2 -> H2O2 + O2. Pressure-dependent (JPL)."""
    k0 = 3.0e-13 * np.exp(460.0 / T)
    k1 = 2.1e-33 * M * np.exp(920.0 / T)
    return k0 + k1  # neglects H2O enhancement (~1.4x when humid)


def k_OH_CO(T, M):
    """OH + CO -> H + CO2 (effective). JPL: combined low/high-pressure form.
    Here we use the standard tropospheric surface-level value 2.4e-13 + M-dep."""
    k_low = 5.9e-33 * (T / 300.0) ** (-1.4)
    k_high = 1.1e-12 * (T / 300.0) ** 1.3
    # Troe-like combination
    return k_low * M / (1.0 + k_low * M / k_high) + 1.5e-13


def k_OH_CH4(T):
    """OH + CH4 -> products. JPL: 2.45e-12 * exp(-1775/T)."""
    return 2.45e-12 * np.exp(-1775.0 / T)


def k_OH_HCHO(T):
    """OH + CH2O -> HO2 + CO + H2O. JPL 19-5: 5.5e-12 * exp(125/T).
    Near-room-T value ~ 9e-12 cm3/molec/s.  Makes HCHO both an OH sink
    AND an HO2 source (OH+HCHO produces one HO2 directly)."""
    return 5.5e-12 * np.exp(125.0 / T)


def k_OH_H2O2(T):
    """OH + H2O2 -> HO2 + H2O. JPL 19-5: 1.8e-12 (T-independent).
    Small contribution in winter polluted air (H2O2 depleted)."""
    return 1.8e-12 * np.ones_like(T)


def k_OH_NO2(T, M):
    """OH + NO2 + M -> HNO3 + M. Pressure-dependent.
    Uses Troe-form with JPL 19-5 parameters for the low-/high-pressure limits."""
    k0 = 1.8e-30 * (T / 300.0) ** (-3.0) * M
    kinf = 2.8e-11
    Fc = 0.6
    ratio = k0 / kinf
    k = (k0 / (1.0 + ratio)) * Fc ** (1.0 / (1.0 + (np.log10(ratio)) ** 2))
    return k


def k_OH_SO2(T, M):
    """OH + SO2 + M -> HSO3 + M (initiates aqueous/gas sulfate production).
    Pressure-dependent Troe form with JPL 19-5 parameters."""
    k0 = 3.3e-31 * (T / 300.0) ** (-4.3) * M
    kinf = 1.6e-12
    Fc = 0.6
    ratio = k0 / kinf
    k = (k0 / (1.0 + ratio)) * Fc ** (1.0 / (1.0 + (np.log10(ratio)) ** 2))
    return k


def k_NO2_O3(T):
    """NO2 + O3 -> NO3 + O2. JPL: 1.2e-13 * exp(-2450/T)."""
    return 1.2e-13 * np.exp(-2450.0 / T)


def Keq_N2O5(T):
    """NO3 + NO2 <-> N2O5 equilibrium constant (cm3/molec).
    JPL: K = 3.5e-27 * exp(11000/T)."""
    return 3.5e-27 * np.exp(11000.0 / T)


def k_RO2_NO(T):
    """RO2 + NO -> NO2 + alkoxy (effective). ~8.1e-12 * exp(270/T)."""
    return 8.1e-12 * np.exp(270.0 / T)


def k_RO2_HO2(T):
    """RO2 + HO2 -> ROOH + O2 (effective). ~2.9e-13 * exp(1300/T)."""
    return 2.9e-13 * np.exp(1300.0 / T)


def mean_speed_N2O5(T):
    """Mean molecular speed of N2O5, cm/s. For heterogeneous uptake rate."""
    # v_mean = sqrt(8 k_B T / (pi m))
    M_kg = 108.01e-3 / AVOGADRO     # mass of one N2O5 molecule (kg)
    v_ms = np.sqrt(8.0 * KB * T / (np.pi * M_kg))
    return v_ms * 100.0              # cm/s


# -----------------------------------------------------------------------------
# Unit conversions
# -----------------------------------------------------------------------------

def air_density(T, P):
    """Air number density [M] (molec/cm3) from T (K) and P (Pa)."""
    # n/V = P/(R T) [mol/m3] -> *NA -> molec/m3; /1e6 -> molec/cm3
    return P * AVOGADRO / (R_GAS * T) * 1e-6


def water_vapor_conc(T, P, RH_percent):
    """H2O number density (molec/cm3) from T (K), P (Pa), RH (percent)."""
    T_C = T - 273.15
    # Bolton (1980) saturation vapor pressure over liquid water (Pa)
    e_sat = 611.2 * np.exp(17.67 * T_C / (T_C + 243.5))
    e = np.clip(RH_percent, 0.0, 110.0) / 100.0 * e_sat
    return e * AVOGADRO / (R_GAS * T) * 1e-6


def _element_to_molcm3(c_ugm3_element, MW_element):
    """Convert element-mass ug/m3 to molec/cm3.

    Derivation:
        C [ug/m3] = C * 1e-6 g/m3
        mol/m3    = (C * 1e-6) / MW
        molec/m3  = NA * (C * 1e-6) / MW
        molec/cm3 = molec/m3 / 1e6 = C * NA * 1e-12 / MW

    One-atom-per-molecule assumption (works for NOx/NH/HNO3 tracked as N mass,
    or SO2/pSO4 tracked as S mass).
    """
    return c_ugm3_element * AVOGADRO * 1e-12 / MW_element


def _compound_to_molcm3(c_ugm3_compound, MW_compound):
    """Convert compound-mass ug/m3 to molec/cm3. Same derivation as _element_to_molcm3."""
    return c_ugm3_compound * AVOGADRO * 1e-12 / MW_compound


# -----------------------------------------------------------------------------
# Main diagnostics
# -----------------------------------------------------------------------------

@dataclass
class OxidantFields:
    """Diagnostic oxidant fields per bin per cell (3D arrays, molec/cm3 except where noted)."""
    OH:   np.ndarray
    HO2:  np.ndarray
    RO2:  np.ndarray
    NO:   np.ndarray
    NO2:  np.ndarray
    f_NO2: np.ndarray              # dimensionless NO2/NOx ratio
    NO3:  np.ndarray
    N2O5: np.ndarray
    k_n2o5_het: np.ndarray         # s^-1
    # Diagnostic rates: exposed so operator assembly and RHS construction
    # can reuse them without re-deriving.
    k_oh_so2: np.ndarray           # cm3/molec/s — SO2 oxidation
    k_oh_no2_for_hno3: np.ndarray  # cm3/molec/s — NOx loss to HNO3
    k_oh_co: np.ndarray            # cm3/molec/s — CO oxidation


def pss_no_no2(NOx_molcm3, O3_molcm3, jNO2, T):
    """Photostationary state NO/NO2 split given NOx total.

    f_NO2 = 1 / (1 + jNO2 / (k_NO+O3 * [O3]))
    At night (jNO2 = 0): f_NO2 = 1 (all NOx as NO2).
    """
    k1 = k_NO_O3(T)
    denom = k1 * O3_molcm3
    # Guard against near-zero O3 or near-zero jNO2
    safe_denom = np.where(denom > 1e-30, denom, 1e-30)
    ratio = jNO2 / safe_denom
    f_NO2 = 1.0 / (1.0 + ratio)
    # Clamp to [0, 1] in case of numerical issues
    f_NO2 = np.clip(f_NO2, 0.0, 1.0)
    NO  = (1.0 - f_NO2) * NOx_molcm3
    NO2 = f_NO2 * NOx_molcm3
    return f_NO2, NO, NO2


def compute_oh_ho2(NO, NO2, O3_molcm3, SO2_molcm3, CO_molcm3, CH4_molcm3,
                   VOC_molcm3: Dict[str, np.ndarray],
                   H2O_molcm3, M_molcm3, T, jO1D,
                   HONO_molcm3=None, jHONO=None,
                   HCHO_molcm3=None, jHCHO=None,
                   H2O2_molcm3=None, jH2O2=None,
                   max_iter=6, tol=1e-3):
    """Resolve the implicit OH-HO2 coupling via fixed-point iteration.

    Returns (OH, HO2, RO2) in molec/cm3.

    Primary OH sources:
      (i)   O3 + hv + H2O -> 2 OH     (Levy mechanism):
              P_O1D = 2 * f_O1D->OH * jO1D * [O3]
      (ii)  HONO + hv -> OH + NO       (only if [HONO] and jHONO supplied):
              P_HONO = jHONO * [HONO]
      (iii) H2O2 + hv -> 2 OH          (small in winter polluted air):
              P_H2O2 = 2 * jH2O2 * [H2O2]

    Primary HO2 sources (feed HO2 steady state which cycles back to OH
    via HO2 + NO):
      (a) OH + CO, OH + CH4 -> HO2            (existing)
      (b) CH2O + hv -> H + HCO -> 2 HO2         (new, Phase 3f)
      (c) OH + HCHO -> HO2 + CO                 (new, Phase 3f — also an
          OH sink)

    HCHO is the missing OH source that drives the noon OH peak in polluted
    urban conditions.  Without HCHO, HONO dominates the primary budget and
    OH's diurnal shape follows HNO2's (which accumulates through the
    afternoon), producing an inverted diurnal.  With HCHO photolysis
    peaking at noon, OH's noon peak is recovered.

    OH steady state:
        [OH] = (P_O1D + P_HONO + P_H2O2 + k_HO2+NO [HO2][NO]) / L_OH
    where L_OH also gains k_OH+HCHO · [HCHO].

    HO2 steady state (analytical quadratic):
        P_HO2 = (k_CO [CO] + k_CH4 [CH4]) [OH] + 2 jHCHO [HCHO]
                + k_OH+HCHO [HCHO] [OH]
        L_HO2 = k_HO2+NO [NO] [HO2] + 2 k_HO2+HO2 [HO2]^2
        => quadratic: a [HO2]^2 + b [HO2] - P = 0, a = 2 k_HH, b = k_HN [NO]
    """
    # Rate constants
    k_HN = k_HO2_NO(T)
    k_HH = k_HO2_HO2(T, M_molcm3)
    k_CO_ = k_OH_CO(T, M_molcm3)
    k_CH4_ = k_OH_CH4(T)
    k_N2 = k_OH_NO2(T, M_molcm3)
    k_S  = k_OH_SO2(T, M_molcm3)
    k_HCHO_ = k_OH_HCHO(T)
    k_H2O2_ = k_OH_H2O2(T)

    # Optional Phase 3f inputs — treat missing values as zero.
    HCHO = HCHO_molcm3 if HCHO_molcm3 is not None else np.zeros_like(T)
    jHCHO_ = jHCHO if jHCHO is not None else np.zeros_like(T)
    H2O2 = H2O2_molcm3 if H2O2_molcm3 is not None else np.zeros_like(T)
    jH2O2_ = jH2O2 if jH2O2 is not None else np.zeros_like(T)

    # O(1D) branching to OH
    k_O1D_H2O = 1.63e-10 * np.exp(60.0 / T)
    k_O1D_M   = 2.15e-11 * np.exp(110.0 / T)
    f_O1D_OH = k_O1D_H2O * H2O_molcm3 / (
        k_O1D_H2O * H2O_molcm3 + k_O1D_M * M_molcm3 + 1e-30
    )
    P_prim = 2.0 * f_O1D_OH * jO1D * O3_molcm3

    # HONO photolysis: HONO + hv -> OH + NO. One-to-one yield. Dominant at dawn
    # and in polluted conditions; negligible when [HONO] is absent.
    if HONO_molcm3 is not None and jHONO is not None:
        P_prim = P_prim + jHONO * HONO_molcm3

    # H2O2 photolysis: H2O2 + hv -> 2 OH.  Small in winter polluted air.
    P_prim = P_prim + 2.0 * jH2O2_ * H2O2

    # VOC OH-sink sum and RO2 production rate (independent of OH since both use [OH] linearly)
    voc_sink = np.zeros_like(T)
    ro2_prod_coef = np.zeros_like(T)  # sum_i y_i k_i [VOC_i], such that P_RO2 = coef * [OH]
    for name, n_voc in VOC_molcm3.items():
        if n_voc is None:
            continue
        k_i = K_VOC_OH_298.get(name, 1e-12)  # fallback if species unknown
        y_i = Y_RO2.get(name, 1.0)
        voc_sink   += k_i * n_voc
        ro2_prod_coef += y_i * k_i * n_voc

    # RO2 loss rate constants (using effective values; small T dependence)
    k_RN = k_RO2_NO(T)
    k_RH = k_RO2_HO2(T)

    L_OH_static = k_CO_ * CO_molcm3 + k_CH4_ * CH4_molcm3 + voc_sink \
                  + k_N2 * NO2 + k_S * SO2_molcm3 \
                  + k_HCHO_ * HCHO + k_H2O2_ * H2O2 + 1e-30  # s^-1

    # Initial OH guess: primary production balanced by static loss only (no HO2 recycle).
    OH = P_prim / L_OH_static
    OH = np.maximum(OH, 1.0)  # floor: 1 molec/cm3, avoid zero

    # Fixed-point iteration
    # Photolysis-driven HO2 source (OH-independent; constant during
    # the inner iteration).  CH2O + hv -> H + HCO -> 2 HO2.
    P_HO2_photo = 2.0 * jHCHO_ * HCHO
    for it in range(max_iter):
        # HO2 production = OH-driven (CO, CH4, HCHO, H2O2 sinks that yield
        # HO2) + photolysis (HCHO radical channel).
        P_HO2 = ((k_CO_ * CO_molcm3 + k_CH4_ * CH4_molcm3
                  + k_HCHO_ * HCHO + k_H2O2_ * H2O2) * OH
                 + P_HO2_photo)
        a = 2.0 * k_HH
        b = k_HN * NO
        # HO2 = (-b + sqrt(b^2 + 4 a P)) / (2 a)
        disc = np.sqrt(b * b + 4.0 * a * P_HO2)
        HO2 = (-b + disc) / (2.0 * a + 1e-30)
        HO2 = np.maximum(HO2, 0.0)

        # OH with HO2 recycle
        OH_new = (P_prim + k_HN * HO2 * NO) / L_OH_static

        # Convergence check
        rel = np.abs(OH_new - OH) / (OH + 1e-30)
        OH = OH_new
        if rel.max() < tol:
            break

    # RO2 (steady-state, using converged OH and HO2)
    RO2_denom = k_RN * NO + k_RH * HO2 + 1e-30
    RO2 = ro2_prod_coef * OH / RO2_denom
    return OH, HO2, RO2


def no3_n2o5_steady_state(NO2, O3_molcm3, T, S_a,
                          jNO3=0.0, gamma_N2O5=0.02):
    """Steady-state NO3 and N2O5 concentrations plus N2O5 hydrolysis rate.

    Production: NO2 + O3 -> NO3 + O2.
    Loss channels on NO3:
      (a) NO3 photolysis (jNO3 ~ 7e-2 s^-1 at noon, dominant in daylight)
      (b) Heterogeneous N2O5 hydrolysis (via N2O5 equilibrium, dominant at night)
      (c) (Minor: NO3 + DMS, NO3 + NO, NO3 + VOC — neglected for first cut)

    Sign convention: at noon jNO3 dominates and NO3 -> 0 rapidly (correct).
    At night jNO3 = 0 and the hetero channel balances (N2O5 accumulates).

    k_het = 1/4 * gamma * v_bar * S_a, with v_bar the N2O5 mean molecular
    speed (cm/s) and S_a the aerosol surface area density (cm^-1).
    gamma_N2O5 default 0.02 is mid-range; real values 0.001-0.1 depending
    on composition/RH. Future work: update gamma from PM composition.

    Equilibrium: [N2O5] = Keq(T) * [NO3] * [NO2], so N2O5 tracks NO3.
    """
    P_NO3 = k_NO2_O3(T) * NO2 * O3_molcm3
    v_N2O5 = mean_speed_N2O5(T)
    k_het = 0.25 * gamma_N2O5 * v_N2O5 * S_a  # per N2O5 molecule
    Keq = Keq_N2O5(T)
    # Effective NO3 loss: photolysis (direct on NO3) + hetero-via-N2O5
    #   dNO3/dt = P - jNO3*NO3 - k_het * [N2O5]
    #           = P - jNO3*NO3 - k_het * Keq * NO2 * NO3
    L_NO3 = jNO3 + k_het * Keq * NO2 + 1e-30
    NO3 = P_NO3 / L_NO3
    N2O5 = Keq * NO3 * NO2
    return NO3, N2O5, k_het


# Back-compat alias
nighttime_no3_n2o5 = no3_n2o5_steady_state


def aerosol_surface_area_from_pm(pm25_ugm3,
                                 rho_particle_kgm3=1500.0,
                                 r_eff_um=0.2):
    """Approximate S_a (cm2/cm3) from total PM2.5 mass using 3 * PM / (rho * r_eff).

    S_a[1/m] = 3 * PM[ug/m3 * 1e-9 kg/ug] / (rho[kg/m3] * r_eff[m])
            = 3 * PM * 1e-9 / (rho * r_eff)

    Convert to 1/cm: multiply by 0.01 (since 1/m = 0.01 /cm).
    Hence S_a[cm^-1] = 3e-11 * PM / (rho * r_eff[m]).

    For r_eff = 0.2 um = 2e-7 m, rho = 1500 kg/m3:
        S_a[cm^-1] = 3e-11 / (1500 * 2e-7) * PM = 1e-7 * PM[ug/m3]

    Cross-check: SAS polluted PM = 100 ug/m3 -> S_a = 1e-5 /cm = 10 um2/cm3,
    consistent with Seinfeld & Pandis rough order of magnitude.
    """
    r_eff_m = r_eff_um * 1e-6
    return 3e-11 * pm25_ugm3 / (rho_particle_kgm3 * r_eff_m)


def diagnose_oxidants_per_bin(
    grid, c_NOx_ugN, c_SO2_ugS, pm25_ugm3,
    O3_ugm3, CO_ugm3, CH4_ugm3, VOC_speciated_ugm3: Dict[str, np.ndarray],
    jNO2, jO1D, jNO3=None,
    HONO_ugm3=None, jHONO=None,
    HCHO_ugm3=None, jHCHO=None,
    H2O2_ugm3=None, jH2O2=None,
    gamma_N2O5=0.02,
) -> OxidantFields:
    """Compute all diagnostic oxidants for one bin on the full 3D grid.

    Parameters
    ----------
    grid : GridData
        Provides Temperature, RH, Pressure, ...  all (nz, ny, nx).
    c_NOx_ugN, c_SO2_ugS : (nz, ny, nx) element mass ug/m3
    pm25_ugm3 : (nz, ny, nx) PM2.5 mass for S_a feedback
    O3_ugm3, CO_ugm3, CH4_ugm3 : (nz, ny, nx) compound-mass backgrounds
    VOC_speciated_ugm3 : dict of name -> (nz, ny, nx) compound mass
    jNO2, jO1D : (nz, ny, nx) photolysis rates (s^-1)

    Returns
    -------
    OxidantFields
    """
    T = grid.Temperature
    RH = grid.RH
    P = grid.Pressure if hasattr(grid, 'Pressure') and grid.Pressure.size > 0 \
        else np.full_like(T, 101325.0)

    M = air_density(T, P)
    H2O = water_vapor_conc(T, P, RH)

    # Element-mass -> molec/cm3 for orbit species
    NOx_n = _element_to_molcm3(c_NOx_ugN, MW_N)
    SO2_n = _element_to_molcm3(c_SO2_ugS, MW_S)

    # Compound-mass backgrounds
    O3_n  = _compound_to_molcm3(O3_ugm3,  MW_O3)
    CO_n  = _compound_to_molcm3(CO_ugm3,  MW_CO)
    CH4_n = _compound_to_molcm3(CH4_ugm3, MW_CH4)
    VOC_n = {k: _compound_to_molcm3(v, MW_VOC[k])
             for k, v in VOC_speciated_ugm3.items() if k in MW_VOC}

    # Photostationary state: split NOx into NO + NO2
    f_NO2, NO_n, NO2_n = pss_no_no2(NOx_n, O3_n, jNO2, T)

    # HONO (optional): HNO2 is a major OH source in polluted dawn bins.
    # Supplied as compound mass (MW_HNO2 = 47.01).
    HONO_n = None
    if HONO_ugm3 is not None:
        HONO_n = _compound_to_molcm3(HONO_ugm3, 47.01)

    # HCHO (compound mass; MW 30.026) — Phase 3f OH/HO2 closure upgrade.
    HCHO_n = None
    if HCHO_ugm3 is not None:
        HCHO_n = _compound_to_molcm3(HCHO_ugm3, 30.026)

    # H2O2 (compound mass; MW 34.015) — minor OH source in winter SAS but
    # wired through for completeness.  Currently unused by the preprocessor
    # pipeline (Phase 3f ships HCHO only); kwarg retained for later.
    H2O2_n = None
    if H2O2_ugm3 is not None:
        H2O2_n = _compound_to_molcm3(H2O2_ugm3, 34.015)

    # Iterative OH-HO2 closure
    OH_n, HO2_n, RO2_n = compute_oh_ho2(
        NO_n, NO2_n, O3_n, SO2_n, CO_n, CH4_n, VOC_n,
        H2O, M, T, jO1D,
        HONO_molcm3=HONO_n, jHONO=jHONO,
        HCHO_molcm3=HCHO_n, jHCHO=jHCHO,
        H2O2_molcm3=H2O2_n, jH2O2=jH2O2,
    )

    # Aerosol surface area and NO3/N2O5 steady state (daytime photolysis +
    # nighttime hetero).
    S_a = aerosol_surface_area_from_pm(pm25_ugm3)
    jNO3_arr = jNO3 if jNO3 is not None else np.zeros_like(T)
    NO3_n, N2O5_n, k_het = no3_n2o5_steady_state(
        NO2_n, O3_n, T, S_a, jNO3=jNO3_arr, gamma_N2O5=gamma_N2O5,
    )

    # Expose the kinetic rate constants needed by operator/RHS assembly
    k_oh_so2 = k_OH_SO2(T, M)
    k_oh_no2 = k_OH_NO2(T, M)
    k_oh_co_rate = k_OH_CO(T, M)

    return OxidantFields(
        OH=OH_n, HO2=HO2_n, RO2=RO2_n,
        NO=NO_n, NO2=NO2_n, f_NO2=f_NO2,
        NO3=NO3_n, N2O5=N2O5_n, k_n2o5_het=k_het,
        k_oh_so2=k_oh_so2, k_oh_no2_for_hno3=k_oh_no2,
        k_oh_co=k_oh_co_rate,
    )


# -----------------------------------------------------------------------------
# Rate builders for operator / RHS assembly
# -----------------------------------------------------------------------------

def build_chemistry_rates(ox: OxidantFields, O3_ugm3, grid, jNO3=None):
    """Turn diagnostic oxidants into linear first-order rates for operator assembly.

    Produces two rate fields (1/s) used by `assemble_species_operators`:

      k_so2_gas : gas-phase SO2 oxidation via OH.
                  k = k_OH+SO2(T, M) * [OH]

      k_nox_to_no3 : NOx -> TotalNO3 conversion (element-mass basis).
                    Two channels, both linear in [NOx]:
                    (a) OH + NO2 -> HNO3       [daytime]
                        k_a = k_OH+NO2 * [OH] * f_NO2
                    (b) NO3 -> N2O5 -> 2 HNO3  [nighttime hetero hydrolysis]
                        k_b = 2 * k_NO2+O3(T) * f_NO2 * [O3] * f_branch
                        f_branch = k_het*Keq*[NO2] / (jNO3 + k_het*Keq*[NO2])

    The factor 2 in channel (b) is the N-atom stoichiometry: 1 N2O5
    hydrolysis event delivers 2 HNO3 into TotalNO3 and consumes 2 NO2
    worth of NOx; per unit time, per [NOx_N], the effective conversion
    rate is 2 * (production of N2O5 from one NO2 × branch to hetero) /
    [NOx_N] which simplifies to the form above.

    Parameters
    ----------
    ox : OxidantFields
        Output of diagnose_oxidants_per_bin for this bin.
    O3_ugm3 : ndarray (nz, ny, nx)
        O3 compound mass used in the oxidant diagnosis (from HEMCO
        climatology or self-consistent orbit O3).
    grid : GridData
        Bin grid (Temperature, Pressure).
    jNO3 : ndarray (nz, ny, nx) or None
        NO3 photolysis rate. If None, falls back to 0 (nighttime-only
        branch, conservative overestimate of hetero channel at dawn/dusk).

    Returns
    -------
    k_so2_gas : ndarray (nz, ny, nx)  s^-1
    k_nox_to_no3 : ndarray (nz, ny, nx)  s^-1
    """
    T = grid.Temperature
    O3_n = _compound_to_molcm3(O3_ugm3, MW_O3)

    # --- SO2 gas-phase ---
    k_so2_gas = ox.k_oh_so2 * ox.OH

    # --- NOx -> TotalNO3 ---
    # Channel (a): OH + NO2 -> HNO3
    k_a = ox.k_oh_no2_for_hno3 * ox.OH * ox.f_NO2

    # Channel (b): nighttime N2O5 -> 2 HNO3
    Keq = Keq_N2O5(T)
    # k_n2o5_het is per N2O5 molecule (s^-1); ox.NO2 is molec/cm3.
    k_branch_num = ox.k_n2o5_het * Keq * ox.NO2
    jNO3_arr = np.asarray(jNO3) if jNO3 is not None else np.zeros_like(T)
    f_branch = k_branch_num / (jNO3_arr + k_branch_num + 1e-30)
    k_b = 2.0 * k_NO2_O3(T) * ox.f_NO2 * O3_n * f_branch

    k_nox_to_no3 = k_a + k_b

    return k_so2_gas, k_nox_to_no3


def build_co_loss_rate(ox: OxidantFields):
    """CO chemistry loss rate (1/s), linear in [CO]:

        k_co_loss = k_OH_CO(T, M) * [OH]

    Returned as an (nz, ny, nx) field suitable for
    ``assemble_species_operators(..., co_loss_rate=...)``.

    Production of CO from VOC oxidation (small per unit carbon, dominated
    by CH4 + CO emission at hemispheric scales) is handled in the orbit
    RHS if needed, not here.  For the South Asia domain, local CO is
    dominated by direct emissions, so the production term is omitted as
    second-order.
    """
    return ox.k_oh_co * ox.OH


def build_o3_rates(ox: OxidantFields, grid, jO1D=None, f_RO2_NO2=0.9,
                    bc_target_ugm3_3d=None, bc_rate_3d=None):
    """O3 chemistry loss diagonal + source term for Stage 3 transport closure.

    Loss (linear in [O3], 1/s):
        k_loss = k_NO+O3(T) · [NO] + jO1D · f_O1D→OH
    Source (compound mass, ug/m³/s, independent of [O3]):
        s = MW_O3 / AVOGADRO · 1e12 · (
              k_HO2+NO(T) · [HO2] · [NO]
            + f_RO2→NO2 · k_RO2+NO(T) · [RO2] · [NO]
          )

    Newtonian relaxation toward a 3D climatological target, rate field
    supplied per-cell (``bc_rate_3d``).  For each cell (k, j, i) with
    nonzero rate:
        k_loss[k,j,i]   += bc_rate_3d[k,j,i]
        s_o3_source[k,j,i] += bc_rate_3d[k,j,i] · bc_target_ugm3_3d[k,j,i]
    At steady state in a strongly-nudged cell with no other terms,
    O3 -> target.  The caller distributes rates spatially to represent
    (a) top-of-model influx via stratospheric descent (rate concentrated
    at the top N layers, decaying downward) and (b) lateral inflow from
    global climatology (rate concentrated at the outermost M cells of
    each horizontal boundary, decaying inward).  Without lateral BCs,
    a regional domain loses O3 through mid-latitude jet export at the
    UT faster than any top-only BC can replenish.

    Parameters
    ----------
    ox : OxidantFields
        From `diagnose_oxidants_per_bin` — uses NO, HO2, RO2 in molec/cm³.
    grid : GridData
        For T, RH, P.
    jO1D : ndarray (nz, ny, nx) or None
        O3 → O(1D) photolysis rate (1/s).  None → photolytic loss disabled.
    f_RO2_NO2 : float
        Fraction of RO2+NO that yields NO2 (the rest → organic nitrates).
        0.9 is a reasonable polluted-urban default (Atkinson 2007).
    bc_target_ugm3_3d : ndarray (nz, ny, nx) or None
        3D climatological O3 target (compound mass, ug/m³).  Only cells
        where rate > 0 are used.
    bc_rate_3d : ndarray (nz, ny, nx) or None
        Per-cell Newtonian relaxation rate (1/s).  None or all-zero
        disables the BC entirely.  Typically built by the caller to
        combine top-BC (rate concentrated at top layers) and lateral-BC
        (rate concentrated at horizontal edges) via cell-wise max.

    Returns
    -------
    k_o3_loss : ndarray (nz, ny, nx)  s^-1
    s_o3_source : ndarray (nz, ny, nx)  ug compound/m³/s
    """
    T = grid.Temperature
    P = grid.Pressure if hasattr(grid, 'Pressure') and grid.Pressure.size > 0 \
        else np.full_like(T, 101325.0)
    RH = grid.RH if grid.RH.size > 0 else np.full_like(T, 50.0)

    M = air_density(T, P)
    H2O = water_vapor_conc(T, P, RH)

    # O(1D) branching to OH (same formula as compute_oh_ho2)
    k_O1D_H2O = 1.63e-10 * np.exp(60.0 / T)
    k_O1D_M   = 2.15e-11 * np.exp(110.0 / T)
    f_O1D_OH = k_O1D_H2O * H2O / (k_O1D_H2O * H2O + k_O1D_M * M + 1e-30)

    jO1D_arr = np.asarray(jO1D) if jO1D is not None else np.zeros_like(T)

    # Loss diagonal (s^-1).  Only the fraction of O(1D) that goes to OH
    # represents an O3 sink: the rest relaxes back to O(3P) + M → O3 again.
    k_loss_NO = k_NO_O3(T) * ox.NO
    k_loss_photo = jO1D_arr * f_O1D_OH
    k_o3_loss = k_loss_NO + k_loss_photo

    # Source (molec/cm³/s → ug/m³/s compound mass).
    # Both HO2+NO and RO2+NO generate one NO2, which photolyses back to NO+O
    # and yields one O3 in steady-state — so the O3 production rate equals
    # the NO2 production rate.
    src_molcm3 = k_HO2_NO(T) * ox.HO2 * ox.NO \
               + f_RO2_NO2 * k_RO2_NO(T) * ox.RO2 * ox.NO
    # x * MW / Avogadro * 1e12: inverse of x_ugm3 * Avogadro * 1e-12 / MW
    s_o3_source = src_molcm3 * MW_O3 / AVOGADRO * 1e12

    # Newtonian BC: cell-wise relaxation toward the 3D target at rate
    # bc_rate_3d.  Caller builds the rate field to combine top and
    # lateral nudging however is appropriate (typically a cell-wise max
    # of separately-computed top and lateral contributions).
    if bc_rate_3d is not None and bc_target_ugm3_3d is not None:
        rate = np.asarray(bc_rate_3d)
        target = np.asarray(bc_target_ugm3_3d)
        k_o3_loss   += rate
        s_o3_source += rate * target

    return k_o3_loss, s_o3_source
