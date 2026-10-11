"""Molecular weight conversion factors (element mass -> compound mass).

Standard molecular-weight values.
The solver works in element mass (N for nitrogen species, S for sulfur)
internally.  At output time, these factors convert to compound mass.
"""

# Atomic weights
MW_N = 14.0067   # Nitrogen
MW_S = 32.0655   # Sulfur

# Particle-phase conversions
N_TO_NH4 = 18.03851 / MW_N   # MW_NH4 / MW_N  ~= 1.28785
N_TO_NO3 = 62.00501 / MW_N   # MW_NO3 / MW_N  ~= 4.42681
S_TO_SO4 = 96.0632 / MW_S    # MW_SO4 / MW_S  ~= 2.99584

# Gas-phase conversions
N_TO_NH3 = 17.03056 / MW_N   # MW_NH3 / MW_N  ~= 1.21589
N_TO_NOx = 46.0055 / MW_N    # MW_NOx / MW_N  ~= 3.28454
S_TO_SO2 = 64.0644 / MW_S    # MW_SO2 / MW_S  ~= 1.99792

# GEOS-Chem PM2.5 convention at 35 % relative humidity (aerosol_mod.F90):
# SIA_GROWTH 1.10 on sulfate, nitrate and ammonium, ORG_GROWTH 1.05 on
# organic aerosol; SSA_GROWTH 1.86 on sea salt is applied at emission. The
# forward 35 % RH field (cli._pm25_at_rh35), the marginal delta
# (marginal._compute_delta_pm25) and the adjoint receptor
# (growth_jacobian.apply_growth_transpose) all read these two numbers, so
# the health chain stays in lockstep. 1.0.3: the deaths gradient and the
# adjoint receptor are defined on this grown mass (Sumil, 2026-10-10).
SIA_GROWTH_RH35 = 1.10
ORG_GROWTH_RH35 = 1.05
