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
