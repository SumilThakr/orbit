"""VBS parent-class stoichiometric SOA yields.

Per-class mass yields (kg SOA per kg VOC) into each of the 5 VBS bins, in
low-to-high C* order matching orbit.core.deposition.IDX_VBS_BINS:

    (C* = 0.1, 1, 10, 100, 1000) µg/m³ at 298 K.

Every class has a high-NOx and a low-NOx set, and the solver interpolates
between them with the RO2 + NO branching ratio of the cell and bin
(orbit.emissions.vbs_distribution.compute_nox_regime). Yields are tied to the
source paper's bin centres and never transposed.

Sources (1.0.3; the 1.0.2 table mixed Tsimpidi, Grieshop and Robinson values
with fixed low-NOx sets for the biogenic classes):

- Tsimpidi et al. (2010), ACP 10, 525, Table 2: four-product basis set at
  C* = 1, 10, 100, 1000 µg/m³, high- and low-NOx, per SAPRC99 lumped
  species. ORBIT's C* = 0.1 bin receives nothing from these.
    anthropogenic  = ARO1 (toluene, benzene, monosubstituted benzenes), the
                     largest contributors to the CEDS SOA-forming VOC. ARO2
                     differs only at C* = 10 (0.195 / 0.300). A species-
                     weighted anthropogenic row is a 1.0.4 candidate.
    monoterpene    = TERP
    isoprene       = ISOP
    biomass burning = Tsimpidi species weighted by the 2022 GFED5.1 mass
                     shares of the seven species summed into the ORBIT BB VOC
                     file over the model domain (scripts/bb_voc_yields.py):
                     benzene 0.232 and toluene 0.130 -> ARO1, xylenes 0.035
                     -> ARO2, higher alkenes 0.147 -> mean(OLE1, OLE2),
                     higher alkanes 0.068 -> mean(ALK4, ALK5), isoprene 0.104
                     -> ISOP, monoterpenes 0.285 -> TERP. The alkene and
                     alkane mappings are ORBIT assumptions (GFED lumps
                     >= C4 alkenes and >= C5 alkanes).
- Hodzic et al. (2016), ACP 16, 7917, Table 1, IVOC: wall-loss-corrected
  GECKO-A yields. Their log C* <= -2 and -1 products are summed into ORBIT's
  lowest bin (0.1). Hodzic's yields include multigenerational aging, so
  ORBIT's explicit aging adds some double counting for IVOC products; most of
  their mass sits in the terminal bin, so the effect is small (stated
  assumption).
"""
from __future__ import annotations
import numpy as np

# 5-tuple per parent class and NOx regime, low-to-high C*: (0.1, 1, 10, 100, 1000).
# Mass yield (kg SOA per kg parent VOC).
VBS_PARENT_YIELDS = {
    # Tsimpidi (2010) Table 2, ARO1
    "anthro_high_nox":          np.array([0.000, 0.003, 0.165, 0.300, 0.435]),
    "anthro_low_nox":           np.array([0.000, 0.075, 0.225, 0.375, 0.525]),
    # Tsimpidi (2010) Table 2, TERP
    "bio_monoterpene_high_nox": np.array([0.000, 0.012, 0.122, 0.201, 0.500]),
    "bio_monoterpene_low_nox":  np.array([0.000, 0.107, 0.092, 0.359, 0.600]),
    # Tsimpidi (2010) Table 2, ISOP
    "bio_isoprene_high_nox":    np.array([0.000, 0.001, 0.023, 0.015, 0.000]),
    "bio_isoprene_low_nox":     np.array([0.000, 0.009, 0.030, 0.015, 0.000]),
    # Tsimpidi (2010) Table 2 weighted by GFED5.1 2022 species shares
    # (scripts/bb_voc_yields.py reproduces these rows)
    "biomass_burning_high_nox": np.array([0.000, 0.005, 0.112, 0.187, 0.346]),
    "biomass_burning_low_nox":  np.array([0.000, 0.063, 0.138, 0.266, 0.423]),
    # Hodzic et al. (2016) Table 1, IVOC (log C* <= -1 summed into bin 0.1)
    "ivoc_high_nox":            np.array([0.276, 0.069, 0.019, 0.010, 0.012]),
    "ivoc_low_nox":             np.array([0.488, 0.046, 0.010, 0.007, 0.008]),
}


# Parent classes and their (high-NOx, low-NOx) yield sets. Every class
# switches with the cell's RO2 + NO branching ratio (1.0.3); through 1.0.2
# only the anthropogenic class did and the others used fixed sets.
NOX_REGIME_PAIRS = {
    "anthro":          ("anthro_high_nox", "anthro_low_nox"),
    "bio_monoterpene": ("bio_monoterpene_high_nox", "bio_monoterpene_low_nox"),
    "bio_isoprene":    ("bio_isoprene_high_nox", "bio_isoprene_low_nox"),
    "biomass_burning": ("biomass_burning_high_nox", "biomass_burning_low_nox"),
    "ivoc":            ("ivoc_high_nox", "ivoc_low_nox"),
}


def regime_yields(parent_class: str, F_high_nox) -> np.ndarray:
    """Effective yields for a parent class given the high-NOx fraction F.

    F may be a scalar, (ny, nx) or (n_bins, ny, nx); the result has the
    5-bin axis inserted after any leading bin axis: (5,), (5, ny, nx) or
    (n_bins, 5, ny, nx).
    """
    hi_key, lo_key = NOX_REGIME_PAIRS[parent_class]
    y_hi, y_lo = VBS_PARENT_YIELDS[hi_key], VBS_PARENT_YIELDS[lo_key]
    F = np.asarray(F_high_nox, dtype=np.float64)
    if F.ndim == 0:
        return F * y_hi + (1.0 - F) * y_lo
    if F.ndim == 2:
        return F[None] * y_hi[:, None, None] + (1.0 - F[None]) * y_lo[:, None, None]
    if F.ndim == 3:
        return (F[:, None] * y_hi[None, :, None, None]
                + (1.0 - F[:, None]) * y_lo[None, :, None, None])
    raise ValueError(f"F_high_nox must have 0, 2 or 3 dimensions, got {F.ndim}")


def get_yields(parent_class: str) -> np.ndarray:
    """Return the 5-tuple of bin yields for a yield set (e.g. 'anthro_high_nox').
    Raises KeyError."""
    return VBS_PARENT_YIELDS[parent_class].copy()


def list_parent_classes() -> list[str]:
    """Sorted names of every yield set."""
    return sorted(VBS_PARENT_YIELDS.keys())
