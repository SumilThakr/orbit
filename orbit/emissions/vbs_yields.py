"""VBS parent-class chamber-derived stoichiometric yields.

Per-class mass yields (kg SoA per kg VOC) into each of the 5 VBS bins,
in low-to-high C* order matching orbit.core.deposition.IDX_VBS_BINS:

    (C*=0.1, C*=1, C*=10, C*=100, C*=1000) µg/m³.

Yields are tied to the source paper's bin centers — never transposed.

References:
- Tsimpidi, A. P. et al. (2010). Atmos. Chem. Phys. 10, 525–546.
  α-pinene low-NOx, anthropogenic high/low-NOx, isoprene low-NOx.
- Grieshop, A. P. et al. (2009). ACP 9, 1263–1277.
  Wood-smoke biomass-burning yields.
- Robinson, A. L. et al. (2007). Science 315, 1259–1262.
  IVOC source scaling × 1.5 from POA, with effective yield distribution.
"""
from __future__ import annotations
import numpy as np

# 5-tuple per parent class, low-to-high C*: (C*=0.1, 1, 10, 100, 1000).
# Mass yield (kg SoA per kg parent VOC).
VBS_PARENT_YIELDS = {
    # Anthropogenic VOC, NOx-regime split per Tsimpidi 2010 Table 1
    # (high-NOx aromatic-dominated; low-NOx aged urban / regional).
    "anthro_high_nox": np.array([0.000, 0.000, 0.000, 0.067, 0.350]),
    "anthro_low_nox":  np.array([0.000, 0.000, 0.075, 0.150, 0.300]),

    # Monoterpenes (α-pinene low-NOx; representative of SE forest BVOC).
    "bio_monoterpene": np.array([0.000, 0.107, 0.092, 0.359, 0.600]),

    # Isoprene low-NOx (Tsimpidi 2010 / Henze 2008 baseline).
    "bio_isoprene":    np.array([0.000, 0.018, 0.060, 0.024, 0.025]),

    # Biomass burning (Grieshop 2009 wood-smoke; representative for
    # IGP rice/wheat residue burning + Indo-Burma forest fires).
    "biomass_burning": np.array([0.000, 0.040, 0.180, 0.220, 0.450]),

    # IVOC effective yield (Robinson 2007 + follow-on calibrations).
    # Robinson's IVOC mass scaling × yields ≈ POA × 0.4 effective.
    "ivoc":            np.array([0.000, 0.110, 0.060, 0.040, 0.040]),
}


# Parent classes that participate in cell-dependent NOx-regime switching.
# When the cell's [NO2]/[OH] ratio indicates high-NOx, we use the *_high_nox
# variant; low-NOx → *_low_nox; for parents in the table without a regime
# split (monoterpene, BB, IVOC), the single yield set is used regardless.
NOX_REGIME_PAIRS = {
    "anthro": ("anthro_high_nox", "anthro_low_nox"),
    # Isoprene NOx-regime sensitivity is large (Surratt 2010, Liu 2021)
    # but Tsimpidi 2010 doesn't ship a high-NOx Pankow set we trust;
    # treat as low-NOx-only for now and document as a §12 limitation.
}


def get_yields(parent_class: str) -> np.ndarray:
    """Return the 5-tuple of bin yields for a parent class. Raises KeyError."""
    return VBS_PARENT_YIELDS[parent_class].copy()


def list_parent_classes() -> list[str]:
    """Sorted names of every parent class with a yield set."""
    return sorted(VBS_PARENT_YIELDS.keys())
