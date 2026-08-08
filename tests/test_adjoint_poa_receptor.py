"""The adjoint deaths receptor must count POA.

Companion to test_marginal_poa_accounting.py. That test fixed the forward
marginal's δPM2.5; the adjoint had the same hole in two places and was
found while derisking the full-year marginal-damage regeneration
(2026-08-03):

  1. `growth_jacobian.apply_growth_transpose` lit up PM25, TotalNH, pSO4,
     TotalNO3 and the VBS bins, but not POA — so λ_POA had a zero source
     term and ∂J/∂e_POA would have come back as a silent zero.
  2. `adjoint.EMITTED_SPECIES` had no POA key, so POA damages could not be
     requested at all, and `PM25_primary` — which after the split is only
     the non-POA remainder — was the natural thing to reach for instead.

Both matter because POA carries 84.7% of CEDS anthropogenic primary mass,
and the error is selective by sector: residential, transport and industry
are POA-heavy, power is not, so sector *rankings* move, not just
magnitudes.

The receptor coefficient must match marginal._compute_delta_pm25 exactly
(1.0, same as PrimaryPM25). An asymmetry there does not merely lose
accuracy — it breaks adjoint/marginal duality, which is the property the
verify_adjoint_compare.py duality check exists to test.
"""

import numpy as np
import pytest

from orbit.core.deposition import (
    IDX_PM25, IDX_POA, IDX_PSO4, IDX_TOTAL_NH, IDX_TOTAL_NO3, N_SPECIES,
)
from orbit.core.growth_jacobian import apply_growth_transpose

N_BINS = 8
NZ, NY, NX = 2, 3, 4
SURFACE = NY * NX


class _Indexer:
    nz, ny, nx = NZ, NY, NX
    N = NZ * NY * NX


class _Grid:
    """Minimal grid stub: explicit partitioning, no VBS state."""
    def __init__(self):
        self.NHPartitioning = np.zeros((NZ, NY, NX))
        self.NO3Partitioning = np.zeros((NZ, NY, NX))
        self.F_p_vbs = None
        self.M_OA_3d = None


def _receptor(S_value=1.0):
    S = np.full((N_BINS, NY, NX), S_value, dtype=np.float64)
    grids = [_Grid() for _ in range(N_BINS)]
    return apply_growth_transpose(
        S, grids, _Indexer(), None,
        p_nh4_surface_per_bin=np.zeros((N_BINS, NY, NX)),
        p_no3_surface_per_bin=np.zeros((N_BINS, NY, NX)),
    )


def test_poa_receptor_is_not_zero():
    """The regression: a zero POA receptor yields ∂J/∂e_POA = 0 silently."""
    r = _receptor()
    assert np.any(r[:, IDX_POA, :] != 0.0), (
        "POA receptor is identically zero — adjoint POA damages would be "
        "a silent zero for 84.7% of anthropogenic primary mass"
    )


def test_poa_receptor_matches_primary_pm25_without_vbs_state():
    """With no baseline VBS state there is no feedback: D = 1, so the POA
    coefficient collapses to 1.0, same as PrimaryPM25."""
    r = _receptor()
    np.testing.assert_allclose(r[:, IDX_POA, :], r[:, IDX_PM25, :])


def test_poa_receptor_is_surface_only():
    """Deaths are a surface endpoint; aloft entries must stay zero."""
    r = _receptor()
    assert np.all(r[:, IDX_POA, SURFACE:] == 0.0)


def test_poa_receptor_scales_with_the_deaths_gradient():
    """Linear in S, like every other species term."""
    np.testing.assert_allclose(
        _receptor(S_value=3.0)[:, IDX_POA, :SURFACE],
        3.0 * _receptor(S_value=1.0)[:, IDX_POA, :SURFACE],
    )


def test_split_primary_mass_is_receptor_invariant():
    """1 µg/m³ of primary is worth the same however it splits POA/other.

    The property the pre-fix adjoint violated: all the weight sat on
    PM25_primary, so moving mass into the POA slot lost it entirely.
    """
    r = _receptor()
    lumped = 5.0 * r[:, IDX_PM25, :SURFACE]
    split = 1.5 * r[:, IDX_PM25, :SURFACE] + 3.5 * r[:, IDX_POA, :SURFACE]
    np.testing.assert_allclose(lumped, split)


def test_poa_is_requestable_from_the_adjoint_driver():
    """EMITTED_SPECIES must expose POA, else it cannot be asked for."""
    from orbit.modes.adjoint import EMITTED_SPECIES
    assert "POA" in EMITTED_SPECIES
    assert EMITTED_SPECIES["POA"] == IDX_POA
    assert EMITTED_SPECIES["PM25_primary"] == IDX_PM25
    assert EMITTED_SPECIES["POA"] != EMITTED_SPECIES["PM25_primary"]


@pytest.mark.parametrize("idx", [IDX_TOTAL_NH, IDX_PSO4, IDX_TOTAL_NO3])
def test_other_species_untouched_by_the_poa_fix(idx):
    """Guard: adding POA must not disturb the inorganic receptor terms."""
    r = _receptor()
    if idx in (IDX_TOTAL_NH, IDX_TOTAL_NO3):
        # partitioning passed as zero → these terms are zero by construction
        assert np.all(r[:, idx, :] == 0.0)
    else:
        assert np.any(r[:, idx, :SURFACE] != 0.0)


def test_receptor_shape_is_full_species_table():
    r = _receptor()
    assert r.shape == (N_BINS, N_SPECIES, _Indexer.N)


# ---------------------------------------------------------------------------
# Pankow feedback: POA must carry 1/D, the same amplification as the VBS bins.
# POA is primary in transport but sits in M_OA, so adding POA raises the
# absorbing mass and condenses semi-volatiles. Omitting 1/D understated POA
# damages by ~4.5% at POA top-decile cells on the January 2022 baseline.
# ---------------------------------------------------------------------------

from orbit.core.deposition import IDX_VBS_BINS
from orbit.core.growth_jacobian import _compute_D_per_bin
from orbit.modes.marginal import C_STAR_VALS

N_VBS = len(IDX_VBS_BINS)


class _GridWithVBS(_Grid):
    """Grid carrying enough VBS state that D < 1 (real feedback)."""
    def __init__(self, m_oa=8.0, c_bin=3.0):
        super().__init__()
        self.F_p_vbs = np.full((N_VBS, NZ, NY, NX), 0.5)
        self.M_OA_3d = np.full((NZ, NY, NX), m_oa)
        self.c_bin = c_bin


def _receptor_with_vbs(S_value=1.0, c_bin=3.0):
    S = np.full((N_BINS, NY, NX), S_value, dtype=np.float64)
    grids = [_GridWithVBS(c_bin=c_bin) for _ in range(N_BINS)]
    baseline = np.full((N_BINS, N_VBS, NZ, NY, NX), c_bin, dtype=np.float64)
    r = apply_growth_transpose(
        S, grids, _Indexer(), baseline,
        p_nh4_surface_per_bin=np.zeros((N_BINS, NY, NX)),
        p_no3_surface_per_bin=np.zeros((N_BINS, NY, NX)),
    )
    D = _compute_D_per_bin(
        grids[0].M_OA_3d.reshape(NZ, NY, NX)[0].ravel(),
        baseline[0, :, 0, :, :].reshape(N_VBS, -1),
        np.asarray(C_STAR_VALS, dtype=np.float64),
    )
    return r, D


def test_feedback_is_actually_active_in_the_fixture():
    """Guard the guard: if D == 1 the next two tests prove nothing."""
    _, D = _receptor_with_vbs()
    assert np.all(D < 1.0)


def test_poa_receptor_carries_one_over_D():
    """The cross-term: coefficient is 1/D, not 1."""
    r, D = _receptor_with_vbs()
    np.testing.assert_allclose(r[0, IDX_POA, :SURFACE], 1.0 / D, rtol=1e-12)


def test_poa_receptor_exceeds_primary_pm25_under_feedback():
    """POA is worth strictly more than inert primary per unit mass, because
    it also condenses semi-volatiles. Direction matters: the pre-fix code
    understated POA."""
    r, _ = _receptor_with_vbs()
    assert np.all(r[:, IDX_POA, :SURFACE] > r[:, IDX_PM25, :SURFACE])


def test_poa_and_vbs_share_the_same_denominator():
    """POA coefficient 1/D and VBS coefficient F_p/D differ only by F_p."""
    r, _ = _receptor_with_vbs()
    f_p = 0.5                                    # fixture value
    np.testing.assert_allclose(
        r[:, IDX_VBS_BINS[0], :SURFACE],
        f_p * r[:, IDX_POA, :SURFACE],
        rtol=1e-12,
    )


def test_stronger_feedback_amplifies_poa_more():
    """More absorbing VBS mass → smaller D → larger POA sensitivity."""
    weak, _ = _receptor_with_vbs(c_bin=1.0)
    strong, _ = _receptor_with_vbs(c_bin=6.0)
    assert np.all(strong[:, IDX_POA, :SURFACE] > weak[:, IDX_POA, :SURFACE])
