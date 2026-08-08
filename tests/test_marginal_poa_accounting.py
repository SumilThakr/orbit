"""Marginal δPM2.5 must count POA.

Before the 2026-08-03 split, POA lived inside PrimaryPM25 and was counted
automatically. After it, `_compute_delta_pm25` summed PrimaryPM25 + SOA +
inorganics and silently dropped POA — about 30% of primary mass at polluted
IGP cells (13.8 of 18.8 µg/m³ in January), and only for POA-emitting sectors:
residential, transport, industry. Every marginal damage from those sectors
would have been understated, with nothing to indicate it.

Caught while handing the work to the server for the full-year marginal-damage
regeneration.
"""

import numpy as np
import pytest

from orbit.core.deposition import (
    IDX_PM25, IDX_POA, IDX_PSO4, N_SPECIES,
)
from orbit.modes.marginal import _compute_delta_pm25

NZ, NY, NX = 2, 3, 4
N = NZ * NY * NX
N_BINS = 8


class _Grid:
    """Minimal grid stub: no partitioning, no VBS feedback."""
    def __init__(self):
        self.NHPartitioning = np.zeros((NZ, NY, NX))
        self.NO3Partitioning = np.zeros((NZ, NY, NX))
        self.F_p_vbs = None
        self.M_OA_3d = None


def _delta(**species):
    d = np.zeros((N_SPECIES, N_BINS + 1, N), dtype=np.float64)
    for idx, val in species.items():
        d[idx, :, :] = val
    return d


def _run(delta):
    _, mean = _compute_delta_pm25(delta, [_Grid() for _ in range(N_BINS)],
                                  NZ, NY, NX, N)
    return mean


def test_poa_contributes_to_delta_pm25():
    """The regression: a POA-only perturbation must not vanish."""
    out = _run(_delta(**{str(IDX_POA): 0.0}) if False else _delta())
    np.testing.assert_allclose(out, 0.0)          # sanity: zero in, zero out
    d = np.zeros((N_SPECIES, N_BINS + 1, N))
    d[IDX_POA, :, :] = 2.0
    assert _run(d).mean() == pytest.approx(2.0)


def test_poa_and_primary_pm25_are_weighted_identically():
    """Both are primary, non-volatile and inert: 1 µg/m³ each, same effect."""
    a = np.zeros((N_SPECIES, N_BINS + 1, N)); a[IDX_PM25, :, :] = 3.0
    b = np.zeros((N_SPECIES, N_BINS + 1, N)); b[IDX_POA, :, :] = 3.0
    np.testing.assert_allclose(_run(a), _run(b))


def test_split_mass_equals_lumped_mass():
    """Splitting a primary perturbation across the two slots is invariant.

    This is the property the pre-split behaviour silently violated.
    """
    lumped = np.zeros((N_SPECIES, N_BINS + 1, N))
    lumped[IDX_PM25, :, :] = 5.0
    split = np.zeros((N_SPECIES, N_BINS + 1, N))
    split[IDX_PM25, :, :] = 1.5
    split[IDX_POA, :, :] = 3.5
    np.testing.assert_allclose(_run(lumped), _run(split))


def test_legacy_13_species_baseline_still_loads():
    """Pre-split baselines have no POA row; that must not raise."""
    legacy = np.zeros((13, N_BINS + 1, N))
    legacy[IDX_PM25, :, :] = 4.0
    assert _run(legacy).mean() == pytest.approx(4.0)


def test_inorganics_unaffected_by_the_change():
    """Guard against the fix perturbing anything it should not."""
    d = np.zeros((N_SPECIES, N_BINS + 1, N))
    d[IDX_PSO4, :, :] = 1.0
    before_style = _run(d)
    assert before_style.mean() > 0          # S_TO_SO4 weighted, non-zero
    d2 = d.copy(); d2[IDX_POA, :, :] = 0.0
    np.testing.assert_allclose(_run(d2), before_style)


# ---------------------------------------------------------------------------
# Pankow feedback (added 2026-08-03). POA is primary in transport but sits in
# M_OA = C_POA + Σ F_p,i C_i, so adding POA raises the absorbing mass and
# condenses semi-volatiles. Differentiating that fixed point gives
# dM_OA/dC_POA = 1/D, so POA carries the same 1/D the VBS bins do.
#
# These mirror tests/test_adjoint_poa_receptor.py — the two linearisations
# MUST agree or adjoint/marginal duality breaks.
# ---------------------------------------------------------------------------

from orbit.core.deposition import IDX_VBS_BINS
from orbit.core.growth_jacobian import _compute_D_per_bin
from orbit.modes.marginal import C_STAR_VALS

N_VBS = len(IDX_VBS_BINS)


class _GridVBS(_Grid):
    """Grid with enough VBS state that D < 1."""
    def __init__(self, m_oa=8.0):
        super().__init__()
        self.F_p_vbs = np.full((N_VBS, NZ, NY, NX), 0.5)
        self.M_OA_3d = np.full((NZ, NY, NX), m_oa)


def _run_vbs(delta, c_bin=3.0):
    grids = [_GridVBS() for _ in range(N_BINS)]
    baseline = np.full((N_BINS, N_VBS, N), c_bin, dtype=np.float64)
    _, mean = _compute_delta_pm25(delta, grids, NZ, NY, NX, N,
                                  baseline_vbs_c=baseline)
    D = _compute_D_per_bin(
        grids[0].M_OA_3d.ravel(),
        baseline[0],
        np.asarray(C_STAR_VALS, dtype=np.float64),
    )
    return mean, D


def test_poa_carries_the_partitioning_feedback():
    """A POA-only perturbation is amplified by 1/D, not passed through at 1."""
    d = np.zeros((N_SPECIES, N_BINS + 1, N))
    d[IDX_POA, :, :] = 1.0
    mean, D = _run_vbs(d)
    expected = (1.0 / D).reshape(NZ, NY, NX)
    np.testing.assert_allclose(mean, expected, rtol=1e-12)


def test_poa_now_exceeds_inert_primary_under_feedback():
    """The direction of the fix: POA is worth strictly more than PrimaryPM25
    per unit mass, because it also condenses semi-volatiles."""
    a = np.zeros((N_SPECIES, N_BINS + 1, N)); a[IDX_PM25, :, :] = 1.0
    b = np.zeros((N_SPECIES, N_BINS + 1, N)); b[IDX_POA, :, :] = 1.0
    assert np.all(_run_vbs(b)[0] > _run_vbs(a)[0])


def test_organic_response_is_symmetric_in_poa_and_vbs():
    """Total organic δ = (δC_POA + Σ F_p,i δC_i)/D. With F_p = 0.5, one unit
    of POA must equal two units of a VBS bin."""
    poa = np.zeros((N_SPECIES, N_BINS + 1, N)); poa[IDX_POA, :, :] = 1.0
    vbs = np.zeros((N_SPECIES, N_BINS + 1, N)); vbs[IDX_VBS_BINS[0], :, :] = 2.0
    np.testing.assert_allclose(_run_vbs(poa)[0], _run_vbs(vbs)[0], rtol=1e-12)


def test_no_feedback_state_falls_back_to_coefficient_one():
    """Without baseline VBS state D is unavailable; POA must still count,
    at coefficient 1 rather than vanishing."""
    d = np.zeros((N_SPECIES, N_BINS + 1, N))
    d[IDX_POA, :, :] = 2.0
    assert _run(d).mean() == pytest.approx(2.0)
