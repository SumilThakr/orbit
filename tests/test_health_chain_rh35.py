"""The 35 % RH health-chain mass: receptor and marginal delta in lockstep.

1.0.3 defines the deaths gradient on the forward `pm25_rh35_mean` field
(GEOS-Chem convention: 1.10 on inorganics, 1.05 on organic aerosol, sea salt
grown at emission). The adjoint receptor and the marginal δPM2.5 must use
the same two numbers, read from one place, or adjoint/marginal duality and
the deaths gradient's linearisation point disagree.
"""
import numpy as np
import pytest

from orbit.core.constants import (
    N_TO_NH4, N_TO_NO3, S_TO_SO4, SIA_GROWTH_RH35, ORG_GROWTH_RH35,
)
from orbit.core.deposition import (
    IDX_PM25, IDX_POA, IDX_PSO4, IDX_TOTAL_NH, IDX_TOTAL_NO3, N_SPECIES,
)
from orbit.core.growth_jacobian import apply_growth_transpose
from orbit.modes.marginal import _compute_delta_pm25
from orbit import cli

N_BINS = 8
NZ, NY, NX = 2, 3, 4
N = NZ * NY * NX
SURF = NY * NX


class _Indexer:
    nz, ny, nx = NZ, NY, NX
    N = N


class _Grid:
    def __init__(self, p_nh, p_no3):
        self.NHPartitioning = np.full((NZ, NY, NX), p_nh)
        self.NO3Partitioning = np.full((NZ, NY, NX), p_no3)
        self.F_p_vbs = None
        self.M_OA_3d = None


def test_forward_field_uses_the_shared_constants():
    assert cli.SIA_GROWTH_RH35 == SIA_GROWTH_RH35 == 1.10
    assert cli.ORG_GROWTH_RH35 == ORG_GROWTH_RH35 == 1.05
    out = cli._pm25_at_rh35(pm25_dry=np.array([10.0]), c_pm=np.array([2.0]),
                            c_poa=np.array([3.0]), c_soa=np.array([1.0]))
    # inorganics = 10 - 2 - 3 - 1 = 4 -> 2 + 1.05*4 + 1.10*4
    assert out[0] == pytest.approx(2.0 + 1.05 * 4.0 + 1.10 * 4.0)


@pytest.mark.parametrize("rh35", [False, True])
def test_receptor_and_marginal_delta_are_dual(rh35):
    """<receptor, δc> over the surface equals the marginal δPM2.5 there."""
    rng = np.random.default_rng(3)
    p_nh, p_no3 = 0.6, 0.3
    grids = [_Grid(p_nh, p_no3) for _ in range(N_BINS)]
    S = rng.uniform(0.5, 2.0, size=(N_BINS, NY, NX))
    recep = apply_growth_transpose(
        S, grids, _Indexer(), None,
        p_nh4_surface_per_bin=np.full((N_BINS, NY, NX), p_nh),
        p_no3_surface_per_bin=np.full((N_BINS, NY, NX), p_no3),
        pm25_rh35=rh35,
    )
    delta = np.zeros((N_SPECIES, N_BINS + 1, N))
    for idx in (IDX_PM25, IDX_POA, IDX_PSO4, IDX_TOTAL_NH, IDX_TOTAL_NO3):
        delta[idx] = rng.uniform(-1, 1, size=(N_BINS + 1, N))
    d_orbit, _ = _compute_delta_pm25(delta, grids, NZ, NY, NX, N, pm25_rh35=rh35)
    for tau in range(N_BINS):
        lhs = sum(recep[tau, sp] @ delta[sp, tau + 1] for sp in range(N_SPECIES))
        rhs = (S[tau].ravel() * d_orbit[tau, 0].ravel()).sum()
        assert lhs == pytest.approx(rhs, rel=1e-12)


def test_rh35_receptor_scales_only_the_grown_species():
    grids = [_Grid(0.5, 0.5) for _ in range(N_BINS)]
    S = np.ones((N_BINS, NY, NX))
    kw = dict(p_nh4_surface_per_bin=np.full((N_BINS, NY, NX), 0.5),
              p_no3_surface_per_bin=np.full((N_BINS, NY, NX), 0.5))
    dry = apply_growth_transpose(S, grids, _Indexer(), None, **kw)
    wet = apply_growth_transpose(S, grids, _Indexer(), None, pm25_rh35=True, **kw)
    np.testing.assert_allclose(wet[:, IDX_PM25], dry[:, IDX_PM25])
    np.testing.assert_allclose(wet[:, IDX_POA], 1.05 * dry[:, IDX_POA])
    for idx in (IDX_PSO4, IDX_TOTAL_NH, IDX_TOTAL_NO3):
        np.testing.assert_allclose(wet[:, idx], 1.10 * dry[:, idx])
    assert wet[0, IDX_PSO4, 0] == pytest.approx(1.10 * S_TO_SO4)
    assert wet[0, IDX_TOTAL_NH, 0] == pytest.approx(1.10 * 0.5 * N_TO_NH4)
    assert wet[0, IDX_TOTAL_NO3, 0] == pytest.approx(1.10 * 0.5 * N_TO_NO3)
