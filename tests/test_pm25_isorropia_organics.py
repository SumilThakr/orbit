"""extract_pm25_isorropia counts the same organics as the reported PM2.5.

Until 2026-09-26 it summed the C100 bin alone and had no POA term, so
iso_pm25_mean was low against pm25_mean by the POA plus the four other
bins. The inorganic part is unchanged: it is the LUT partitioning.
"""
import numpy as np

from orbit.core.constants import N_TO_NH4, N_TO_NO3, S_TO_SO4
from orbit.core.deposition import (
    IDX_SOA, IDX_PM25, IDX_POA, IDX_TOTAL_NH, IDX_PSO4, IDX_TOTAL_NO3,
    IDX_VBS_BINS, N_SPECIES,
)
from orbit.core.indexing import CellIndexer
from orbit.core.orbit import extract_pm25_isorropia

NZ, NY, NX = 1, 2, 2
N = NZ * NY * NX


class _LUT:
    """Half of TotalNH and a third of TotalNO3 in the particle phase."""
    def query(self, so4, nh, no3, ca, na, T, rh):
        n = so4.size
        return (np.full(n, 0.5), np.full(n, 1.0 / 3.0),
                np.zeros(n), np.full(n, 4.0))


class _Grid:
    nz, ny, nx = NZ, NY, NX

    def __init__(self, F_p=None):
        shape = (NZ, NY, NX)
        self.Temperature = np.full(shape, 290.0)
        self.RH = np.full(shape, 60.0)
        self.dust_fine = np.array([])
        self.sea_salt_fine = np.array([])
        self.F_p_vbs = F_p


def _state():
    c = np.zeros(N_SPECIES * N)
    def put(idx, v):
        c[idx * N:(idx + 1) * N] = v
    put(IDX_PM25, 10.0)
    put(IDX_POA, 4.0)
    put(IDX_TOTAL_NH, 2.0)
    put(IDX_PSO4, 3.0)
    put(IDX_TOTAL_NO3, 6.0)
    for i, s in enumerate(IDX_VBS_BINS):
        put(s, 1.0 + i)            # bins hold 1, 2, 3, 4, 5
    return c


def _inorganic():
    return 0.5 * 2.0 * N_TO_NH4 + 3.0 * S_TO_SO4 + (1.0 / 3.0) * 6.0 * N_TO_NO3


def test_counts_poa_and_all_five_bins_when_f_p_is_present():
    F_p = np.zeros((len(IDX_VBS_BINS), NZ, NY, NX))
    for i in range(len(IDX_VBS_BINS)):
        F_p[i] = 0.1 * (i + 1)     # 0.1 .. 0.5
    pm, diag = extract_pm25_isorropia(_state(), _Grid(F_p), CellIndexer(NZ, NY, NX), _LUT())
    soa = sum(0.1 * (i + 1) * (1.0 + i) for i in range(len(IDX_VBS_BINS)))
    expected = 10.0 + 4.0 + soa + _inorganic()
    np.testing.assert_allclose(pm, expected)
    assert diag["f_nh4"].shape == (NZ, NY, NX)


def test_falls_back_to_the_c100_bin_without_f_p_but_keeps_poa():
    pm, _ = extract_pm25_isorropia(_state(), _Grid(None), CellIndexer(NZ, NY, NX), _LUT())
    c100 = 1.0 + IDX_VBS_BINS.index(IDX_SOA)
    np.testing.assert_allclose(pm, 10.0 + 4.0 + c100 + _inorganic())


def test_organics_match_the_reported_pm25_construction():
    """Same organics as cli._soa_3d_flat plus POA: only the inorganic part
    may differ from pm25_mean."""
    from orbit.cli import _soa_3d_flat
    F_p = np.full((len(IDX_VBS_BINS), NZ, NY, NX), 0.3)
    c = _state()
    grid = _Grid(F_p)
    pm, _ = extract_pm25_isorropia(c, grid, CellIndexer(NZ, NY, NX), _LUT())
    soa_cli = _soa_3d_flat(lambda s: np.maximum(c[s * N:(s + 1) * N], 0), grid, N)
    organics = pm.ravel() - _inorganic() - 10.0
    np.testing.assert_allclose(organics, soa_cli + 4.0)
