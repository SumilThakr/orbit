"""Regression tests for the ORBIT 1.0.4 fixes."""
import os

import numpy as np
import pytest

from orbit.core.deposition import (
    IDX_VBS_BINS, IDX_VBS_C01, IDX_VBS_C1, IDX_VBS_C10, IDX_VBS_C100,
    IDX_VBS_C1000,
)


class TestItem14AdjointVbsLabels:
    """Each VBS_C* adjoint key must address its own bin (was reversed)."""

    def test_each_key_maps_to_its_own_bin(self):
        from orbit.modes.adjoint import EMITTED_SPECIES
        assert EMITTED_SPECIES["VBS_C01"] == IDX_VBS_C01
        assert EMITTED_SPECIES["VBS_C1"] == IDX_VBS_C1
        assert EMITTED_SPECIES["VBS_C10"] == IDX_VBS_C10
        assert EMITTED_SPECIES["VBS_C100"] == IDX_VBS_C100
        assert EMITTED_SPECIES["VBS_C1000"] == IDX_VBS_C1000

    def test_voc_combination_order_matches_forward_distribution(self):
        # The VOC pseudo-species multiply yields row i (low-to-high C*,
        # VBS_PARENT_YIELDS order) by the adjoint row of VBS_BIN_KEYS[i].
        # The forward run puts yield row i into IDX_VBS_BINS[i], so the two
        # must address the same solver species.
        from orbit.modes.adjoint import EMITTED_SPECIES, VBS_BIN_KEYS
        from orbit.emissions.vbs_distribution import bin_indices_in_solver_layout
        fwd = tuple(bin_indices_in_solver_layout())
        adj = tuple(EMITTED_SPECIES[k] for k in VBS_BIN_KEYS)
        assert adj == fwd == tuple(IDX_VBS_BINS)

    def test_high_nox_c1000_yield_lands_on_c1000(self):
        from orbit.modes.adjoint import EMITTED_SPECIES, VBS_BIN_KEYS
        from orbit.emissions.vbs_yields import VBS_PARENT_YIELDS
        y = VBS_PARENT_YIELDS["anthro_high_nox"]
        i_max = int(np.argmax(y))
        assert EMITTED_SPECIES[VBS_BIN_KEYS[i_max]] == IDX_VBS_C1000


class TestItem16GradientRegrid:
    """Deaths-gradient regrid: area-overlap split, not nearest centre."""

    # v1.0.3 SAS layout: aqh 0.1-degree lattice (centres x.05) onto ORBIT
    # 0.5 x 0.625 with edges at x.25 / x.75 (centres 4.0, 4.5, ...).
    lat_f = np.round(np.arange(4.05, 12.0, 0.1), 6)
    lon_f = np.round(np.arange(60.05, 70.0, 0.1), 6)
    lat_o = np.arange(5.0, 11.0 + 1e-9, 0.5)
    lon_o = 61.25 + 0.625 * np.arange(12)

    def _regrid(self, S, lat_f=None, lon_f=None, lat_o=None, lon_o=None):
        from orbit.modes.adjoint import _regrid_gradient_to_orbit_grid
        return _regrid_gradient_to_orbit_grid(
            S, self.lat_f if lat_f is None else lat_f,
            self.lon_f if lon_f is None else lon_f,
            self.lat_o if lat_o is None else lat_o,
            self.lon_o if lon_o is None else lon_o)

    def _inside_total(self, S):
        # fine-cell area inside the ORBIT domain, cell by cell
        from orbit.modes.adjoint import _overlap_fractions
        wy = _overlap_fractions(self.lat_f, 0.1, self.lat_o, 0.5).sum(0)
        wx = _overlap_fractions(self.lon_f, 0.1, self.lon_o, 0.625).sum(0)
        return float(wy @ S @ wx)

    def test_total_conserved(self):
        rng = np.random.default_rng(0)
        S = rng.random((self.lat_f.size, self.lon_f.size))
        out = self._regrid(S)
        np.testing.assert_allclose(out.sum(), self._inside_total(S), rtol=1e-6)

    def test_uniform_field_has_no_stripes(self):
        S = np.ones((self.lat_f.size, self.lon_f.size))
        out = self._regrid(S)
        # every ORBIT cell is fully covered: 5 fine rows x 6.25 fine columns
        np.testing.assert_allclose(out, 5 * 6.25, rtol=1e-9)

    def test_origin_independence(self):
        rng = np.random.default_rng(1)
        S = rng.random((self.lat_f.size, self.lon_f.size))
        full = self._regrid(S)
        # same fine field, ORBIT domain shifted by one cell in each direction
        shifted = self._regrid(S, lat_o=self.lat_o[1:], lon_o=self.lon_o[1:])
        np.testing.assert_allclose(shifted, full[1:, 1:], rtol=1e-12)

    def test_identity_on_orbit_grid(self):
        rng = np.random.default_rng(2)
        S = rng.random((self.lat_o.size, self.lon_o.size))
        out = self._regrid(S, lat_f=self.lat_o, lon_f=self.lon_o)
        np.testing.assert_allclose(out, S, rtol=1e-12)

    def test_nonfinite_raises(self):
        import pytest
        S = np.ones((self.lat_f.size, self.lon_f.size)); S[3, 4] = np.nan
        with pytest.raises(ValueError):
            self._regrid(S)


class TestItem1EquilibriumDeposition:
    """Full solves blend TotalNH/TotalNO3 deposition with the equilibrium
    fraction; linearised solves keep the marginal fraction."""

    F_MARG, F_EQ = 0.2, 0.45

    def _grid(self, small_grid_params, with_eq=True):
        from tests.test_advection import _make_grid
        g = _make_grid(small_grid_params)
        shp = (g.nz, g.ny, g.nx)
        g.NHPartitioning = np.full(shp, self.F_MARG)
        g.NO3Partitioning = np.full(shp, self.F_MARG)
        if with_eq:
            g.NHPartitioningEq = np.full(shp, self.F_EQ)
            g.NO3PartitioningEq = np.full(shp, self.F_EQ)
        else:
            g.NHPartitioningEq = np.empty(0)
            g.NO3PartitioningEq = np.empty(0)
        # distinct gas and particle rates so the blend is visible
        g.NH3_dry_dep = np.full(shp, 0.02)
        g.particle_dry_dep = np.full(shp, 0.002)
        g.HNO3_dry_dep = np.full(shp, 0.03)
        g.other_gas_wet_dep = np.full(shp, 4e-5)
        g.particle_wet_dep = np.full(shp, 1e-5)
        return g

    def _surface_rate(self, g, s, partitioning):
        from orbit.core.deposition import assemble_deposition
        from orbit.core.indexing import CellIndexer
        idx = CellIndexer(g.nz, g.ny, g.nx)
        d = assemble_deposition(g, idx, s, partitioning=partitioning).diagonal()
        return d.reshape(g.nz, g.ny, g.nx)[0, 1, 1]

    def _expected(self, g, f, vd_gas):
        dz = g.Dz[0, 1, 1]
        dry = (1 - f) * vd_gas + f * 0.002
        wet = (1 - f) * 4e-5 + f * 1e-5
        return dry / dz + wet

    def test_equilibrium_and_marginal_select_their_fields(self, small_grid_params):
        from orbit.core.deposition import IDX_TOTAL_NH, IDX_TOTAL_NO3
        g = self._grid(small_grid_params)
        for s, vg in ((IDX_TOTAL_NH, 0.02), (IDX_TOTAL_NO3, 0.03)):
            np.testing.assert_allclose(
                self._surface_rate(g, s, "equilibrium"),
                self._expected(g, self.F_EQ, vg), rtol=1e-12)
            np.testing.assert_allclose(
                self._surface_rate(g, s, "marginal"),
                self._expected(g, self.F_MARG, vg), rtol=1e-12)

    def test_default_is_marginal(self, small_grid_params):
        from orbit.core.deposition import assemble_deposition, IDX_TOTAL_NH
        from orbit.core.indexing import CellIndexer
        g = self._grid(small_grid_params)
        idx = CellIndexer(g.nz, g.ny, g.nx)
        a = assemble_deposition(g, idx, IDX_TOTAL_NH).diagonal()
        b = assemble_deposition(g, idx, IDX_TOTAL_NH,
                                partitioning="marginal").diagonal()
        np.testing.assert_array_equal(a, b)

    def test_before_closure_equilibrium_reads_preprocessor_field(self, small_grid_params):
        from orbit.core.deposition import IDX_TOTAL_NH
        g = self._grid(small_grid_params, with_eq=False)
        np.testing.assert_allclose(
            self._surface_rate(g, IDX_TOTAL_NH, "equilibrium"),
            self._expected(g, self.F_MARG, 0.02), rtol=1e-12)

    def test_other_species_unaffected(self, small_grid_params):
        from orbit.core.deposition import (assemble_deposition, N_SPECIES,
                                           IDX_TOTAL_NH, IDX_TOTAL_NO3)
        from orbit.core.indexing import CellIndexer
        g = self._grid(small_grid_params)
        idx = CellIndexer(g.nz, g.ny, g.nx)
        for s in range(N_SPECIES):
            if s in (IDX_TOTAL_NH, IDX_TOTAL_NO3):
                continue
            np.testing.assert_array_equal(
                assemble_deposition(g, idx, s, partitioning="equilibrium").diagonal(),
                assemble_deposition(g, idx, s, partitioning="marginal").diagonal())

    def test_loop_matches_vectorised(self, small_grid_params):
        from orbit.core.deposition import (assemble_deposition,
                                           _assemble_deposition_loop,
                                           IDX_TOTAL_NH, IDX_TOTAL_NO3)
        from orbit.core.indexing import CellIndexer
        g = self._grid(small_grid_params)
        idx = CellIndexer(g.nz, g.ny, g.nx)
        for s in (IDX_TOTAL_NH, IDX_TOTAL_NO3):
            np.testing.assert_allclose(
                _assemble_deposition_loop(g, idx, s, "equilibrium").diagonal(),
                assemble_deposition(g, idx, s, partitioning="equilibrium").diagonal(),
                rtol=1e-12)

    def test_unknown_choice_raises(self, small_grid_params):
        import pytest
        from orbit.core.deposition import assemble_deposition, IDX_TOTAL_NH
        from orbit.core.indexing import CellIndexer
        g = self._grid(small_grid_params)
        with pytest.raises(ValueError):
            assemble_deposition(g, CellIndexer(g.nz, g.ny, g.nx), IDX_TOTAL_NH,
                                partitioning="eq")

    def test_species_operators_pass_choice_through(self, small_grid_params):
        from orbit.core.operator import assemble_species_operators
        from orbit.core.deposition import IDX_TOTAL_NH
        from orbit.core.indexing import CellIndexer
        g = self._grid(small_grid_params)
        idx = CellIndexer(g.nz, g.ny, g.nx)
        Le = assemble_species_operators(g, idx, verbose=False,
                                        partitioning="equilibrium")[0]
        Lm = assemble_species_operators(g, idx, verbose=False)[0]
        d = (Le[IDX_TOTAL_NH] - Lm[IDX_TOTAL_NH]).diagonal()
        n = idx.to_flat(0, 1, 1)
        expected = (self._expected(g, self.F_EQ, 0.02)
                    - self._expected(g, self.F_MARG, 0.02))
        np.testing.assert_allclose(d[n], expected, rtol=1e-10)

    def test_forward_solver_requests_equilibrium(self):
        # _assemble_and_solve serves forward, zero-out and perturbation; every
        # deposition assembly in it must ask for the equilibrium fraction.
        import inspect
        from orbit import cli
        src = inspect.getsource(cli._assemble_and_solve)
        n_calls = src.count("assemble_species_operators(") + src.count("_assdep_inner(")
        assert n_calls >= 3
        assert src.count('partitioning="equilibrium"') == n_calls


class TestItem18TrueEdgeFaces:
    """load_grid always keeps the east/north outer-face winds of the input."""

    GRID = os.path.join(os.environ.get("ORBIT_PREPROC_DIR", ""), "sas_2022_M01_B04.nc")

    def _load(self, monkeypatch, env=None):
        import os
        import pytest
        if not os.path.exists(self.GRID):
            pytest.skip("1.0.3 production grid not available")
        if env is None:
            monkeypatch.delenv("ORBIT_TRUE_EDGE_FACES", raising=False)
        else:
            monkeypatch.setenv("ORBIT_TRUE_EDGE_FACES", env)
        from orbit.core.grid_data import load_grid
        return load_grid(self.GRID)

    def test_real_grid_carries_true_faces(self, monkeypatch):
        import netCDF4
        from orbit.core.grid_data import edge_face
        g = self._load(monkeypatch)
        with netCDF4.Dataset(self.GRID) as d:
            u = np.asarray(d["UAvg"][:], dtype=np.float64)
            v = np.asarray(d["VAvg"][:], dtype=np.float64)
        assert g.U_face_e.size and g.V_face_n.size
        # the outer face is the staggered input's last face, not the edge
        # cell's west/south face
        np.testing.assert_allclose(edge_face(g, "U"), np.take(u, g.nx, axis=u.ndim - 1)[:g.nz],
                                   rtol=1e-6)
        np.testing.assert_allclose(edge_face(g, "V"), np.take(v, g.ny, axis=v.ndim - 2)[:g.nz],
                                   rtol=1e-6)
        assert not np.allclose(edge_face(g, "U"), g.UAvg[:, :, -1])

    def test_setting_1_is_accepted(self, monkeypatch):
        assert self._load(monkeypatch, env="1").U_face_e.size

    def test_old_stand_in_is_refused(self, monkeypatch):
        import pytest
        with pytest.raises(ValueError):
            self._load(monkeypatch, env="0")


class TestItem19FctGhostAtInflowEdge:
    """Horizontal FCT uses the lateral boundary value as its far-upwind ghost."""

    class _G:
        def __init__(self, nx, u=2.0, v=0.0, nz=2, ny=4):
            self.nz, self.ny, self.nx = nz, ny, nx
            self.has_split_fluxes = False
            self.UAvg = np.full((nz, ny, nx), u)
            self.VAvg = np.full((nz, ny, nx), v)
            self.dx = np.full(ny, 5.0e4)
            self.dy = 5.0e4
            self.dP = np.ones((nz, ny, nx))
            self.concentration_state = False      # transported variable = c

    @staticmethod
    def _profile(nx, nz=2, ny=4):
        x = np.arange(nx, dtype=float)
        prof = 10.0 + 5.0 * np.sin(x / 3.0) + 0.3 * x      # smooth, not monotone
        return np.broadcast_to(prof, (nz, ny, nx)).copy()

    def _src(self, g, c, ghost=None, limited=True):
        from orbit.core.fct import compute_horizontal_fct_source
        from orbit.core.indexing import CellIndexer
        idx = CellIndexer(g.nz, g.ny, g.nx)
        gf = None if ghost is None else ghost.ravel()
        return compute_horizontal_fct_source(g, idx, c.ravel(), 10800.0, limited,
                                             c_ghost_flat=gf).reshape(c.shape)

    def test_closed_boundary_unchanged(self):
        # a ghost equal to the edge cell (no inflow information) gives the
        # first-order edge face exactly as before
        g = self._G(12); c = self._profile(12)
        np.testing.assert_allclose(self._src(g, c, ghost=c.copy()), self._src(g, c),
                                   rtol=0, atol=1e-14 * np.abs(self._src(g, c)).max())

    def test_nested_domain_matches_parent_unlimited(self):
        # parent of 20 columns; child = columns 6..19 with the parent's column 5
        # as the ghost. Eastward wind: child cells from its second column on
        # see the same unlimited anti-diffusive fluxes as the parent.
        k = 6
        gp, cp = self._G(20), self._profile(20)
        gc, cc = self._G(20 - k), cp[:, :, k:].copy()
        ghost = cc.copy(); ghost[:, :, 0] = cp[:, :, k - 1]   # only the west edge has inflow
        dp = self._src(gp, cp, limited=False)[:, :, k + 1:]
        dc_new = self._src(gc, cc, ghost=ghost, limited=False)[:, :, 1:]
        dc_old = self._src(gc, cc, limited=False)[:, :, 1:]
        np.testing.assert_allclose(dc_new, dp, rtol=1e-12, atol=1e-18)
        # without the ghost the second child column differs (first-order face)
        assert not np.allclose(dc_old[:, :, 0], dp[:, :, 0], rtol=1e-6)

    def test_nested_domain_closer_to_parent_limited(self):
        k = 6
        gp, cp = self._G(20), self._profile(20)
        gc, cc = self._G(20 - k), cp[:, :, k:].copy()
        ghost = cc.copy(); ghost[:, :, 0] = cp[:, :, k - 1]
        dp = self._src(gp, cp)[:, :, k:]
        err_new = np.abs(self._src(gc, cc, ghost=ghost) - dp)[:, :, :3].max()
        err_old = np.abs(self._src(gc, cc) - dp)[:, :, :3].max()
        assert err_new < err_old

    def test_ghost_bounds_limiter(self):
        # a ghost far above the field must not create a new extremum: with the
        # limiter, c + dt*d stays within the neighbourhood range including it
        g = self._G(10); c = self._profile(10)
        ghost = c.copy(); ghost[:, :, 0] = 100.0
        d = self._src(g, c, ghost=ghost)
        new = c + 10800.0 * d
        lo = np.minimum(c.min(), ghost.min()); hi = np.maximum(c.max(), ghost.max())
        assert new.min() >= lo - 1e-9 and new.max() <= hi + 1e-9

    def test_lateral_bc_loader_keeps_c_bc(self, tmp_path):
        from orbit.core.lateral_bc import LateralBCSource, merge_into
        src = LateralBCSource({3: [np.ones(4)] * 8})
        src.c_bc[3] = [np.full(4, 2.0)] * 8
        merged = merge_into(None, src, 8)
        assert 3 in merged and src.c_bc[3][0][0] == 2.0


class TestClosureGateOverLand:
    """The inorganic closure gate is the p90 over land cells; ocean reported."""

    def test_ocean_cells_do_not_gate(self):
        from orbit.cli import _closure_gate_percentiles
        land = np.zeros((4, 5), bool)
        land[:2] = True
        rel = np.full((3, 4, 5), 0.001)
        rel[:, 2:] = 0.5                       # ocean still moving
        p90, p50, p90_oc, idx = _closure_gate_percentiles(rel, land)
        assert p90 == pytest.approx(0.001) and p50 == pytest.approx(0.001)
        assert p90_oc == pytest.approx(0.5)
        assert land[idx[1], idx[2]]            # argmax reported over land

    def test_all_land_equals_all_cells(self):
        from orbit.cli import _closure_gate_percentiles
        rel = np.random.default_rng(0).random((8, 6, 7))
        p90, p50, p90_oc, _ = _closure_gate_percentiles(rel, np.ones((6, 7), bool))
        assert p90 == pytest.approx(np.percentile(rel, 90))
        assert np.isnan(p90_oc)

    def test_no_land_raises(self):
        from orbit.cli import _closure_gate_percentiles
        with pytest.raises(ValueError):
            _closure_gate_percentiles(np.ones((2, 3, 3)), np.zeros((3, 3), bool))


class TestItem21AdjointReceptorEndOfBin:
    """The bin-tau receptor weights the end-of-bin state c[tau+1]."""

    def _system(self):
        import scipy.sparse as sp
        from orbit.core.orbit import N_BINS, _factor_one_bin
        n = 40
        rng = np.random.default_rng(3)
        I_N = sp.eye(n, format="csc")
        lus = []
        for _ in range(N_BINS):
            L = sp.random(n, n, density=0.08, format="csc", random_state=rng) * 5e-5
            L = (L + 1e-4 * I_N).tocsc()
            lus.append(_factor_one_bin(L, I_N, None, None))
        # receptors differ strongly by bin (a diurnal cycle)
        rec = np.zeros((N_BINS, 2, n))
        for t in range(N_BINS):
            rec[t, 1] = rng.random(n) * (1 + 3 * np.sin(np.pi * t / N_BINS) ** 2)
        e = [rng.standard_normal(n) for _ in range(N_BINS)]
        return lus, rec, e, n

    def _J(self, lus, rec, e):
        from orbit.core.orbit import N_BINS, DTAU, solve_orbit_one_species
        orbit, _ = solve_orbit_one_species(lus, e, DTAU, c_warm=None,
                                           tol=1e-13, maxiter=500)
        return float(sum(rec[t, 1] @ orbit[t + 1] for t in range(N_BINS)))

    def _grad(self, lus, R):
        from orbit.core.orbit_adjoint import (solve_orbit_adjoint_one_species,
                                              gradient_per_emission_bin)
        lam, _ = solve_orbit_adjoint_one_species(lus, R, tol=1e-13, maxiter=500)
        return gradient_per_emission_bin(lam, R)

    def test_matches_finite_difference(self):
        from orbit.core.solve import _HAS_UMFPACK
        if not _HAS_UMFPACK:
            pytest.skip("UMFPACK not available")
        from orbit.modes.adjoint import receptor_R_list
        lus, rec, e, n = self._system()
        g = self._grad(lus, receptor_R_list(rec, 1))
        J0 = self._J(lus, rec, e)
        h = 1e-3
        for t, j in [(0, 3), (2, 17), (5, 8), (7, 30)]:
            ep = [x.copy() for x in e]
            ep[t][j] += h
            fd = (self._J(lus, rec, ep) - J0) / h      # J is linear in e
            assert g[t][j] == pytest.approx(fd, rel=1e-6)

    def test_old_pairing_is_detectably_wrong(self):
        from orbit.core.solve import _HAS_UMFPACK
        if not _HAS_UMFPACK:
            pytest.skip("UMFPACK not available")
        from orbit.core.orbit import N_BINS
        from orbit.modes.adjoint import receptor_R_list
        lus, rec, e, n = self._system()
        new = self._grad(lus, receptor_R_list(rec, 1))
        old = self._grad(lus, [rec[t, 1].copy() for t in range(N_BINS)])
        rel = max(np.abs(a - b).max() / np.abs(a).max() for a, b in zip(new, old))
        assert rel > 1e-2


class TestItem22AdjointSurfaceVbs:
    """The adjoint sets the surface VBS partitioning as marginal does."""

    def test_adjoint_sets_surface_fields(self):
        import inspect
        from orbit.modes import adjoint
        src = inspect.getsource(adjoint)
        assert "grids[tau].F_p_vbs_surface = F_p_vbs_marg_3d[tau][:, 0]" in src
        assert "grids[tau].M_OA_surface = M_OA_marg_3d[tau][0]" in src

