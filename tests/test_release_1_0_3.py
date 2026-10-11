"""Regression tests for the 1.0.3 model fixes (docs/notes/1.0.3_fixes/).

Each test pins one behaviour that the methods audit found wrong in 1.0.2:
the NOx-regime switch, VBS aging on the vapour fraction only, NOx and
organic-vapour wet deposition, per-bin plume rise, the temperature-driven
NH3 diurnal profile, and the Briggs buoyancy flux.
"""
import numpy as np
import pytest

from orbit.core.deposition import (
    IDX_NOX, IDX_VBS_BINS, IDX_VBS_C1000, IDX_VBS_C100, N_VBS_BINS,
    VBS_AGING_PAIRS, _get_wet_dep_rate_3d, _get_wet_dep_rate_scalar,
)
from orbit.core.grid_data import GridData


# ---------------------------------------------------------------------------
# NOx regime: RO2 + NO branching ratio with consistent units
# ---------------------------------------------------------------------------

def _regime_grid(no_ppbv, oh, T=298.0, P=101325.0, ny=2, nx=2, nz=2):
    g = GridData(nz=nz, ny=ny, nx=nx)
    g.archive_NO = np.full((nz, ny, nx), no_ppbv)
    g.archive_OH = np.full((nz, ny, nx), oh)
    g.Temperature = np.full((nz, ny, nx), T)
    g.Pressure = np.full((nz, ny, nx), P)
    return g


class TestNoxRegime:
    def test_polluted_surface_is_high_nox(self):
        """20 ppbv NO at midday OH: RO2 + NO dominates (beta near 1)."""
        from orbit.emissions.vbs_distribution import compute_nox_regime
        F = compute_nox_regime(_regime_grid(20.0, 5.0e6))
        assert F.shape == (2, 2)
        assert np.all(F > 0.95)

    def test_no_free_air_is_low_nox(self):
        from orbit.emissions.vbs_distribution import compute_nox_regime
        F = compute_nox_regime(_regime_grid(0.0, 5.0e6))
        assert np.all(F == 0.0)

    def test_half_point_matches_branching_ratio(self):
        """beta = 0.5 where k_NO [NO] = k_HO2 [HO2]; check the unit chain."""
        from orbit.emissions.vbs_distribution import (
            compute_nox_regime, HO2_TO_OH_RATIO, K_BOLTZ,
            K_RO2_NO_A, K_RO2_NO_B, K_RO2_HO2_A, K_RO2_HO2_B,
        )
        T, P, oh = 298.0, 101325.0, 2.0e6
        k_no = K_RO2_NO_A * np.exp(K_RO2_NO_B / T)
        k_ho2 = K_RO2_HO2_A * np.exp(K_RO2_HO2_B / T)
        n_air = P / (K_BOLTZ * T) * 1e-6
        no_half_ppbv = k_ho2 * HO2_TO_OH_RATIO * oh / k_no / n_air * 1e9
        F = compute_nox_regime(_regime_grid(no_half_ppbv, oh, T=T, P=P))
        np.testing.assert_allclose(F, 0.5, rtol=1e-9)
        # The 1.0.2 form, log10(NO2[ppbv]/OH[molec cm-3]) > 1.5, could never
        # reach the high-NOx regime: the ratio is ~1e-7 in polluted air.
        assert no_half_ppbv < 1.0   # sub-ppbv NO already tips the balance

    def test_missing_archive_is_regime_neutral(self):
        from orbit.emissions.vbs_distribution import compute_nox_regime
        g = GridData(nz=1, ny=2, nx=3)
        np.testing.assert_array_equal(compute_nox_regime(g), 0.5)

    def test_per_bin_stack(self):
        from orbit.emissions.vbs_distribution import compute_nox_regime_per_bin
        grids = [_regime_grid(20.0, 5e6), _regime_grid(0.0, 5e6)]
        F = compute_nox_regime_per_bin(grids)
        assert F.shape == (2, 2, 2)
        assert np.all(F[0] > 0.95) and np.all(F[1] == 0.0)


# ---------------------------------------------------------------------------
# VBS aging acts on the vapour fraction only
# ---------------------------------------------------------------------------

class TestVbsAgingGasOnly:
    @pytest.fixture
    def grid_oh(self, small_grid_params):
        from tests.test_advection import _make_grid
        g = _make_grid(small_grid_params)
        g.archive_OH = np.full((g.nz, g.ny, g.nx), 1.0e6)
        g.archive_NO2 = np.full((g.nz, g.ny, g.nx), 1.0e10)
        return g

    def _loss_diag(self, g, src):
        from orbit.core.operator import assemble_species_operators
        from orbit.core.indexing import CellIndexer
        idx = CellIndexer(g.nz, g.ny, g.nx)
        zero = np.zeros((g.nz, g.ny, g.nx))
        _, K_sources, _, _ = assemble_species_operators(
            g, idx, scheme="exp", nox_to_no3_rate=zero)
        dst = dict(VBS_AGING_PAIRS)[src]
        return -K_sources[(dst, src)].diagonal()

    def test_particle_phase_mass_does_not_age(self, grid_oh, monkeypatch):
        monkeypatch.setenv("ORBIT_VBS_AGING_GAIN", "1.0")
        full = self._loss_diag(grid_oh, IDX_VBS_C1000).copy()
        assert np.all(full > 0)
        F_p = np.zeros((N_VBS_BINS, grid_oh.nz, grid_oh.ny, grid_oh.nx))
        F_p[IDX_VBS_BINS.index(IDX_VBS_C1000)] = 1.0     # all particulate
        grid_oh.F_p_vbs = F_p
        np.testing.assert_array_equal(self._loss_diag(grid_oh, IDX_VBS_C1000), 0.0)

    def test_rate_scales_with_vapour_fraction(self, grid_oh, monkeypatch):
        monkeypatch.setenv("ORBIT_VBS_AGING_GAIN", "1.0")
        full = self._loss_diag(grid_oh, IDX_VBS_C100).copy()
        F_p = np.zeros((N_VBS_BINS, grid_oh.nz, grid_oh.ny, grid_oh.nx))
        F_p[IDX_VBS_BINS.index(IDX_VBS_C100)] = 0.75
        grid_oh.F_p_vbs = F_p
        np.testing.assert_allclose(self._loss_diag(grid_oh, IDX_VBS_C100), 0.25 * full)

    def test_other_bins_unaffected_by_one_bins_fraction(self, grid_oh, monkeypatch):
        monkeypatch.setenv("ORBIT_VBS_AGING_GAIN", "1.0")
        full = self._loss_diag(grid_oh, IDX_VBS_C1000).copy()
        F_p = np.zeros((N_VBS_BINS, grid_oh.nz, grid_oh.ny, grid_oh.nx))
        F_p[IDX_VBS_BINS.index(IDX_VBS_C100)] = 0.9
        grid_oh.F_p_vbs = F_p
        np.testing.assert_allclose(self._loss_diag(grid_oh, IDX_VBS_C1000), full)


# ---------------------------------------------------------------------------
# Wet deposition: NOx none, semivolatile organics particle-only
# ---------------------------------------------------------------------------

def _wet_grid():
    n = (2, 2, 2)
    g = GridData(nz=2, ny=2, nx=2,
                 particle_wet_dep=np.full(n, 1.0e-4),
                 other_gas_wet_dep=np.full(n, 3.0e-5),
                 SO2_wet_dep=np.full(n, 1.0e-5),
                 NHPartitioning=np.full(n, 0.5))
    return g


class TestWetDeposition:
    def test_nox_is_not_wet_deposited(self):
        g = _wet_grid()
        np.testing.assert_array_equal(_get_wet_dep_rate_3d(g, IDX_NOX), 0.0)
        assert _get_wet_dep_rate_scalar(g, IDX_NOX, 0, 0, 0) == 0.0

    def test_vbs_vapour_is_not_wet_deposited(self):
        g = _wet_grid()
        F_p = np.zeros((N_VBS_BINS, 2, 2, 2))
        F_p[IDX_VBS_BINS.index(IDX_VBS_C100)] = 0.25
        g.F_p_vbs = F_p
        rate = _get_wet_dep_rate_3d(g, IDX_VBS_C100)
        np.testing.assert_allclose(rate, 0.25 * 1.0e-4)   # no gas term

    def test_vbs_falls_back_to_particle_rate_without_partitioning(self):
        g = _wet_grid()
        np.testing.assert_allclose(_get_wet_dep_rate_3d(g, IDX_VBS_C100), 1.0e-4)


# ---------------------------------------------------------------------------
# Temperature-driven NH3 diurnal profile (Zhu et al. 2015, Eq. 3)
# ---------------------------------------------------------------------------

class TestHenryTemperatureProfile:
    def _grids(self, temps):
        out = []
        for T in temps:
            g = GridData(nz=1, ny=2, nx=3)
            g.Temperature = np.full((1, 2, 3), T)
            out.append(g)
        return out

    def test_factors_have_unit_bin_mean_per_cell(self):
        from orbit.emissions.loader import DynamicProfile, dynamic_profile_factors
        grids = self._grids([285, 287, 292, 298, 303, 301, 295, 289])
        f = dynamic_profile_factors(DynamicProfile("nh3", "henry_temperature"), grids, 8)
        assert f.shape == (8, 2, 3)
        np.testing.assert_allclose(f.mean(axis=0), 1.0, atol=1e-12)

    def test_warmer_bin_emits_more(self):
        from orbit.emissions.loader import DynamicProfile, dynamic_profile_factors
        grids = self._grids([285, 287, 292, 298, 303, 301, 295, 289])
        f = dynamic_profile_factors(DynamicProfile("nh3", "henry_temperature"), grids, 8)
        assert np.argmax(f[:, 0, 0]) == 4 and np.argmin(f[:, 0, 0]) == 0
        # H(T) grows about 11% per K, so 285 K to 303 K is a factor of ~8;
        # bin-mean temperatures in practice span less than this.
        assert 5.0 < f[4, 0, 0] / f[0, 0, 0] < 12.0

    def test_henry_weight_formula(self):
        from orbit.emissions.loader import henry_temperature_weight
        T = 298.0
        np.testing.assert_allclose(henry_temperature_weight(T),
                                   161500.0 / T * np.exp(-10380.0 / T))

    def test_single_grid_degenerates_to_flat_with_warning(self):
        from orbit.emissions.loader import DynamicProfile, dynamic_profile_factors
        g = self._grids([290])[0]
        with pytest.warns(UserWarning, match="per-bin meteorology"):
            f = dynamic_profile_factors(DynamicProfile("nh3", "henry_temperature"), [g] * 8, 8)
        np.testing.assert_array_equal(f, 1.0)

    def test_yaml_kind_entry_parses(self, tmp_path):
        from orbit.emissions.loader import DiurnalConfig, DynamicProfile
        p = tmp_path / "d.yaml"
        p.write_text(
            "ist_offset_hours: 5.5\n"
            "profiles:\n"
            "  nh3_t:\n    kind: henry_temperature\n"
            "  flat:\n    hours: [" + ", ".join(["1.0"] * 24) + "]\n"
            "source_profiles:\n  ceds_nh3_anthro_2022_monthly.nc: nh3_t\n")
        cfg = DiurnalConfig.from_yaml(str(p))
        prof = cfg.profile_for("ceds_nh3_anthro_2022_monthly.nc", 1)
        assert isinstance(prof, DynamicProfile) and prof.kind == "henry_temperature"

    def test_shipped_config_maps_nh3_to_temperature_profile(self):
        import os
        from orbit.emissions.loader import DiurnalConfig, DynamicProfile
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cfg = DiurnalConfig.from_yaml(os.path.join(here, "orbit", "data", "diurnal_sas.yaml"))
        prof = cfg.profile_for("ceds_nh3_anthro_2022_monthly.nc", 7)
        assert isinstance(prof, DynamicProfile)
        # the tabulated profiles all have an exact unit mean
        for name, arr in cfg.profiles.items():
            assert abs(float(arr.mean()) - 1.0) < 1e-6, name


# ---------------------------------------------------------------------------
# Plume rise per bin
# ---------------------------------------------------------------------------

class TestPlumeRisePerBin:
    def test_elevated_source_follows_each_bins_meteorology(self, tmp_path):
        """A stack file loaded with 8 bin grids of different stability must
        land in different layers for the different bins; with one grid for
        every bin the layers are identical (the 1.0.2 behaviour)."""
        import xarray as xr
        from tests.test_emission_loading import _make_mock_grid_with_met, MockIndexer
        from orbit.emissions.sources import EmissionSource
        from orbit.emissions.loader import load_emissions_diurnal

        ny, nx, nz = 3, 3, 12
        lat = np.arange(20.0, 20.0 + ny * 1.0, 1.0)
        lon = np.arange(70.0, 70.0 + nx * 1.0, 1.0)
        ds = xr.Dataset({"pm25_anthro": (("time", "lat", "lon"),
                                         np.full((1, ny, nx), 1.0e-10, dtype=np.float32),
                                         {"units": "kg m-2 s-1"})},
                        coords={"time": [0], "lat": lat, "lon": lon})
        path = tmp_path / "stack_high.nc"
        ds.to_netcdf(path)
        src = EmissionSource(path=str(path), format="netcdf", elevated=True,
                             stack_height=220.0, stack_diam=6.0,
                             stack_temp=410.0, stack_vel=20.0)

        grids = []
        for tau in range(8):
            g = _make_mock_grid_with_met(nz=nz, ny=ny, nx=nx, Dz=np.full((nz, ny, nx), 100.0))
            # bins 0-3 stable and calm, bins 4-7 unstable and windy
            if tau < 4:
                g.Sclass[:] = 1.0
                g.S1[:] = 2.0e-4
                g.WindSpeed[:] = 2.0
            else:
                g.Sclass[:] = 0.0
                g.WindSpeed[:] = 8.0
            g.WindSpeedInverse = 1.0 / g.WindSpeed
            g.WindSpeedMinusThird = g.WindSpeed ** (-1.0 / 3.0)
            g.WindSpeedMinusOnePointFour = g.WindSpeed ** (-1.4)
            grids.append(g)
        idx = MockIndexer(nz, ny, nx)
        N = idx.N

        def top_layer(e_bin):
            from orbit.emissions.sources import N_ORBIT_SPECIES
            col = e_bin.reshape(N_ORBIT_SPECIES, nz, ny, nx)[1, :, 1, 1]
            return int(np.argmax(col))

        e_per_bin = load_emissions_diurnal([src], grids, idx, n_bins=8)
        layers = [top_layer(e_per_bin[t]) for t in range(8)]
        assert len(set(layers[:4])) == 1 and len(set(layers[4:])) == 1
        assert layers[0] != layers[4]

        e_one = load_emissions_diurnal([src], grids[0], idx, n_bins=8)
        assert len({top_layer(e_one[t]) for t in range(8)}) == 1

        # The emitted mass is the same either way (budget preserved).
        vol = np.tile(grids[0].volume.ravel(), 7)
        np.testing.assert_allclose(
            (e_per_bin.mean(axis=0) * np.tile(np.stack([g.volume.ravel() for g in grids]).mean(axis=0), 7)).sum(),
            (e_one.mean(axis=0) * vol).sum(), rtol=1e-9)


# ---------------------------------------------------------------------------
# Briggs buoyancy flux
# ---------------------------------------------------------------------------

def test_buoyancy_flux_uses_stack_temperature_in_the_denominator():
    from orbit.emissions.plumerise import calc_delta_h, g
    T, u, H, D, Ts, Vs = 285.0, 4.0, 220.0, 6.0, 410.0, 20.0
    col = dict(temperature=[T], wind_speed=[u], s_class=[0.0], s1=[0.0],
               wind_speed_minus_one_point_four=[u ** -1.4],
               wind_speed_minus_third=[u ** (-1 / 3)], wind_speed_inverse=[1 / u])
    dh = calc_delta_h(0, col["temperature"], col["wind_speed"], col["s_class"],
                      col["s1"], H, Ts, Vs, D, col["wind_speed_minus_one_point_four"],
                      col["wind_speed_minus_third"], col["wind_speed_inverse"])
    F = g * (Ts - T) / Ts * Vs * (D / 2) ** 2
    assert dh == pytest.approx(7.4 * (F * H ** 2) ** (1 / 3) / u, rel=1e-6)


def test_hno3_dry_dep_is_read_from_the_grid_file():
    """The restricted-read whitelist must include HNO3DryDep, or the 1.0.3
    grids' Wesely HNO3 velocity is silently replaced by the 3 cm/s fallback
    (this happened in the first two 1.0.3 forward years)."""
    from orbit.core.grid_data import _GRID_INPUT_VARS
    assert "HNO3DryDep" in _GRID_INPUT_VARS
