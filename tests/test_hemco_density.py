"""The HEMCO background conversion uses the model's own air density."""
import numpy as np

from orbit.hemco import _molmol_to_ugm3, density_on_levels


def test_density_is_taken_from_the_nearest_model_layer():
    orbit_sigma = np.array([0.99, 0.9, 0.7, 0.4])
    rho = np.stack([np.full((2, 3), v) for v in (1.15, 1.0, 0.8, 0.5)])
    lev = np.array([0.995, 0.95, 0.8, 0.6, 0.35])
    out = density_on_levels(lev, orbit_sigma, rho)
    assert out.shape == (5, 2, 3)
    np.testing.assert_array_equal(out[:, 0, 0], [1.15, 1.15, 1.0, 0.8, 0.5])


def test_conversion_scales_with_density_not_a_constant():
    X = np.array([1.0e-9])
    lo = _molmol_to_ugm3(X, "O3", rho_air=np.array([0.5]))
    hi = _molmol_to_ugm3(X, "O3", rho_air=np.array([1.0]))
    np.testing.assert_allclose(hi, 2.0 * lo)
    np.testing.assert_allclose(_molmol_to_ugm3(X, "O3"), _molmol_to_ugm3(X, "O3", rho_air=1.2))
