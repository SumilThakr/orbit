"""Verify _UmfpackLU.solve(b, trans='T') gives A^T x = b.

Needed by orbit.modes.adjoint for the marginal-deaths pipeline:
the adjoint per-bin step is P_τ^T λ, and we want to reuse the
existing UMFPACK numeric factorization rather than re-factor on L^T.
"""
import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from orbit.core.solve import _UmfpackLU, umfpack_symbolic, _HAS_UMFPACK


@pytest.mark.skipif(not _HAS_UMFPACK, reason="UMFPACK not available")
def test_umfpack_transpose_solve_matches_spsolve():
    rng = np.random.default_rng(2024)
    n = 200
    # An asymmetric sparse matrix with diagonal dominance (so it's invertible).
    A = sp.random(n, n, density=0.05, format="csc", random_state=2024)
    A = A + 5.0 * sp.eye(n, format="csc")
    sym = umfpack_symbolic(A, verbose=False)
    lu = _UmfpackLU(A, sym)

    b = rng.standard_normal(n)
    x_fwd = lu.solve(b, trans="N")
    x_trans = lu.solve(b, trans="T")
    x_ref_fwd = spla.spsolve(A.tocsc(), b)
    x_ref_trans = spla.spsolve(A.T.tocsc(), b)

    np.testing.assert_allclose(x_fwd, x_ref_fwd, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(x_trans, x_ref_trans, rtol=1e-10, atol=1e-12)


@pytest.mark.skipif(not _HAS_UMFPACK, reason="UMFPACK not available")
def test_umfpack_solve_rejects_unknown_trans():
    A = sp.eye(5, format="csc")
    sym = umfpack_symbolic(A, verbose=False)
    lu = _UmfpackLU(A, sym)
    b = np.ones(5)
    with pytest.raises(ValueError, match="trans must be"):
        lu.solve(b, trans="H")


@pytest.mark.skipif(not _HAS_UMFPACK, reason="UMFPACK not available")
def test_umfpack_transpose_default_is_forward():
    A = sp.diags([2.0, 3.0, 5.0]).tocsc()
    sym = umfpack_symbolic(A, verbose=False)
    lu = _UmfpackLU(A, sym)
    b = np.array([2.0, 6.0, 10.0])
    np.testing.assert_allclose(lu.solve(b), [1.0, 2.0, 2.0])
