"""Verify the umfpack_metis path: qinit correctness and ordering quality.

Regression test for the 2026-08-01 finding: umfpack_di_qsymbolic with a
METIS nested-dissection Qinit but the default (AUTO) strategy resolved to
the unsymmetric strategy, which postorders the column etree and re-pivots
per front. That destroys the ND ordering: on the January operator the
per-species numeric factorization ran 8x SLOWER than UMFPACK's built-in
default (360s vs 46s). Only the symmetric strategy preserves Qinit, so
umfpack_symbolic forces it whenever qinit is supplied.

The quality assertion uses UMFPACK's own symbolic fill estimate
(Info[UMFPACK_LNZ_ESTIMATE] + Info[UMFPACK_UNZ_ESTIMATE]), which is
deterministic, rather than wall-clock timing.
"""
import numpy as np
import pytest
import scipy.sparse as sp

from orbit.core.solve import (
    _HAS_METIS, _HAS_UMFPACK, _UmfpackLU, compute_metis_ordering,
    umfpack_free_symbolic, umfpack_symbolic,
)


def _laplacian_3d(n):
    """7-point Laplacian on an n^3 grid, shifted to be nonsingular."""
    I = sp.eye(n, format="csr", dtype=np.float64)
    D = sp.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(n, n), format="csr")
    A = (sp.kron(sp.kron(D, I), I) + sp.kron(sp.kron(I, D), I)
         + sp.kron(sp.kron(I, I), D))
    return (A + 0.01 * sp.eye(n ** 3)).tocsc()


@pytest.mark.skipif(not (_HAS_UMFPACK and _HAS_METIS),
                    reason="needs UMFPACK and pymetis")
def test_qinit_metis_solves_correctly():
    A = _laplacian_3d(12)
    n = A.shape[0]
    perm = compute_metis_ordering(A, verbose=False)
    sym = umfpack_symbolic(A, verbose=False, qinit=perm)
    lu = _UmfpackLU(A, sym)
    b = np.ones(n)
    x = lu.solve(b)
    assert np.abs(A @ x - b).max() < 1e-10
    # Exact memory accounting from the numeric Info array (D1): the final
    # factor is at least as large as the input matrix, and factorization
    # working memory is at least the final factor.
    assert lu.numeric_bytes > A.nnz * 8
    assert lu.peak_bytes >= lu.numeric_bytes
    del lu
    umfpack_free_symbolic(sym)


@pytest.mark.skipif(not (_HAS_UMFPACK and _HAS_METIS),
                    reason="needs UMFPACK and pymetis")
def test_qinit_metis_preserved_and_fill_competitive():
    # Large enough that a destroyed ND ordering shows up unmistakably in
    # the fill estimate: with the AUTO-strategy bug, the METIS qinit gave
    # ~4x MORE estimated fill than the default on this operator.
    A = _laplacian_3d(20)
    perm = compute_metis_ordering(A, verbose=False)

    sym_default = umfpack_symbolic(A, verbose=False)
    sym_metis = umfpack_symbolic(A, verbose=False, qinit=perm)
    try:
        # The symmetric strategy must actually be in force (strategy code
        # 3 = symmetric), otherwise UMFPACK silently degrades the ordering.
        assert sym_metis["strategy_used"] == 3
        # ND must be at least competitive with the built-in ordering; the
        # broken configuration exceeds it several-fold.
        assert sym_metis["lunz_est"] <= 1.5 * sym_default["lunz_est"], (
            f"METIS qinit fill estimate {sym_metis['lunz_est']:.3e} vs "
            f"default {sym_default['lunz_est']:.3e}: ordering not preserved?"
        )
    finally:
        umfpack_free_symbolic(sym_default)
        umfpack_free_symbolic(sym_metis)
