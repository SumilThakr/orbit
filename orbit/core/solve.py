"""Sparse LU factorization (UMFPACK/SuperLU) + symbolic-reuse and ordering helpers.

The legacy single-shot decoupled solve path (factorize/solve/solve_decoupled/
extract_pm25) was retired in the production-branch cleanup (2026-05-22); the
periodic-orbit solver (orbit.core.orbit / scripts/run_orbit.py) is the
only solver. This module now provides just the factorization backend
(UMFPACK symbolic reuse, METIS ordering, the LU wrapper classes).
"""

import time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# --- METIS nested-dissection ordering ---
# METIS produces far better fill-reducing orderings than COLAMD for 3D PDE
# operators with dense vertical couplings (ACM2 mixing).  On SAS (227K cells):
# COLAMD: 786M nnz(LU), 789s  |  METIS ND: 210M nnz(LU), 82s
try:
    import pymetis
    _HAS_METIS = True
except ImportError:
    _HAS_METIS = False

# --- UMFPACK support via ctypes ---
# UMFPACK's multifrontal method with symbolic/numeric split: compute the
# symbolic factorization (ordering + elimination tree) once, then reuse it
# for all 6 species since they share identical sparsity patterns.
# Benchmark (SAS 72K cells): symbolic 0.2s + numeric 2.1s*6 = 14s total,
# vs SuperLU 9.8s*6 = 59s (4.1x speedup).
import ctypes as _ctypes
import ctypes.util as _ctypes_util
import os as _os
import sys as _sys

# Tried in order: a cluster-provided SuiteSparse install (if present),
# then the active conda/virtualenv prefix (conda-forge suitesparse,
# pulled in by scikit-umfpack), then whatever the system linker can find.
_UMFPACK_LIB_CANDIDATES = [
    "/common/software/install/migrated/suitesparse/4.5.3_intel2016update3"
    "/lib/libumfpack.so",
    _os.path.join(_sys.prefix, "lib", "libumfpack.so"),
    _ctypes_util.find_library("umfpack"),
]
_UMFPACK_LIB_PATH = _UMFPACK_LIB_CANDIDATES[0]
_umfpack_lib = None
for _cand in _UMFPACK_LIB_CANDIDATES:
    if not _cand:
        continue
    try:
        _umfpack_lib = _ctypes.CDLL(_cand)
        _UMFPACK_LIB_PATH = _cand
        break
    except OSError:
        continue
_HAS_UMFPACK = _umfpack_lib is not None

_UMFPACK_A = 0   # solve A x = b
_UMFPACK_At = 1  # solve A^T x = b   (used by the adjoint mode)

# UMFPACK Control[] array layout (UMFPACK 5.x; SuiteSparse 4.5.3 bundles 5.7.x;
# slots verified unchanged in UMFPACK 6.3 / SuiteSparse 7.10).
# We only need the ORDERING and STRATEGY slots here; other slots use library
# defaults loaded via umfpack_di_defaults().
_UMFPACK_CONTROL_LEN = 20
_UMFPACK_ORDERING = 10
_UMFPACK_ORDERING_CODES = {
    "colamd": 0,
    "amd":    1,
    "user":   2,
    "metis":  3,
    "best":   4,
    "none":   5,
    "given":  6,
}
_UMFPACK_STRATEGY = 5
_UMFPACK_STRATEGY_SYMMETRIC = 3
_UMFPACK_INFO_LEN = 90
_UMFPACK_LNZ_ESTIMATE = 23
_UMFPACK_UNZ_ESTIMATE = 24
_UMFPACK_STRATEGY_USED = 18
_UMFPACK_ORDERING_USED = 19
_UMFPACK_SIZE_OF_UNIT = 3
_UMFPACK_NUMERIC_SIZE = 40
_UMFPACK_PEAK_MEMORY = 41


def umfpack_symbolic(L: sp.csc_matrix, verbose: bool = True, ordering=None,
                     qinit=None):
    """Compute UMFPACK symbolic factorization from a CSC matrix.

    The symbolic object captures the fill-reducing ordering and elimination
    tree.  It can be reused for numeric factorization of any matrix with
    the same sparsity pattern.

    Parameters
    ----------
    L : csc_matrix
        Any matrix with the target sparsity pattern.
    verbose : bool
    ordering : str or None
        UMFPACK fill-reducing ordering. None (default) uses the library
        default. Other values: "colamd", "amd", "metis" (only if
        libumfpack was linked with METIS at build time), "best", "none".
        Ignored when ``qinit`` is provided.
    qinit : ndarray of int32 or None
        User-supplied column permutation of length n_col. When set, calls
        ``umfpack_di_qsymbolic`` instead of ``umfpack_di_symbolic`` so
        UMFPACK skips its built-in ordering and uses ``qinit`` directly.
        This lets us hand UMFPACK a METIS-quality ordering computed via
        pymetis while keeping UMFPACK's compact frontal storage. The
        permutation must satisfy ``Qinit[i] = original column to use at
        position i`` (the same convention compute_metis_ordering returns).
        The symmetric strategy is forced alongside: per the UMFPACK docs,
        only the symmetric strategy preserves Qinit -- the unsymmetric
        strategy postorders the column etree and re-pivots per front,
        which destroys a nested-dissection ordering (measured 6x SLOWER
        than the built-in default on a 36^3 Laplacian, vs 1.8x faster
        with the strategy forced).

    Returns
    -------
    symbolic : dict
        Opaque handle with keys 'ptr', 'Ap', 'Ai', 'n' for use with
        umfpack_factorize_decoupled(), plus symbolic-analysis diagnostics
        'lunz_est' (estimated nnz(L)+nnz(U)) and 'strategy_used' (the
        UMFPACK strategy code: 1=unsymmetric, 3=symmetric).
    """
    if not _HAS_UMFPACK:
        raise RuntimeError("UMFPACK not available: " + _UMFPACK_LIB_PATH)

    L = L.tocsc()
    L.sum_duplicates()
    L.sort_indices()
    n = L.shape[0]
    Ap = L.indptr.astype(np.int32)
    Ai = L.indices.astype(np.int32)

    if ordering is None or qinit is not None:
        control_ptr = None
    else:
        key = ordering.lower()
        if key not in _UMFPACK_ORDERING_CODES:
            raise ValueError(
                f"Unknown UMFPACK ordering {ordering!r}; "
                f"valid: {sorted(_UMFPACK_ORDERING_CODES)}"
            )
        control = np.zeros(_UMFPACK_CONTROL_LEN, dtype=np.float64)
        # Populate with library defaults so we don't zero out other tunings
        # (pivot tolerance, alloc init, etc.) when overriding a single slot.
        _umfpack_lib.umfpack_di_defaults(
            control.ctypes.data_as(_ctypes.POINTER(_ctypes.c_double))
        )
        control[_UMFPACK_ORDERING] = float(_UMFPACK_ORDERING_CODES[key])
        control_ptr = control.ctypes.data_as(
            _ctypes.POINTER(_ctypes.c_double)
        )
        # Keep a reference so the underlying buffer isn't GC'd before
        # symbolic completes (ctypes pointer doesn't pin the np.ndarray).

    t0 = time.time()
    sym_ptr = _ctypes.c_void_p()
    info = np.zeros(_UMFPACK_INFO_LEN, dtype=np.float64)
    info_ptr = info.ctypes.data_as(_ctypes.POINTER(_ctypes.c_double))
    if qinit is not None:
        qinit_i32 = np.asarray(qinit, dtype=np.int32)
        if qinit_i32.shape != (n,):
            raise ValueError(
                f"qinit length {qinit_i32.shape[0]} != n_col {n}"
            )
        # Only the symmetric strategy preserves Qinit (see docstring).
        control = np.zeros(_UMFPACK_CONTROL_LEN, dtype=np.float64)
        _umfpack_lib.umfpack_di_defaults(
            control.ctypes.data_as(_ctypes.POINTER(_ctypes.c_double))
        )
        control[_UMFPACK_STRATEGY] = float(_UMFPACK_STRATEGY_SYMMETRIC)
        ret = _umfpack_lib.umfpack_di_qsymbolic(
            _ctypes.c_int(n), _ctypes.c_int(n),
            Ap.ctypes.data_as(_ctypes.POINTER(_ctypes.c_int)),
            Ai.ctypes.data_as(_ctypes.POINTER(_ctypes.c_int)),
            L.data.ctypes.data_as(_ctypes.POINTER(_ctypes.c_double)),
            qinit_i32.ctypes.data_as(_ctypes.POINTER(_ctypes.c_int)),
            _ctypes.byref(sym_ptr),
            control.ctypes.data_as(_ctypes.POINTER(_ctypes.c_double)),
            info_ptr,
        )
        tag_extra = " qinit=user-supplied strategy=symmetric"
    else:
        ret = _umfpack_lib.umfpack_di_symbolic(
            _ctypes.c_int(n), _ctypes.c_int(n),
            Ap.ctypes.data_as(_ctypes.POINTER(_ctypes.c_int)),
            Ai.ctypes.data_as(_ctypes.POINTER(_ctypes.c_int)),
            L.data.ctypes.data_as(_ctypes.POINTER(_ctypes.c_double)),
            _ctypes.byref(sym_ptr), control_ptr, info_ptr,
        )
        tag_extra = f" ordering={ordering}" if ordering is not None else ""
    if ret != 0:
        raise RuntimeError(f"umfpack_di_symbolic failed with ret={ret}")

    if verbose:
        print(f"  UMFPACK symbolic{tag_extra}: {time.time() - t0:.2f}s")

    return {"ptr": sym_ptr, "Ap": Ap, "Ai": Ai, "n": n,
            "lunz_est": float(info[_UMFPACK_LNZ_ESTIMATE]
                              + info[_UMFPACK_UNZ_ESTIMATE]),
            "strategy_used": int(info[_UMFPACK_STRATEGY_USED]),
            "ordering_used": int(info[_UMFPACK_ORDERING_USED])}


class _UmfpackLU:
    """UMFPACK numeric factorization with solve(), matching SuperLU interface."""

    def __init__(self, L: sp.csc_matrix, symbolic: dict):
        L = L.tocsc()
        L.sum_duplicates()
        L.sort_indices()
        n = symbolic["n"]
        Ap = symbolic["Ap"]
        Ai = symbolic["Ai"]

        self._numeric = _ctypes.c_void_p()
        info = np.zeros(_UMFPACK_INFO_LEN, dtype=np.float64)
        ret = _umfpack_lib.umfpack_di_numeric(
            Ap.ctypes.data_as(_ctypes.POINTER(_ctypes.c_int)),
            Ai.ctypes.data_as(_ctypes.POINTER(_ctypes.c_int)),
            L.data.ctypes.data_as(_ctypes.POINTER(_ctypes.c_double)),
            symbolic["ptr"],
            _ctypes.byref(self._numeric), None,
            info.ctypes.data_as(_ctypes.POINTER(_ctypes.c_double)),
        )
        if ret != 0:
            raise RuntimeError(f"umfpack_di_numeric failed with ret={ret}")
        # Exact memory accounting from UMFPACK itself (Info is in Units).
        _unit = info[_UMFPACK_SIZE_OF_UNIT] or 1.0
        self.numeric_bytes = float(info[_UMFPACK_NUMERIC_SIZE] * _unit)
        self.peak_bytes = float(info[_UMFPACK_PEAK_MEMORY] * _unit)

        # Keep references for solve: umfpack_di_solve reads Ap/Ai/Ax; the
        # csc wrapper object itself is never touched again, so pinning it
        # would only hold its (duplicate) index arrays alive.
        self._Ap = Ap
        self._Ai = Ai
        self._Ax = L.data  # keep reference so it's not GC'd
        self.shape = (n, n)

    def solve(self, b: np.ndarray, trans: str = "N") -> np.ndarray:
        """Back-substitute through the factorization.

        Parameters
        ----------
        b : ndarray
            Right-hand side, shape (n,).
        trans : {"N", "T"}
            "N" (default) solves A x = b; "T" solves A^T x = b. The transposed
            solve reuses the same numeric factor — UMFPACK ships
            ``sys=UMFPACK_At`` for this exact purpose, so no re-factor cost.
            Used by the marginal-deaths adjoint mode in orbit.modes.adjoint.
        """
        if trans == "N":
            sys_code = _UMFPACK_A
        elif trans == "T":
            sys_code = _UMFPACK_At
        else:
            raise ValueError(f"trans must be 'N' or 'T'; got {trans!r}")
        x = np.empty_like(b)
        ret = _umfpack_lib.umfpack_di_solve(
            _ctypes.c_int(sys_code),
            self._Ap.ctypes.data_as(_ctypes.POINTER(_ctypes.c_int)),
            self._Ai.ctypes.data_as(_ctypes.POINTER(_ctypes.c_int)),
            self._Ax.ctypes.data_as(_ctypes.POINTER(_ctypes.c_double)),
            x.ctypes.data_as(_ctypes.POINTER(_ctypes.c_double)),
            b.ctypes.data_as(_ctypes.POINTER(_ctypes.c_double)),
            self._numeric, None, None,
        )
        if ret != 0:
            raise RuntimeError(f"umfpack_di_solve failed with ret={ret}")
        return x

    def __del__(self):
        if self._numeric and _umfpack_lib is not None:
            _umfpack_lib.umfpack_di_free_numeric(
                _ctypes.byref(self._numeric),
            )

    @property
    def nnz(self):
        # UMFPACK doesn't expose nnz(LU) easily; return 0 as placeholder
        return 0


def umfpack_free_symbolic(symbolic: dict):
    """Free the UMFPACK symbolic factorization handle."""
    if symbolic and symbolic.get("ptr") and _umfpack_lib is not None:
        _umfpack_lib.umfpack_di_free_symbolic(
            _ctypes.byref(symbolic["ptr"]),
        )


def compute_metis_ordering(L: sp.csc_matrix, verbose: bool = True) -> np.ndarray:
    """Compute METIS nested-dissection permutation from sparsity pattern.

    Parameters
    ----------
    L : csc_matrix
        Any matrix with the target sparsity pattern (e.g., the transport block T).
    verbose : bool

    Returns
    -------
    perm : ndarray of int32, shape (N,)
        Permutation vector: new_row[i] = old_row[perm[i]].
    """
    if not _HAS_METIS:
        raise ImportError("pymetis is required for METIS ordering: pip install pymetis")

    t0 = time.time()
    N = L.shape[0]

    # Build symmetric adjacency (no self-loops)
    S = (L + L.T).tocsr()
    S.setdiag(0)
    S.eliminate_zeros()

    # Convert to pymetis adjacency list format
    adjacency = []
    for i in range(N):
        adjacency.append(S.indices[S.indptr[i]:S.indptr[i + 1]].tolist())

    perm, _ = pymetis.nested_dissection(adjacency)
    perm = np.array(perm, dtype=np.int32)

    if verbose:
        print(f"  METIS nested dissection: {time.time() - t0:.1f}s")

    return perm


class _PermutedLU:
    """SuperLU factorization with METIS reordering."""

    def __init__(self, L: sp.csc_matrix, perm: np.ndarray):
        self._perm = perm
        self._iperm = np.argsort(perm)
        L_p = L[perm][:, perm].tocsc()
        L_p.sum_duplicates()
        L_p.sort_indices()
        self._lu = spla.splu(L_p, permc_spec="NATURAL")
        self.shape = L.shape

    def solve(self, b: np.ndarray) -> np.ndarray:
        x_p = self._lu.solve(b[self._perm])
        x = np.empty_like(x_p)
        x[self._perm] = x_p
        return x

    @property
    def nnz(self):
        return self._lu.nnz


def _splu(L: sp.csc_matrix, perm: np.ndarray = None):
    """Factorize with optional METIS permutation, else default SuperLU."""
    if perm is not None:
        return _PermutedLU(L, perm)
    return spla.splu(L)


