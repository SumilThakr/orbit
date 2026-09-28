"""Diurnal periodic orbit solver.

Finds a periodic trajectory through 8 backward-Euler steps by solving
(I - M) c_0 = s via GMRES, where M = P_8 P_7 ... P_1 is the monodromy
operator.  Each P_tau = (I + L_tau * dtau)^{-1} is one bin's backward-Euler
propagator.  The monodromy matvec is 8 sequential backsolves; M is never
formed explicitly.
"""

import os
import queue
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from orbit.core.deposition import (
    N_SPECIES, IDX_SOA, IDX_PM25, IDX_POA, IDX_TOTAL_NH, IDX_PSO4,
    IDX_TOTAL_NO3, IDX_VBS_BINS,
)
from orbit.core.constants import N_TO_NH4, N_TO_NO3, S_TO_SO4

DTAU = 10800.0  # 3 hours in seconds
N_BINS = 8

# Opt 2 (see notes/2026-04-24_orbit_solver_optimizations.md): optional
# thread-parallel LU factorization across the 8 per-bin operators of a
# species.  UMFPACK's numeric factor releases the GIL, so N threads can
# saturate N CPUs.  Default 1 = sequential (byte-identical to pre-Opt-2);
# set ORBIT_FACTOR_THREADS=4 in the SLURM submit to opt in.
_N_FACTOR_THREADS = int(os.environ.get("ORBIT_FACTOR_THREADS", "1"))

# Opt 4 (species-level parallelism): run independent species concurrently.
# The DAG has 7 independent species (Org, PM2.5, NH, SO2, NOx, O3, CO) and
# 2 sinks (pSO4 ← SO2, TotalNO3 ← NOx).  Running wave-0 species in parallel
# collapses the critical path from sum-of-solves to max-solve + wave-1.
# Default 1 = sequential.  Set ORBIT_SPECIES_THREADS to enable; typical
# 4 for 4 CPUs, memory-bounded (~2 GB extra LU per concurrent species).
# When species-parallel is on, keep ORBIT_FACTOR_THREADS=1 or CPUs will be
# oversubscribed (species × factor can exceed physical cores).
_N_SPECIES_THREADS = int(os.environ.get("ORBIT_SPECIES_THREADS", "1"))

_SPECIES_NAMES = ["VBS_C100", "PM2.5", "NH", "SO2", "NOx", "pSO4", "TotalNO3",
                  "O3", "CO", "VBS_C10", "VBS_C1", "VBS_C01", "VBS_C1000",
                  "POA"]
assert len(_SPECIES_NAMES) == N_SPECIES, (
    f"_SPECIES_NAMES has {len(_SPECIES_NAMES)} entries, expected {N_SPECIES}"
)

# Solve order: solve sources before receivers for each coupling.
# SO2 (3) before pSO4 (5); NOx (4) before TotalNO3 (6). O3 (7) and
# CO (8) are diagonal-only (no transported precursors); their chemistry
# comes from the oxidants module — O3 via HO2+NO/RO2+NO production,
# CO via a simple k_OH_CO*[OH] loss added to the operator diagonal.
#
# 1D-VBS aging chain: C1000 (12) → C100 (0) → C10 (9) → C1 (10) →
# C01 (11). Place C1000 (the source root) first among VBS bins so the
# topological-sort places it in wave 0; the rest cascade naturally.
# The acyclic-DAG wave-builder iterates to fixed point and handles any
# ordering, but a sensible solve_order keeps log lines readable.
# Solve order: sources before sinks, and the VBS aging cascade descends
# 1000 -> 100 -> 10 -> 1 -> 0.1. POA (13) is inert and uncoupled, so its
# position is free; it sits next to PrimaryPM25 because their operators are
# identical and adjacency keeps the LU cache warm.
_SOLVE_ORDER = [0, 1, 13, 2, 3, 4, 5, 6, 7, 8, 12, 9, 10, 11]

# A species missing from _SOLVE_ORDER is never solved, and only surfaces
# much later as a KeyError while assembling the orbit dict. Fail loudly at
# import instead.
assert sorted(_SOLVE_ORDER) == list(range(N_SPECIES)), (
    f"_SOLVE_ORDER must cover every species exactly once: "
    f"missing {sorted(set(range(N_SPECIES)) - set(_SOLVE_ORDER))}, "
    f"unexpected {sorted(set(_SOLVE_ORDER) - set(range(N_SPECIES)))}"
)


def _factor_one_bin(L_s_tau, I_N, umfpack_sym, perm):
    """Build A_tau = I + L_s_tau * DTAU and factor it.

    Extracted out of ``solve_orbit_all_species`` so that either a
    sequential loop or a ThreadPoolExecutor can call it per bin.

    Safe to call concurrently on distinct ``L_s_tau`` matrices: the
    scipy CSC construction path is thread-local, and both UMFPACK's
    numeric factor and SuperLU's ``splu`` release the GIL.
    """
    from orbit.core.solve import _UmfpackLU, _splu
    A_tau = I_N + L_s_tau * DTAU
    A_tau = A_tau.tocsc()
    A_tau.sum_duplicates()
    A_tau.sort_indices()
    if umfpack_sym is not None:
        return _UmfpackLU(A_tau, umfpack_sym)
    return _splu(A_tau, perm=perm)


def apply_monodromy(v, lu_list_8):
    """Apply monodromy operator M = P_8 ... P_1 to vector v.

    8 sequential backsolves: pure homogeneous propagation (no emissions).
    """
    x = v.copy()
    for tau in range(N_BINS):
        x = lu_list_8[tau].solve(x)
    return x


def compute_s(lu_list_8, e_tau_list, dtau=DTAU):
    """Compute orbit forcing: one forward sweep from zero with emissions.

    Equivalent to running one representative day from zero concentration.
    Returns s such that (I - M) c_0 = s for the periodic orbit.
    """
    N = lu_list_8[0].shape[0]
    x = np.zeros(N, dtype=np.float64)
    for tau in range(N_BINS):
        x = lu_list_8[tau].solve(x + dtau * e_tau_list[tau])
    return x


def solve_orbit_one_species(lu_list_8, e_tau_list, dtau=DTAU,
                             c_warm=None, tol=1e-6, maxiter=200):
    """GMRES solve for one species' periodic orbit.

    Solves (I - M) c_0 = s where M is the monodromy operator.

    Parameters
    ----------
    lu_list_8 : list of 8 LU factorizations (each with .solve(b))
    e_tau_list : list of 8 emission rate vectors (one per bin)
    dtau : float
        Time step per bin (seconds).
    c_warm : ndarray or None
        Warm-start for c_0.  If None, uses s (one-cycle-from-zero).
    tol : float
        Relative tolerance (atol = tol * ||s||).
    maxiter : int
        Maximum GMRES iterations.

    Returns
    -------
    orbit : list of 9 ndarray
        [c_0, c_1, ..., c_8] where c_8 ~ c_0.
    info : dict
        gmres_iters, residual, periodicity, neg_frac.
    """
    N = lu_list_8[0].shape[0]

    # Compute s = one cycle from zero
    s = compute_s(lu_list_8, e_tau_list, dtau)
    s_norm = np.linalg.norm(s)

    # Trivial case: no emissions -> zero orbit
    if s_norm < 1e-30:
        zero = np.zeros(N, dtype=np.float64)
        orbit = [zero.copy() for _ in range(N_BINS + 1)]
        return orbit, {"gmres_iters": 0, "residual": 0.0,
                       "periodicity": 0.0, "neg_frac": 0.0}

    # (I - M) linear operator
    def matvec(v):
        return v - apply_monodromy(v, lu_list_8)

    op = spla.LinearOperator((N, N), matvec=matvec, dtype=np.float64)

    # Warm start
    x0 = c_warm if c_warm is not None else s.copy()

    # Iteration counter
    iter_count = [0]

    def _counter(_):
        iter_count[0] += 1

    c_0, info_code = spla.gmres(
        op, s, x0=x0,
        rtol=1e-30, atol=tol * s_norm,
        restart=30, maxiter=maxiter,
        callback=_counter, callback_type='pr_norm',
    )

    if info_code != 0:
        print(f"    WARNING: GMRES did not converge (info={info_code}, "
              f"iters={iter_count[0]})")

    # Recover full orbit: forward propagation from c_0
    orbit = [c_0.copy()]
    x = c_0.copy()
    for tau in range(N_BINS):
        x = lu_list_8[tau].solve(x + dtau * e_tau_list[tau])
        orbit.append(x.copy())

    # Diagnostics
    residual_vec = op.matvec(c_0) - s
    residual = np.linalg.norm(residual_vec) / s_norm

    c_0_norm = np.linalg.norm(c_0)
    periodicity = np.linalg.norm(orbit[-1] - c_0) / (c_0_norm + 1e-30)

    all_vals = np.concatenate(orbit)
    neg_frac = float(np.mean(all_vals < 0))

    return orbit, {
        "gmres_iters": iter_count[0],
        "residual": float(residual),
        "periodicity": float(periodicity),
        "neg_frac": neg_frac,
    }


def _solve_one_species(s, L_species_per_bin, K_sources_per_bin,
                        emissions_SN, N, umfpack_sym, perm, I_N,
                        lu_cache_entry, sources_for_target, orbits,
                        c_warm_SN, tol, maxiter, extra_rhs_for_s,
                        skip_solve, cached_orbit_s, verbose,
                        endbin_keys=None):
    """Solve orbit for one species.

    Pure function (modulo read of `orbits` for source species — callers
    must ensure source species are already solved, e.g. by wave ordering).

    Returns (s, orbit_list, info_dict, lu_list_8 or None, log_lines).
      - orbit_list : list of N_BINS+1 (N,) arrays
      - info_dict  : {gmres_iters, residual, periodicity, factor_skipped,
                      t_factor, t_solve, neg_count, total_count}
      - lu_list_8  : the factored LUs (may be None if skip_solve)
      - log_lines  : list of strings (buffered so parallel calls don't
                     interleave stdout)
    """
    from orbit.core.solve import _UmfpackLU, _splu  # noqa: F401

    log_lines = []
    if verbose:
        log_lines.append(f"  [{_SPECIES_NAMES[s]}]")

    # 0. Copy-verbatim short-circuit for unchanged species.
    if skip_solve and cached_orbit_s is not None:
        if verbose:
            log_lines.append("    [cache] skipped factor + solve (operator+RHS unchanged)")
        info = {
            "gmres_iters": 0, "residual": 0.0, "periodicity": 0.0,
            "factor_skipped": True, "t_factor": 0.0, "t_solve": 0.0,
            "neg_count": 0, "total_count": 0,
        }
        return s, [c.copy() for c in cached_orbit_s], info, lu_cache_entry, log_lines

    # 1. Build per-bin emission vectors (with source couplings).
    # emissions_SN may be 1D (N_SPECIES*N,) flat-across-bins (legacy) or
    # 2D (N_BINS, N_SPECIES*N) for diurnal-resolved input.  Resolve once
    # so the inner loop just indexes by tau.
    if emissions_SN.ndim == 1:
        e_s_per_bin = [emissions_SN[s * N:(s + 1) * N]] * N_BINS
    else:
        e_s_per_bin = [emissions_SN[tau, s * N:(s + 1) * N]
                       for tau in range(N_BINS)]
    sources = sources_for_target.get(s, [])
    if sources:
        e_tau_list = []
        for tau in range(N_BINS):
            src_term = np.zeros(N, dtype=np.float64)
            for src in sources:
                K = K_sources_per_bin[tau][(s, src)]
                src_orbit = orbits.get(src)
                if src_orbit is None:
                    # Source not present in the snapshot. Either the source
                    # is in a later wave (chemistry DAG bug) or we're in a
                    # cyclic-DAG configuration without a seed. The marginal
                    # Picard driver always passes a seed; raise loudly if
                    # we hit this in a non-iterated path.
                    raise KeyError(
                        f"orbits[{src}] missing for target s={s} at tau={tau}. "
                        f"Seed-orbits machinery (Picard iteration) must be "
                        f"used for cyclic source DAGs (e.g. iso-coupled "
                        f"marginal NH↔NO3)."
                    )
                # Time-level convention. Default: start-of-bin τ — this
                # is right for chemistry-DAG K-blocks (NOx → NO3, SO2 →
                # pSO4) where K · c_source represents a mass-production
                # rate integrated over the bin (explicit treatment,
                # consistent between forward and marg).
                #
                # For iso K-blocks (linearisation of an implicit
                # operator coefficient), the c_baseline factor inside K
                # is built at end-of-bin τ (orbits[idx][tau+1] in
                # dcomp_isorropia._concentration_for_bin) and the
                # corresponding linearisation of zo's per-bin equation
                # has δc_other at end-of-bin too — so the K · δc_other
                # forcing must read src_orbit[tau+1] to keep the time-
                # levels matched within the linearised term.
                #
                # `endbin_keys` (a set of (target, source) tuples) tags
                # which K-blocks should switch convention. Marg passes
                # the iso keys here. Default None preserves chem-DAG
                # behaviour bit-identical.
                if endbin_keys is not None and (s, src) in endbin_keys:
                    src_term -= K @ src_orbit[tau + 1]
                else:
                    src_term -= K @ src_orbit[tau]
            e_tau_list.append(e_s_per_bin[tau] + src_term)
    else:
        e_tau_list = [e.copy() for e in e_s_per_bin]

    if extra_rhs_for_s is not None:
        for tau in range(N_BINS):
            e_tau_list[tau] = e_tau_list[tau] + extra_rhs_for_s[tau]

    # 1.5 Zero-source short-circuit BEFORE factorization (memory-audit
    # fix 1, 2026-08-01). The orbit of the linear periodic system with an
    # identically-zero RHS is zero regardless of the operator, and the
    # solver's own s_norm check would return exactly this — but only
    # after 8 bins were factored (~35 s and a ~1.5-2 GB transient per
    # species). O3 and CO hit this in every solve of this release.
    if all(not np.any(e) for e in e_tau_list):
        zero = np.zeros(N, dtype=np.float64)
        orbit = [zero.copy() for _ in range(N_BINS + 1)]
        if verbose:
            log_lines.append(
                "    [zero-source] skipped factor + solve (RHS identically zero)")
        info_out = {
            "gmres_iters": 0, "residual": 0.0, "periodicity": 0.0,
            "factor_skipped": True, "t_factor": 0.0, "t_solve": 0.0,
            "neg_count": 0, "total_count": (N_BINS + 1) * N,
            "lu_bytes": 0.0, "zero_source_skip": True,
        }
        return s, orbit, info_out, lu_cache_entry, log_lines

    # 2. Factor (or reuse cache).
    t0 = time.time()
    if lu_cache_entry is not None:
        lu_list_8 = lu_cache_entry
        factor_skipped = True
    else:
        if _N_FACTOR_THREADS > 1:
            with ThreadPoolExecutor(max_workers=_N_FACTOR_THREADS) as pool:
                futures = [pool.submit(_factor_one_bin,
                                        L_species_per_bin[tau][s],
                                        I_N, umfpack_sym, perm)
                           for tau in range(N_BINS)]
                lu_list_8 = [f.result() for f in futures]
        else:
            lu_list_8 = [
                _factor_one_bin(L_species_per_bin[tau][s],
                                 I_N, umfpack_sym, perm)
                for tau in range(N_BINS)
            ]
        factor_skipped = False
    t_factor = time.time() - t0

    # 3. GMRES orbit solve.
    c_warm_s = None
    if c_warm_SN is not None:
        c_warm_s = c_warm_SN[s * N:(s + 1) * N].copy()
    t1 = time.time()
    orbit, info = solve_orbit_one_species(
        lu_list_8, e_tau_list, DTAU,
        c_warm=c_warm_s, tol=tol, maxiter=maxiter,
    )
    t_solve = time.time() - t1

    neg_count = sum(np.sum(c < 0) for c in orbit)
    total_count = sum(c.size for c in orbit)

    # Exact per-species LU footprint (UMFPACK reports its own sizes;
    # SuperLU wrappers lack the attribute and report as 0 = unknown).
    lu_bytes = sum(getattr(lu, "numeric_bytes", 0.0) for lu in lu_list_8)
    # UMFPACK's own peak workspace during the numeric phase, the largest of
    # the 8 bins: the transient that the allocator's high-water mark keeps.
    lu_peak_bytes = max((getattr(lu, "peak_bytes", 0.0) for lu in lu_list_8), default=0.0)

    if verbose:
        factor_note = " (cached)" if factor_skipped else ""
        lu_note = (f", lu={lu_bytes / 1e6:.0f}MB, factor peak={lu_peak_bytes / 1e6:.0f}MB"
                   if lu_bytes else "")
        log_lines.append(
            f"    Factor: {t_factor:.1f}s{factor_note}{lu_note}, "
            f"GMRES: {info['gmres_iters']} iter "
            f"({t_solve:.2f}s), resid={info['residual']:.1e}, "
            f"period={info['periodicity']:.1e}, "
            f"neg={neg_count}/{total_count} "
            f"({neg_count/total_count*100:.1f}%)"
        )

    info_out = {
        "gmres_iters": info["gmres_iters"],
        "residual": info["residual"],
        "periodicity": info["periodicity"],
        "factor_skipped": factor_skipped,
        "t_factor": t_factor, "t_solve": t_solve,
        "neg_count": neg_count, "total_count": total_count,
        "lu_bytes": lu_bytes,
    }
    return s, orbit, info_out, lu_list_8, log_lines


def _build_solve_waves(sources_for_target, solve_order, skip_species):
    """Split species into dependency waves.

    Species in `skip_species` that have no prerequisite sources
    (i.e. their copy-verbatim doesn't need a solved source) go into
    wave 0.  Species in skip_species *with* sources go into the
    earliest wave where those sources exist — but since skip means
    the cached orbit is already available, we don't need source orbits
    to assemble RHS.  So all skip species can live in wave 0.

    For solved species, placement is determined by the DAG:
      wave 0: no source dependencies on other solved species
      wave k: all dependencies are in waves < k

    The DAG may contain cycles (iso coupling NH↔NO3 — each is a source
    of the other). Cyclic edges are removed before the topological
    placement; the marginal driver's Picard outer iteration provides
    seed orbits for the cyclic species so the within-wave snapshot
    read of ``orbits[cyclic_src]`` finds the seed rather than a
    KeyError.
    """
    # Detect mutual cycles in the DAG (a→b and b→a). With our current
    # iso-coupling DAG only NH↔NO3 form a cycle, but generalise so
    # future couplings don't silently regress wave construction.
    cyclic_edges = set()
    for tgt, srcs in sources_for_target.items():
        for src in srcs:
            if tgt in sources_for_target.get(src, []):
                cyclic_edges.add((tgt, src))
    # Effective DAG (acyclic) — used only for wave placement.
    effective_srcs = {
        tgt: [s for s in srcs if (tgt, s) not in cyclic_edges]
        for tgt, srcs in sources_for_target.items()
    }
    # Iterate the placement to fixpoint. A single pass over solve_order
    # under-counts max_src_wave when a source has higher solve_order
    # index than its target (e.g. NH at idx 2 with iso source pSO4 at
    # idx 5): the source's wave isn't known yet and defaults to 0,
    # under-placing the target. Iterating until placements stop
    # changing converges in O(diameter) passes — typically 2–3.
    placed = {}
    for _ in range(len(solve_order) + 2):
        changed = False
        for s in solve_order:
            if s in skip_species:
                if placed.get(s) != 0:
                    placed[s] = 0
                    changed = True
                continue
            srcs = effective_srcs.get(s, [])
            if not srcs:
                if placed.get(s) != 0:
                    placed[s] = 0
                    changed = True
                continue
            max_src_wave = max(placed.get(src, 0) for src in srcs)
            new_wave = max_src_wave + 1
            if placed.get(s) != new_wave:
                placed[s] = new_wave
                changed = True
        if not changed:
            break
    # Group by wave, preserving solve_order within each wave for stable
    # logging.
    max_wave = max(placed.values()) if placed else 0
    waves = [[] for _ in range(max_wave + 1)]
    for s in solve_order:
        waves[placed[s]].append(s)
    return waves


def solve_orbit_all_species(L_species_per_bin, K_sources_per_bin,
                             emissions_SN, N, umfpack_sym=None,
                             perm=None, tol=1e-6, maxiter=200,
                             verbose=True, c_warm_SN=None,
                             lu_cache=None, skip_solve_species=None,
                             cached_orbits=None, return_lu_cache=False,
                             keep_lu_species=None,
                             extra_rhs_per_bin_per_species=None,
                             seed_orbits=None,
                             endbin_keys=None):
    """Solve periodic orbits for all N_SPECIES species.

    Species are solved in _SOLVE_ORDER. Couplings are handled generically:
    for each target species s, the effective per-bin emission is
        e_tau = e_s - sum over sources src:  K[(s,src),tau] @ c_src[tau]
    where K[(target,source),tau] is an off-diagonal block passed via
    K_sources_per_bin.

    Parameters
    ----------
    L_species_per_bin : list of N_BINS lists, each of N_SPECIES csc_matrix (N, N)
    K_sources_per_bin : list of N_BINS dicts, each dict[(target, source)] -> csc_matrix (N, N)
        Off-diagonal blocks keyed by (target_species, source_species). Each
        target may be the receiving end of multiple sources (summed).
    emissions_SN : ndarray
        Monthly emission rates.  Either 1D shape (N_SPECIES*N,) with the
        same vector applied to every bin (legacy bin-flat path), or 2D
        shape (N_BINS, N_SPECIES*N) for diurnal-resolved input where each
        bin has its own emission vector.
    N : int
    umfpack_sym, perm, tol, maxiter, verbose : standard arguments
    c_warm_SN : ndarray, shape (N_SPECIES*N,), or None
        Warm start concentrations.
    lu_cache : dict or None
        Optional LU cache {species_idx: [lu_for_bin0, ..., lu_for_bin7]}.
        For species present in the cache the factorization is skipped and
        the cached LU objects are reused (Phase 3d: unchanged-species
        short-circuit across outer iterations).  LU objects are trusted to
        match the matrix in ``L_species_per_bin``; caller is responsible
        for invalidating stale entries.
    skip_solve_species : set or None
        Species indices whose orbits should be copied verbatim from
        ``cached_orbits`` rather than re-solved.  Only safe for species
        whose operator AND RHS are both unchanged from the prior iter.
    cached_orbits : dict or None
        {species_idx: [c_0, ..., c_8]} used when ``skip_solve_species`` is
        provided.  Entries must exist for every species in that set.
    return_lu_cache : bool
        If True, return the fresh LU cache under key ``"lu_cache"``.
    keep_lu_species : set or None
        Only retain LU factorisations for species in this set.  Others
        are explicitly freed after their own orbit solve to cap peak
        memory.  Default None → retain all (legacy behaviour).  Phase
        3d should pass the "unchanged across iters" set here so we keep
        only the cache entries that will actually be reused.
    extra_rhs_per_bin_per_species : dict {species_idx: [vec_tau0, ..., vec_tau7]} or None
        Per-species, per-bin RHS additions for species with bin-varying
        sources that cannot travel through the uniform ``emissions_SN``
        channel.  Phase 3e: O3 chemistry production (HO2+NO, RO2+NO) +
        optional top-of-model Newtonian BC.  Each vector has shape (N,)
        in ug compound/m³/s.  Species not in the dict get no extra RHS.
    seed_orbits : dict {species_idx: list of N_BINS+1 ndarray} or None
        Pre-populates the ``orbits`` dict before wave processing starts.
        Within a wave, ``_solve_one_species`` reads from a *snapshot* of
        ``orbits`` taken at wave start, so cyclic-DAG species in the same
        wave (e.g. NH↔NO3 via iso coupling) see the seed for their
        cross-coupled source rather than tripping ``KeyError``.  This is
        the foundation for Picard outer iteration in the marginal driver:
        seed = zeros for the first iter, then seed = previous-iter orbits
        for subsequent iters until ``max |Δδc|`` falls below tolerance.
        Skipped species (in ``skip_solve_species``) get their cached value
        applied at wave 0 via the copy-verbatim short-circuit; the seed
        is overwritten and irrelevant for them.

    Returns
    -------
    dict with keys:
        orbits : dict {species_idx: list of N_BINS+1 ndarray}
        gmres_iters : ndarray (N_SPECIES,)
        gmres_resid : ndarray (N_SPECIES,)
        periodicity : ndarray (N_SPECIES,)
        lu_cache : dict (if return_lu_cache=True)
    """

    I_N = sp.eye(N, format="csc")

    orbits = {}
    gmres_iters = np.zeros(N_SPECIES, dtype=int)
    gmres_resid = np.zeros(N_SPECIES, dtype=np.float64)
    periodicity_arr = np.zeros(N_SPECIES, dtype=np.float64)

    # Pre-populate orbits from seed_orbits (Picard iteration support).
    # Copies so the caller's seed isn't mutated by _merge_result.
    if seed_orbits is not None:
        for s_seed, seed_orbit_s in seed_orbits.items():
            orbits[s_seed] = [c.copy() for c in seed_orbit_s]

    if skip_solve_species is None:
        skip_solve_species = set()

    # Precompute per-target source lists for O(1) lookup during solve.
    # sources_for_target[target] = list of source species that feed target.
    sources_for_target = {}
    for (target, source) in K_sources_per_bin[0].keys():
        sources_for_target.setdefault(target, []).append(source)

    fresh_lu_cache: dict = {}

    # Build DAG waves.  Sequential mode (_N_SPECIES_THREADS<=1) still goes
    # through the same wave structure but with max_workers=1, so each
    # species's output is emitted immediately after its own solve — the
    # log order is identical to the pre-refactor sequential version as
    # long as within-wave solve time doesn't reorder completions.  With
    # threads>1 the per-species log block is still emitted atomically
    # (joined log_lines per species) in solve-completion order.
    waves = _build_solve_waves(sources_for_target, _SOLVE_ORDER,
                               skip_solve_species)
    # Clamp concurrency to avoid oversubscribing: with Opt 2 factor-threads
    # > 1, running N species * M factor_threads can exceed physical cores.
    # Leave that to the operator — just warn if both are high.
    n_species_threads = max(1, _N_SPECIES_THREADS)

    def _merge_result(s, orbit, info, lu_list, log_lines):
        """Apply one species' results to shared state."""
        orbits[s] = orbit
        gmres_iters[s] = info["gmres_iters"]
        gmres_resid[s] = info["residual"]
        periodicity_arr[s] = info["periodicity"]
        if lu_list is not None:
            fresh_lu_cache[s] = lu_list
        # Keep-LU filter: drop caches for species the caller doesn't plan
        # to reuse, so peak memory doesn't hold 9 species × 8 bins of LUs.
        if keep_lu_species is not None and s not in keep_lu_species:
            fresh_lu_cache.pop(s, None)
        # Free L blocks — we never touch L[tau][s] again in this call.
        # Exception: a zero-source-skipped species with no cached LU never
        # factored, and a later call over the same assembly (Picard outer
        # loop) may need its operator once its coupled source is nonzero.
        if not (info.get("zero_source_skip") and s not in fresh_lu_cache):
            for tau in range(N_BINS):
                L_species_per_bin[tau][s] = None
        if verbose and log_lines:
            print("\n".join(log_lines), flush=True)

    # Execution model: dataflow (ORBIT_DATAFLOW=1) or wave-barrier (default).
    # Dataflow submits each species as soon as its acyclic dependencies have
    # completed, rather than waiting for the entire wave; downstream cores
    # come online earlier, especially for the VBS cascade and pSO4/TotalNO3
    # which each have a single wave-0 dependency.
    _use_dataflow = os.environ.get("ORBIT_DATAFLOW", "0") == "1"

    if _use_dataflow:
        import concurrent.futures as _cf
        # Build the acyclic effective DAG (cycles broken — those use
        # seed_orbits to break the read dependency).
        _cyclic_edges = set()
        for tgt, srcs in sources_for_target.items():
            for src in srcs:
                if tgt in sources_for_target.get(src, []):
                    _cyclic_edges.add((tgt, src))
        effective_srcs = {
            tgt: [s for s in srcs if (tgt, s) not in _cyclic_edges]
            for tgt, srcs in sources_for_target.items()
        }

        def _build_task_args(s, orbits_snap, skip_s, cached_s):
            lu_entry = (lu_cache[s] if (lu_cache is not None and s in lu_cache)
                        else None)
            extra_s = (extra_rhs_per_bin_per_species.get(s)
                       if extra_rhs_per_bin_per_species is not None else None)
            return dict(
                s=s, L_species_per_bin=L_species_per_bin,
                K_sources_per_bin=K_sources_per_bin,
                emissions_SN=emissions_SN, N=N,
                umfpack_sym=umfpack_sym, perm=perm, I_N=I_N,
                lu_cache_entry=lu_entry,
                sources_for_target=sources_for_target,
                orbits=orbits_snap, c_warm_SN=c_warm_SN,
                tol=tol, maxiter=maxiter,
                extra_rhs_for_s=extra_s,
                skip_solve=skip_s, cached_orbit_s=cached_s,
                verbose=verbose,
                endbin_keys=endbin_keys,
            )

        pending = {s for s in _SOLVE_ORDER}
        completed = set()

        # Pre-merge skip-solve species (no actual work, just copy cached
        # orbits). Doing this up front lets downstream consumers see them
        # as "completed" immediately without scheduling a no-op task.
        for s in _SOLVE_ORDER:
            if (s in skip_solve_species and cached_orbits is not None
                    and s in cached_orbits):
                args = _build_task_args(s, {}, True, cached_orbits[s])
                _merge_result(*_solve_one_species(**args))
                completed.add(s)
                pending.discard(s)

        # Initial ready set: pending species with all effective deps done.
        # Use list to preserve _SOLVE_ORDER bias for stable log/scheduling.
        ready = [s for s in _SOLVE_ORDER
                 if s in pending
                 and all(src in completed
                         for src in effective_srcs.get(s, []))]

        with ThreadPoolExecutor(max_workers=n_species_threads) as pool:
            in_flight = {}   # future -> species
            while ready or in_flight:
                # Fill threadpool to capacity from the ready set.
                while ready and len(in_flight) < n_species_threads:
                    s = ready.pop(0)
                    pending.discard(s)
                    orbits_snap = dict(orbits)  # capture deps at submit time
                    fut = pool.submit(_solve_one_species,
                                       **_build_task_args(s, orbits_snap,
                                                           False, None))
                    in_flight[fut] = s
                # Wait for the next completion.
                if in_flight:
                    done, _ = _cf.wait(list(in_flight),
                                        return_when=_cf.FIRST_COMPLETED)
                    for fut in done:
                        s_done = in_flight.pop(fut)
                        result = fut.result()
                        _merge_result(*result)
                        completed.add(s_done)
                        del result
                        # Promote any pending species whose deps are now all
                        # in completed.
                        for tgt in _SOLVE_ORDER:
                            if tgt not in pending or tgt in ready:
                                continue
                            srcs = effective_srcs.get(tgt, [])
                            if all(src in completed for src in srcs):
                                ready.append(tgt)
                    # Drop the finished Futures now: each keeps its result
                    # (8 factors) alive until it is collected, and `done` is
                    # otherwise rebound only after the next wait.
                    del done, fut
    else:
        # Legacy wave-barrier execution (preserved for A/B testing).
        for wave_idx, wave in enumerate(waves):
            if not wave:
                continue

            # Snapshot of orbits at wave start. Within-wave reads
            # (`_solve_one_species` looking up `orbits[src]` for cross-couplings)
            # use this snapshot, not the live orbits dict that `_merge_result`
            # writes into. This makes within-wave concurrency Picard-clean for
            # cyclic DAGs: NH and NO3 in the same wave both see the seed for
            # the *other* species rather than racing on whichever finishes
            # first. Across waves, the snapshot includes prior-wave updates,
            # which is fine because the chemistry DAG was acyclic by
            # construction.
            orbits_snapshot = dict(orbits)

            # Per-species kwargs prepared up front (cheap) so threads only do
            # numeric work.
            task_args = []
            for s in wave:
                skip_s = (s in skip_solve_species
                          and cached_orbits is not None and s in cached_orbits)
                cached_s = cached_orbits[s] if skip_s else None
                lu_entry = (lu_cache[s] if (lu_cache is not None and s in lu_cache)
                            else None)
                extra_s = (extra_rhs_per_bin_per_species.get(s)
                           if extra_rhs_per_bin_per_species is not None else None)
                task_args.append(dict(
                    s=s, L_species_per_bin=L_species_per_bin,
                    K_sources_per_bin=K_sources_per_bin,
                    emissions_SN=emissions_SN, N=N,
                    umfpack_sym=umfpack_sym, perm=perm, I_N=I_N,
                    lu_cache_entry=lu_entry,
                    sources_for_target=sources_for_target,
                    orbits=orbits_snapshot, c_warm_SN=c_warm_SN,
                    tol=tol, maxiter=maxiter,
                    extra_rhs_for_s=extra_s,
                    skip_solve=skip_s, cached_orbit_s=cached_s,
                    verbose=verbose,
                    endbin_keys=endbin_keys,
                ))

            if n_species_threads <= 1 or len(wave) <= 1:
                # Sequential within wave (or single-species wave).
                for args in task_args:
                    result = _solve_one_species(**args)
                    _merge_result(*result)
                    del result
            else:
                # Parallel within wave. Merge each species AS IT COMPLETES
                # so its factors and orbit are freed immediately
                # (memory-audit fix 4, 2026-08-01): with submit-order
                # merging, an early finisher parked its ~1.5-2 GB result
                # inside the Future until the main thread reached it. Log
                # blocks are still printed in submit order so run logs
                # stay comparable across runs.
                #
                # Results travel through a queue rather than the Futures.
                # A Future keeps its result until it is garbage-collected,
                # and every Future of the wave lives until the pool exits,
                # so returning the factors through them pinned ~1.6 GB per
                # finished species until the wave's last species was done:
                # the peak scaled with the wave size, not the thread count
                # (January 2026-09-28 trace: 10.6 GB with 2 threads, where
                # two species in flight need ~5 GB).
                done_q = queue.Queue()

                def _run_and_hand_off(idx, args):
                    done_q.put((idx, _solve_one_species(**args)))

                with ThreadPoolExecutor(
                    max_workers=min(n_species_threads, len(wave))
                ) as pool:
                    for i, args in enumerate(task_args):
                        pool.submit(_run_and_hand_off, i, args)
                    pending_logs = {}
                    next_print = 0
                    for _ in range(len(task_args)):
                        idx, res = done_q.get()
                        s_res, orbit_res, info_res, lu_res, logs = res
                        _merge_result(s_res, orbit_res, info_res, lu_res, None)
                        del res, s_res, orbit_res, info_res, lu_res
                        pending_logs[idx] = logs
                        while next_print in pending_logs:
                            logs_p = pending_logs.pop(next_print)
                            if verbose and logs_p:
                                print("\n".join(logs_p), flush=True)
                            next_print += 1

    result = {
        "orbits": orbits,
        "gmres_iters": gmres_iters,
        "gmres_resid": gmres_resid,
        "periodicity": periodicity_arr,
    }
    if return_lu_cache:
        result["lu_cache"] = fresh_lu_cache
    return result


# =============================================================================
# ISORROPIA PM2.5 extraction
# =============================================================================

# Crustal Ca fraction of fine dust (Fountoukis & Nenes 2007)
_DUST_TO_CA = 0.036
# Na fraction of fine sea salt (Seinfeld & Pandis Table 9.7)
_SEASALT_TO_NA = 0.306


def extract_pm25_isorropia(c_6N, grid, indexer, lut):
    """Extract PM2.5 using ISORROPIA LUT for inorganic partitioning.

    The organic part is the same as in the reported ``pm25_mean``: primary
    organic aerosol plus the F_p-weighted sum over the 5 VBS bins when the
    grid carries ``F_p_vbs`` (the C100 bin alone otherwise). Inorganic
    f_nh4/f_no3 come from the 7D ISORROPIA LUT re-queried at the orbit
    concentrations and bin-averaged meteorology, so the field differs from
    ``pm25_mean`` only through the inorganic partitioning. Until 2026-09-26
    the organic part was the C100 bin alone with no POA, which made the
    field a few µg/m³ low in organic-rich cells.

    Parameters
    ----------
    c_6N : ndarray, shape (N_SPECIES*N,)
        Coupled concentration vector (element mass, ug/m3), species-major
        in the IDX_* order of deposition.py.
    grid : GridData
        Bin grid with RH, Temperature, dust_fine, sea_salt_fine.
    indexer : CellIndexer
    lut : IsorropiaLUT
        Loaded 7D lookup table.

    Returns
    -------
    pm25_3d : ndarray, shape (nz, ny, nx)
        Total PM2.5 concentration (compound mass, ug/m3).
    diagnostics : dict
        f_nh4, f_no3, aerosol_water, ph arrays (nz, ny, nx).
    """
    N = indexer.N
    nz, ny, nx = grid.nz, grid.ny, grid.nx

    def _species(idx):
        return np.maximum(c_6N[idx * N:(idx + 1) * N], 0.0)

    n_species_present = c_6N.size // N
    c_pm    = _species(IDX_PM25)
    c_nh    = _species(IDX_TOTAL_NH)
    c_pso4  = _species(IDX_PSO4)
    c_no3   = _species(IDX_TOTAL_NO3)   # TotalNO3 = HNO3 + pNO3
    # Organics as in cli._soa_3d_flat: F_p-weighted particle mass over the 5
    # VBS bins when the grid carries F_p_vbs, else the C100 bin alone.
    F_p = getattr(grid, "F_p_vbs", None)
    if (F_p is not None and F_p.shape[0] == len(IDX_VBS_BINS)
            and n_species_present > max(IDX_VBS_BINS)):
        c_soa = np.zeros(N, dtype=np.float64)
        for i, s in enumerate(IDX_VBS_BINS):
            c_soa = c_soa + F_p[i].ravel() * _species(s)
    else:
        c_soa = _species(IDX_SOA)
    c_poa = _species(IDX_POA) if n_species_present > IDX_POA else np.zeros(N)

    # ISORROPIA inputs: element mass, ug/m3.
    # total_SO4 = pSO4 only (SO2 not in equilibrium pool).
    # total_NO3 = HNO3 + pNO3 — the correct thermodynamic pool now that
    # TotalNO3 is an explicit species separate from NOx.
    total_so4 = c_pso4
    total_nh = c_nh
    total_no3 = c_no3

    # Meteorology
    T_flat = grid.Temperature.ravel()
    RH_flat = grid.RH.ravel() / 100.0  # percent -> fraction

    # Crustal and sea salt (0 if fields not available)
    if grid.dust_fine.size == N:
        Ca_flat = grid.dust_fine.ravel() * _DUST_TO_CA
    else:
        Ca_flat = np.zeros(N)
    if grid.sea_salt_fine.size == N:
        Na_flat = grid.sea_salt_fine.ravel() * _SEASALT_TO_NA
    else:
        Na_flat = np.zeros(N)

    # Query LUT
    f_nh4, f_no3, aerosol_water, ph = lut.query(
        total_so4, total_nh, total_no3, Ca_flat, Na_flat, T_flat, RH_flat,
    )

    # PM2.5 = primary + POA + SOA + ISORROPIA NH4 + SO4 + ISORROPIA NO3.
    pm25 = (c_pm
            + c_poa
            + c_soa
            + f_nh4 * c_nh * N_TO_NH4
            + c_pso4 * S_TO_SO4
            + f_no3 * c_no3 * N_TO_NO3)

    diagnostics = {
        "f_nh4": f_nh4.reshape((nz, ny, nx)),
        "f_no3": f_no3.reshape((nz, ny, nx)),
        "aerosol_water": aerosol_water.reshape((nz, ny, nx)),
        "ph": ph.reshape((nz, ny, nx)),
    }
    return pm25.reshape((nz, ny, nx)), diagnostics


def compare_partitioning_gc(bin_path, grid, lut):
    """Compare GEOS-Chem prescribed partitioning with ISORROPIA LUT at GC concentrations.

    Isolates the thermodynamic effect: same concentrations (GEOS-Chem),
    different partitioning model (prescribed vs LUT).

    Parameters
    ----------
    bin_path : str
        Path to preprocessor bin file (has gNH, pNH, gNO, pNO, gS, pS).
    grid : GridData
        Bin grid with RH, Temperature, dust_fine, sea_salt_fine.
    lut : IsorropiaLUT

    Returns
    -------
    dict with prescribed and LUT partitioning arrays (nz, ny, nx).
    """
    import xarray as xr

    ds = xr.open_dataset(bin_path)
    nz, ny, nx = grid.nz, grid.ny, grid.nx
    N = nz * ny * nx

    # GEOS-Chem total concentrations (element mass, ug/m3).
    # total_SO4 is aerosol sulfate only (pS); SO2 gas is not an ISORROPIA input.
    # total_NO3: the GC 2016 per-bin archive does not carry HNO3, so this
    # underspecifies the HNO3+pNO3 pool. We use pNO3 alone as a lower bound —
    # the resulting LUT f_NO3 comparison against NOPartitioning (a lumped
    # NOx+pNO3 marginal) is apples-to-oranges but retained as a sanity check.
    total_nh_gc = (ds["gNH"].values + ds["pNH"].values).ravel()
    total_no3_gc = ds["pNO"].values.ravel()
    total_s_gc = ds["pS"].values.ravel()

    # Prescribed partitioning — NOPartitioning is the legacy lumped NOx+pNO3
    # marginal (kept for legacy interop). The HNO3+pNO3 thermodynamic
    # marginal is produced at runtime by the Phase 3d outer iteration, not
    # in the preprocessor.
    p_nh_gc = ds["NHPartitioning"].values
    p_no_gc = ds["NOPartitioning"].values
    ds.close()

    # Meteorology + crustal/seasalt from grid
    T_flat = grid.Temperature.ravel()
    RH_flat = grid.RH.ravel() / 100.0

    if grid.dust_fine.size == N:
        Ca_flat = grid.dust_fine.ravel() * _DUST_TO_CA
    else:
        Ca_flat = np.zeros(N)
    if grid.sea_salt_fine.size == N:
        Na_flat = grid.sea_salt_fine.ravel() * _SEASALT_TO_NA
    else:
        Na_flat = np.zeros(N)

    # Query LUT at GC concentrations
    f_nh4, f_no3, _, _ = lut.query(
        total_s_gc, total_nh_gc, total_no3_gc, Ca_flat, Na_flat, T_flat, RH_flat,
    )

    return {
        "gc_p_nh": p_nh_gc,
        "gc_p_no": p_no_gc,
        "lut_f_nh4_at_gc": f_nh4.reshape((nz, ny, nx)),
        "lut_f_no3_at_gc": f_no3.reshape((nz, ny, nx)),
    }
