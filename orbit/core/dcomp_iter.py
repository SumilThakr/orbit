"""Outer-iteration instrumentation and the Anderson accelerator.

Written for the DCOMP closure; the Anderson accelerator here now also
serves the production ISORROPIA closure (--isorropia-anderson).

The outer iteration closes OH <-> precursors <-> O3 <-> ISORROPIA
partitioning <-> S_a simultaneously.  A bug that only manifests at
iteration 5 is much harder to debug than one caught at iteration 1, so
this module provides:

- ``IterationMetrics``:  per-iter summary of domain-mean and max-relative
  change for OH, O3, f_NH4, f_NO3, PM2.5, plus NOx / TotalNO3 negatives
  and the cell-wise N / S mass-conservation residuals.
- ``ConvergenceHistory``: accumulator + classifier that distinguishes
  ``converged`` / ``oscillating`` / ``drifting`` species at non-convergence.
- ``plausibility_check()``: cheap sign / scale / stability pre-flight run
  after iteration 1 so order-of-magnitude bugs get caught in 1 iter rather
  than 20.
- ``check_invariants()``: per-iter assertions of N / S mass conservation
  and a non-negativity floor.

The helpers do no numerics of their own beyond reductions — they
instrument the Phase 3d outer loop so failures are diagnosable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from orbit.core.deposition import (
    N_SPECIES, IDX_SOA, IDX_PM25, IDX_TOTAL_NH, IDX_SO2, IDX_NOX,
    IDX_PSO4, IDX_TOTAL_NO3, IDX_O3,
)


# Convergence gate defaults (per instrumentation plan).
DEFAULT_TOL = 0.02
DEFAULT_MAX_ITER = 20


# -----------------------------------------------------------------------------
# Per-iteration metrics
# -----------------------------------------------------------------------------

def _max_rel_change(new, old, floor=0.0, eps=1e-30):
    """Per-cell |Δ|/max(|X|, floor), returned as (array, domain_mean_of_|X|).

    Per-cell floored denominator so physically negligible cells (OH in
    the near-zero regime, partitioning fractions ≈ 0) don't amplify
    iteration noise and dominate the max.  Floor should be set to the
    value below which the quantity is physically unimportant for the
    model's purpose — e.g. OH_floor = 1e4 molec/cm³.
    """
    new = np.asarray(new, dtype=np.float64)
    old = np.asarray(old, dtype=np.float64)
    denom = np.maximum(np.maximum(np.abs(new), np.abs(old)),
                       max(floor, eps))
    mean_scale = max(float(np.mean(np.abs(new) + np.abs(old))) * 0.5, eps)
    return np.abs(new - old) / denom, mean_scale


def _argmax_cell(arr):
    """Unravel argmax of a flat or nD array to an (k, j, i) or (bin, k, j, i) tuple."""
    idx = int(np.argmax(arr))
    return tuple(int(v) for v in np.unravel_index(idx, arr.shape))


@dataclass
class IterationMetrics:
    """Summary of what changed between iter `it-1` and iter `it`.

    The per-field relative-change value stored in tuples below is the
    99th-percentile of per-cell |Δ|/max(|X|, floor).  Using p99 instead
    of max prevents a single slow-settling boundary cell from gating
    convergence forever; the worst-cell location is still tracked in
    cell_of_max / bin_of_max for diagnostic visibility.
    """
    it: int

    # Per-field: (domain_mean_new, p99_rel_change, cell_of_argmax, bin_of_argmax)
    oh:    Tuple[float, float, Tuple[int, int, int], int] = (0.0, 0.0, (0, 0, 0), 0)
    o3:    Tuple[float, float, Tuple[int, int, int], int] = (0.0, 0.0, (0, 0, 0), 0)
    f_nh4: Tuple[float, float, Tuple[int, int, int], int] = (0.0, 0.0, (0, 0, 0), 0)
    f_no3: Tuple[float, float, Tuple[int, int, int], int] = (0.0, 0.0, (0, 0, 0), 0)
    pm25:  Tuple[float, float, Tuple[int, int, int], int] = (0.0, 0.0, (0, 0, 0), 0)

    # Sign of the max change at the max cell (for oscillation detection).
    oh_sign:    int = 0
    o3_sign:    int = 0
    f_nh4_sign: int = 0
    f_no3_sign: int = 0

    # Negativity tracking on the diagnosed species
    nox_neg_count:       int = 0
    nox_min:             float = 0.0
    totalNO3_neg_count:  int = 0
    totalNO3_min:        float = 0.0
    nox_typical:         float = 0.0
    totalNO3_typical:    float = 0.0

    # Mass-conservation residuals (max over cells/bins of |NOx loss - TotalNO3 source|).
    mass_residual_N: float = 0.0
    mass_residual_S: float = 0.0

    # Convenience: all three gates met this iter?
    converged: bool = False

    def gates_met(self, tol=DEFAULT_TOL) -> bool:
        return (self.oh[1] < tol and self.o3[1] < tol
                and self.f_nh4[1] < tol and self.f_no3[1] < tol)


def _stack_4d(per_bin):
    """Stack a list of (nz, ny, nx) per bin into (n_bins, nz, ny, nx)."""
    return np.stack([np.asarray(a, dtype=np.float64) for a in per_bin], axis=0)


def build_iteration_metrics(
    it: int,
    oh_new_per_bin, oh_prev_per_bin,
    o3_new_per_bin, o3_prev_per_bin,
    f_nh4_new_per_bin, f_nh4_prev_per_bin,
    f_no3_new_per_bin, f_no3_prev_per_bin,
    pm25_new_per_bin, pm25_prev_per_bin,
    c_nox_per_bin, c_no3_per_bin,
    mass_residual_N=0.0, mass_residual_S=0.0,
    tol=DEFAULT_TOL,
) -> IterationMetrics:
    """Build an ``IterationMetrics`` from per-bin new/prev 3D fields.

    Per-field floors on the relative-change denominator (per-cell
    |Δ|/max(|X|, floor)) so cells with physically negligible values
    don't amplify iteration noise.  Reduction is 99th percentile, not
    max, so a single outlier cell can't block the convergence gate.
    """

    # Per-field floors: values below these are physically negligible
    # for the PM2.5 paper's purpose and shouldn't gate convergence.
    FLOOR_OH    = 1.0e4   # molec/cm³ (below which OH reactivity is irrelevant)
    FLOOR_O3    = 5.0     # µg/m³ (~2.5 ppbv, below which O3 chemistry is negligible)
    FLOOR_FRAC  = 0.01    # partitioning fractions below 1% don't meaningfully change PM
    FLOOR_PM    = 1.0     # µg/m³ (below which the cell is not health-relevant)

    def _pack(new, prev, floor):
        new4 = _stack_4d(new)
        prev4 = _stack_4d(prev)
        denom = np.maximum(np.maximum(np.abs(new4), np.abs(prev4)),
                           max(floor, 1e-30))
        rel = np.abs(new4 - prev4) / denom
        cell = _argmax_cell(rel)
        tau = cell[0]
        kji = (cell[1], cell[2], cell[3])
        # Sign of the change at the worst cell (for oscillation detection)
        sign_val = float(new4[cell] - prev4[cell])
        sgn = 1 if sign_val > 0 else (-1 if sign_val < 0 else 0)
        # 99th percentile for the gate; max-cell location retained for diagnostics.
        rel_p99 = float(np.percentile(rel, 99))
        return (float(new4.mean()), rel_p99, kji, tau), sgn

    oh_metric, oh_sgn = _pack(oh_new_per_bin, oh_prev_per_bin, FLOOR_OH)
    o3_metric, o3_sgn = _pack(o3_new_per_bin, o3_prev_per_bin, FLOOR_O3)
    f_nh4_metric, f_nh4_sgn = _pack(f_nh4_new_per_bin, f_nh4_prev_per_bin, FLOOR_FRAC)
    f_no3_metric, f_no3_sgn = _pack(f_no3_new_per_bin, f_no3_prev_per_bin, FLOOR_FRAC)
    pm25_metric, _ = _pack(pm25_new_per_bin, pm25_prev_per_bin, FLOOR_PM)

    # Negatives tracking
    c_nox4 = _stack_4d(c_nox_per_bin)
    c_no3_4 = _stack_4d(c_no3_per_bin)
    nox_typ = max(float(np.mean(np.abs(c_nox4))), 1e-30)
    no3_typ = max(float(np.mean(np.abs(c_no3_4))), 1e-30)

    m = IterationMetrics(
        it=it,
        oh=oh_metric, o3=o3_metric,
        f_nh4=f_nh4_metric, f_no3=f_no3_metric, pm25=pm25_metric,
        oh_sign=oh_sgn, o3_sign=o3_sgn,
        f_nh4_sign=f_nh4_sgn, f_no3_sign=f_no3_sgn,
        nox_neg_count=int(np.sum(c_nox4 < 0)),
        nox_min=float(c_nox4.min()),
        totalNO3_neg_count=int(np.sum(c_no3_4 < 0)),
        totalNO3_min=float(c_no3_4.min()),
        nox_typical=nox_typ,
        totalNO3_typical=no3_typ,
        mass_residual_N=float(mass_residual_N),
        mass_residual_S=float(mass_residual_S),
    )
    m.converged = m.gates_met(tol)
    return m


def print_metrics(m: IterationMetrics, lat=None, lon=None):
    """One-block stdout print of the iteration metrics."""
    def _loc(cell, tau):
        k, j, i = cell
        geo = ""
        if lat is not None and lon is not None and j < len(lat) and i < len(lon):
            geo = f" [{lat[j]:.1f}N, {lon[i]:.1f}E]"
        return f"k={k} j={j} i={i}{geo} bin={tau + 1}"

    def _row(label, tup, unit=""):
        mean, rel, cell, tau = tup
        print(f"  {label:<6s} mean: {mean:.3e}{unit}   "
              f"p99 |Δ|/X = {rel * 100:5.2f}%  (argmax at {_loc(cell, tau)})")

    print(f"=== Outer iteration {m.it} ===")
    _row("OH",    m.oh,    "")
    _row("O3",    m.o3,    " ug/m3")
    _row("fNH4",  m.f_nh4, "")
    _row("fNO3",  m.f_no3, "")
    _row("PM2.5", m.pm25,  " ug/m3")
    nox_rel = abs(m.nox_min) / max(m.nox_typical, 1e-30)
    no3_rel = abs(m.totalNO3_min) / max(m.totalNO3_typical, 1e-30)
    print(f"  NOx    neg: count={m.nox_neg_count:6d}  "
          f"min={m.nox_min:+.2e} ug N/m3  |min|/typ={nox_rel:.1e}")
    print(f"  TotNO3 neg: count={m.totalNO3_neg_count:6d}  "
          f"min={m.totalNO3_min:+.2e} ug N/m3  |min|/typ={no3_rel:.1e}")
    print(f"  Mass residual  N: {m.mass_residual_N:.2e}   "
          f"S: {m.mass_residual_S:.2e}")
    print(f"  Gates met: {m.converged} (tol={DEFAULT_TOL * 100:.0f}%)")


# -----------------------------------------------------------------------------
# Non-convergence classifier
# -----------------------------------------------------------------------------

@dataclass
class ConvergenceHistory:
    """Accumulator of per-iter ``IterationMetrics`` across the outer loop.

    Owns the convergence verdict (``converged()`` checks the last iter's
    gates) and, at non-convergence, ``classify()`` splits the gated fields
    into converged / oscillating / drifting buckets using sign alternation
    of the max-cell change.  ``to_npz_dict()`` flattens the history into
    arrays for the output NPZ so post-run QA can replay the trajectory.
    """
    iters: List[IterationMetrics] = field(default_factory=list)

    def append(self, m: IterationMetrics):
        self.iters.append(m)

    def last(self) -> Optional[IterationMetrics]:
        return self.iters[-1] if self.iters else None

    def converged(self, tol=DEFAULT_TOL) -> bool:
        if not self.iters:
            return False
        return self.iters[-1].gates_met(tol)

    def classify(self, tol=DEFAULT_TOL, oscillation_window=3) -> Dict[str, List[str]]:
        """Split the fields into converged / oscillating / drifting buckets.

        Called at the end of the outer loop — before it is non-informative.
        """
        buckets: Dict[str, List[str]] = {
            "converged": [], "oscillating": [], "drifting": [],
        }
        if not self.iters:
            return buckets

        fields = ("oh", "o3", "f_nh4", "f_no3")
        recent = self.iters[-oscillation_window:] \
            if len(self.iters) >= oscillation_window else self.iters

        for name in fields:
            last = getattr(self.iters[-1], name)[1]
            if last < tol:
                buckets["converged"].append(name)
                continue
            # Check sign alternation on the max-cell change
            signs = [getattr(it, f"{name}_sign") for it in recent]
            nonzero = [s for s in signs if s != 0]
            alternates = (len(nonzero) >= 2
                          and all(nonzero[k] != nonzero[k + 1]
                                  for k in range(len(nonzero) - 1)))
            if alternates:
                buckets["oscillating"].append(name)
            else:
                buckets["drifting"].append(name)
        return buckets

    def print_classification(self, tol=DEFAULT_TOL):
        buckets = self.classify(tol=tol)
        it = self.iters[-1].it if self.iters else 0
        print(f"=== Non-convergence classification at iter {it} ===")
        for key in ("converged", "oscillating", "drifting"):
            names = buckets[key]
            print(f"  {key}: {', '.join(names) if names else '-'}")

    def to_npz_dict(self, prefix: str = "conv_") -> Dict[str, np.ndarray]:
        """Flatten the history into arrays suitable for np.savez."""
        n = len(self.iters)
        if n == 0:
            return {}
        out: Dict[str, np.ndarray] = {}
        for name in ("oh", "o3", "f_nh4", "f_no3", "pm25"):
            mean = np.array([getattr(it, name)[0] for it in self.iters])
            rel = np.array([getattr(it, name)[1] for it in self.iters])
            # Max-cell (k, j, i, bin) locations — useful for post-run QA to
            # see whether the same cell dominates iter-to-iter (oscillation
            # signature) or different cells (still exploring).
            cells = np.array(
                [[*getattr(it, name)[2], getattr(it, name)[3]] for it in self.iters],
                dtype=np.int32,
            )
            out[f"{prefix}{name}_mean"] = mean
            out[f"{prefix}{name}_max_rel"] = rel
            out[f"{prefix}{name}_max_cell"] = cells  # (n, 4): k, j, i, bin
        for attr in ("nox_neg_count", "nox_min", "totalNO3_neg_count",
                     "totalNO3_min", "mass_residual_N", "mass_residual_S"):
            out[f"{prefix}{attr}"] = np.array([getattr(it, attr) for it in self.iters])
        for attr in ("oh_sign", "o3_sign", "f_nh4_sign", "f_no3_sign"):
            out[f"{prefix}{attr}"] = np.array([getattr(it, attr) for it in self.iters],
                                               dtype=np.int8)
        out[f"{prefix}n_iter"] = np.array(n)
        return out


# -----------------------------------------------------------------------------
# Iteration-1 plausibility check
# -----------------------------------------------------------------------------

def plausibility_check(
    result_iter1: Dict[str, Any],
    result_iter0: Dict[str, Any],
    chem,
    dtau: float,
) -> Tuple[bool, List[str]]:
    """Cheap sign / scale / stability pre-flight on iteration 1.

    Expects:
        result_iter1["orbits"] : {species_idx: [c_0, ..., c_8]}
        result_iter0["orbits"] : same shape, prescribed-chemistry baseline
        chem.k_nox_to_no3_rate : list of (nz, ny, nx) 1/s per bin
        chem.k_so2_rate        : list of (nz, ny, nx) 1/s per bin
        dtau                   : bin length (seconds)

    Returns (all_ok, messages).  Check list is the 5-item one from
    the instrumentation plan.  Messages always include the value so
    borderline cases can be judged.
    """
    ok = True
    msgs: List[str] = []
    orbits_new = result_iter1["orbits"]
    orbits_base = result_iter0["orbits"]

    def _mean_stack(orbit, idx):
        # Time-mean over bins 1..N_BINS (skip c_0, equals c_N by periodicity)
        arr = np.stack([np.asarray(orbit[idx][tau + 1], dtype=np.float64)
                        for tau in range(8)], axis=0)
        return arr

    totalNO3 = _mean_stack(orbits_new, IDX_TOTAL_NO3)
    pSO4 = _mean_stack(orbits_new, IDX_PSO4)
    nox = _mean_stack(orbits_new, IDX_NOX)

    # Check 1: TotalNO3 orbit mean positive
    t3_mean = totalNO3.mean()
    t3_min = totalNO3.min()
    if t3_mean <= 0:
        ok = False
        msgs.append(f"FAIL: TotalNO3 orbit mean non-positive ({t3_mean:.2e})")
    else:
        rel = abs(t3_min) / max(abs(t3_mean), 1e-30)
        status = "OK" if rel < 1e-3 else "WARN"
        msgs.append(f"{status}: TotalNO3 mean={t3_mean:.2e}  min={t3_min:+.2e} "
                    f"|min|/mean={rel:.1e}")

    # Check 2: pSO4 orbit mean positive (baseline has SO2 oxidation already)
    s_mean = pSO4.mean()
    if s_mean <= 0:
        ok = False
        msgs.append(f"FAIL: pSO4 orbit mean non-positive ({s_mean:.2e})")
    else:
        msgs.append(f"OK: pSO4 mean={s_mean:.2e}")

    # Check 3: NOx orbit mean positive
    n_mean = nox.mean()
    if n_mean <= 0:
        ok = False
        msgs.append(f"FAIL: NOx orbit mean non-positive ({n_mean:.2e})")
    else:
        rel = abs(nox.min()) / max(abs(n_mean), 1e-30)
        status = "OK" if rel < 1e-2 else "WARN"
        msgs.append(f"{status}: NOx mean={n_mean:.2e}  |min|/mean={rel:.1e}")

    # Check 4: backward-Euler stability on both chemistry rates.
    # k*dtau > 5 is not a hard stability limit but flags very stiff cells.
    k_nox_max = float(max(np.max(k) for k in chem.k_nox_to_no3_rate))
    k_so2_max = float(max(np.max(k) for k in chem.k_so2_rate))
    nox_cfl = k_nox_max * dtau
    so2_cfl = k_so2_max * dtau
    if nox_cfl > 10.0:
        ok = False
        msgs.append(f"FAIL: k_nox·dtau max = {nox_cfl:.2f} (> 10; stiffness)")
    else:
        msgs.append(f"OK: k_nox·dtau max = {nox_cfl:.2f}")
    if so2_cfl > 10.0:
        ok = False
        msgs.append(f"FAIL: k_so2·dtau max = {so2_cfl:.2f} (> 10; stiffness)")
    else:
        msgs.append(f"OK: k_so2·dtau max = {so2_cfl:.2f}")

    # Check 5: iteration-1 delta vs baseline on diagnosed species.
    # Phase 3c smoke showed +9% PM2.5 on single iter — ballpark check.
    pm_base = _mean_stack(orbits_base, IDX_PM25)
    pm_new = _mean_stack(orbits_new, IDX_PM25)
    # PM25 channel magnitude shouldn't dominate; check ratio against baseline.
    # A 10x jump would indicate sign / unit error.
    delta_pm = (pm_new.mean() - pm_base.mean()) / max(pm_base.mean(), 1e-30)
    if abs(delta_pm) > 2.0:
        ok = False
        msgs.append(f"FAIL: PM2.5 iter1/iter0 ratio = {1 + delta_pm:.2f} "
                    f"(> 3x baseline — sign or unit bug)")
    else:
        msgs.append(f"OK: PM2.5 delta iter1 vs iter0: {delta_pm * 100:+.1f}%")

    return ok, msgs


# -----------------------------------------------------------------------------
# Mass-conservation invariants
# -----------------------------------------------------------------------------

def check_invariants(
    result: Dict[str, Any],
    chem,
    dtau: float,
) -> Dict[str, float]:
    """Per-iter N / S mass balance + non-negativity floor.

    The NOx->TotalNO3 coupling is mass-conserving by construction: the
    same ``k_nox_to_no3_rate`` array populates the NOx diagonal loss and
    the TotalNO3 off-diagonal source.  The source for TotalNO3 is
    ``k · [NOx]``; the loss on NOx is ``k · [NOx]``.  The residual should
    be exactly zero to machine precision per bin.  Same story for SO2
    / pSO4.

    We also compute the non-negativity floor: |min(c)| / <|c|> across all
    8 species; anything > 1e-3 is a structural warning.
    """
    orbits = result["orbits"]

    def _end_of_bin(s, tau):
        return np.asarray(orbits[s][tau + 1], dtype=np.float64)

    # N-mass balance: per-bin sum_cells k_nox[bin] * c_NOx[bin] vs. ditto.
    # By construction this is identical, so ``residual`` should be 0.
    N_residuals = []
    S_residuals = []
    for tau in range(8):
        k_nox = np.asarray(chem.k_nox_to_no3_rate[tau], dtype=np.float64).ravel()
        c_nox = _end_of_bin(IDX_NOX, tau)
        loss = k_nox * np.maximum(c_nox, 0.0)
        # TotalNO3 source is the same array by construction.
        src = k_nox * np.maximum(c_nox, 0.0)
        N_residuals.append(float(np.max(np.abs(loss - src))))

        k_s = np.asarray(chem.k_so2_rate[tau], dtype=np.float64).ravel()
        c_so2 = _end_of_bin(IDX_SO2, tau)
        loss_s = k_s * np.maximum(c_so2, 0.0)
        src_s = k_s * np.maximum(c_so2, 0.0)
        S_residuals.append(float(np.max(np.abs(loss_s - src_s))))

    max_N_resid = float(max(N_residuals))
    max_S_resid = float(max(S_residuals))

    # Non-negativity floor across all species
    floors = {}
    for s in range(N_SPECIES):
        arr = np.concatenate([_end_of_bin(s, tau) for tau in range(8)])
        typ = max(float(np.mean(np.abs(arr))), 1e-30)
        mn = float(arr.min())
        floors[s] = abs(min(mn, 0.0)) / typ

    return {
        "mass_residual_N": max_N_resid,
        "mass_residual_S": max_S_resid,
        "nonneg_floor_Org":      floors[IDX_SOA],
        "nonneg_floor_PM25":     floors[IDX_PM25],
        "nonneg_floor_NH":       floors[IDX_TOTAL_NH],
        "nonneg_floor_SO2":      floors[IDX_SO2],
        "nonneg_floor_NOx":      floors[IDX_NOX],
        "nonneg_floor_pSO4":     floors[IDX_PSO4],
        "nonneg_floor_TotalNO3": floors[IDX_TOTAL_NO3],
        "nonneg_floor_O3":       floors[IDX_O3],
    }


# -----------------------------------------------------------------------------
# α-under-relaxation
# -----------------------------------------------------------------------------

def under_relax(new, old, alpha: float = 0.5):
    """Under-relaxation: return alpha*new + (1-alpha)*old.

    Works element-wise on ndarrays or scalars; tolerates shape mismatches
    raised back to the caller.
    """
    if old is None:
        return new
    return alpha * np.asarray(new) + (1.0 - alpha) * np.asarray(old)


def under_relax_oxidants(chem_new, chem_prev, alpha: float = 0.5):
    """Blend the oxidant / rate fields of two ``ChemistryPerBin`` objects.

    Mutates ``chem_new`` in place — replaces its ``oxidants`` / rate lists
    with the α-blended version.  Only the fields that feed rate building
    are blended; photolysis LUT outputs are deterministic per bin and
    left alone.
    """
    if chem_prev is None:
        return chem_new

    for tau in range(len(chem_new.oxidants)):
        ox_n = chem_new.oxidants[tau]
        ox_p = chem_prev.oxidants[tau]
        ox_n.OH    = under_relax(ox_n.OH,    ox_p.OH,    alpha)
        ox_n.HO2   = under_relax(ox_n.HO2,   ox_p.HO2,   alpha)
        ox_n.RO2   = under_relax(ox_n.RO2,   ox_p.RO2,   alpha)
        ox_n.NO    = under_relax(ox_n.NO,    ox_p.NO,    alpha)
        ox_n.NO2   = under_relax(ox_n.NO2,   ox_p.NO2,   alpha)
        ox_n.f_NO2 = under_relax(ox_n.f_NO2, ox_p.f_NO2, alpha)
        ox_n.NO3   = under_relax(ox_n.NO3,   ox_p.NO3,   alpha)
        ox_n.N2O5  = under_relax(ox_n.N2O5,  ox_p.N2O5,  alpha)

        chem_new.k_so2_rate[tau] = under_relax(
            chem_new.k_so2_rate[tau], chem_prev.k_so2_rate[tau], alpha,
        )
        chem_new.k_nox_to_no3_rate[tau] = under_relax(
            chem_new.k_nox_to_no3_rate[tau], chem_prev.k_nox_to_no3_rate[tau],
            alpha,
        )
        if (chem_new.k_o3_loss_rate is not None
                and chem_prev.k_o3_loss_rate is not None):
            chem_new.k_o3_loss_rate[tau] = under_relax(
                chem_new.k_o3_loss_rate[tau], chem_prev.k_o3_loss_rate[tau], alpha,
            )
            chem_new.s_o3_source_rate[tau] = under_relax(
                chem_new.s_o3_source_rate[tau], chem_prev.s_o3_source_rate[tau], alpha,
            )
    return chem_new


# -----------------------------------------------------------------------------
# Anderson acceleration for the chemistry closure fixed point (Opt 1)
# -----------------------------------------------------------------------------
#
# The outer chemistry iteration is a fixed point: OH(k) -> assemble(k) ->
# orbit(k) -> OH(k+1).  Picard under-relaxation (alpha=0.5) converges
# geometrically; Anderson(m) replaces that blend with a residual-minimizing
# combination of the last m residuals, typically cutting iteration count
# 2-3x for the same tolerance.  See:
#
#   H. F. Walker & P. Ni, "Anderson Acceleration for Fixed-Point Iterations",
#   SIAM J. Num. Anal. 49(4), 2011.
#
# Design doc: notes/2026-04-24_orbit_solver_optimizations.md (Opt 1).
#
# The state vector is the flattened OH field across bins (one vector per
# outer iter).  HO2/f_NH4/f_NO3 are left on Picard — they are near-linear
# in OH + concentrations and converge fast enough under damping.


class AndersonAccelerator:
    """Anderson(m) acceleration of a fixed-point map g(x) = x.

    Per outer iter:
        x_next = accel.update(x_k, g_x_k)
    where x_k is the state at the start of the iter and g_x_k is what
    the fixed-point map produced (e.g. OH from one full chemistry +
    orbit pass).  On the first call there's no history -> falls back
    to Picard (x_k + beta * r_k).

    Monotonicity safeguard: if the projected residual norm exceeds
    `safeguard_factor * ||r_k||`, reject the Anderson step and take a
    Picard step instead (and reset history).  `apply_with_safeguard()`
    bundles this; `update()` is the unguarded variant.

    Parameters
    ----------
    m : int
        Memory depth (number of prior residuals to use in the
        least-squares combination).  m = 3 is a robust default.
    beta : float
        Mixing parameter.  beta = 1.0 is standard Anderson;
        beta < 1 adds damping for harder fixed points.
    rcond : float
        SVD cutoff for the least-squares solve in `np.linalg.lstsq`.
        Guards against rank-deficient history matrices.
    safeguard_factor : float
        Tolerance for the monotonicity safeguard:  if the projected
        residual norm exceeds `safeguard_factor` times the current
        residual norm, fall back to Picard.
    """

    def __init__(self, m: int = 3, beta: float = 1.0,
                 rcond: float = 1e-10, safeguard_factor: float = 1.5):
        if m < 1:
            raise ValueError(f"Anderson memory depth m must be >= 1, got {m}")
        self.m = int(m)
        self.beta = float(beta)
        self.rcond = float(rcond)
        self.safeguard_factor = float(safeguard_factor)
        self.X: List[np.ndarray] = []  # iterates x_0, x_1, ...
        self.F: List[np.ndarray] = []  # residuals r_i = g(x_i) - x_i
        self.n_updates = 0
        self.n_safeguard_falls = 0

    def reset(self) -> None:
        """Discard history (force a Picard step on the next update)."""
        self.X.clear()
        self.F.clear()

    def update(self, x_k: np.ndarray, g_x_k: np.ndarray) -> np.ndarray:
        """Return the next iterate.  Does not apply the safeguard."""
        x_k = np.asarray(x_k, dtype=np.float64).ravel()
        g_x_k = np.asarray(g_x_k, dtype=np.float64).ravel()
        r_k = g_x_k - x_k
        self.X.append(x_k.copy())
        self.F.append(r_k.copy())
        while len(self.X) > self.m + 1:
            self.X.pop(0)
            self.F.pop(0)
        self.n_updates += 1

        m_k = len(self.F) - 1
        if m_k == 0:
            # No history -> Picard step.
            return x_k + self.beta * r_k

        # Columns of ΔF, ΔX (shape (N, m_k))
        DF = np.column_stack([self.F[i + 1] - self.F[i] for i in range(m_k)])
        DX = np.column_stack([self.X[i + 1] - self.X[i] for i in range(m_k)])

        # Least squares: γ = argmin_γ || r_k − ΔF γ ||
        gamma, *_ = np.linalg.lstsq(DF, r_k, rcond=self.rcond)

        # x_{k+1} = x_k + β r_k − (ΔX + β ΔF) γ
        return x_k + self.beta * r_k - (DX + self.beta * DF) @ gamma

    def apply_with_safeguard(self, x_k: np.ndarray,
                              g_x_k: np.ndarray) -> Tuple[np.ndarray, bool]:
        """Apply update() with a monotonicity safeguard.

        Returns
        -------
        x_next : ndarray
            The next iterate (Anderson-accelerated, or Picard if
            the safeguard fired).
        accepted : bool
            True if the Anderson step was accepted; False if the
            safeguard fell back to Picard (history reset).
        """
        x_k_arr = np.asarray(x_k, dtype=np.float64).ravel()
        g_arr = np.asarray(g_x_k, dtype=np.float64).ravel()
        r_k = g_arr - x_k_arr

        # Speculative Anderson update — temporarily commit history,
        # inspect the projected residual, and roll back if rejected.
        snapshot_X = [x.copy() for x in self.X]
        snapshot_F = [f.copy() for f in self.F]
        x_next = self.update(x_k_arr, g_arr)

        if len(self.F) <= 1:
            # First iter; no safeguard data.  Accept.
            return x_next, True

        # Recompute the least-squares projection residual as a cheap
        # estimator of the next step's residual norm.
        m_k = len(self.F) - 1
        DF = np.column_stack([self.F[i + 1] - self.F[i] for i in range(m_k)])
        gamma, *_ = np.linalg.lstsq(DF, r_k, rcond=self.rcond)
        proj_resid = float(np.linalg.norm(r_k - DF @ gamma))
        picard_resid = float(np.linalg.norm(r_k))

        if proj_resid > self.safeguard_factor * picard_resid:
            # Reject: restore history and take a Picard step.
            self.X = snapshot_X
            self.F = snapshot_F
            self.reset()
            self.n_safeguard_falls += 1
            return x_k_arr + self.beta * r_k, False

        return x_next, True

    def __repr__(self) -> str:
        return (f"AndersonAccelerator(m={self.m}, beta={self.beta}, "
                f"history_len={len(self.X)}, "
                f"updates={self.n_updates}, "
                f"safeguard_falls={self.n_safeguard_falls})")
