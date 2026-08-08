"""Zero-out simulation mode.

Consequential response of concentrations to a sector's emission removal
or scaling, including ISORROPIA regime shifts. Re-runs the forward
driver with perturbed emissions and reports
``δc = c_full - c_perturbed`` (note the sign — opposite to marginal).

See ``MODES.md`` (top level) for when to use zero-out vs. marginal,
sign conventions, schema, and the regime-shift diagnostic fields.
"""

from __future__ import annotations

import os
import time
from typing import Callable, Dict, List, Optional

import numpy as np

from orbit.core.grid_data import load_grid
from orbit.core.indexing import CellIndexer
from orbit.core.orbit import N_BINS
from orbit.core.deposition import N_SPECIES
from orbit.emissions.loader import DiurnalConfig
from orbit.emissions.sources import EmissionSource

from orbit.modes.perturbation import (
    Perturbation, build_perturbed_emissions,
)


def run_zero_out_month(
    month: int,
    *,
    baseline_npz_path: str,
    perturbation: Perturbation,
    output_dir: str,
    preproc_path_fn: Callable[[int, int], str],
    constants_path: str,
    baseline_sources: List[EmissionSource],
    forward_runner: Callable,
    forward_kwargs: Optional[Dict] = None,
    diurnal_cfg: Optional[DiurnalConfig] = None,
    keep_perturbed_npz: bool = True,
    output_filename: Optional[str] = None,
    verbose: bool = True,
) -> dict:
    """Compute δc / δPM2.5 by re-running forward with perturbed emissions.

    Parameters
    ----------
    month : int (1-12).
    baseline_npz_path : path to the per-month forward NPZ.
    perturbation : Perturbation. is_empty() → δ = 0 trivially.
    output_dir : where to write the zero-out δNPZ.
    preproc_path_fn : callable(month, bin_idx_1based) → preproc path.
    constants_path : MERRA2 constants file path.
    baseline_sources : the EmissionSource list the baseline used.
    forward_runner : the forward driver to call (typically
        ``run_orbit.run_forward_month``). Must accept
        ``e_perturbed_override`` kwarg and ``output_filename``.
    forward_kwargs : extra kwargs for forward_runner (closure_alpha,
        isorropia_closure_iters, basin_flip_damp, ...). Typically the
        same as the baseline's settings.
    diurnal_cfg : optional. Same diurnal config the baseline used.
    keep_perturbed_npz : if True, the intermediate perturbed-run NPZ is
        kept on disk under ``output_dir/_perturbed/``. Useful for
        re-differencing with different δ definitions later.
    output_filename : override for the δNPZ filename.

    Returns a dict of timings and key diagnostics, or None on failure.
    """
    forward_kwargs = dict(forward_kwargs or {})
    out_name = output_filename or f"zeroout_M{month:02d}.npz"
    out_path = os.path.join(output_dir, out_name)
    os.makedirs(output_dir, exist_ok=True)

    timings: Dict[str, float] = {"month": month}

    # 1. Load baseline NPZ.
    if not os.path.exists(baseline_npz_path):
        if verbose:
            print(f"  Missing baseline: {baseline_npz_path}")
        return None
    t0 = time.time()
    baseline = np.load(baseline_npz_path, allow_pickle=False)
    required = {"c_orbit", "iso_pm25_mean", "iso_pm25_orbit", "lon", "lat"}
    missing = required - set(baseline.files)
    if missing:
        raise ValueError(
            f"Baseline NPZ {baseline_npz_path} missing required keys: {missing}"
        )
    c_full_orbit = np.asarray(baseline["c_orbit"], dtype=np.float64)
    pm25_full_orbit = np.asarray(baseline["iso_pm25_orbit"], dtype=np.float64)
    pm25_full_mean = np.asarray(baseline["iso_pm25_mean"], dtype=np.float64)
    grid_shape = (
        tuple(int(x) for x in baseline["grid_shape"])
        if "grid_shape" in baseline.files else pm25_full_mean.shape
    )
    timings["baseline_load"] = time.time() - t0

    # Optional regime-shift diagnostic — needs per-bin equilibrium
    # partitioning from both runs.
    have_baseline_eq = (
        "f_no3_eq_3d" in baseline.files and "f_nh4_eq_3d" in baseline.files
    )
    f_no3_eq_full = (
        np.asarray(baseline["f_no3_eq_3d"], dtype=np.float64)
        if have_baseline_eq else None
    )
    f_nh4_eq_full = (
        np.asarray(baseline["f_nh4_eq_3d"], dtype=np.float64)
        if have_baseline_eq else None
    )

    # 2. Build e_perturbed. Need an indexer and a representative grid;
    # load just bin 1 for the regridding step (matches forward driver's
    # convention of using grids[0] for the emission spatial regrid).
    t0 = time.time()
    bp1 = preproc_path_fn(month, 1)
    if not os.path.exists(bp1):
        if verbose:
            print(f"  Missing: {bp1}")
        return None
    grid0 = load_grid(bp1, constants_path)
    nz, ny, nx = grid0.nz, grid0.ny, grid0.nx
    if (nz, ny, nx) != grid_shape:
        raise ValueError(
            f"Preproc grid shape ({nz},{ny},{nx}) != baseline grid_shape {grid_shape}"
        )
    indexer = CellIndexer(nz, ny, nx)
    e_perturbed = build_perturbed_emissions(
        perturbation, baseline_sources, grid0, indexer,
        month=month, n_bins=N_BINS, diurnal_cfg=diurnal_cfg,
        verbose=verbose,
    )
    timings["build_e"] = time.time() - t0
    if verbose:
        print(f"  e_perturbed: shape={e_perturbed.shape}, "
              f"|sum|={float(np.abs(e_perturbed).sum()):.3e} "
              f"({timings['build_e']:.1f}s)")

    # 3. Run forward with perturbed emissions. Output lands directly in
    # output_dir/_perturbed/ via the forward driver's `output_dir` kwarg
    # (Phase 4d). The helper does its own grid load + UMFPACK symbolic +
    # ISORROPIA closure exactly as a normal forward run does.
    perturbed_dir = os.path.join(output_dir, "_perturbed")
    os.makedirs(perturbed_dir, exist_ok=True)
    perturbed_filename = f"forward_perturbed_{perturbation.name}_M{month:02d}.npz"
    perturbed_path = os.path.join(perturbed_dir, perturbed_filename)

    if verbose:
        print(f"  Forward(perturbed) → {perturbed_path}")
    t0 = time.time()
    fwd_result = forward_runner(
        month,
        diurnal_cfg=diurnal_cfg,
        e_perturbed_override=e_perturbed,
        output_filename=perturbed_filename,
        output_dir=perturbed_dir,
        **forward_kwargs,
    )
    timings["forward_perturbed"] = time.time() - t0

    if fwd_result is None:
        if verbose:
            print("  Forward(perturbed) failed")
        return None

    # 4. Diff against baseline.
    t0 = time.time()
    perturbed = np.load(perturbed_path, allow_pickle=False)
    c_pert_orbit = np.asarray(perturbed["c_orbit"], dtype=np.float64)
    if "iso_pm25_orbit" not in perturbed.files:
        raise RuntimeError(
            f"Perturbed forward NPZ at {perturbed_path} lacks iso_pm25_orbit / "
            f"iso_pm25_mean — ISORROPIA closure did not run. zero-out's whole "
            f"point is the regime-shift response captured by re-converging iso. "
            f"Likely cause: lut_path was not passed to the forward driver. "
            f"Check forward_kwargs in _run_zero_out_main."
        )
    pm25_pert_orbit = np.asarray(perturbed["iso_pm25_orbit"], dtype=np.float64)
    pm25_pert_mean = np.asarray(perturbed["iso_pm25_mean"], dtype=np.float64)

    # Deferred-backlog: closure-config consistency check at diff time.
    # If both the baseline and the perturbed forward NPZs carry
    # closure_settings_hash (Phase 4b), assert they match. A diff
    # taken across mismatched closure tunables silently bakes the
    # closure-config delta into the apparent regime-shift signal.
    if (
        "closure_settings_hash" in baseline.files
        and "closure_settings_hash" in perturbed.files
    ):
        bh = str(baseline["closure_settings_hash"])
        ph = str(perturbed["closure_settings_hash"])
        if bh != ph:
            print(
                f"  WARN: baseline closure_settings_hash={bh} differs from "
                f"perturbed={ph}. The δc reported below conflates the "
                f"emissions perturbation with a closure-config delta. "
                f"Re-run forward with matched args."
            )

    delta_c_orbit = c_full_orbit - c_pert_orbit
    delta_pm25_orbit = pm25_full_orbit - pm25_pert_orbit
    delta_pm25_mean = pm25_full_mean - pm25_pert_mean

    # Regime-shift diagnostic.
    delta_f_no3_eq = None
    delta_f_nh4_eq = None
    if (have_baseline_eq and "f_no3_eq_3d" in perturbed.files
            and "f_nh4_eq_3d" in perturbed.files):
        delta_f_no3_eq = (
            f_no3_eq_full - np.asarray(perturbed["f_no3_eq_3d"], dtype=np.float64)
        )
        delta_f_nh4_eq = (
            f_nh4_eq_full - np.asarray(perturbed["f_nh4_eq_3d"], dtype=np.float64)
        )
    timings["diff"] = time.time() - t0

    # 5. Save δNPZ.
    t0 = time.time()
    save_dict = {
        "mode": np.array("zero-out"),
        "month": np.array(month),
        "baseline_npz_path": np.array(os.path.abspath(baseline_npz_path)),
        "perturbed_npz_path": np.array(os.path.abspath(perturbed_path)),
        "perturbation_name": np.array(perturbation.name),
        "perturbation_description": np.array(perturbation.description),
        # Phase 4c: sign convention. zero-out is c_full - c_perturbed,
        # so a downward perturbation (factor=0.5) yields δc > 0 — the
        # opposite of marginal. unify_sign() flips when comparing.
        "sign_convention": np.array("baseline_minus_perturbed"),
        "delta_c_orbit": delta_c_orbit.astype(np.float64),
        "delta_pm25_orbit": delta_pm25_orbit.astype(np.float64),
        "delta_pm25_mean": delta_pm25_mean.astype(np.float64),
        "orbit_species_order": np.arange(N_SPECIES),
        "lon": baseline["lon"],
        "lat": baseline["lat"],
        "grid_shape": np.array(grid_shape),
    }
    if delta_f_no3_eq is not None:
        save_dict["delta_f_no3_eq_3d"] = delta_f_no3_eq.astype(np.float32)
        save_dict["delta_f_nh4_eq_3d"] = delta_f_nh4_eq.astype(np.float32)
    if perturbation.factors:
        save_dict["perturbation_factors_keys"] = np.array(list(perturbation.factors.keys()))
        save_dict["perturbation_factors_values"] = np.array(list(perturbation.factors.values()))
    if perturbation.add_sources:
        save_dict["perturbation_add_paths"] = np.array(
            [os.path.abspath(s.path) for s in perturbation.add_sources]
        )
    np.savez_compressed(out_path, **save_dict)
    timings["save"] = time.time() - t0

    if not keep_perturbed_npz and os.path.exists(perturbed_path):
        os.unlink(perturbed_path)

    timings["delta_c_max_abs"] = float(np.abs(delta_c_orbit).max())
    timings["delta_pm25_max_abs"] = float(np.abs(delta_pm25_orbit).max())
    if verbose:
        out_mb = os.path.getsize(out_path) / 1024 / 1024
        print(f"  Saved: {out_path} ({out_mb:.1f} MB, {timings['save']:.1f}s)")
        print(f"  max|δc|={timings['delta_c_max_abs']:.3e}, "
              f"max|δPM25|={timings['delta_pm25_max_abs']:.3e}")
    return timings
