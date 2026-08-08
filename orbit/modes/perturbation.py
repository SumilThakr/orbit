"""Perturbation specifications for `marginal` and `zero-out` modes.

A `Perturbation` describes the emission change to apply against a forward
baseline. Two complementary blocks:

- ``factors``: scale existing baseline source files by basename. Missing
  keys default to 1.0 (no change).
- ``add_sources``: purely additive new emission sources (e.g. a single
  facility). Each is loaded through the same `_accumulate_sources` path
  as baseline sources, so spatial regridding is consistent.

Two output paths share this spec:

- ``build_delta_emissions``: returns δe = sum (factor−1) × source +
  sum add. Used by `marginal` mode.
- ``build_perturbed_emissions``: returns e_perturbed = sum factor ×
  source + sum add. Used by `zero-out` mode.

Both return arrays in the 9-species solver layout (N_BINS, N_SPECIES * N).

YAML schema example (see ``examples/scenarios/*.yaml``):

    name: ag_nh3_minus_50pct
    description: 50% reduction in CEDS + GFED NH3
    factors:
      ceds_nh3_anthro_2022_monthly.nc: 0.50
      gfed5_nh3_bb_2022_monthly.nc: 0.50
    add:
      - path: /path/to/new_facility.nc
        units: kg/m2/s
        species_override: NH3
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from orbit.core.indexing import CellIndexer
from orbit.core.orbit import N_BINS
from orbit.core.deposition import N_SPECIES
from orbit.emissions.loader import load_emissions_diurnal, DiurnalConfig
from orbit.emissions.sources import EmissionSource, N_ORBIT_SPECIES
from orbit.emissions.vbs_distribution import (
    compute_nox_regime, distribute_voc_to_vbs_bins,
)


# Mapping from the loader's emission layout (Org, PM25, NH, SO2, pSO4,
# NO, POA) into the solver species layout. The legacy "NO" emissions slot
# (old index 5) routes into NOx (new index 4); pSO4 stays at new index 5;
# TotalNO3, O3, CO get zero direct emissions.
#
# MUST stay identical to `_old_to_new` in cli.run_forward_month — the
# perturbed run is differenced against a baseline built by that mapping,
# so any divergence lands directly in δ. Slot 6 (POA) was missing here
# until 2026-08-03, silently dropping 266 kg/s (Jan 2022) — 84.7% of
# anthropogenic primary mass — from every zero-out / marginal perturbed
# vector. Pinned by tests/test_perturbation_matches_forward_assembly.py.
_OLD_TO_NEW = {0: 0, 1: 1, 2: 2, 3: 3, 4: 5, 5: 4, 6: 13}


# ── Perturbation spec ──────────────────────────────────────────────────────


@dataclass
class Perturbation:
    """A perturbation against a baseline emissions configuration.

    factors : dict[basename, multiplier]
        Existing baseline source basenames mapped to scaling factors.
        Missing keys default to 1.0 (no change). Marginal contribution
        from a scaled source is ``(factor - 1) × source_contribution``;
        the zero-out path uses ``factor × source_contribution``.

    add_sources : list[EmissionSource]
        Purely additive new emission sources (e.g. a single facility).
        Marginal contribution is ``+1 × source_contribution``. Headline
        use case: "what's this new facility's PM2.5 impact?"
    """
    factors: Dict[str, float] = field(default_factory=dict)
    add_sources: List[EmissionSource] = field(default_factory=list)
    name: str = "perturbation"
    description: str = ""

    def is_empty(self) -> bool:
        non_unit_factors = any(abs(f - 1.0) > 1e-15 for f in self.factors.values())
        return not non_unit_factors and len(self.add_sources) == 0

    @classmethod
    def from_yaml(cls, path: str) -> "Perturbation":
        """Load a perturbation from a YAML config file.

        Schema: see module docstring. ``factors`` keys are basenames
        matched against the baseline emission filenames at runtime.
        ``add`` entries become ``EmissionSource`` instances.
        """
        import yaml
        with open(path, "r") as f:
            cfg = yaml.safe_load(f) or {}

        if not isinstance(cfg, dict):
            raise ValueError(
                f"Perturbation YAML must be a top-level mapping; "
                f"got {type(cfg).__name__} from {path}"
            )

        name = str(cfg.get("name", os.path.basename(path)))
        description = str(cfg.get("description", ""))

        factors: Dict[str, float] = {}
        for key, value in (cfg.get("factors") or {}).items():
            try:
                factors[str(key).strip()] = float(value)
            except (TypeError, ValueError) as e:
                raise ValueError(
                    f"factors[{key!r}] in {path} is not numeric: {value!r}"
                ) from e

        add_sources: List[EmissionSource] = []
        for i, entry in enumerate(cfg.get("add") or []):
            if not isinstance(entry, dict):
                raise ValueError(
                    f"add[{i}] in {path} must be a mapping; got {entry!r}"
                )
            if "path" not in entry:
                raise ValueError(f"add[{i}] in {path} missing required 'path'")
            varmap = None
            species_override = entry.get("species_override")
            if species_override:
                varmap = {str(species_override): str(species_override)}
            add_sources.append(EmissionSource(
                path=str(entry["path"]),
                format=str(entry.get("format", "netcdf")),
                units=str(entry.get("units", "kg/m2/s")),
                variable_mapping=entry.get("variable_mapping") or varmap,
                layer_index=entry.get("layer_index"),
                time_index=entry.get("time_index"),
                bin_axis=bool(entry.get("bin_axis", False)),
                bin_axis_name=str(entry.get("bin_axis_name", "bin")),
            ))

        return cls(
            factors=factors,
            add_sources=add_sources,
            name=name,
            description=description,
        )


def parse_cli_perturbation(
    scale_source_args: Optional[List[str]],
    add_emissions_args: Optional[List[str]],
) -> Perturbation:
    """Build a Perturbation from `--scale-source` / `--add-emissions` flags.

    Each --scale-source is "BASENAME=FACTOR" (multi-occurrence).
    Each --add-emissions is "PATH[:SPECIES]" (multi-occurrence). SPECIES,
    if given, becomes the variable_mapping override for the NetCDF; if
    omitted, the loader auto-detects the species from variable names.
    """
    factors: Dict[str, float] = {}
    for spec in (scale_source_args or []):
        if "=" not in spec:
            raise ValueError(f"--scale-source must be NAME=FACTOR, got: {spec}")
        name, factor_str = spec.split("=", 1)
        try:
            factor = float(factor_str)
        except ValueError as e:
            raise ValueError(f"--scale-source factor not numeric: {spec}") from e
        factors[name.strip()] = factor

    add_sources: List[EmissionSource] = []
    for spec in (add_emissions_args or []):
        if ":" in spec:
            path, species = spec.split(":", 1)
            species = species.strip()
            varmap = {species: species}
        else:
            path = spec
            varmap = None
        add_sources.append(EmissionSource(
            path=path.strip(),
            format="netcdf",
            units="kg/m2/s",
            time_index=None,
            variable_mapping=varmap,
        ))

    return Perturbation(factors=factors, add_sources=add_sources)


# ── Emission builders ──────────────────────────────────────────────────────


def _rebroadcast_6_to_9(emis_6: np.ndarray, n_bins: int, N: int) -> np.ndarray:
    """Reshape (n_bins, N_ORBIT_SPECIES * N) → (n_bins, N_SPECIES * N).

    Routes the legacy 6-species emission slots into the 9-species solver
    layout (TotalNO3, O3, CO get zero direct emissions).
    """
    out = np.zeros((n_bins, N_SPECIES * N), dtype=np.float64)
    for tau in range(n_bins):
        for old_s, new_s in _OLD_TO_NEW.items():
            out[tau, new_s * N:(new_s + 1) * N] = (
                emis_6[tau, old_s * N:(old_s + 1) * N]
            )
    return out


def _route_voc_to_vbs(out_13, perturbation, baseline_sources, grid, indexer,
                      month, n_bins, diurnal_cfg, verbose, baseline_multiplier):
    """Add VOC→VBS contributions to a 13-species emission vector, in place.

    ``load_emissions_diurnal`` zeros slot 0 for VOC sources (their mass is
    routed to the 5 VBS bins downstream of the loader), so the 6→13 rebroadcast
    carries NO VOC mass. Distribute it here exactly as the forward does, via the
    shared ``orbit.emissions.vbs_distribution`` — otherwise VOC perturbations
    would be silent no-ops in marginal/zero-out.

    ``baseline_multiplier(factor)`` gives the multiplier for each perturbed
    baseline source: ``factor - 1`` for δe (marginal), ``factor`` for the full
    perturbed vector (zero-out). ``add_sources`` always enter at +1.
    """
    F_high_nox = compute_nox_regime(grid)
    for source in baseline_sources:
        if source.voc_parent_class is None:
            continue
        factor = perturbation.factors.get(os.path.basename(source.path), 1.0)
        mult = baseline_multiplier(factor)
        if abs(mult) < 1e-15:
            continue
        contrib, _ = distribute_voc_to_vbs_bins(
            source, grid, indexer, diurnal_cfg, month, n_bins, F_high_nox,
            verbose=verbose,
        )
        out_13 += mult * contrib
    for source in perturbation.add_sources:
        if source.voc_parent_class is None:
            continue
        contrib, _ = distribute_voc_to_vbs_bins(
            source, grid, indexer, diurnal_cfg, month, n_bins, F_high_nox,
            verbose=verbose,
        )
        out_13 += contrib


def _add_ivoc_synthesis(out_13, perturbation, baseline_sources, grid, indexer,
                        month, n_bins, diurnal_cfg, verbose, baseline_multiplier):
    """Add synthesised IVOC to a 13-species emission vector, in place.

    Mirror of Stage C in ``cli.run_forward_month``: IVOC is not an
    inventory species — it is synthesised as POA x _IVOC_SCALING
    (Robinson 2007, IVOC ~ 1.5 x POA) and distributed over the VBS bins
    with the "ivoc" parent yields. The forward does this; without the
    same step here, a perturbed run loses ALL IVOC-derived SOA precursor
    relative to its own baseline, and the difference lands in δ.

    Falls back to the pre-POA-split proxy (CEDS PM2.5 x 0.3) when the
    manifest has no POA source, exactly as the forward does, so old
    manifests keep working.

    ``baseline_multiplier(factor)`` gives the multiplier per perturbed
    baseline source: ``factor - 1`` for δe, ``factor`` for the full
    perturbed vector. add_sources always enter at +1.
    """
    from orbit.cli import _IVOC_SCALING
    from orbit.emissions.loader import _factors_for_source
    from orbit.emissions.netcdf import load_netcdf_source
    from orbit.emissions.vbs_distribution import bin_indices_in_solver_layout
    from orbit.emissions.vbs_yields import VBS_PARENT_YIELDS

    N = indexer.N
    ivoc_yields = VBS_PARENT_YIELDS["ivoc"]
    bin_solver_idx = bin_indices_in_solver_layout()

    def _select(sources):
        poa = [s for s in sources
               if "ceds_poa_anthro" in os.path.basename(s.path)]
        if poa:
            return poa, 6, _IVOC_SCALING
        return ([s for s in sources
                 if "ceds_pm25_anthro" in os.path.basename(s.path)],
                1, 0.3 * _IVOC_SCALING)

    # Source selection follows the BASELINE manifest (as the forward
    # does); perturbation add_sources are handled in the same pass with a
    # +1 multiplier so a POA-bearing added source also synthesises IVOC.
    for srcs, mult_of in ((baseline_sources, "baseline"),
                          (perturbation.add_sources, "add")):
        sel, slot, frac = _select(srcs)
        for source in sel:
            if mult_of == "baseline":
                factor = perturbation.factors.get(
                    os.path.basename(source.path), 1.0)
                mult = baseline_multiplier(factor)
            else:
                mult = 1.0
            if abs(mult) < 1e-15:
                continue
            monthly = load_netcdf_source(source, grid, verbose=False)
            ivoc_emis_3d = frac * monthly[slot]
            if diurnal_cfg is None:
                factors = np.ones(n_bins)
            else:
                factors = _factors_for_source(source, diurnal_cfg, month, n_bins)
            for tau in range(n_bins):
                voc_mass_3d = mult * factors[tau] * ivoc_emis_3d
                for i_bin, solver_s in enumerate(bin_solver_idx):
                    out_13[tau, solver_s * N:(solver_s + 1) * N] += (
                        ivoc_yields[i_bin] * voc_mass_3d).ravel()
            if verbose:
                print(f"  [IVOC] {os.path.basename(source.path)} "
                      f"x {frac:g} (mult {mult:+.3f})")


def build_delta_emissions(
    perturbation: Perturbation,
    baseline_sources: List[EmissionSource],
    grid,
    indexer: CellIndexer,
    month: int,
    n_bins: int = N_BINS,
    diurnal_cfg: Optional[DiurnalConfig] = None,
    verbose: bool = False,
) -> np.ndarray:
    """Construct the perturbation emission vector δe for `marginal` mode.

    For each baseline source with factor != 1.0, compute its standalone
    emission contribution and add ``(factor - 1) × contribution`` to δe.
    For each `add_sources` entry, compute and add ``+1 × contribution``.

    Returns (n_bins, N_SPECIES * N) in solver layout.
    """
    N = indexer.N
    delta_old = np.zeros((n_bins, N_ORBIT_SPECIES * N), dtype=np.float64)

    for source in baseline_sources:
        basename = os.path.basename(source.path)
        factor = perturbation.factors.get(basename, 1.0)
        delta = factor - 1.0
        if abs(delta) < 1e-15:
            continue
        contrib = load_emissions_diurnal(
            [source], grid, indexer,
            diurnal_cfg=diurnal_cfg, month=month, n_bins=n_bins,
            verbose=verbose,
        )
        delta_old += delta * contrib
        if verbose:
            print(f"  [δe] scale {basename} by {factor:g} "
                  f"(δ={delta:+.3f}, contrib_max={contrib.max():.3e})")

    for source in perturbation.add_sources:
        contrib = load_emissions_diurnal(
            [source], grid, indexer,
            diurnal_cfg=diurnal_cfg, month=month, n_bins=n_bins,
            verbose=verbose,
        )
        delta_old += contrib
        if verbose:
            basename = os.path.basename(source.path)
            print(f"  [δe] add {basename} "
                  f"(contrib_max={contrib.max():.3e})")

    delta = _rebroadcast_6_to_9(delta_old, n_bins, N)
    # VOC sources contribute nothing above (slot 0 zeroed); route their
    # perturbed mass to the 5 VBS bins, as the forward does.
    _route_voc_to_vbs(delta, perturbation, baseline_sources, grid, indexer,
                      month, n_bins, diurnal_cfg, verbose,
                      baseline_multiplier=lambda f: f - 1.0)
    # Stage C parity with the forward: IVOC synthesised from POA.
    _add_ivoc_synthesis(delta, perturbation, baseline_sources, grid, indexer,
                        month, n_bins, diurnal_cfg, verbose,
                        baseline_multiplier=lambda f: f - 1.0)
    return delta


def build_perturbed_emissions(
    perturbation: Perturbation,
    baseline_sources: List[EmissionSource],
    grid,
    indexer: CellIndexer,
    month: int,
    n_bins: int = N_BINS,
    diurnal_cfg: Optional[DiurnalConfig] = None,
    verbose: bool = False,
) -> np.ndarray:
    """Construct the *full* perturbed emission vector for `zero-out` mode.

    e_perturbed = sum(factor × baseline_source) + sum(add_source).

    Returns (n_bins, N_SPECIES * N) in solver layout. Used by zero-out
    to feed a second forward run.
    """
    N = indexer.N
    e_old = np.zeros((n_bins, N_ORBIT_SPECIES * N), dtype=np.float64)

    for source in baseline_sources:
        basename = os.path.basename(source.path)
        factor = perturbation.factors.get(basename, 1.0)
        if abs(factor) < 1e-15:
            if verbose:
                print(f"  [e] zero {basename}")
            continue
        contrib = load_emissions_diurnal(
            [source], grid, indexer,
            diurnal_cfg=diurnal_cfg, month=month, n_bins=n_bins,
            verbose=verbose,
        )
        e_old += factor * contrib
        if verbose and abs(factor - 1.0) > 1e-15:
            print(f"  [e] scale {basename} by {factor:g}")

    for source in perturbation.add_sources:
        contrib = load_emissions_diurnal(
            [source], grid, indexer,
            diurnal_cfg=diurnal_cfg, month=month, n_bins=n_bins,
            verbose=verbose,
        )
        e_old += contrib
        if verbose:
            basename = os.path.basename(source.path)
            print(f"  [e] add {basename}")

    e = _rebroadcast_6_to_9(e_old, n_bins, N)
    # VOC sources contribute nothing above (slot 0 zeroed); route their full
    # (factor-scaled) mass to the 5 VBS bins, as the forward does.
    _route_voc_to_vbs(e, perturbation, baseline_sources, grid, indexer,
                      month, n_bins, diurnal_cfg, verbose,
                      baseline_multiplier=lambda f: f)
    # Stage C parity with the forward: IVOC synthesised from POA.
    _add_ivoc_synthesis(e, perturbation, baseline_sources, grid, indexer,
                        month, n_bins, diurnal_cfg, verbose,
                        baseline_multiplier=lambda f: f)
    return e


# ----------------------------------------------------------------------
# Phase 4c: cross-mode sign reconciliation.
#
# marginal saves δc, δPM25 in `perturbation_response` form
# (factor=0.99 → δc < 0). zero-out saves them in
# `baseline_minus_perturbed` form (factor=0.5 → δc > 0). For any
# comparison or aggregation across the two modes, callers should
# normalise to a single convention via `unify_sign`.
# ----------------------------------------------------------------------

_SIGN_FIELDS = (
    "delta_c_orbit", "delta_c_mean",
    "delta_pm25_orbit", "delta_pm25_mean",
    "delta_f_no3_eq_3d", "delta_f_nh4_eq_3d",
)

_VALID_SIGN_CONVENTIONS = ("perturbation_response", "baseline_minus_perturbed")


def unify_sign(
    npz_path: str,
    target: str = "perturbation_response",
) -> Dict[str, np.ndarray]:
    """Load a mode-output NPZ and return its δ-fields in `target` convention.

    Parameters
    ----------
    npz_path : path to a marginal/ or zeroout/ NPZ.
    target : either ``"perturbation_response"`` (marginal's native;
        downward perturbation → δ < 0) or ``"baseline_minus_perturbed"``
        (zero-out's native; downward perturbation → δ > 0).

    Returns
    -------
    dict mapping each present δ-field name to its sign-flipped (or
    sign-aligned) array. Non-δ fields are not returned — load the NPZ
    directly for those.
    """
    if target not in _VALID_SIGN_CONVENTIONS:
        raise ValueError(
            f"target must be one of {_VALID_SIGN_CONVENTIONS}, got {target!r}"
        )
    data = np.load(npz_path, allow_pickle=False)
    if "sign_convention" not in data.files:
        raise ValueError(
            f"{npz_path} lacks `sign_convention` — likely a pre-Phase-4c "
            f"NPZ. Re-run the mode driver to refresh."
        )
    src = str(data["sign_convention"])
    if src not in _VALID_SIGN_CONVENTIONS:
        raise ValueError(
            f"{npz_path} has unknown sign_convention={src!r}; "
            f"expected one of {_VALID_SIGN_CONVENTIONS}"
        )
    flip = (src != target)
    out: Dict[str, np.ndarray] = {}
    for k in _SIGN_FIELDS:
        if k in data.files:
            arr = np.asarray(data[k])
            out[k] = -arr if flip else arr.copy()
    return out
