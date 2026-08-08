"""Diurnal periodic orbit solver for South Asia (SAS) 2022.

For each month:
  1. Load 8 bin grids (UTC 3h bins) + MERRA2 constants
  2. Assemble 8 sets of species operators
  3. UMFPACK symbolic factorization (shared across all 48 matrices)
  4. Load monthly emissions
  5. Solve periodic orbit for all 6 species via GMRES
  6. Extract PM2.5 per bin, compute orbit mean
  7. Save orbit NPZ

Output:
  outputs/sas/orbit/orbit_M{MM}.npz

Usage:
  python scripts/run_orbit.py                  # all months with data
  python scripts/run_orbit.py --month 2        # February only
  python scripts/run_orbit.py --resume         # skip existing output; auto-resume un-done months
  python scripts/run_orbit.py --month 1 \
      --resume-from outputs/sas/orbit/orbit_M01.npz \
      --chemistry-iters 12                           # extend prior solve to 12 iters

Checkpoint policy:
  Every output NPZ written by this script — both the final
  orbit_M{MM}.npz and each per-iter orbit_M{MM}_iter{NN}.npz — carries
  the full resume schema (c_orbit + orbit_species_order + iter +
  conv_*).  Do not rm per-iter checkpoints by hand to 'protect' a new
  run: --resume-from gives you the explicit control you need.
"""

import argparse
import os
import resource
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.stdout.reconfigure(line_buffering=True)

import numpy as np

from orbit.core.grid_data import load_grid
from orbit.core.indexing import CellIndexer
from orbit.core.operator import assemble_species_operators
from orbit.core.solve import (
    umfpack_symbolic, umfpack_free_symbolic, _HAS_UMFPACK,
    compute_metis_ordering, _HAS_METIS,
)
from orbit.core.orbit import (
    solve_orbit_all_species, extract_pm25_isorropia, compare_partitioning_gc,
    N_BINS, DTAU,
)
from orbit.core.deposition import (
    N_SPECIES, IDX_SOA, IDX_PM25, IDX_POA, IDX_TOTAL_NH, IDX_SO2, IDX_NOX,
    IDX_PSO4, IDX_TOTAL_NO3, IDX_O3,
)
from orbit.core.dcomp import (
    compute_chemistry_per_bin, chemistry_diagnostics,
    splice_oh_into_chem, rebuild_rates_from_chem,
)
from orbit.core.dcomp_iter import (
    ConvergenceHistory, build_iteration_metrics, print_metrics,
    plausibility_check, check_invariants, under_relax_oxidants,
    AndersonAccelerator, DEFAULT_TOL,
)
from orbit.core.dcomp_isorropia import (
    update_grid_partitioning, partitioning_summary,
)
from orbit.core.dcomp_vbs import (
    update_vbs_partitioning,
)
from orbit.core.photolysis import PhotolysisLUT
from orbit.core.soa_photolysis import load_photolysis_lut, attach_jno2_to_grid, get_a_photo
from orbit.hemco import load_hemco_climatology
from orbit.emissions.loader import (
    load_emissions_diurnal, DiurnalConfig,
)
from orbit.emissions.manifest import (
    load_manifest, source_mass_budget, format_mass_budget,
)
from orbit.runlog import RunRecord, RssSampler
from orbit.core.deposition_maps import (
    compute_deposition_maps, summarise as _summarise_deposition,
)

# ── Cluster paths ──────────────────────────────────────────────────────────
# PREPROC_DIR + filename year-tag are env-var configurable so the same
# script drives 2016, 2016-with-CMFMC, 2022-met, etc. without code edits.
# Defaults preserve the historical behaviour (2016 met, MONTHLY_SAS/2016 dir).
PREPROC_DIR = os.environ.get(
    "ORBIT_PREPROC_DIR",
    "/path/to/data/inputs/grids_2022",
)
PREPROC_YEAR_TAG = os.environ.get("ORBIT_PREPROC_YEAR_TAG", "2022")
CONSTANTS = os.environ.get(
    "ORBIT_CONSTANTS",
    "/path/to/data/inputs/MERRA2.20150101.CN.05x0625.nc4",
)
EMISSION_DIR = os.environ.get(
    "ORBIT_EMISSION_DIR",
    "/path/to/data/emissions/sas",
)
OUTPUT_DIR = os.environ.get(
    "ORBIT_OUTPUT_DIR",
    os.path.join(os.path.dirname(__file__), "..", "outputs", "sas", "orbit"),
)


def _check_input_paths(lut_path=None):
    """Fail early and legibly when a path is still the placeholder default.

    The "/path/to/data/..." defaults are documentation, not usable paths. Left
    unset, ORBIT_CONSTANTS used to surface ~23 minutes of grid loading later as
    a bare netCDF4 FileNotFoundError deep in load_frland, with no hint that an
    environment variable was the cause. Say so at startup instead.

    When ``lut_path`` is given, the ISORROPIA LUT path gets the same
    treatment: a placeholder default means ORBIT_LUT was never set, and
    proceeding would silently skip the ISORROPIA closure. The explicit
    opt-out ``--lut none`` is exempt.
    """
    for var, path in (("ORBIT_PREPROC_DIR", PREPROC_DIR),
                      ("ORBIT_CONSTANTS", CONSTANTS),
                      ("ORBIT_EMISSION_DIR", EMISSION_DIR)):
        if path.startswith("/path/to/data"):
            raise SystemExit(
                f"{var} is unset, so it fell back to the placeholder default\n"
                f"  {path}\n"
                f"which is not a real path. Set {var} to the location of the\n"
                f"data on this machine (see the README's data section)."
            )
        if not os.path.exists(path):
            raise SystemExit(f"{var} points at a missing path:\n  {path}")
    if (lut_path is not None and str(lut_path).lower() != "none"
            and str(lut_path).startswith("/path/to/data")):
        raise SystemExit(
            "ORBIT_LUT is unset, so the ISORROPIA LUT path fell back to the "
            "placeholder default\n"
            f"  {lut_path}\n"
            "which is not a real path. Set ORBIT_LUT (or pass --lut) to the\n"
            "LUT location on this machine (see the README's data section),\n"
            "or pass --lut none to run without the ISORROPIA closure."
        )

# The baseline inventory is described by an emission manifest (a YAML index
# of which files make up a run and how to interpret each). The shipped default
# reproduces the published South Asia 2022 inventory; override with
# --emissions-manifest or ORBIT_EMISSION_MANIFEST. See orbit/emissions/manifest.py.
_MANIFEST = None
_MANIFEST_PATH_OVERRIDE = None
_ALLOW_MISSING_EMISSIONS = False
_RECORD = RunRecord()          # populated in main(); logging + JSON sidecar
_RSS = RssSampler()            # phase-attributed peak RSS (D3, 2026-08-01)
_VERIFY_INPUTS = False
_EMISSION_BUDGET = True


def _default_diurnal_config_path() -> str:
    """The production diurnal emission profiles shipped with ORBIT."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "data", "diurnal_sas.yaml")


def get_manifest():
    """The active emission manifest, loaded on first use."""
    global _MANIFEST
    if _MANIFEST is None:
        _MANIFEST = load_manifest(_MANIFEST_PATH_OVERRIDE)
    return _MANIFEST


def _emission_basenames():
    """Basenames in the active manifest, for provenance and reporting."""
    return [e.file for e in get_manifest().entries]


SPECIES_NAMES = ["VBS_C100", "PM2.5", "NH", "SO2", "NOx", "pSO4", "TotalNO3",
                 "O3", "CO", "VBS_C10", "VBS_C1", "VBS_C01", "VBS_C1000",
                 "POA"]
assert len(SPECIES_NAMES) == N_SPECIES, (
    f"SPECIES_NAMES has {len(SPECIES_NAMES)} entries, expected {N_SPECIES}"
)
MONTH_ABBR = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# Named diagnostic cells — surface, bin 3 (noon) trajectories are printed
# per outer iter and surface slabs are persisted in each iter checkpoint,
# so per-iter drift at specific cells is directly observable from the
# SLURM log and (post-hoc) from checkpoint files.  Indices are for the
# SAS 0.5°×0.625° grid (lat 4–39°N, lon 60–100°E; ny=71, nx=65).
NAMED_CELLS = [
    # (label, j, i) — nominal lat/lon in comment
    ("Delhi",      49, 28),  # 28.6N, 77.5E — polluted urban, IGP
    ("Kanpur",     45, 32),  # 26.5N, 80.0E — IGP, downwind Delhi
    ("Kolkata",    37, 45),  # 22.5N, 88.1E — coastal, east IGP
    ("Arabian-Sea", 22,  8),  # 15.0N, 65.0E — low-NOx reference (marine)
    ("Andaman",    16, 52),  # 12.0N, 92.5E — low-NOx reference (marine)
]


def _preproc_path(month, bin_idx):
    """Path to monthly bin preprocessor file. month: 1-12, bin_idx: 1-8.

    The year tag in the filename comes from ORBIT_PREPROC_YEAR_TAG (default
    '2016'), and the directory from ORBIT_PREPROC_DIR. Both default to the
    historical 2016-met baseline location for backward compatibility.
    """
    return os.path.join(
        PREPROC_DIR,
        f"sas_{PREPROC_YEAR_TAG}_M{month:02d}_B{bin_idx:02d}.nc",
    )


# ── Phase 4b: baseline-config hashing ────────────────────────────────
#
# A forward NPZ is the input to marginal/zero-out. If the user re-runs
# either against an old baseline that doesn't match the current
# emissions list / closure config / code revision, the result is
# silently inconsistent. These three short hashes are persisted in
# every forward NPZ; mode drivers compare and refuse on mismatch.

_HASH_PREFIX_LEN = 16


_EMISSIONS_LOADER_VERSION = "v2-perbin"
# v1 (pre-2026-04-30): legacy ``load_emissions`` bin-flat path collapsed
# bin_axis=True sources (CAMS soil NOx) to bin 0, while perturbed forwards
# used ``load_emissions_diurnal`` and saw the native 8 UTC slabs. Version
# bump forces baseline-hash mismatch so stale (v1) baselines cannot be
# silently reused by zero-out or marginal modes after the fix.

_ISO_CROSS_PARTIAL_DEFINITION = "marg_clipped"
# "eq" (pre-2026-05-01): cross-partials f_*_d* were computed as the
# equilibrium partitioning's cross-derivative ∂f_eq/∂c_other × c_self.
# That's wrong for the iso-coupled marg K-blocks because the operator
# builds with f_marg, so the linearization needs ∂f_marg/∂c_other × c_self.
# "marg" (2026-05-01): nested 2nd-order LUT FD probes computed
# f_marg at perturbed c_other and differenced. Right formula but
# applied to UNCLIPPED f_marg, while the operator uses CLIPPED
# f_marg. At regime-edge cells (Delhi winter sulfate-saturated, ~5/8
# bins) clipped derivative is exactly 0 but unclipped is non-zero;
# K-block over-predicted cross-coupling there, driving the empirical
# 50% marg/zo gap.
# "marg_clipped" (post-2026-05-01 fix): clipped f_marg at each
# c_other ± δ slice via the same symmetric ±δ averaged + clipped
# scheme as the diagonal probe. Cross-derivative is ∂(clipped
# f_marg)/∂c_other, consistent with the operator's clipped f_marg.
# Falsifying diagnostic: R_zo dropped 67× at Delhi when applying
# clipped K-block to existing zo data.
# Threaded into the closure_settings_hash so "marg" / "eq" baselines hash
# differently from "marg" baselines.


def _baseline_emissions_hash(emission_files, month):
    """Stable hex digest of (sorted basenames, month, loader version).

    Hashing basenames only (no absolute path) keeps baselines portable
    across machines that mount the same emissions directory at different
    paths.
    """
    import hashlib
    h = hashlib.sha256()
    for f in sorted(emission_files):
        h.update(os.path.basename(f).encode())
        h.update(b"\0")
    h.update(str(int(month)).encode())
    h.update(b"\0")
    h.update(_EMISSIONS_LOADER_VERSION.encode())
    # Cover the manifest's interpretation too (units, stack parameters, VOC
    # class, bin axis) so a baseline cannot be reused across manifests that
    # differ only in how the same filenames are read.
    h.update(b"\0")
    h.update(get_manifest().content_hash().encode())
    return h.hexdigest()[:_HASH_PREFIX_LEN]


def _closure_settings_hash(closure_alpha, closure_tol, isorropia_anderson,
                           basin_flip_damp, disable_night_nox,
                           isorropia_closure_iters):
    """Hex digest of closure tunables that change physics output.

    Excludes purely-mechanical settings (warm-start path, krylov tol).
    Includes _ISO_CROSS_PARTIAL_DEFINITION so "eq" baselines hash
    differently from "marg" baselines after the 2026-05-01 fix.
    """
    import hashlib
    h = hashlib.sha256()
    for v in (closure_alpha, closure_tol, isorropia_anderson,
              basin_flip_damp, disable_night_nox, isorropia_closure_iters):
        h.update(repr(v).encode())
        h.update(b"\0")
    h.update(_ISO_CROSS_PARTIAL_DEFINITION.encode())
    h.update(b"\0")
    return h.hexdigest()[:_HASH_PREFIX_LEN]


def _git_sha_short():
    """Return current HEAD's short SHA, or "unknown" if not a git repo."""
    import subprocess
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def _named_cell_iso_report(result, grids, iso_per_bin, nz, ny, nx,
                            prev_values, noon_tau=2, print_prefix="    "):
    """Per-iter named-cell diagnostic for the ISORROPIA-only Picard loop.

    Strips out DCOMP-only fields (OH, HO2, k_so2) since on
    feat/orbit-isorropia-archive those rates are archive-prescribed and
    don't change between iso iters. Reports f_NH4_eq, f_NO3_eq (LUT) and
    c_NH / c_pSO4 / c_TotalNO3 (which DO change as the operator updates),
    with Δ% vs the prior iter so partitioning drift at named cells is
    visible directly in the SLURM log.
    """
    from orbit.core.deposition import (
        IDX_TOTAL_NH as _I_NH,
        IDX_PSO4 as _I_SO4,
        IDX_TOTAL_NO3 as _I_NO3,
    )

    def _pct(new, old):
        if old is None or abs(old) < 1e-30:
            return "   —  "
        return f"{(new - old) / old * 100.0:+5.1f}%"

    print(f"{print_prefix}Named-cell surface diagnostics at noon (Bin {noon_tau + 1}):")
    print(f"{print_prefix}  {'cell':<12s} "
          f"{'f_NH4eq':<7s} {'Δ':<6s}  {'f_NO3eq':<7s} {'Δ':<6s}  "
          f"{'pSO4':<8s} {'Δ':<6s}  {'NH':<8s} {'Δ':<6s}  "
          f"{'NO3':<8s} {'Δ':<6s}")
    out_prev = dict(prev_values) if prev_values else {}
    for (label, j, i) in NAMED_CELLS:
        if not (0 <= j < ny and 0 <= i < nx):
            continue
        if iso_per_bin and noon_tau in iso_per_bin:
            fnh4_v = float(iso_per_bin[noon_tau]["f_nh_eq"][0, j, i])
            fno3_v = float(iso_per_bin[noon_tau]["f_no3_eq"][0, j, i])
        else:
            g = grids[noon_tau]
            fnh4_v = float(g.NHPartitioningEq[0, j, i]
                           if g.NHPartitioningEq.size > 0 else 0.0)
            fno3_v = float(g.NO3PartitioningEq[0, j, i]
                           if g.NO3PartitioningEq.size > 0 else 0.0)
        flat = j * nx + i
        pso4_v = float(result["orbits"][_I_SO4][noon_tau + 1][flat])
        nh_v = float(result["orbits"][_I_NH][noon_tau + 1][flat])
        no3_v = float(result["orbits"][_I_NO3][noon_tau + 1][flat])
        prev = out_prev.get(label, {})
        print(f"{print_prefix}  {label:<12s} "
              f"{fnh4_v:7.4f} {_pct(fnh4_v, prev.get('fnh4')):>6s}  "
              f"{fno3_v:7.4f} {_pct(fno3_v, prev.get('fno3')):>6s}  "
              f"{pso4_v:8.3f} {_pct(pso4_v, prev.get('pso4')):>6s}  "
              f"{nh_v:8.3f} {_pct(nh_v, prev.get('nh')):>6s}  "
              f"{no3_v:8.3f} {_pct(no3_v, prev.get('no3')):>6s}")
        out_prev[label] = {"fnh4": fnh4_v, "fno3": fno3_v,
                           "pso4": pso4_v, "nh": nh_v, "no3": no3_v}
    return out_prev


def _named_cell_report(chem, result, grids, iso_per_bin, nz, ny, nx,
                        prev_values, noon_tau=2, print_prefix="    "):
    """Per-iter diagnostic line at named cells (surface, noon bin).

    Extracts OH / HO2 / f_NO2 / f_nh4_eq / f_no3_eq / k_so2 / c_pSO4 /
    c_NH / c_NO / c_O3 at each NAMED_CELLS location, surface layer, at the
    noon bin (default tau=2 = Bin 3 in 1-indexed display).  Computes Δ%
    vs the prior iter's values (stored in ``prev_values``, which this
    function mutates) and prints a single compact line per cell.

    Returns the updated ``prev_values`` dict so the caller can feed it
    back in next iter.  On iter 1 all Δ columns show as "   —  ".
    """
    def _pct(new, old):
        if old is None or abs(old) < 1e-30:
            return "   —  "
        d = (new - old) / old * 100.0
        return f"{d:+5.1f}%"

    noon_bin_label = noon_tau + 1  # 1-indexed for readability
    print(f"{print_prefix}Named-cell surface diagnostics at noon (Bin {noon_bin_label}):")
    header = (f"{print_prefix}  "
              f"{'cell':<12s} {'OH':<8s} {'Δ':<6s}  "
              f"{'HO2':<8s} {'Δ':<6s}  {'f_NH4eq':<7s} {'Δ':<6s}  "
              f"{'f_NO3eq':<7s} {'Δ':<6s}  {'k_so2':<8s} {'Δ':<6s}  "
              f"{'pSO4':<8s} {'Δ':<6s}  {'NH':<8s} {'Δ':<6s}  "
              f"{'NO3':<8s} {'Δ':<6s}")
    print(header)
    from orbit.core.deposition import (
        IDX_TOTAL_NH as _I_NH,
        IDX_PSO4 as _I_SO4,
        IDX_TOTAL_NO3 as _I_NO3,
    )

    out_prev = dict(prev_values) if prev_values else {}
    for (label, j, i) in NAMED_CELLS:
        if not (0 <= j < ny and 0 <= i < nx):
            continue
        ox = chem.oxidants[noon_tau]
        OH_v   = float(ox.OH[0, j, i])
        HO2_v  = float(ox.HO2[0, j, i])
        kso2_v = float(chem.k_so2_rate[noon_tau][0, j, i])
        if iso_per_bin and noon_tau in iso_per_bin:
            fnh4_v = float(iso_per_bin[noon_tau]["f_nh_eq"][0, j, i])
            fno3_v = float(iso_per_bin[noon_tau]["f_no3_eq"][0, j, i])
        else:
            g = grids[noon_tau]
            fnh4_v = float(g.NHPartitioningEq[0, j, i]
                           if g.NHPartitioningEq.size > 0
                           else g.NHPartitioning[0, j, i])
            fno3_v = float(g.NO3PartitioningEq[0, j, i]
                           if g.NO3PartitioningEq.size > 0
                           else (g.NO3Partitioning[0, j, i]
                                 if g.NO3Partitioning.size > 0 else 0.0))
        # Orbit concentrations: reshape (N,) slice to (nz, ny, nx)
        flat_idx = 0 * (ny * nx) + j * nx + i  # k=0 surface
        def _c(species_idx):
            return float(result["orbits"][species_idx][noon_tau + 1][flat_idx])
        pso4_v = _c(_I_SO4)
        nh_v   = _c(_I_NH)
        no3_v  = _c(_I_NO3)

        prev = out_prev.get(label, {})
        line = (f"{print_prefix}  "
                f"{label:<12s} "
                f"{OH_v:8.2e} {_pct(OH_v, prev.get('OH')):>6s}  "
                f"{HO2_v:8.2e} {_pct(HO2_v, prev.get('HO2')):>6s}  "
                f"{fnh4_v:7.4f} {_pct(fnh4_v, prev.get('fnh4')):>6s}  "
                f"{fno3_v:7.4f} {_pct(fno3_v, prev.get('fno3')):>6s}  "
                f"{kso2_v:8.2e} {_pct(kso2_v, prev.get('kso2')):>6s}  "
                f"{pso4_v:8.3f} {_pct(pso4_v, prev.get('pso4')):>6s}  "
                f"{nh_v:8.3f} {_pct(nh_v, prev.get('nh')):>6s}  "
                f"{no3_v:8.3f} {_pct(no3_v, prev.get('no3')):>6s}")
        print(line)
        out_prev[label] = {
            "OH": OH_v, "HO2": HO2_v, "fnh4": fnh4_v, "fno3": fno3_v,
            "kso2": kso2_v, "pso4": pso4_v, "nh": nh_v, "no3": no3_v,
        }
    return out_prev


# Elevated-source assignments. Each entry binds a filename to physical
# stack parameters (height, diameter, exit temperature, exit velocity)
# that ASME (1973) plume rise consumes. The model computes the
# effective injection layer per cell from these scalars + local met
# (stability, wind, T) — see orbit/emissions/plumerise.py.
#
# Configurable via env var ORBIT_ELEVATED_SOURCES, format:
#   "<filename>:H,D,T,V;<filename>:H,D,T,V;..."
# where H=stack_height [m], D=stack_diam [m], T=stack_temp [K],
# V=stack_vel [m/s]. Or pass `H` alone to use the dataclass default
# diam/temp/vel (0,0,0 → no rise, just lookup at H).
#
# Default (empty): no elevated sources — backward compatibility.
#
# Typical Indian coal plant params (NTPC Singrauli / Korba / Talcher):
#   stack_height = 220-275 m       (taller for newer / Super Thermal units)
#   stack_diam   = 6-9 m           (large flue stacks for ~500-660 MW units)
#   stack_temp   = 380-420 K       (post-FGD; pre-FGD ~430-450 K)
#   stack_vel    = 18-25 m/s       (typical exit velocity)
# Conservative defaults: 250, 7, 400, 20.
def _parse_elevated_spec(spec):
    out = {}
    if not spec:
        return out
    for entry in spec.split(";"):
        entry = entry.strip()
        if not entry or ":" not in entry:
            continue
        fn, params = entry.split(":", 1)
        parts = [p.strip() for p in params.split(",")]
        try:
            H = float(parts[0])
            D = float(parts[1]) if len(parts) > 1 else 0.0
            T = float(parts[2]) if len(parts) > 2 else 0.0
            V = float(parts[3]) if len(parts) > 3 else 0.0
            out[fn.strip()] = (H, D, T, V)
        except (ValueError, IndexError):
            continue
    return out


# Retired: stack parameters now live in the emission manifest, per source.
# _parse_elevated_spec is kept because the manifest's stack block and this
# legacy string form share the same (height, diam, temp, vel) semantics.


# Per-source VOC parent class for the VBS yield-at-emission scheme.
# Each VOC emission file is assigned to a parent class; chamber-derived stoichiometric yields
# (orbit.emissions.vbs_yields.VBS_PARENT_YIELDS) then distribute its
# emitted mass across the 5 VBS bins. Files NOT listed here have no
# voc_parent_class and contribute nothing to SoA — important so that
# legacy "TotalOrg" mass doesn't accidentally land in the C*=100 bin.
#
# Configurable via env var ORBIT_VOC_SOURCES, format:
#   "<filename>:<class>;<filename>:<class>;..."
# where <class> is one of: anthro_high_nox, anthro_low_nox, anthro
# (auto-switching), bio_monoterpene, bio_isoprene, biomass_burning, ivoc.
#
# When <class> = "anthro", cell-dependent NOx-regime switching kicks in:
# yields_eff = F_HIGH_NOX × yields_high_nox + (1-F) × yields_low_nox
# with F_HIGH_NOX from a sigmoid on archive [NO2]/[OH] (see
# _compute_nox_regime below).
#
# Default (empty env var): no SoA emissions — domain-mean SoA = 0. The
# user must opt in explicitly by listing the desired VOC sources.
def _parse_voc_parent_classes(spec):
    out = {}
    if not spec:
        return out
    for entry in spec.split(";"):
        entry = entry.strip()
        if not entry or ":" not in entry:
            continue
        fn, cls = entry.split(":", 1)
        out[fn.strip()] = cls.strip()
    return out


# Retired: VOC parent classes now live in the emission manifest, per source.

# VOC→VBS distribution is shared with the marginal/zero-out perturbation
# builder — single source of truth in orbit/emissions/vbs_distribution.py
# (extracted 2026-05-20 so a VOC perturbation routes mass to the VBS bins
# exactly as the forward does). Aliased to the historical underscore names so
# the rest of this module is unchanged.
from orbit.emissions.vbs_distribution import (
    compute_nox_regime as _compute_nox_regime,
    bin_indices_in_solver_layout as _bin_indices_in_solver_layout,
    distribute_voc_to_vbs_bins as _distribute_voc_to_vbs_bins,
)

# IVOC scaling factor (Robinson 2007 central = 1.5; range 0.5–4.5).
_IVOC_SCALING = float(os.environ.get("ORBIT_IVOC_SCALING", "1.5"))


def _extract_vbs_c_mean_surface(c_mean, indexer):
    """Pull the 5 VBS bin surface fields from c_mean.

    Returns (5, ny, nx) in low-to-high C* order matching IDX_VBS_BINS.
    """
    from orbit.core.deposition import IDX_VBS_BINS
    nz, ny, nx = indexer.nz, indexer.ny, indexer.nx
    out = np.zeros((5, ny, nx), dtype=np.float64)
    for i_bin, s in enumerate(IDX_VBS_BINS):
        c_3d = c_mean[s].reshape(nz, ny, nx)
        out[i_bin] = c_3d[0]   # surface only
    return out


def _compute_soa_mean_surface(c_mean, indexer, grid):
    """Total SoA particle mass at the surface = Σ_i F_p,i × C_i.

    Reads grid.F_p_vbs_surface (set by update_vbs_partitioning); falls
    back to all-particle if absent (regression-test path).
    """
    c_vbs = _extract_vbs_c_mean_surface(c_mean, indexer)   # (5, ny, nx)
    F_p = getattr(grid, "F_p_vbs_surface", None)
    if F_p is None or F_p.shape != c_vbs.shape:
        # Fallback: pure particle (Variant A semantics)
        return c_vbs.sum(axis=0)
    return (F_p * c_vbs).sum(axis=0)


def _soa_3d_flat(get_bin, grid, N):
    """F_p-weighted SoA particle mass (flat, shape (N,)) = Σ_i F_p,i × C_i over
    the 5 VBS bins — the canonical SoA that enters PM2.5, matching
    _compute_soa_mean_surface but at all levels.

    ``get_bin(species_idx)`` returns that bin's total (gas+particle) conc as a
    flat (N,) array. Uses grid.F_p_vbs (5, nz, ny, nx). Falls back to the C100
    bin alone (legacy Variant-A single-tracer SoA) when F_p_vbs is unavailable
    — e.g. the pre-closure baseline PM2.5, before the VBS Pankow closure runs.
    """
    from orbit.core.deposition import IDX_VBS_BINS, IDX_SOA
    F_p = getattr(grid, "F_p_vbs", None)
    if F_p is None or F_p.shape[0] != len(IDX_VBS_BINS):
        return get_bin(IDX_SOA)
    soa = np.zeros(N, dtype=np.float64)
    for i, s in enumerate(IDX_VBS_BINS):
        soa = soa + F_p[i].ravel() * get_bin(s)
    return soa


def _build_emission_sources(month, verbose=True):
    """Build the EmissionSource list for a month from the active manifest.

    Missing required files raise MissingEmissionsError unless
    --allow-missing-emissions was passed: a run with part of the inventory
    silently absent would produce a plausible but wrong answer.
    """
    manifest = get_manifest()
    sources, report = manifest.build_sources(
        EMISSION_DIR, month, allow_missing=_ALLOW_MISSING_EMISSIONS,
    )
    if verbose:
        print(manifest.describe(report))
        print(f"  directory: {EMISSION_DIR}")
        _RECORD.emissions["manifest"] = manifest.name
        _RECORD.emissions["manifest_hash"] = manifest.content_hash()
        _RECORD.emissions["loaded"] = list(report.loaded)
        _RECORD.emissions["missing_required"] = list(report.missing_required)
        if report.missing_required:
            print("  WARNING: proceeding with REQUIRED emissions missing "
                  "(--allow-missing-emissions)")
    return sources


_VERSION = "1.0.0"


def _record_configuration(args, months):
    """Populate the run record, splitting knobs by whether they move numbers.

    The split is the point: a reader must be able to tell at a glance which
    settings changed the answer and which only changed the speed. Anything
    that affects results belongs in the first table -- that is a maintenance
    rule, not just formatting.
    """
    def src(flag, is_set):
        return flag if is_set else "default"

    R = _RECORD
    R.set_config("months", months, True, source="--month" if args.month else "default")
    R.set_config("horizontal transport",
                 "FCT (van Leer-MUSCL + Zalesak)" if args.horizontal_fct
                 else "first-order Patankar exponential",
                 True, src("--horizontal-fct", args.horizontal_fct))
    R.set_config("inorganic closure",
                 f"{'Anderson' if args.isorropia_anderson else 'Picard'}, "
                 f"{args.isorropia_closure_iters} iters",
                 True, src("--isorropia-anderson", args.isorropia_anderson))
    R.set_config("closure alpha/tol", f"{args.closure_alpha} / {args.closure_tol}",
                 True, "default")
    R.set_config("chemistry iterations",
                 f"{args.chemistry_iters}"
                 + (" (prescribed oxidants; O3/CO inert)"
                    if args.chemistry_iters == 0 else ""),
                 True, src("--chemistry-iters", args.chemistry_iters != 0))
    R.set_config("ISORROPIA LUT", os.path.basename(str(args.lut)), True, "--lut")
    for label, env, default in (
        ("VBS k_age", "ORBIT_VBS_K_AGE", "4.0e-11"),
        ("VBS fragmentation", "ORBIT_VBS_FRAG", "0.75"),
        ("VBS photolysis A", "ORBIT_VBS_A_PHOTO", "0"),
        ("IVOC scaling", "ORBIT_IVOC_SCALING", "1.5"),
    ):
        val = os.environ.get(env)
        shown = val if val is not None else default
        # The photolytic sink needs a TUV LUT that is absent by default, so
        # report whether it will actually ACT rather than what was asked for.
        # The banner previously said "4.0e-4" on a run whose sink was silently
        # disabled by a missing LUT, producing output identical to no-sink.
        if env == "ORBIT_VBS_A_PHOTO" and val is not None and float(val) > 0:
            from orbit.core.soa_photolysis import _DEFAULT_LUT_PATH
            lut_p = os.environ.get("ORBIT_PHOTOLYSIS_LUT", _DEFAULT_LUT_PATH)
            if not os.path.exists(lut_p):
                shown = f"{val} (INACTIVE: TUV LUT missing at {lut_p})"
        R.set_config(label, shown, True,
                     env if val is not None else "default")
    R.set_config("emission manifest", get_manifest().name, True,
                 "--emissions-manifest" if _MANIFEST_PATH_OVERRIDE else "default")

    # performance only
    R.set_config("LU backend pref", os.environ.get("ORBIT_LU_BACKEND", "auto"),
                 False, "ORBIT_LU_BACKEND" if "ORBIT_LU_BACKEND" in os.environ
                 else "default")
    R.set_config("UMFPACK available", _HAS_UMFPACK, False, "detected")
    R.set_config("pymetis available", _HAS_METIS, False, "detected")
    R.set_config("species threads", os.environ.get("ORBIT_SPECIES_THREADS", "1"),
                 False, "ORBIT_SPECIES_THREADS")
    R.set_config("factor threads", os.environ.get("ORBIT_FACTOR_THREADS", "1"),
                 False, "ORBIT_FACTOR_THREADS")
    R.set_config("execution model",
                 "dataflow" if os.environ.get("ORBIT_DATAFLOW") == "1"
                 else "wave-barrier", False, "ORBIT_DATAFLOW")
    R.set_config("bins / dtau", f"{N_BINS} / {DTAU:.0f}s", False, "fixed")
    R.set_config("output dir", os.path.abspath(OUTPUT_DIR), False,
                 "ORBIT_OUTPUT_DIR")


def _record_inputs(args):
    """Fingerprint the inputs so the log identifies the data, not just paths."""
    R = _RECORD
    if args.lut and os.path.exists(str(args.lut)):
        R.set_input("ISORROPIA", str(args.lut), full=_VERIFY_INPUTS)
    if os.path.exists(CONSTANTS):
        R.set_input("constants", CONSTANTS, full=_VERIFY_INPUTS)
    R.inputs["grids"] = {"path": PREPROC_DIR,
                         "year_tag": PREPROC_YEAR_TAG,
                         "method": "directory"}
    R.inputs["emissions"] = {"path": EMISSION_DIR,
                             "manifest": get_manifest().name,
                             "manifest_hash": get_manifest().content_hash(),
                             "method": "directory"}


def _check_solver_capabilities():
    """Warn when the solver is running in a slower/heavier configuration.

    The default (auto) backend is UMFPACK with a pymetis nested-dissection
    ordering; missing either dependency degrades to a slower, heavier
    configuration, so both draw a warning that names the cost and the fix.
    """
    pref = os.environ.get("ORBIT_LU_BACKEND", "auto").lower()
    if pref in ("auto", "umfpack") and not _HAS_UMFPACK:
        _RECORD.warn(
            "solver",
            "UMFPACK not available; falling back to SuperLU.",
            impact="factorisation is slower and uses roughly 1.5-2x more "
                   "memory; a forward month already peaks near 17 GB, so this "
                   "may exhaust a 20-24 GB machine.",
            fix="conda install -c conda-forge scikit-umfpack "
                "(or pip install -e \".[fast]\")",
            silence="set ORBIT_LU_BACKEND=superlu to choose it deliberately.",
        )
    if pref == "auto" and _HAS_UMFPACK and not _HAS_METIS:
        _RECORD.warn(
            "solver",
            "pymetis not available; auto backend falls back to "
            "UMFPACK+COLAMD.",
            impact="roughly 1.3x slower and 25% more peak memory than "
                   "UMFPACK+METIS (January: 28.2 vs 21.3 min, 15.0 vs "
                   "12.0 GB).",
            fix='pip install -e ".[metis]"',
            silence="set ORBIT_LU_BACKEND=umfpack to choose COLAMD "
                    "deliberately.",
        )
    if pref in ("superlu_metis", "umfpack_metis") and not _HAS_METIS:
        _RECORD.warn(
            "solver",
            f"ORBIT_LU_BACKEND={pref} requested but pymetis is unavailable.",
            impact="the run will fail; METIS orderings need pymetis.",
            fix='pip install -e ".[metis]"',
            silence="unset ORBIT_LU_BACKEND to use the default backend.",
        )


def _log_paths(month):
    """(log_path, json_path) for a month, next to the output NPZ."""
    base = os.path.join(OUTPUT_DIR, f"orbit_M{month:02d}")
    return base + ".log", base + ".run.json"


def _peak_rss_mb():
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return ru / 1024 / 1024
    return ru / 1024


def _fmt_time(seconds):
    if seconds >= 60:
        return f"{seconds / 60:.1f}m"
    return f"{seconds:.1f}s"


# ── Main processing ───────────────────────────────────────────────────────



def _pno3_pnh4_surf_per_bin(orbits, grids, N, nz, ny, nx):
    """Return (pNO3_surf, pNH4_surf), each (N_BINS, ny, nx) compound mass µg/m³.

    Used as the convergence target for the ISORROPIA-only Picard loop:
    these are the speciated outputs the paper makes claims about (NOx
    scenarios → pNO3, NH3 scenarios → pNH4), so the closure must
    converge each one independently — gating on Σ PM2.5 would let them
    cancel in the sum.
    """
    from orbit.core.constants import N_TO_NH4, N_TO_NO3
    from orbit.core.deposition import IDX_TOTAL_NH, IDX_TOTAL_NO3
    pno3 = np.zeros((N_BINS, ny, nx), dtype=np.float64)
    pnh4 = np.zeros((N_BINS, ny, nx), dtype=np.float64)
    for tau in range(N_BINS):
        g = grids[tau]
        c_nh = np.maximum(orbits[IDX_TOTAL_NH ][tau + 1], 0.0).reshape(nz, ny, nx)
        c_no3 = np.maximum(orbits[IDX_TOTAL_NO3][tau + 1], 0.0).reshape(nz, ny, nx)
        # Use equilibrium fractions (mass-extraction convention).
        p_nh = (g.NHPartitioningEq if g.NHPartitioningEq.size > 0
                else g.NHPartitioning)
        if g.NO3PartitioningEq.size > 0:
            p_no3 = g.NO3PartitioningEq
        elif g.NO3Partitioning.size > 0:
            p_no3 = g.NO3Partitioning
        else:
            p_no3 = np.ones((nz, ny, nx))   # baseline: all-particle proxy
        pnh4[tau] = (p_nh * c_nh * N_TO_NH4)[0]
        pno3[tau] = (p_no3 * c_no3 * N_TO_NO3)[0]
    return pno3, pnh4


class _BasinFlipDetector:
    """Detect per-cell basin-flipping in ISORROPIA partitioning across iso
    closure iterations and freeze flipping cells at the centroid of the
    bracketing basin values.

    Detection at iter k (k >= 4) requires the per-cell f_*_eq trajectory
    [c_{k-3}, c_{k-2}, c_{k-1}, c_k] to satisfy ALL of:

      (a) sign(c_{k-2} - c_{k-3}) != sign(c_{k-1} - c_{k-2})  [flip 1]
      (b) sign(c_{k-1} - c_{k-2}) != sign(c_k - c_{k-1})      [flip 2]
      (c) |d2| / |d1| in [min_ratio, max_ratio]               [stable amp]
      (d) |d3| / |d2| in [min_ratio, max_ratio]               [stable amp]
      (e) |d2| > min_delta                                    [non-trivial]

    Two consecutive flips with stable amplitude is the signature of a
    true LUT basin-flipping cell. Decaying oscillation (|d_k| / |d_{k-1}|
    < 0.9) and one-off Anderson overshoot-correction (single flip) are
    excluded. Tightened from the reviewer's [0.7, 1.3] to [0.9, 1.1]
    after M5 (jobs 7703707, 7705275) showed [0.7, 1.3] still admits
    decaying oscillation as the iteration converges and per-iter deltas
    naturally tighten in magnitude.

    Detection at iter 4 earliest. For 6-iter runs, eligible at iters 4,
    5, 6. Trade-off: a cell whose basin-flipping FIRST appears at iter 6
    gets at most one observable flip and won't be flagged — the user's
    M5 Andaman cell is borderline (oscillation onset iter 5). Worth it
    to keep the mask diagnostic-meaningful for the paper.

    Once flagged, the cell is frozen at 0.5*(c_{k-1} + c_k) for the rest
    of the closure and re-applied after the post-loop Final ISORROPIA
    pass and the ISORROPIA PM2.5 extraction.

    Both NO3 and NH4 are tracked independently. For each, the equilibrium
    fraction (used by mass extraction) and marginal fraction (used by
    operator deposition) are frozen at the same iter-(k-1, k) centroid —
    both come from the same LUT lookup so they flip together; freezing
    both keeps operator and extraction consistent at the centroid.
    """

    def __init__(self, n_bins, nz, ny, nx,
                 min_ratio=0.9, max_ratio=1.1, min_delta=1e-3):
        self.n_bins = n_bins
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.min_delta = min_delta
        shape = (n_bins, nz, ny, nx)
        self._hist_no3_eq = []
        self._hist_no3_marg = []
        self._hist_nh4_eq = []
        self._hist_nh4_marg = []
        self.flip_no3 = np.zeros(shape, dtype=bool)
        self.flip_nh4 = np.zeros(shape, dtype=bool)
        self.frozen_f_no3_eq = np.full(shape, np.nan, dtype=np.float32)
        self.frozen_f_no3_marg = np.full(shape, np.nan, dtype=np.float32)
        self.frozen_f_nh4_eq = np.full(shape, np.nan, dtype=np.float32)
        self.frozen_f_nh4_marg = np.full(shape, np.nan, dtype=np.float32)
        self.basin_low_no3 = np.full(shape, np.nan, dtype=np.float32)
        self.basin_high_no3 = np.full(shape, np.nan, dtype=np.float32)
        self.basin_low_nh4 = np.full(shape, np.nan, dtype=np.float32)
        self.basin_high_nh4 = np.full(shape, np.nan, dtype=np.float32)

    def update_and_freeze(self, new_iso, grids):
        """Append current iter's fractions, detect new flipping cells, and
        overwrite all flagged cells with the centroid in `new_iso` AND
        `grids` in place. Returns (n_new_no3, n_new_nh4)."""
        n_bins = self.n_bins
        f_no3_eq = np.stack(
            [new_iso[t]["f_no3_eq"] for t in range(n_bins)], axis=0)
        f_no3_marg = np.stack(
            [new_iso[t]["f_no3_marg"] for t in range(n_bins)], axis=0)
        f_nh4_eq = np.stack(
            [new_iso[t]["f_nh_eq"] for t in range(n_bins)], axis=0)
        f_nh4_marg = np.stack(
            [new_iso[t]["f_nh_marg"] for t in range(n_bins)], axis=0)

        self._hist_no3_eq.append(f_no3_eq.copy())
        self._hist_no3_marg.append(f_no3_marg.copy())
        self._hist_nh4_eq.append(f_nh4_eq.copy())
        self._hist_nh4_marg.append(f_nh4_marg.copy())

        counts = {"no3": 0, "nh4": 0}
        if len(self._hist_no3_eq) >= 4:
            for label in ("no3", "nh4"):
                hist_eq = (self._hist_no3_eq if label == "no3"
                           else self._hist_nh4_eq)
                hist_marg = (self._hist_no3_marg if label == "no3"
                             else self._hist_nh4_marg)
                flip_mask = (self.flip_no3 if label == "no3"
                             else self.flip_nh4)
                frozen_eq = (self.frozen_f_no3_eq if label == "no3"
                             else self.frozen_f_nh4_eq)
                frozen_marg = (self.frozen_f_no3_marg if label == "no3"
                               else self.frozen_f_nh4_marg)
                basin_lo = (self.basin_low_no3 if label == "no3"
                            else self.basin_low_nh4)
                basin_hi = (self.basin_high_no3 if label == "no3"
                            else self.basin_high_nh4)
                c_km3 = hist_eq[-4]
                c_km2 = hist_eq[-3]
                c_km1 = hist_eq[-2]
                c_k = hist_eq[-1]
                d1 = c_km2 - c_km3
                d2 = c_km1 - c_km2
                d3 = c_k - c_km1
                abs_d1 = np.abs(d1)
                abs_d2 = np.abs(d2)
                abs_d3 = np.abs(d3)
                # (a, b) Two consecutive sign flips required.
                flip_1 = (d1 * d2) < 0.0
                flip_2 = (d2 * d3) < 0.0
                # (c, d) Stable amplitude on both flips: ratios in
                #     [min_ratio, max_ratio] (default [0.9, 1.1]).
                #     Excludes decaying oscillation (ratio < 0.9) and the
                #     bulk Anderson overshoot-correction artefact.
                ratio_1 = abs_d2 / np.maximum(abs_d1, 1e-12)
                ratio_2 = abs_d3 / np.maximum(abs_d2, 1e-12)
                mag_1 = (ratio_1 >= self.min_ratio) & (ratio_1 <= self.max_ratio)
                mag_2 = (ratio_2 >= self.min_ratio) & (ratio_2 <= self.max_ratio)
                # (e) Non-trivial central delta: rules out cells where
                #     all three iterates are essentially equal (numerical
                #     noise can manufacture sign flips at zero amplitude).
                non_trivial = abs_d2 > self.min_delta
                new_flips = (flip_1 & flip_2 & mag_1 & mag_2
                             & non_trivial & ~flip_mask)
                counts[label] = int(new_flips.sum())
                flip_mask |= new_flips
                mid_eq = (0.5 * (c_km1 + c_k)).astype(np.float32)
                mid_marg = (0.5 * (hist_marg[-2]
                                   + hist_marg[-1])).astype(np.float32)
                lo = np.minimum(c_km1, c_k).astype(np.float32)
                hi = np.maximum(c_km1, c_k).astype(np.float32)
                frozen_eq[new_flips] = mid_eq[new_flips]
                frozen_marg[new_flips] = mid_marg[new_flips]
                basin_lo[new_flips] = lo[new_flips]
                basin_hi[new_flips] = hi[new_flips]

        if len(self._hist_no3_eq) > 4:
            self._hist_no3_eq = self._hist_no3_eq[-4:]
            self._hist_no3_marg = self._hist_no3_marg[-4:]
            self._hist_nh4_eq = self._hist_nh4_eq[-4:]
            self._hist_nh4_marg = self._hist_nh4_marg[-4:]

        for tau in range(n_bins):
            self._apply_to_bin(tau, new_iso, grids)
        return counts["no3"], counts["nh4"]

    def _apply_to_bin(self, tau, new_iso, grids):
        m_no3 = self.flip_no3[tau]
        if m_no3.any():
            new_iso[tau]["f_no3_eq"][m_no3] = self.frozen_f_no3_eq[tau][m_no3]
            new_iso[tau]["f_no3_marg"][m_no3] = self.frozen_f_no3_marg[tau][m_no3]
            grids[tau].NO3PartitioningEq[m_no3] = self.frozen_f_no3_eq[tau][m_no3]
            grids[tau].NO3Partitioning[m_no3] = self.frozen_f_no3_marg[tau][m_no3]
        m_nh4 = self.flip_nh4[tau]
        if m_nh4.any():
            new_iso[tau]["f_nh_eq"][m_nh4] = self.frozen_f_nh4_eq[tau][m_nh4]
            new_iso[tau]["f_nh_marg"][m_nh4] = self.frozen_f_nh4_marg[tau][m_nh4]
            grids[tau].NHPartitioningEq[m_nh4] = self.frozen_f_nh4_eq[tau][m_nh4]
            grids[tau].NHPartitioning[m_nh4] = self.frozen_f_nh4_marg[tau][m_nh4]

    def reapply_to_grids(self, grids):
        """Re-overwrite grids[tau].* at flagged cells. Called after the
        post-iso-loop Final ISORROPIA pass which would otherwise restore
        raw LUT values at frozen cells."""
        for tau in range(self.n_bins):
            m_no3 = self.flip_no3[tau]
            if m_no3.any():
                grids[tau].NO3PartitioningEq[m_no3] = self.frozen_f_no3_eq[tau][m_no3]
                grids[tau].NO3Partitioning[m_no3] = self.frozen_f_no3_marg[tau][m_no3]
            m_nh4 = self.flip_nh4[tau]
            if m_nh4.any():
                grids[tau].NHPartitioningEq[m_nh4] = self.frozen_f_nh4_eq[tau][m_nh4]
                grids[tau].NHPartitioning[m_nh4] = self.frozen_f_nh4_marg[tau][m_nh4]

    def total_frozen(self):
        return int(self.flip_no3.sum()), int(self.flip_nh4.sum())


def _pm25_per_bin(orbits, grids, N, nz, ny, nx):
    """Compute 4D PM2.5 (bin, k, j, i) from the orbit, using equilibrium
    partitioning when available and marginal otherwise.  Matches the
    extraction used in post-processing."""
    from orbit.core.constants import N_TO_NH4, N_TO_NO3, S_TO_SO4
    from orbit.core.deposition import (
        IDX_PM25, IDX_TOTAL_NH, IDX_PSO4, IDX_TOTAL_NO3,
    )
    out = np.zeros((N_BINS, nz, ny, nx), dtype=np.float64)
    for tau in range(N_BINS):
        g = grids[tau]
        # SoA = Σ_i F_p,i × C_i over the 5 VBS bins (canonical soa_mean), not
        # just the C100 bin (the old Variant-A single-tracer convention).
        c_soa  = _soa_3d_flat(lambda s: np.maximum(orbits[s][tau + 1], 0.0), g, N)
        c_pm   = np.maximum(orbits[IDX_PM25     ][tau + 1], 0.0)
        c_poa  = (np.maximum(orbits[IDX_POA][tau + 1], 0.0)
                  if len(orbits) > IDX_POA else 0.0)
        c_nh   = np.maximum(orbits[IDX_TOTAL_NH ][tau + 1], 0.0)
        c_pso4 = np.maximum(orbits[IDX_PSO4     ][tau + 1], 0.0)
        c_no3  = np.maximum(orbits[IDX_TOTAL_NO3][tau + 1], 0.0)

        p_nh = (g.NHPartitioningEq.ravel()
                if g.NHPartitioningEq.size > 0 else g.NHPartitioning.ravel())
        if g.NO3PartitioningEq.size > 0:
            p_no3 = g.NO3PartitioningEq.ravel()
        elif g.NO3Partitioning.size > 0:
            p_no3 = g.NO3Partitioning.ravel()
        else:
            p_no3 = np.ones(N)

        pm25 = (c_pm + c_poa + c_soa
                + p_nh * c_nh * N_TO_NH4
                + c_pso4 * S_TO_SO4
                + p_no3 * c_no3 * N_TO_NO3)
        out[tau] = pm25.reshape((nz, ny, nx))
    return out


def _assemble_and_solve(grids, indexer, emissions_SN, N,
                         umfpack_sym, perm, c_warm_SN, tol, maxiter,
                         k_so2_per_bin=None, k_nox_per_bin=None,
                         k_o3_loss_per_bin=None, s_o3_source_per_bin=None,
                         k_co_loss_per_bin=None,
                         verbose=True,
                         lu_cache=None, skip_solve_species=None,
                         cached_orbits=None, return_lu_cache=False,
                         keep_lu_species=None,
                         disable_night_nox=False,
                         cached_assembly=None,
                         rebuild_deposition_species=None,
                         return_assembly=False,
                         fct_enabled=False):
    """Helper: assemble N_SPECIES per-bin operators (optionally with chemistry rates)
    then solve the periodic orbit.

    If both k_so2_per_bin and k_nox_per_bin are None, reproduces the baseline
    prescribed-chemistry behaviour (grid.SO2oxidation only; NOx->TotalNO3
    inactive).  If provided, the per-bin override rates are passed to
    `assemble_species_operators` to activate the DCOMP chemistry couplings.

    Phase 3e: if ``k_o3_loss_per_bin`` is provided, O3 chemistry loss
    diagonal is installed in the operator and ``s_o3_source_per_bin`` is
    routed to the orbit RHS as an ``extra_rhs_per_bin_per_species`` entry
    for IDX_O3.

    lu_cache / skip_solve_species / cached_orbits / return_lu_cache are
    forwarded to ``solve_orbit_all_species`` for the Phase 3d unchanged-
    species short-circuit across outer iterations.

    cached_assembly / rebuild_deposition_species / return_assembly are the
    iso-loop partial-reassembly hooks (Opt: avoid rebuilding stable
    operators across iso iters on feat/orbit-isorropia-archive). When
    ``cached_assembly`` (a dict {'L_species_per_bin': [...], 'K_sources_per_bin':
    [...], 'D_per_bin': [...], 'T_per_bin': [...]}) is supplied alongside
    ``rebuild_deposition_species`` (e.g. {IDX_TOTAL_NH, IDX_TOTAL_NO3}),
    only those species' deposition matrices are re-assembled per bin and
    their L_species[s] = T + D[s] re-summed; the rest of the assembled
    state is reused verbatim. Saves the full N_SPECIES-wide deposition +
    transport assembly per iso iter (~30-60s wall on the SAS grid).
    With ``return_assembly=True`` the function adds ``"assembly"`` to its
    return dict carrying the cacheable state for the caller to pass back
    on the next invocation.
    """
    use_partial = (cached_assembly is not None
                   and rebuild_deposition_species is not None
                   and len(rebuild_deposition_species) > 0)
    if use_partial:
        from orbit.core.deposition import assemble_deposition as _assdep_inner
        # Partial-rebuild now handles K_age aging-loss diagonals for VBS
        # bins via cached_assembly["K_age_loss_per_bin"]. K_age is
        # partition-independent (depends on archive_OH and constant
        # k_age), so caching across iso iters is safe under the production
        # chemistry_iters=0 path. The footgun guard from the pre-K_age
        # implementation has been removed.
        #
        # Reuse cached L for stable species; rebuild deposition + L only
        # for species in rebuild_deposition_species. Cache must be
        # well-formed (L/K/D/T/K_age_loss for all 8 bins from prior call).
        L_species_per_bin = [list(L) for L in cached_assembly["L_species_per_bin"]]
        K_sources_per_bin = list(cached_assembly["K_sources_per_bin"])
        D_per_bin = [list(D) for D in cached_assembly["D_per_bin"]]
        T_per_bin = list(cached_assembly["T_per_bin"])
        K_age_loss_per_bin = list(cached_assembly.get("K_age_loss_per_bin",
                                                      [{}] * N_BINS))
        for tau in range(N_BINS):
            for s in rebuild_deposition_species:
                D_new = _assdep_inner(grids[tau], indexer, s)
                D_per_bin[tau][s] = D_new
                # L_s = T + D_s + (optional K_age_loss_s for VBS bins).
                # NH/TotalNO3 don't have K_age, so the addition is a no-op.
                # NOx→TotalNO3 and SO2→pSO4 rate matrices are diagonal-only
                # off-diagonals stored in K_sources and don't enter L.
                L_new = T_per_bin[tau] + D_new
                K_age_s = K_age_loss_per_bin[tau].get(s)
                if K_age_s is not None:
                    L_new = L_new + K_age_s
                L_species_per_bin[tau][s] = L_new
    else:
        L_species_per_bin = []
        K_sources_per_bin = []
        D_per_bin = []
        T_per_bin = []
        K_age_loss_per_bin = []
        for tau in range(N_BINS):
            ks = None if k_so2_per_bin is None else k_so2_per_bin[tau]
            kn = None if k_nox_per_bin is None else k_nox_per_bin[tau]
            # --disable-night-nox ablation: override the operator's internal
            # fallback by building the archive-driven rate here with the
            # nighttime channel zeroed. Only fires when caller hasn't passed
            # an explicit per-bin rate (i.e. default archive-baseline path).
            if kn is None and disable_night_nox:
                from orbit.core.nox_to_no3_rate import build_nox_to_no3_rate
                kn = build_nox_to_no3_rate(grids[tau], disable_night=True)
            ko3 = None if k_o3_loss_per_bin is None else k_o3_loss_per_bin[tau]
            kco = None if k_co_loss_per_bin is None else k_co_loss_per_bin[tau]
            if return_assembly:
                (L_species, K_sources, T_block, _d, D_list,
                 K_age_loss) = assemble_species_operators(
                    grids[tau], indexer, verbose=False, scheme="exp",
                    nox_to_no3_rate=kn, so2_ox_rate=ks, o3_loss_rate=ko3,
                    co_loss_rate=kco,
                    return_D_per_species=True,
                )
                D_per_bin.append(D_list)
                K_age_loss_per_bin.append(K_age_loss)
            else:
                L_species, K_sources, T_block, _d = assemble_species_operators(
                    grids[tau], indexer, verbose=False, scheme="exp",
                    nox_to_no3_rate=kn, so2_ox_rate=ks, o3_loss_rate=ko3,
                    co_loss_rate=kco,
                )
            L_species_per_bin.append(L_species)
            K_sources_per_bin.append(K_sources)
            T_per_bin.append(T_block)

    extra_rhs = None
    if s_o3_source_per_bin is not None:
        # Flatten per-bin (nz, ny, nx) -> (N,) for the orbit RHS.
        extra_rhs = {
            IDX_O3: [np.asarray(s).ravel() for s in s_o3_source_per_bin]
        }

    # Horizontal FCT (deferred correction): add the limited anti-diffusive
    # source d_AD = -div(C⊙A) to the orbit RHS for every transported species,
    # evaluated at the PREVIOUS outer iterate (cached_orbits) — a frozen source
    # within this solve, recomputed each closure iteration. Gated off by
    # default. See orbit.core.fct.compute_horizontal_fct_source.
    if fct_enabled and cached_orbits is not None:
        from orbit.core import fct as _fct
        if extra_rhs is None:
            extra_rhs = {}
        _fct_norm = 0.0
        _fct_nspec = 0
        for s, orbit_s in cached_orbits.items():
            if orbit_s is None:
                continue
            d_list = []
            for tau in range(N_BINS):
                c_tau = np.asarray(orbit_s[tau + 1], dtype=np.float64)  # end-of-bin
                d_list.append(_fct.compute_horizontal_fct_source(
                    grids[tau], indexer, c_tau, DTAU))
            _fct_norm += float(np.linalg.norm(np.concatenate(d_list)))
            _fct_nspec += 1
            if s in extra_rhs:
                extra_rhs[s] = [extra_rhs[s][tau] + d_list[tau]
                                for tau in range(N_BINS)]
            else:
                extra_rhs[s] = d_list
        if verbose:
            print(f"    [FCT] anti-diffusive source applied to {_fct_nspec} "
                  f"species, total |d_AD| = {_fct_norm:.3e}")

    # Snapshot assembly BEFORE solve_orbit_all_species mutates
    # L_species_per_bin[tau][s] = None as species finish. The shallow
    # list-copy preserves CSC matrix refs even after the original list
    # entries are nulled by the solver. Without this, the next iso iter's
    # cached_assembly would carry None for solved species.
    if return_assembly:
        if use_partial:
            # Inherit K_age cache from input (partition-independent and
            # already in cached_assembly), since the partial path didn't
            # rebuild it.
            kage_for_snapshot = list(cached_assembly.get(
                "K_age_loss_per_bin", [{}] * N_BINS))
        else:
            kage_for_snapshot = list(K_age_loss_per_bin)
        assembly_snapshot = {
            "L_species_per_bin": [list(Ls) for Ls in L_species_per_bin],
            "K_sources_per_bin": list(K_sources_per_bin),
            "D_per_bin": [list(Ds) for Ds in D_per_bin] if D_per_bin else None,
            "T_per_bin": list(T_per_bin),
            "K_age_loss_per_bin": kage_for_snapshot,
        }

    result = solve_orbit_all_species(
        L_species_per_bin, K_sources_per_bin,
        emissions_SN, N,
        umfpack_sym=umfpack_sym, perm=perm,
        tol=tol, maxiter=maxiter, verbose=verbose,
        c_warm_SN=c_warm_SN,
        lu_cache=lu_cache, skip_solve_species=skip_solve_species,
        cached_orbits=cached_orbits, return_lu_cache=return_lu_cache,
        keep_lu_species=keep_lu_species,
        extra_rhs_per_bin_per_species=extra_rhs,
    )
    if return_assembly:
        result["assembly"] = assembly_snapshot
    return result


def _checkpoint_path(month, it):
    """Path to an intermediate per-iter checkpoint NPZ."""
    return os.path.join(OUTPUT_DIR, f"orbit_M{month:02d}_iter{it:02d}.npz")


def _find_latest_checkpoint(month):
    """Return (iter, path) of the latest checkpoint on disk, or (None, None)."""
    best = None
    for it in range(50, 0, -1):  # scan down from a large cap
        p = _checkpoint_path(month, it)
        if os.path.exists(p):
            best = (it, p)
            break
    return best if best else (None, None)


def _save_checkpoint(month, it, result, chem, iso_per_bin, history,
                      closure_mode, closure_alpha, closure_tol, converged,
                      grid_shape=None):
    """Write orbit_M{MM}_iter{NN}.npz with everything needed to resume.

    Kept deliberately small: full orbits (~40 MB) + per-bin surface slabs
    of the key diagnostics (OH, HO2, f_NO2, f_nh4_eq, f_no3_eq, k_so2,
    k_nox) + full convergence history.  Slabs are ~300 KB compressed
    each, enough to reconstruct any named-cell trajectory post-hoc
    without rerunning.  Full 3D oxidant fields are still reserved for
    the final NPZ (too big to write 10+ times per run).
    """
    save = {
        "month": np.array(month),
        "iter": np.array(it),
        "converged": np.array(converged, dtype=bool),
        "closure_mode": np.array(closure_mode),
        "closure_alpha": np.array(closure_alpha),
        "closure_tol": np.array(closure_tol),
    }
    if grid_shape is not None:
        save["grid_shape"] = np.array(grid_shape)
    # Orbits: stack 8 species x 9 steps x N
    orbits = result["orbits"]
    c_orbit = np.stack(
        [np.stack(orbits[s], axis=0) for s in sorted(orbits.keys())], axis=0,
    )
    save["c_orbit"] = c_orbit
    save["orbit_species_order"] = np.array(sorted(orbits.keys()))
    save["gmres_iters"] = result["gmres_iters"]
    save["gmres_resid"] = result["gmres_resid"]
    save["periodicity"] = result["periodicity"]

    # Chem state: per-bin surface-mean diagnostics (full 3D dropped to keep
    # checkpoint small).  Also stash the full surface slab (k=0) of OH / HO2
    # / f_NO2 per bin — ~300 KB per field compressed, enough to reconstruct
    # any named-cell diurnal trajectory post-hoc without an iter-by-iter
    # rerun.
    if chem is not None:
        oh_surf_slab = np.stack([ox.OH[0].astype(np.float32)
                                 for ox in chem.oxidants], axis=0)
        ho2_surf_slab = np.stack([ox.HO2[0].astype(np.float32)
                                  for ox in chem.oxidants], axis=0)
        fno2_surf_slab = np.stack([ox.f_NO2[0].astype(np.float32)
                                   for ox in chem.oxidants], axis=0)
        k_so2_surf_slab = np.stack([chem.k_so2_rate[tau][0].astype(np.float32)
                                    for tau in range(len(chem.oxidants))], axis=0)
        k_nox_surf_slab = np.stack([chem.k_nox_to_no3_rate[tau][0].astype(np.float32)
                                    for tau in range(len(chem.oxidants))], axis=0)
        save["chem_oh_surf_per_bin"] = oh_surf_slab      # (N_BINS, ny, nx)
        save["chem_ho2_surf_per_bin"] = ho2_surf_slab
        save["chem_f_no2_surf_per_bin"] = fno2_surf_slab
        save["chem_k_so2_surf_per_bin"] = k_so2_surf_slab
        save["chem_k_nox_surf_per_bin"] = k_nox_surf_slab
        for tau, ox in enumerate(chem.oxidants):
            save[f"chem_bin{tau + 1:02d}_OH_surf_mean"] = np.array(float(ox.OH[0].mean()))
            save[f"chem_bin{tau + 1:02d}_HO2_surf_mean"] = np.array(float(ox.HO2[0].mean()))
            save[f"chem_bin{tau + 1:02d}_f_NO2_surf_mean"] = np.array(float(ox.f_NO2[0].mean()))
            save[f"chem_bin{tau + 1:02d}_k_so2_surf_mean"] = np.array(
                float(chem.k_so2_rate[tau][0].mean()))
            save[f"chem_bin{tau + 1:02d}_k_nox_surf_mean"] = np.array(
                float(chem.k_nox_to_no3_rate[tau][0].mean()))

    # Partitioning state (equilibrium + marginal surface slabs + means).
    # Named-cell partitioning drift at Delhi is the main Q1 question.
    if iso_per_bin:
        f_nh4_eq_slab_list = []
        f_no3_eq_slab_list = []
        for tau in sorted(iso_per_bin.keys()):
            d = iso_per_bin[tau]
            f_nh4_eq_slab_list.append(d["f_nh_eq"][0].astype(np.float32))
            f_no3_eq_slab_list.append(d["f_no3_eq"][0].astype(np.float32))
            save[f"iso_bin{tau + 1:02d}_f_nh4_eq_surf"] = np.array(float(d["f_nh_eq"][0].mean()))
            save[f"iso_bin{tau + 1:02d}_f_nh4_marg_surf"] = np.array(float(d["f_nh_marg"][0].mean()))
            save[f"iso_bin{tau + 1:02d}_f_no3_eq_surf"] = np.array(float(d["f_no3_eq"][0].mean()))
            save[f"iso_bin{tau + 1:02d}_f_no3_marg_surf"] = np.array(float(d["f_no3_marg"][0].mean()))
        if f_nh4_eq_slab_list:
            save["iso_f_nh4_eq_surf_per_bin"] = np.stack(f_nh4_eq_slab_list, axis=0)
            save["iso_f_no3_eq_surf_per_bin"] = np.stack(f_no3_eq_slab_list, axis=0)

    # Convergence history
    save.update(history.to_npz_dict(prefix="conv_"))

    path = _checkpoint_path(month, it)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.savez_compressed(path, **save)
    size_mb = os.path.getsize(path) / 1024 / 1024
    print(f"  [checkpoint] iter {it} -> {path} ({size_mb:.1f} MB)")


def _load_checkpoint(path):
    """Restore minimal state for resuming the outer iteration.

    Returns a dict with:
        - orbits: {species_idx: [c_0,...,c_8]} (reconstructed from c_orbit)
        - gmres_iters, gmres_resid, periodicity
        - iter, converged
    History is re-hydrated from the ``conv_*`` arrays below.
    """
    d = np.load(path, allow_pickle=False)
    c_orbit = d["c_orbit"]  # (n_species, n_steps, N)
    species_order = d["orbit_species_order"]
    orbits = {}
    for s_idx, s in enumerate(species_order.tolist()):
        orbits[int(s)] = [c_orbit[s_idx, t].copy() for t in range(c_orbit.shape[1])]
    return {
        "orbits": orbits,
        "gmres_iters": d["gmres_iters"].copy(),
        "gmres_resid": d["gmres_resid"].copy(),
        "periodicity": d["periodicity"].copy(),
        "iter": int(d["iter"]),
        "converged": bool(d["converged"]),
        "conv_dict": {k: d[k] for k in d.files if k.startswith("conv_")},
    }


def _history_from_conv_dict(conv_dict):
    """Reconstruct a ConvergenceHistory from the flattened conv_* arrays.

    Only what we need to continue iterating + produce the final NPZ.  The
    per-iter cell locations are not persisted (the checkpoint keeps them
    implicit via the conv_*_max_rel scalars) — which is fine for resume
    but means the final classification output may lose oscillation
    evidence for iters that pre-date the restart.
    """
    from orbit.core.dcomp_iter import IterationMetrics
    history = ConvergenceHistory()
    if not conv_dict:
        return history
    n = int(conv_dict.get("conv_n_iter", np.array(0)))
    for k in range(n):
        def _g(name):
            arr = conv_dict.get(f"conv_{name}")
            return float(arr[k]) if arr is not None else 0.0
        m = IterationMetrics(it=k + 1)
        m.oh = (_g("oh_mean"), _g("oh_max_rel"), (0, 0, 0), 0)
        m.o3 = (_g("o3_mean"), _g("o3_max_rel"), (0, 0, 0), 0)
        m.f_nh4 = (_g("f_nh4_mean"), _g("f_nh4_max_rel"), (0, 0, 0), 0)
        m.f_no3 = (_g("f_no3_mean"), _g("f_no3_max_rel"), (0, 0, 0), 0)
        m.pm25 = (_g("pm25_mean"), _g("pm25_max_rel"), (0, 0, 0), 0)
        m.nox_neg_count = int(_g("nox_neg_count"))
        m.nox_min = _g("nox_min")
        m.totalNO3_neg_count = int(_g("totalNO3_neg_count"))
        m.totalNO3_min = _g("totalNO3_min")
        m.mass_residual_N = _g("mass_residual_N")
        m.mass_residual_S = _g("mass_residual_S")
        m.converged = m.gates_met(tol=DEFAULT_TOL)
        history.append(m)
    return history


def run_forward_month(month, resume=False, warm=False, lut_path=None,
                   chemistry_iters=0, photolysis_lut_path=None,
                   hemco_dir=None, closure_mode="full",
                   closure_alpha=0.5, closure_tol=DEFAULT_TOL,
                   resume_from=None,
                   top_bc_days=10.0,
                   top_bc_layers=4,
                   top_bc_decay_factor=3.0,
                   lateral_bc_days=1.0,
                   lateral_bc_depth=3,
                   krylov_tol_intermediate=1e-4,
                   closure_accel="picard",
                   anderson_m=3,
                   disable_night_nox=False,
                   isorropia_closure_iters=6,
                   isorropia_anderson=False,
                   basin_flip_damp=True,
                   diurnal_cfg=None,
                   e_perturbed_override=None,
                   output_filename=None,
                   output_dir=None,
                   iso_cross_partials=True,
                   fold_meander_in_K=False,
                   unified_vertical_patankar=False,
                   horizontal_fct=False):
    """Process one month: load grids -> assemble -> orbit solve -> save.

    Resume policy:
      * ``--resume-from <path>``  : load state from the named checkpoint
        (any NPZ produced by this script — final or per-iter — works as
        long as it carries ``c_orbit`` + ``orbit_species_order`` + ``iter``
        + ``conv_*``).  Does NOT skip existing output; extends it.
      * ``--resume``              : skip months whose final output exists;
        for un-done months, auto-discover the latest per-iter checkpoint.
        Convenience for multi-month reruns; avoid if another experiment's
        per-iter checkpoints may be on disk (use ``--resume-from`` instead).

    Returns dict of timings or None on failure.

    Phase 3 hooks (zero-out mode):
      * ``e_perturbed_override``: if non-None, supplies the full perturbed
        emissions array (shape (N_BINS, N_SPECIES * N) or (N_SPECIES * N,))
        in solver layout. The internal source-loading + 6→9 rebroadcast
        is skipped. The forward pass runs identically otherwise — same
        ISORROPIA closure, same NPZ schema.
      * ``output_filename``: override for the output file name within
        OUTPUT_DIR (default ``orbit_M{MM}.npz``). Useful for zero-out's
        perturbed-run output (kept on disk separately so analysis can
        re-difference without re-running).
      * ``output_dir``: override for the output directory. When None,
        falls back to module-level ``OUTPUT_DIR``. Phase 4d threading
        for zero-out's perturbed forward (avoids the file-move dance).
    """
    # Branch note: --enable-orbit-co and --prescribed-oh lived here on
    # feat/isorropia-orbit. Dropped on feat/orbit-isorropia-archive: CO
    # scenarios are out of scope without archive 3h HCHO/HO2, and the
    # monthly-HEMCO prescribed-OH hook was demonstrated misleading.
    enable_orbit_co = False
    prescribed_oh = False
    _check_input_paths(lut_path)
    out_name = output_filename or f"orbit_M{month:02d}.npz"
    out_dir = output_dir or OUTPUT_DIR
    out_path = os.path.join(out_dir, out_name)
    os.makedirs(out_dir, exist_ok=True)

    # --resume-from bypasses the "skip-if-output-exists" short-circuit:
    # if you specified an explicit resume path, you want to extend that
    # state, not skip it.
    if resume and resume_from is None and os.path.exists(out_path):
        print("  Skipping (output exists)")
        return {"month": month, "skipped": True}

    # Check all 8 bin files exist
    bin_paths = [_preproc_path(month, b + 1) for b in range(N_BINS)]
    for bp in bin_paths:
        if not os.path.exists(bp):
            print(f"  Missing: {bp}")
            return None

    timings = {"month": month}

    # 1. Load 8 grids
    t0 = time.time()
    # Clear-sky j(NO2) LUT for the VBS SoA photolytic sink. The sink is OFF in
    # production (ORBIT_VBS_A_PHOTO=0), so the LUT is loaded ONLY when the sink
    # is explicitly enabled (A_PHOTO > 0) — production needs neither the file
    # nor the TUV-LUT generation pipeline. Attached per bin below; consumed in
    # deposition.assemble_deposition.
    _photo_lut = load_photolysis_lut() if get_a_photo() > 0 else None
    if get_a_photo() > 0 and _photo_lut is None:
        # Asking for the sink and silently not getting it is worse than not
        # running: the config banner reports the requested A_PHOTO, so the run
        # LOOKS like a sink experiment and produces output bit-identical to
        # the no-sink case. That cost a 21-minute April run on 2026-08-03,
        # and the module default is a "/path/to/data" placeholder, so the
        # miss is the norm rather than the exception. Refuse instead.
        raise SystemExit(
            f"ORBIT_VBS_A_PHOTO={get_a_photo():g} requests the VBS SOA "
            f"photolytic sink, but the TUV j(NO2) LUT was not found.\n"
            f"Set ORBIT_PHOTOLYSIS_LUT to a real photolysis_tuv.npz, or unset "
            f"ORBIT_VBS_A_PHOTO to run without the sink.\n"
            f"Running on would silently produce a no-sink answer labelled as "
            f"a sink experiment.")
    _RSS.set_phase("grid")
    try:
        _year = int(PREPROC_YEAR_TAG)
    except ValueError:
        _year = 2016
    grids = []
    for tau in range(N_BINS):
        g = load_grid(bin_paths[tau], CONSTANTS)
        if _photo_lut is not None:
            attach_jno2_to_grid(g, _photo_lut, _year, month, tau)
        if fold_meander_in_K:
            # Flip to the non-split-flux code path so K_meander is folded
            # into K_face (convdiff) and lateral-boundary loss falls back
            # to mean-flux.
            g.has_split_fluxes = False
        if unified_vertical_patankar:
            # PROTOTYPE: assemble a single unified vertical Patankar operator
            # (omega advection + Kzz diffusion) in place of the separate
            # operators.
            g.unified_vertical_patankar = True
        grids.append(g)
    timings["grid_load"] = time.time() - t0
    if fold_meander_in_K:
        print("  has_split_fluxes overridden to False — K_meander folded into K_face")
    if unified_vertical_patankar:
        print("  unified_vertical_patankar ON — vertical omega+Kzz assembled as "
              "one Patankar operator (PROTOTYPE)")

    # Verify shape consistency
    nz, ny, nx = grids[0].nz, grids[0].ny, grids[0].nx
    for tau in range(1, N_BINS):
        assert (grids[tau].nz, grids[tau].ny, grids[tau].nx) == (nz, ny, nx), \
            f"Bin {tau} shape mismatch: {(grids[tau].nz, grids[tau].ny, grids[tau].nx)} != {(nz, ny, nx)}"

    indexer = CellIndexer(nz, ny, nx)
    N = indexer.N
    print(f"  Grid: {nz}x{ny}x{nx} = {N:,} cells, "
          f"8 bins loaded ({timings['grid_load']:.1f}s)")

    # 2. Factorize a baseline operator for UMFPACK symbolic reuse.
    # (The symbolic is shared across all 64 species-bin pairs via common
    # sparsity pattern.  We build one representative operator only.)
    t0 = time.time()
    from orbit.core.operator import assemble_transport_block
    from orbit.core.deposition import assemble_deposition as _assdep
    T_sample = assemble_transport_block(grids[0], indexer, scheme="exp")
    L_sample = T_sample + _assdep(grids[0], indexer, 0)

    # Backend preference. Default = "auto" → UMFPACK + METIS-via-qsymbolic
    # when pymetis is available, else UMFPACK+COLAMD.
    #
    # History: the 2026-05-15 postmortem measured UMFPACK+METIS (qinit)
    # at 10x COLAMD factor wall (job 9203389) and attributed it to qinit
    # "disabling internal pivoting freedom". The real cause, found
    # 2026-08-01: qsymbolic was
    # called with a NULL Control, so the AUTO strategy resolved to
    # unsymmetric, which does not preserve Qinit (column etree postorder +
    # per-front re-pivoting). With Control[UMFPACK_STRATEGY]=SYMMETRIC the
    # same ordering factors 1.5x FASTER than COLAMD at 20% less peak RSS
    # (January: 21.3 vs 28.2 min, 12.0 vs 15.0 GB). See solve.py and
    # tests/test_umfpack_metis_qinit.py.
    #   - SuperLU+METIS  → OOM in production at 41.9 GB despite 2.3× lower
    #                      fill nnz (supernodal storage ~12-16 B/nnz vs
    #                      UMFPACK frontal ~8 B/nnz; fill nnz is NOT a
    #                      cross-backend memory proxy). Job 9199858.
    # Explicit values:
    #   umfpack          — UMFPACK with its default ordering (COLAMD)
    #   umfpack_metis    — METIS perm via pymetis → UMFPACK qsymbolic
    #   superlu_metis    — SuperLU+METIS via _PermutedLU (OOM-fragile)
    #   superlu          — plain SuperLU+COLAMD
    backend_pref = os.environ.get("ORBIT_LU_BACKEND", "auto").lower()
    _RSS.set_phase("symbolic")
    umfpack_sym = None
    perm = None
    if backend_pref == "auto":
        # Default: UMFPACK with a pymetis nested-dissection ordering.
        # Measured on the January operator (2026-08-01): 21.3 vs 28.2 min
        # wall, 12.0 vs 15.0 GB peak against UMFPACK+COLAMD, identical
        # GMRES iteration counts.
        if _HAS_UMFPACK and _HAS_METIS:
            metis_perm = compute_metis_ordering(L_sample, verbose=True)
            umfpack_sym = umfpack_symbolic(L_sample, verbose=True,
                                           qinit=metis_perm)
            backend = "UMFPACK+METIS (qinit)"
        elif _HAS_UMFPACK:
            umfpack_sym = umfpack_symbolic(L_sample, verbose=True)
            backend = "UMFPACK+COLAMD (auto fallback, pymetis missing)"
        else:
            backend = "SuperLU+COLAMD (auto fallback, UMFPACK missing)"
    elif backend_pref == "umfpack":
        if not _HAS_UMFPACK:
            raise RuntimeError("ORBIT_LU_BACKEND=umfpack but UMFPACK unavailable")
        umfpack_sym = umfpack_symbolic(L_sample, verbose=True)
        backend = "UMFPACK+COLAMD"
    elif backend_pref == "superlu":
        backend = "SuperLU+COLAMD"
    elif backend_pref == "superlu_metis":
        if not _HAS_METIS:
            raise RuntimeError(
                "ORBIT_LU_BACKEND=superlu_metis but pymetis unavailable"
            )
        perm = compute_metis_ordering(L_sample, verbose=True)
        backend = "SuperLU+METIS"
    elif backend_pref == "umfpack_metis":
        if not (_HAS_METIS and _HAS_UMFPACK):
            raise RuntimeError(
                "ORBIT_LU_BACKEND=umfpack_metis requires both pymetis and "
                "UMFPACK to be available"
            )
        metis_perm = compute_metis_ordering(L_sample, verbose=True)
        umfpack_sym = umfpack_symbolic(L_sample, verbose=True,
                                        qinit=metis_perm)
        backend = "UMFPACK+METIS (qinit)"
    else:
        raise ValueError(
            f"Unknown ORBIT_LU_BACKEND={backend_pref!r}; "
            f"valid: auto, umfpack, umfpack_metis, superlu_metis, superlu"
        )
    timings["symbolic"] = time.time() - t0
    fill_note = ""
    if umfpack_sym is not None:
        timings["lu_fill_est"] = umfpack_sym["lunz_est"]
        fill_note = (f", est nnz(LU) {umfpack_sym['lunz_est'] / 1e6:.0f}M"
                     f"/factor")
    print(f"  Backend: {backend} ({timings['symbolic']:.1f}s{fill_note})")
    _RSS.set_phase("emissions")

    # 3. Load emissions. Two-stage:
    #
    # Stage A: standard loader on non-VOC sources only. Returns a
    # (N_BINS, 6*N) array under the legacy ORBIT 6-species layout. Slot 0
    # is forced to zero by the loader (it would have received only
    # legacy "TotalOrg" mass, which has no place in the VBS scheme).
    # Rebroadcast 6 → 13 slots placing the existing species at their
    # DCOMP indices. New VBS slots (9-12) start at zero.
    #
    # Stage B: per VOC source individually, distribute the source's VOC
    # mass across the 5 VBS bins (slot 0 = C*=100 + slots 9-12) per
    # chamber-derived stoichiometric yields (vbs_yields.VBS_PARENT_YIELDS).
    # Cell-dependent NOx-regime switching applied at this step using
    # archive [NO2]/[OH] (parent class "anthro" only).
    #
    # The split keeps per-source identity for VOC files (needed to apply
    # the right yield set per-source), while reusing the existing
    # 6-species accumulator path for everything else.
    #
    # 13-species DCOMP+VBS layout:
    #   0   IDX_VBS_C100  (gets C*=100 yields from VOC sources)
    #   1   PM25
    #   2   TotalNH
    #   3   SO2
    #   4   NOx
    #   5   pSO4
    #   6   TotalNO3
    #   7   O3
    #   8   CO
    #   9   IDX_VBS_C10
    #   10  IDX_VBS_C1
    #   11  IDX_VBS_C01
    #   12  IDX_VBS_C1000
    #
    # We always use ``load_emissions_diurnal`` (with diurnal_cfg=None when
    # not configured) so that bin_axis=True sources (CAMS soil NOx) keep
    # their native 8 UTC slabs in both baseline and perturbed forwards.
    t0 = time.time()
    sources = _build_emission_sources(month)

    # Per-source mass budget. A file listing cannot reveal a wrong `units`
    # declaration, a bad regrid, or a truncated inventory; a mass total makes
    # all three obvious. Costs one extra read of the inventory, so it is
    # skippable with --no-emission-budget.
    if _EMISSION_BUDGET and e_perturbed_override is None:
        try:
            _rows, _totals = source_mass_budget(sources, grids[0], indexer)
            print(format_mass_budget(_rows, _totals))
            _RECORD.emissions["by_source_tg_yr"] = {
                f"{n}:{lab}": tg for n, lab, tg, _ in _rows}
            _RECORD.emissions["by_species_tg_yr"] = _totals
        except Exception as exc:
            print(f"  WARNING: emission mass budget not computed: {exc}")

    # loader 6/7-species layout -> solver 14-species layout
    _old_to_new = {0: 0, 1: 1, 2: 2, 3: 3, 4: 5, 5: 4, 6: 13}
    if e_perturbed_override is not None:
        # Phase 3 zero-out hook: caller pre-built the full perturbed
        # emissions array. Skip the source loop entirely.
        emissions_SN = np.asarray(e_perturbed_override, dtype=np.float64)
        if emissions_SN.shape not in (
            (N_SPECIES * N,), (N_BINS, N_SPECIES * N),
        ):
            raise ValueError(
                f"e_perturbed_override has shape {emissions_SN.shape}; "
                f"expected ({N_SPECIES * N},) or ({N_BINS}, {N_SPECIES * N})"
            )
        bin_mean = (emissions_SN.mean(axis=0) if emissions_SN.ndim == 2
                    else emissions_SN)
        # Mass-rate diagnostic (kg/s) — invariant under vertical injection
        # changes, unlike Σ(per-volume rates) which depends on receiving Dz.
        vol_per_species = np.tile(grids[0].volume.ravel(), N_SPECIES)
        mass_per_s_kg = float((bin_mean * vol_per_species).sum() * 1e-9)
        print(f"  Emissions OVERRIDE: shape={emissions_SN.shape}, "
              f"|sum_per_vol|={float(np.abs(bin_mean).sum()):.3e} ug/m^3/s, "
              f"total mass = {mass_per_s_kg:.4e} kg/s")
    else:
        # Stage A: standard loader on all sources. Slot 0 of every source
        # is forced to zero by the loader; VOC sources are still in the
        # list (they just don't contribute to slot 0 here). Their non-VOC
        # slots, if any, still flow normally — but VOC files only have
        # the gOrg/voc variable, so all of their contribution is slot 0
        # which is zeroed.
        emissions_old_per_bin = load_emissions_diurnal(
            sources, grids[0], indexer,
            diurnal_cfg=diurnal_cfg, month=month, n_bins=N_BINS,
            verbose=(diurnal_cfg is not None),
        )  # (N_BINS, 6*N) — slot 0 is zero
        emissions_SN = np.zeros((N_BINS, N_SPECIES * N), dtype=np.float64)
        for tau in range(N_BINS):
            for old_s, new_s in _old_to_new.items():
                emissions_SN[tau, new_s * N:(new_s + 1) * N] = (
                    emissions_old_per_bin[tau, old_s * N:(old_s + 1) * N]
                )

        # Stage B: per-VOC-source distribution across the 5 VBS bins.
        F_HIGH_NOX = _compute_nox_regime(grids[0])
        voc_diag = []
        for source in sources:
            if source.voc_parent_class is None:
                continue
            contrib_per_bin, voc_mass_total = _distribute_voc_to_vbs_bins(
                source, grids[0], indexer,
                diurnal_cfg=diurnal_cfg, month=month, n_bins=N_BINS,
                F_HIGH_NOX=F_HIGH_NOX,
                verbose=(diurnal_cfg is not None),
            )
            emissions_SN += contrib_per_bin
            voc_diag.append((os.path.basename(source.path),
                             source.voc_parent_class, voc_mass_total))

        # Stage C: synthesize IVOC from POA × _IVOC_SCALING, using the
        # "ivoc" parent class (Robinson 2007: IVOC ≈ 1.5 × POA).
        #
        # Before 2026-08-02 there was no POA field, so this read the CEDS
        # PM2.5 anthro tiers and multiplied by an ASSUMED POA share of 0.30.
        # The share is now inventory-derived — POA is 1.8×OC from the same
        # CEDS files the PM2.5 tiers were built from — and measures 0.847,
        # so the old proxy understated IVOC by ~2.8x.
        from orbit.emissions.vbs_yields import VBS_PARENT_YIELDS
        ivoc_yields = VBS_PARENT_YIELDS["ivoc"]   # (5,)
        bin_solver_idx = _bin_indices_in_solver_layout()
        ivoc_total = 0.0
        # Prefer a real POA source when the manifest provides one; fall back
        # to the legacy PM2.5 proxy (× 0.3) when it does not, so old
        # manifests keep working unchanged.
        _poa_sources = [x for x in sources
                        if "ceds_poa_anthro" in os.path.basename(x.path)]
        _use_poa = bool(_poa_sources)
        if _use_poa:
            _ivoc_srcs, _ivoc_slot, _ivoc_frac = _poa_sources, 6, _IVOC_SCALING
        else:
            _ivoc_srcs = [x for x in sources
                          if "ceds_pm25_anthro" in os.path.basename(x.path)]
            _ivoc_slot, _ivoc_frac = 1, 0.3 * _IVOC_SCALING
        n_ivoc_sources = 0
        for source in _ivoc_srcs:
            n_ivoc_sources += 1
            from orbit.emissions.netcdf import load_netcdf_source
            from orbit.emissions.loader import _factors_for_source
            monthly = load_netcdf_source(source, grids[0], verbose=False)
            poa_3d = monthly[_ivoc_slot]        # 6 = POA (new), 1 = PM2.5 proxy
            ivoc_emis_3d = _ivoc_frac * poa_3d
            if diurnal_cfg is None:
                factors = np.ones(N_BINS)
            else:
                factors = _factors_for_source(source, diurnal_cfg, month, N_BINS)
            for tau in range(N_BINS):
                voc_mass_3d = factors[tau] * ivoc_emis_3d
                ivoc_total += float(voc_mass_3d.sum())
                for i_bin, solver_s in enumerate(bin_solver_idx):
                    bin_3d = ivoc_yields[i_bin] * voc_mass_3d   # (nz, ny, nx)
                    emissions_SN[tau, solver_s * N:(solver_s + 1) * N] += (
                        bin_3d.ravel())

        bin_mean = emissions_SN.mean(axis=0)
        flat_total = bin_mean.sum()
        vol_per_species = np.tile(grids[0].volume.ravel(), N_SPECIES)
        mass_per_s_kg = float((bin_mean * vol_per_species).sum() * 1e-9)
        cfg_tag = "with diurnal_cfg" if diurnal_cfg is not None else "diurnal_cfg=None (bin-flat for non-bin_axis sources)"
        print(f"  Per-bin emissions ({cfg_tag}): bin-mean Σ(per-volume) = "
              f"{flat_total:.3e} ug/m^3/s | total mass = {mass_per_s_kg:.4e} kg/s "
              f"shape={emissions_SN.shape}")
        if voc_diag:
            print("  VBS VOC sources distributed:")
            for fname, pcls, vmass in voc_diag:
                print(f"    {fname:50s}  class={pcls:18s}  ΣVOC mass(per-vol·s)={vmass:.3e}")
            _ivoc_src_label = ("POA" if _use_poa
                               else "CEDS PM2.5 (0.3 assumed POA share)")
            print(f"  IVOC synth from {_ivoc_src_label} × {_ivoc_frac:.2f} "
                  f"(IVOC scaling {_IVOC_SCALING:g}): "
                  f"ΣIVOC mass(per-vol·s)={ivoc_total:.3e} "
                  f"from {n_ivoc_sources} source(s)")
            if n_ivoc_sources > 0 and ivoc_total == 0.0:
                _RECORD.warn(
                    "emissions",
                    "IVOC synthesis contributed zero mass despite matching "
                    f"{n_ivoc_sources} IVOC precursor source(s).",
                    impact="the VBS receives no IVOC precursor, so IVOC "
                           "scaling has no effect on OA.",
                    fix="verify the species slot read by the IVOC stage "
                        "against the ORBIT 6-species layout and the "
                        "ceds_pm25_anthro inputs.",
                )
    timings["emissions"] = time.time() - t0
    print(f"  Emissions loaded: {len(sources)} files ({timings['emissions']:.1f}s)")

    # 4. Warm start (optional, for Phase 7 inter-month chaining)
    c_warm_SN = None
    if warm:
        prev_month = month - 1 if month > 1 else 12
        prev_path = os.path.join(OUTPUT_DIR, f"orbit_M{prev_month:02d}.npz")
        if os.path.exists(prev_path):
            d_prev = np.load(prev_path)
            if d_prev["c_mean"].shape[0] == N_SPECIES:
                c_warm_SN = d_prev["c_mean"].ravel()
                print(f"  Warm start: previous month orbit mean (M{prev_month:02d})")
            else:
                print(f"  Warm start: prior orbit has {d_prev['c_mean'].shape[0]} species, "
                      f"need {N_SPECIES}; starting cold")
        else:
            print("  Warm start: no prior orbit found, using s (one-cycle-from-zero)")

    # 5. Load DCOMP chemistry inputs if chemistry iterations requested.
    photolysis_lut = None
    hemco_clim = None
    if chemistry_iters > 0:
        if photolysis_lut_path and os.path.exists(photolysis_lut_path):
            photolysis_lut = PhotolysisLUT.load(photolysis_lut_path)
            print(f"  Photolysis LUT: {photolysis_lut_path}")
        else:
            print("  Photolysis LUT missing -> chemistry disabled")
            chemistry_iters = 0
        if chemistry_iters > 0:
            try:
                hemco_clim = load_hemco_climatology(
                    2016, month, grids[0].lat, grids[0].lon,
                    hemco_dir=hemco_dir or "/path/to/data/GCClassic_Output/14.0.0",
                )
                print(f"  HEMCO climatology: 2016-{month:02d} loaded")
            except Exception as e:
                print(f"  HEMCO load failed: {e} -> chemistry disabled")
                chemistry_iters = 0

    # Combined top + lateral Newtonian BC for O3.  Build a full 3D
    # target (HEMCO monthly-mean O3 at each ORBIT layer's domain-mean
    # sigma) and a full 3D rate field that superimposes:
    #   (a) top-of-model stratospheric-descent nudging (rate concentrated
    #       at top N layers, decaying downward from 1/(top_bc_days) at
    #       the top to 1/(top_bc_days * decay_factor) at the bottom of
    #       the nudged band);
    #   (b) lateral inflow from global climatology (rate concentrated at
    #       outermost M cells of each of the 4 horizontal edges, linearly
    #       decaying from 1/(lateral_bc_days) at the outermost to 0 at
    #       the interior edge).
    # Cell-wise max of (a) + (b) so overlapping top-corner cells take
    # the stronger rate rather than double-counting.  Diagnosed from
    # the multi-layer-only smoke: Delhi UT equilibrated at 4-11% of
    # HEMCO despite nudging, because mid-lat westerly jet advects O3
    # out of the domain faster than top-BC can replenish.  Lateral BC
    # at ~1/(1 day) pins inflow boundary cells to climatology so
    # advected air arriving at Delhi carries realistic O3.
    o3_bc_target_3d = None
    o3_bc_rate_3d = None
    bc_active = (
        chemistry_iters > 0 and hemco_clim is not None and (
            (top_bc_days > 0 and top_bc_layers > 0)
            or (lateral_bc_days > 0 and lateral_bc_depth > 0)
        )
    )
    if bc_active:
        from orbit.hemco import layer_slice_by_sigma
        P = grids[0].Pressure
        P_surf_dom = float(P[0].mean())

        # Full 3D target: HEMCO-at-sigma for every ORBIT layer (not just
        # nudged ones — lateral BC may nudge at any k, not just top).
        target_3d = np.zeros((nz, ny, nx), dtype=np.float64)
        for k in range(nz):
            sigma_k = float(P[k].mean() / P_surf_dom)
            slab_2d = layer_slice_by_sigma(
                hemco_clim, "O3", target_sigma=sigma_k,
            )
            if slab_2d is not None:
                target_3d[k] = slab_2d

        # (a) Top-BC rate field.
        rate_top_3d = np.zeros((nz, ny, nx), dtype=np.float64)
        if top_bc_days > 0 and top_bc_layers > 0:
            N_top = int(top_bc_layers)
            rate_top = 1.0 / (top_bc_days * 86400.0)
            rate_bot = rate_top / top_bc_decay_factor
            if N_top == 1:
                rate_top_3d[-1, :, :] = rate_top
            else:
                for i in range(N_top):
                    frac = i / (N_top - 1)
                    rate_top_3d[nz - 1 - i, :, :] = rate_top * (
                        (rate_bot / rate_top) ** frac
                    )
            print(f"  Top-BC: {N_top} layers, top_days={top_bc_days:g}, "
                  f"decay_factor={top_bc_decay_factor:g}")
            for i in range(N_top):
                k = nz - 1 - i
                days_k = 1.0 / (rate_top_3d[k, 0, 0] * 86400.0)
                print(f"    k={k} (σ={P[k].mean()/P_surf_dom:.3f}, "
                      f"P={P[k].mean()/100:.0f} hPa): rate=1/({days_k:.1f} d), "
                      f"target mean={float(target_3d[k].mean()):.1f} ug/m3")

        # (b) Lateral-BC rate field.  Each of the 4 boundaries (south,
        # north, west, east) gets a decaying rate at depths d = 0..M-1.
        # Depth d=0 is the outermost cell, d=M-1 the innermost.
        # Linear decay: rate(d) = rate_out · (1 − d/(M−1)) ; at d=M-1
        # rate is 0 which means effectively (M-1) nudged layers with
        # the outermost M-1.  Use M+1 inclusive so the innermost cell
        # still gets a small rate.  Simpler: (M cells, linear decay to
        # but not including 0 at one further cell).  Using strictly
        # linear: rate(d) = rate_out · (M − d) / M for d in [0, M−1].
        rate_lat_3d = np.zeros((nz, ny, nx), dtype=np.float64)
        if lateral_bc_days > 0 and lateral_bc_depth > 0:
            M = int(lateral_bc_depth)
            rate_out = 1.0 / (lateral_bc_days * 86400.0)
            for d in range(M):
                # (M - d) / M: d=0 → 1.0, d=M-1 → 1/M, d=M → 0.
                rate_d = rate_out * (M - d) / M
                # Apply at all layers (no vertical restriction — lateral
                # inflow affects the full column).
                # South edge: j = d
                rate_lat_3d[:, d, :] = np.maximum(rate_lat_3d[:, d, :], rate_d)
                # North edge: j = ny-1-d
                rate_lat_3d[:, ny - 1 - d, :] = np.maximum(
                    rate_lat_3d[:, ny - 1 - d, :], rate_d)
                # West edge: i = d
                rate_lat_3d[:, :, d] = np.maximum(rate_lat_3d[:, :, d], rate_d)
                # East edge: i = nx-1-d
                rate_lat_3d[:, :, nx - 1 - d] = np.maximum(
                    rate_lat_3d[:, :, nx - 1 - d], rate_d)
            # Count nudged cells for reporting
            n_lat_cells = int((rate_lat_3d[0] > 0).sum())
            print(f"  Lateral-BC: depth={M} cells, outermost_days={lateral_bc_days:g}, "
                  f"rate_outermost=1/({lateral_bc_days:g} d)={rate_out:.2e}/s")
            print(f"    {n_lat_cells}/{ny*nx} cells nudged per layer "
                  f"({100.0*n_lat_cells/(ny*nx):.1f}%)")
            # Target means along the 4 boundaries at selected layers
            for k in [nz - 1, nz - 4, nz // 2, 0]:
                edge_vals = np.concatenate([
                    target_3d[k, 0, :], target_3d[k, -1, :],
                    target_3d[k, :, 0], target_3d[k, :, -1],
                ])
                print(f"    layer k={k} (P={P[k].mean()/100:.0f} hPa) boundary "
                      f"target mean={edge_vals.mean():.1f} ug/m3 "
                      f"min/max={edge_vals.min():.1f}/{edge_vals.max():.1f}")

        # Combine via cell-wise max — avoids double-counting at
        # top-corner cells where both top-BC and lateral-BC apply.
        rate_3d = np.maximum(rate_top_3d, rate_lat_3d)

        if rate_3d.sum() > 0.0:
            o3_bc_target_3d = target_3d
            o3_bc_rate_3d = rate_3d
        else:
            print("  O3 BC: all rates 0 -> BC disabled")

    # Resume path: load state from an explicit checkpoint (--resume-from)
    # or auto-discover the latest per-iter checkpoint (--resume).  Either
    # way, the baseline solve is skipped along with every outer iteration
    # up to and including the checkpoint iter.
    resume_iter = 0
    resumed_history = None
    r_path = None
    if resume_from is not None:
        r_path = resume_from
        if not os.path.exists(r_path):
            print(f"  [resume-from] path does not exist: {r_path}")
            return None
        print(f"  [resume-from] {r_path}")
    elif resume:
        _, r_path = _find_latest_checkpoint(month)
        if r_path is not None:
            print(f"  [resume] auto-discovered checkpoint: {r_path}")
        else:
            print("  [resume] no checkpoint found; running full baseline")

    if r_path is not None:
        cp = _load_checkpoint(r_path)
        result = {
            "orbits": cp["orbits"],
            "gmres_iters": cp["gmres_iters"],
            "gmres_resid": cp["gmres_resid"],
            "periodicity": cp["periodicity"],
        }
        resumed_history = _history_from_conv_dict(cp["conv_dict"])
        resume_iter = cp["iter"]
        print(f"  [resume] picking up at iter {resume_iter} (next: iter {resume_iter + 1})")
        # The LU cache was not persisted; first resumed iter does a
        # clean factorisation.  Small cost, big simplicity.
        baseline_lu_cache = {}
        # Resume path doesn't persist the cached operator assembly either.
        # That just means the iso loop falls back to full reassembly each
        # iter (harmless, slower). Resume is rare; not worth persisting.
        baseline_assembly = None
        # We also skip the plausibility check on resume (it's an iter-1
        # only check).

    if resume_iter == 0:
        # 6. Baseline solve (iteration 0) — prescribed chemistry from grid.
        # Request the LU cache back so subsequent iterations can reuse
        # factorisations for species whose operator didn't change.  Only
        # keep LUs for species that are guaranteed-unchanged across
        # iterations (Org, PM25, pSO4, O3) — those whose operators depend
        # on chemistry rates (SO2, NOx) or partitioning (NH, TotalNO3)
        # will be refactored at iter 1, so holding their LUs now just
        # wastes ~2-4 GB of RSS.
        # Phase 3e: O3 operator changes every iter (chemistry loss diagonal
        # tracks oxidants), so drop it from the baseline keep-across set.
        # On feat/orbit-isorropia-archive (chemistry_iters == 0) the iso
        # loop uses skip_solve_species for the 7 stable species, so it
        # never reads from baseline_lu_cache. Keep nothing in that case to
        # avoid holding unused LUs (~1 GB) for the rest of the run.
        if chemistry_iters > 0:
            keep_across_iters = {IDX_SOA, IDX_PM25, IDX_POA, IDX_PSO4}
        else:
            keep_across_iters = set()
        t0 = time.time()
        _RSS.set_phase("iter0")
        print("  [iter 0] Baseline solve (prescribed chemistry)")
        # The iso loop's partial-reassembly fastpath is disabled at its
        # call site (iso_rebuild_dep = None, 2026-05-19 postmortem), so
        # nothing consumes a returned assembly snapshot. Not requesting it
        # keeps the snapshot from pinning all 104 matrices through the
        # solve, which defeated the solver's per-species streaming drain
        # (memory-audit fix 3, 2026-08-01; ~870 MB). Re-request it here if
        # the fastpath is ever re-enabled.
        request_assembly = False
        result = _assemble_and_solve(
            grids, indexer, emissions_SN, N,
            umfpack_sym, perm, c_warm_SN,
            tol=1e-6, maxiter=200,
            return_lu_cache=True,
            keep_lu_species=keep_across_iters,
            disable_night_nox=disable_night_nox,
            return_assembly=request_assembly,
        )
        baseline_lu_cache = result.get("lu_cache", {})
        baseline_assembly = result.get("assembly", None)
        timings["orbit_solve_iter0"] = time.time() - t0
        print(f"  [iter 0] Solve: {_fmt_time(timings['orbit_solve_iter0'])}")

    # Snapshot the original preproc partitioning before any closure mutates
    # it (update_grid_partitioning overwrites NHPartitioning with the LUT
    # marginal). These snapshots back the "prescribed=" diagnostic and the
    # pm25_mean_baseline reference so the headline "with-closure vs without-
    # closure" comparison stays meaningful even after the closure runs.
    orig_NHPartitioning_per_bin = [g.NHPartitioning.copy() for g in grids]
    orig_NHPartitioningEq_per_bin = [
        g.NHPartitioningEq.copy() if g.NHPartitioningEq.size > 0
        else g.NHPartitioning.copy()
        for g in grids
    ]
    orig_NOPartitioning_per_bin = [g.NOPartitioning.copy() for g in grids]
    # Baseline PM2.5 reference: orbit-avg from the BASELINE solve using the
    # ORIGINAL (preproc) partitioning, before the ISORROPIA closure runs.
    # This is the proper "without ISORROPIA closure" reference for the
    # headline comparison; once the closure mutates grids, the post-solve
    # `pm25_mean` no longer represents that.
    if resume_iter == 0:
        pm25_orbit_baseline = _pm25_per_bin(
            result["orbits"], grids, N, nz, ny, nx,
        )
        pm25_mean_baseline = pm25_orbit_baseline.mean(axis=0)
    else:
        pm25_orbit_baseline = None
        pm25_mean_baseline = None

    # 7. Chemistry iterations — Phase 3d outer closure.
    # The closure iterates diagnostic oxidants + ISORROPIA partitioning
    # + operator re-assembly + orbit re-solve, under-relaxed by alpha,
    # to a three-gate convergence criterion on OH / O3 / partitioning.
    chem_summary_per_iter = []
    history = resumed_history if resumed_history is not None else ConvergenceHistory()
    prev_chem = None
    prev_iso = None
    baseline_result = result

    # Opt 1: Anderson accelerator for the OH chemistry closure.
    # State vector is the flattened per-bin OH field.  Only constructed
    # when the user asks for it via --closure-accel anderson.
    # Only report the scheme when the loop that uses it will actually run.
    # chemistry_iters == 0 leaves the OH loop empty, and announcing a
    # closure that never executes reads as production behaviour in the log.
    accel = None
    if closure_accel == "anderson":
        accel = AndersonAccelerator(m=anderson_m)
    if chemistry_iters > 0:
        print("  OH closure acceleration: "
              + (f"Anderson(m={anderson_m}) on OH "
                 f"(safeguard -> Picard on rejection)" if accel is not None
                 else f"Picard + alpha={closure_alpha}"))
    else:
        print("  OH closure: not run (--chemistry-iters 0); "
              "oxidants are archive-prescribed")

    # Optional ISORROPIA LUT for the closure (independent of the post-solve
    # ISORROPIA extraction path). `closure_mode == "full"` engages it,
    # regardless of chemistry_iters: the partitioning closure and the OH
    # closure are logically separate (the chemistry_iters > 0 gate that
    # used to live here conflated them).
    closure_lut = None
    if closure_mode == "full":
        closure_lut_path = lut_path
        if (closure_lut_path and closure_lut_path.lower() != "none"
                and os.path.exists(closure_lut_path) and grids[0].RH.size > 0):
            from orbit.core.isorropia_lut import IsorropiaLUT
            # Closure path only needs f_nh4 and f_no3.  Skipping
            # aerosol_water and ph halves the LUT memory footprint
            # (~1.7 GB -> ~0.85 GB of interpolator data per field pair).
            closure_lut = IsorropiaLUT(
                closure_lut_path, fields=("f_nh4", "f_no3"),
            )
            print(f"  Closure LUT: {closure_lut_path} (2 fields)")
        else:
            print("  Closure LUT unavailable -> partitioning closure skipped "
                  "(TotalNO3 will deposit as 100% HNO3, fast bound)")

    # ── ISORROPIA partitioning closure (chemistry_iters == 0 path) ────────
    # Picard on (f_NH, f_NO3) with the LUT, with operator + orbit reassembly
    # at each step. Oxidant rates stay archive-prescribed throughout.
    # Required because the preprocessor doesn't supply HNO3↔pNO3 partitioning
    # (GC archive lacks HNO3 separately), so without this loop the TotalNO3
    # operator falls back to all-HNO3 deposition (`_get_no3_partitioning`
    # returns zeros) — biases TotalNO3 low, then the prescribed mass split
    # assumes 100% particle, which is internally inconsistent.
    # Methods note (paper, §"Equilibrium closure"): the LUT query inside
    # `update_grid_partitioning` writes BOTH `*Partitioning` (marginal,
    # d(f·c)/dc — used by the deposition operator) and `*PartitioningEq`
    # (equilibrium, f at current totals — used by mass extraction). the archive
    # uses one quantity for both; ORBIT doesn't. See `dcomp_isorropia.py`
    # docstring for the derivation. ISORROPIA inputs use totalSO4 = pSO4
    # alone (SO2 is not in the equilibrium pool — verified bug fix kept).
    #
    # Convergence gate: the paper's purpose is SPECIATED source-receptor
    # analysis (NOx scenarios drive pNO3, NH3 scenarios drive pNH4), so
    # the closure must converge each scientifically-claimed species
    # independently — gating on Σ PM2.5 would let pNO3 ↑ and pNH4 ↓
    # cancel in the sum, leaving the speciated answers depending on
    # which iter you happened to stop at.
    #
    # Metric: max(p90(|ΔpNO3|/max(pNO3, F)), p90(|ΔpNH4|/max(pNH4, F)))
    # at the surface, F = 0.5 µg/m³. This matches the convention from
    # dcomp_iter (FLOOR_PM = 1 µg/m³ for total PM2.5; pNO3/pNH4 are each
    # ~half of PM2.5 over IGP, so 0.5 is the analogous per-species floor).
    # Mass floor (not partitioning floor): a clean-marine cell with pNO3
    # ≈ 0.05 µg/m³ doesn't gate (well below 0.5); a Delhi cell with pNO3
    # ≈ 15 µg/m³ does (Δ of 0.3 → relative ≈ 2%). p90 (not p99): a single
    # oscillating coastal cell at the boundary of two LUT basins is a
    # fixed-point pathology and shouldn't gate the answer when 90% of
    # cells are converged — see the iso loop's `_gate_metrics` docstring.
    FLOOR_PSPECIES_UGM3 = 0.5
    iso_closure_history = []   # per-iter (mean_fno3, no3_p90, nh4_p90)
    iso_named_cell_state = {}
    iso_accel = None
    iso_basin_flip = None
    if (chemistry_iters == 0
            and closure_lut is not None
            and isorropia_closure_iters > 0
            and resume_iter == 0):
        prev_iso = None
        prev_pno3_surf, prev_pnh4_surf = _pno3_pnh4_surf_per_bin(
            result["orbits"], grids, N, nz, ny, nx,
        )
        accel_label = (f"Anderson(m={anderson_m})" if isorropia_anderson
                       else "Picard")
        if isorropia_anderson:
            iso_accel = AndersonAccelerator(m=anderson_m)
        if basin_flip_damp:
            iso_basin_flip = _BasinFlipDetector(N_BINS, nz, ny, nx)
        print(f"  ISORROPIA partitioning closure: max {isorropia_closure_iters} "
              f"iters, alpha={closure_alpha}, accel={accel_label}, "
              f"basin_flip_damp={'on' if basin_flip_damp else 'off'}")
        print(f"    Gate: max(p90(|ΔpNO3|/max(pNO3, {FLOOR_PSPECIES_UGM3} µg/m³)), "
              f"same for pNH4) < {closure_tol}")
        _RSS.set_phase("iso")
        for iso_it in range(1, isorropia_closure_iters + 1):
            t_iso = time.time()
            new_iso = update_grid_partitioning(
                grids, result["orbits"], closure_lut,
                # Anderson does its own mixing; pass alpha=1 to the LUT
                # update so we get the raw fixed-point query (g(x_k)),
                # then the accelerator combines it with history. Picard
                # path keeps the under-relaxation here.
                alpha=1.0 if isorropia_anderson else closure_alpha,
                prev_per_bin=None if isorropia_anderson else prev_iso,
            )
            if isorropia_anderson:
                # State vector = surface f_NO3 across bins (the slowest-
                # converging mode). Anderson combines history into a step
                # that's then written back to ALL bins/levels via a
                # direct LUT requery at the new f_NO3.  Cheap: one extra
                # LUT call per iter.
                f_no3_surf_new = np.stack(
                    [new_iso[tau]["f_no3_eq"][0] for tau in range(N_BINS)], axis=0,
                )
                if prev_iso is None:
                    # Iter 1: no history. Take alpha-blended Picard step.
                    f_no3_surf_prev = np.zeros_like(f_no3_surf_new)
                    accel_x_next = (closure_alpha * f_no3_surf_new
                                    + (1.0 - closure_alpha) * f_no3_surf_prev)
                    anderson_accepted = True
                else:
                    f_no3_surf_prev = np.stack(
                        [prev_iso[tau]["f_no3_eq"][0] for tau in range(N_BINS)], axis=0,
                    )
                    accel_x_next, anderson_accepted = iso_accel.apply_with_safeguard(
                        f_no3_surf_prev.ravel(), f_no3_surf_new.ravel(),
                    )
                    accel_x_next = np.clip(
                        accel_x_next.reshape(f_no3_surf_new.shape), 0.0, 1.0,
                    )
                # Blend the accelerated surface f_NO3 back into the new
                # 3D fields by scaling: the column profile shape is
                # preserved from the LUT query, just the magnitude gets
                # nudged toward the Anderson estimate.
                for tau in range(N_BINS):
                    surf_new = f_no3_surf_new[tau]
                    surf_acc = accel_x_next[tau]
                    safe = np.where(surf_new > 1e-6, surf_new, 1.0)
                    scale = np.where(surf_new > 1e-6, surf_acc / safe, 1.0)
                    # Apply scale to all levels.
                    new_iso[tau]["f_no3_eq"] = np.clip(
                        new_iso[tau]["f_no3_eq"] * scale[None, :, :], 0.0, 1.0,
                    )
                    new_iso[tau]["f_no3_marg"] = np.clip(
                        new_iso[tau]["f_no3_marg"] * scale[None, :, :], 0.0, 1.0,
                    )
                    grids[tau].NO3PartitioningEq = new_iso[tau]["f_no3_eq"]
                    grids[tau].NO3Partitioning = new_iso[tau]["f_no3_marg"]
                if not anderson_accepted:
                    print(f"    [anderson] safeguard fired -> Picard fallback "
                          f"(falls={iso_accel.n_safeguard_falls})")
            prev_iso = new_iso
            if iso_basin_flip is not None:
                n_new_no3, n_new_nh4 = iso_basin_flip.update_and_freeze(
                    new_iso, grids,
                )
                tot_no3, tot_nh4 = iso_basin_flip.total_frozen()
                if n_new_no3 or n_new_nh4 or tot_no3 or tot_nh4:
                    print(f"    [basin-flip] new this iter: "
                          f"NO3={n_new_no3} NH4={n_new_nh4}  "
                          f"cumulative frozen: NO3={tot_no3} NH4={tot_nh4}")
            mean_fno3 = float(np.stack(
                [prev_iso[tau]["f_no3_eq"][0] for tau in range(N_BINS)], axis=0
            ).mean())
            partitioning_summary(prev_iso)
            iso_named_cell_state = _named_cell_iso_report(
                result, grids, prev_iso, nz, ny, nx,
                iso_named_cell_state, noon_tau=2,
            )
            # Build warm start from the prior orbit's mean — saves ~30%
            # GMRES iters on the resolve when partitioning has only
            # nudged the operator slightly.
            c_warm_iter = np.empty(N_SPECIES * N, dtype=np.float64)
            for s in range(N_SPECIES):
                mean_s = np.mean(
                    [result["orbits"][s][tau + 1] for tau in range(N_BINS)],
                    axis=0,
                )
                c_warm_iter[s * N:(s + 1) * N] = mean_s

            # Use krylov_tol_intermediate (Opt 3) on iso-loop solves.  The
            # loop's own convergence gate (~closure_tol = 2%) is well above
            # any 1e-4 GMRES residual, so tightening to 1e-6 here would
            # waste Krylov work on noise.  The downstream post-loop ISO
            # extraction recomputes f at the saved orbit; if you want
            # tighter saved orbits, pass --krylov-tol-intermediate 1e-6.
            #
            # Skip-solve optimisation: only TotalNH and TotalNO3 operators
            # change in the iso loop (their deposition diagonals blend
            # in the LUT marginal partitioning). Org/PM25/SO2/NOx/pSO4/O3/CO
            # operators are stable across iso iters because:
            #   - Org/PM25: no chemistry, no source coupling
            #   - SO2: SO2_dry_dep + grid.SO2oxidation (preproc)
            #   - NOx: NOx_dry_dep + archive-driven NOx→TotalNO3 rate
            #     (built once at baseline, no partitioning dependence)
            #   - pSO4: particle dep + K_source from SO2 (SO2 orbit stable
            #     → source stable)
            #   - O3/CO: zero everything (already 0-iter solve)
            # For these 7 species, both operator AND RHS are unchanged
            # between iso iters, so we copy their orbits verbatim from
            # the prior result rather than refactoring + re-solving.
            # Expected speedup: per-iter resolve ~4.5 min → ~1 min.
            # Full rebuild every iso iter (partial-rebuild fastpath
            # was empirically slower: numerically bit-identical, but
            # sparse-matrix addition recomposition cost outweighs the
            # transport-cache savings.
            # The fastpath infrastructure (cached_assembly with K_age
            # caching) is preserved in _assemble_and_solve for potential
            # future use, just disabled at this call site.
            iso_skip_solve = set()
            iso_rebuild_dep = None
            # Streaming DAG cascade: pass an empty keep_lu_species so the
            # solver pops each species's LU factor immediately after its
            # orbit is merged, rather than holding 13 × 8 = 104 LU factors
            # (~27 GB) in fresh_lu_cache until end-of-call. The iso loop
            # rebuilds operators next iter anyway, so we never re-use these
            # LUs. Expected peak-RSS reduction ~12-15 GB, ~0 wall cost.
            # No cached_assembly / return_assembly: the full-reassembly
            # branch ignores the cache, and holding the previous iter's
            # snapshot across the rebuild doubled the assembly footprint
            # (memory-audit fix 3). The fastpath infrastructure in
            # _assemble_and_solve is intact; re-wire both if re-enabling.
            baseline_assembly = None
            result = _assemble_and_solve(
                grids, indexer, emissions_SN, N,
                umfpack_sym, perm, c_warm_iter,
                tol=krylov_tol_intermediate, maxiter=200,
                disable_night_nox=disable_night_nox,
                skip_solve_species=iso_skip_solve,
                cached_orbits=result["orbits"],
                cached_assembly=None,
                rebuild_deposition_species=iso_rebuild_dep,
                keep_lu_species=set(),
                fct_enabled=horizontal_fct,
            )
            # Per-species mass-convergence metric. Computing AFTER the
            # resolve so it reflects the full closed-loop response of
            # this iter's partitioning + reassembly + orbit solve.
            new_pno3_surf, new_pnh4_surf = _pno3_pnh4_surf_per_bin(
                result["orbits"], grids, N, nz, ny, nx,
            )
            def _gate_metrics(new, prev, floor, lon, lat):
                """Return (p90, p50, argmax_info) of |Δ|/max(|new|,|prev|,floor).

                Using p90 (not p99) as the gate: a single oscillating
                coastal cell at the boundary of two LUT basins (e.g. the
                Andaman Sea cell at 6°N/95°E observed on 2026-04-26
                bouncing between sea-salt-dominated and sulfate-dominated
                LUT regimes) is a fixed-point pathology in <0.01% of cells
                and shouldn't gate the answer when the bulk is converged.
                p90 still requires 90% of cells under tol — strong enough
                for paper claims, tolerant of regional pathologies.
                argmax kept for diagnostic logging.
                """
                denom = np.maximum(np.maximum(np.abs(new), np.abs(prev)), floor)
                rel = np.abs(new - prev) / denom
                p90 = float(np.percentile(rel, 90))
                p50 = float(np.percentile(rel, 50))
                idx = np.unravel_index(np.argmax(rel), rel.shape)
                tau, j, i = idx
                rel_max = float(rel[idx])
                new_v = float(new[idx])
                prev_v = float(prev[idx])
                lon_v = float(lon[i]) if i < len(lon) else float("nan")
                lat_v = float(lat[j]) if j < len(lat) else float("nan")
                return p90, p50, (lat_v, lon_v, tau, new_v, prev_v, rel_max)
            no3_p90, no3_p50, no3_arg = _gate_metrics(
                new_pno3_surf, prev_pno3_surf, FLOOR_PSPECIES_UGM3,
                grids[0].lon, grids[0].lat)
            nh4_p90, nh4_p50, nh4_arg = _gate_metrics(
                new_pnh4_surf, prev_pnh4_surf, FLOOR_PSPECIES_UGM3,
                grids[0].lon, grids[0].lat)
            print(f"    pNO3 argmax: lat={no3_arg[0]:.1f}°, lon={no3_arg[1]:.1f}°, "
                  f"bin={no3_arg[2]+1}, prev={no3_arg[4]:.3f}, "
                  f"new={no3_arg[3]:.3f} µg/m³ "
                  f"(rel={no3_arg[5]:.3f}, p50={no3_p50:.3e})")
            print(f"    pNH4 argmax: lat={nh4_arg[0]:.1f}°, lon={nh4_arg[1]:.1f}°, "
                  f"bin={nh4_arg[2]+1}, prev={nh4_arg[4]:.3f}, "
                  f"new={nh4_arg[3]:.3f} µg/m³ "
                  f"(rel={nh4_arg[5]:.3f}, p50={nh4_p50:.3e})")
            gate_metric = max(no3_p90, nh4_p90)
            iso_closure_history.append((iso_it, mean_fno3, no3_p90, nh4_p90))

            # VBS Pankow closure: piggyback on the iso outer iteration.
            # Reads orbit-mean concentrations from per-bin orbits, writes
            # grid.M_OA_3d + grid.F_p_vbs for the next operator rebuild.
            # Each grid (per UTC bin) gets its own M_OA + F_p_vbs since the
            # bin-mean concentrations differ slightly across bins.
            for tau in range(N_BINS):
                # Build c_mean for this bin from result["orbits"] (per-species
                # array of orbit endpoints per bin).
                c_bin = np.empty(N_SPECIES * N, dtype=np.float64)
                for s in range(N_SPECIES):
                    c_bin[s * N:(s + 1) * N] = np.maximum(
                        result["orbits"][s][tau + 1], 0.0)
                vbs_diag = update_vbs_partitioning(grids[tau], c_bin, indexer)
            # Use the last bin's diagnostic for the closure-history line
            # (representative; M_OA varies by ~10% across bins typically).
            print(f"  [iso {iso_it}] resolve: {_fmt_time(time.time() - t_iso)} "
                  f"(GMRES tol={krylov_tol_intermediate:.0e})  "
                  f"f_NO3 surf mean={mean_fno3:.3f}  "
                  f"p90|ΔpNO3|={no3_p90:.3e}  p90|ΔpNH4|={nh4_p90:.3e}  "
                  f"(gate=max={gate_metric:.3e})  "
                  f"M_OA surf mean={vbs_diag['M_OA_surface_mean']:.2f} µg/m³ "
                  f"(VBS iters={vbs_diag['iters_run']})")
            if gate_metric < closure_tol and iso_it >= 2:
                print(f"  [iso] converged at iter {iso_it} "
                      f"(max(p90|ΔpNO3|, p90|ΔpNH4|)={gate_metric:.3e} "
                      f"< {closure_tol})")
                break
            prev_pno3_surf = new_pno3_surf
            prev_pnh4_surf = new_pnh4_surf
        else:
            # for/else: ran to max iters without break
            print(f"  [iso] hit max iters ({isorropia_closure_iters}) "
                  f"with last gate metric={gate_metric:.3e} >= "
                  f"{closure_tol} — proceeding (IGP cells may still be "
                  f"converged; check the named-cell trajectory)")
        # The final α=1.0 pass that aligns grid.*PartitioningEq with the
        # converged orbit is run further down (the existing post-loop block
        # at "Final ISORROPIA pass (consistent with converged orbit)") —
        # gated only on `closure_lut is not None`, so it now fires for
        # both chemistry_iters>0 and chemistry_iters==0 paths.

    # Cache the iter-(k-1) oxidants / O3 / partitioning / PM2.5 for Δ metrics
    prev_oh_per_bin = None
    prev_o3_per_bin = None
    prev_fnh4_per_bin = None
    prev_fno3_per_bin = None
    prev_pm25_per_bin = None
    # Named-cell per-iter trajectory state (Delhi, Kanpur, ...).  Each
    # entry is {cell_label: {'OH': ..., 'HO2': ..., 'fnh4': ..., ...}}
    # from the prior iter, used to print Δ% columns in the log.
    prev_named_cell = {}

    # Initial prev_* from baseline: OH / f_NH / f_NO3 use the prescribed
    # grid fields so iter-1 Δ is measured against the Phase 3c starting
    # point.
    prev_fnh4_per_bin = [
        (g.NHPartitioningEq if g.NHPartitioningEq.size > 0
         else g.NHPartitioning).copy()
        for g in grids
    ]
    prev_fno3_per_bin = [
        np.zeros((nz, ny, nx)) if g.NO3PartitioningEq.size == 0
        else g.NO3PartitioningEq.copy()
        for g in grids
    ]
    prev_o3_per_bin = [
        np.maximum(baseline_result["orbits"][IDX_O3][tau + 1], 0.0).reshape((nz, ny, nx))
        for tau in range(N_BINS)
    ]
    prev_pm25_per_bin = _pm25_per_bin(baseline_result["orbits"], grids, N, nz, ny, nx)

    # Seed the LU / orbit caches for the first iteration so the inner-loop
    # short-circuit works regardless of whether we ran baseline or resumed.
    prev_cache = baseline_lu_cache
    prev_orbits_cache = baseline_result["orbits"]

    # Plausibility check result — filled at iter 1, otherwise left as
    # "skipped".  Persisted to the final NPZ for later audits.
    plausibility_ok = True
    plausibility_msgs = ["(not run: resume or chemistry-iters=0)"]

    last_iter_completed = resume_iter
    for it in range(resume_iter + 1, chemistry_iters + 1):
        t_iter = time.time()
        print(f"  [iter {it}] Computing oxidants + chemistry rates")
        # Phase 3e bootstrap: iter 1 uses HEMCO O3 (orbit O3 from baseline is
        # zero because chemistry was inactive).  Iter 2+ uses ORBIT's own
        # orbit-derived O3 (self-consistent OH<->O3 loop).
        # CO closure (PM-first scope): iter 2+ uses orbit CO so emission
        # scenarios feed back through OH.  Disabled by default until a CO
        # emission inventory is wired in — with zero CO emissions the orbit
        # solves to c_CO = 0 everywhere, which would wrongly remove the
        # CO sink from the OH budget.  HEMCO CO is the safe fallback.
        use_orbit_o3 = (it >= 2)
        use_orbit_co = enable_orbit_co and (it >= 2)
        chem = compute_chemistry_per_bin(
            grids, result["orbits"], photolysis_lut,
            hemco_clim.species_ugm3,
            year=2016, month=month, day=15, o3_column_du=275.0,
            use_orbit_o3=use_orbit_o3,
            use_orbit_co=use_orbit_co,
            o3_bc_target_3d=o3_bc_target_3d,
            o3_bc_rate_3d=o3_bc_rate_3d,
            prescribed_oh=prescribed_oh,
        )

        if closure_accel == "anderson":
            # Map-change bootstrap: iter 1 uses HEMCO O3, iter 2+ uses
            # orbit-derived O3.  The fixed-point map g(x) genuinely
            # changes between iter 1 and iter 2, so Anderson's history
            # from iter 1 is from a different map than iter 2's update.
            # Reset the accelerator at iter 2 so iter 2 becomes a fresh
            # Picard step and iter 3+ accelerates on the stationary map.
            # Without this reset, Anderson extrapolates across a
            # discontinuity and produces a contaminated update
            # (observed: ~20-29% PM2.5 shift at IGP cells in iter 3).
            if it == 2 and accel is not None and len(accel.X) > 0:
                accel.reset()
                print("    [anderson] reset history at map-change boundary "
                      "(iter 2: HEMCO->orbit O3)")
            # Opt 1: Anderson(m) acceleration of the OH fixed point.
            # State vector = flattened OH across bins.  First iter has
            # no prior OH, so the update is Picard (beta=1 * residual).
            # Safeguard: if projected residual > 1.5 * Picard residual,
            # fall back to a Picard step and reset history.
            oh_current_flat = np.concatenate([
                np.asarray(chem.oxidants[t].OH).ravel()
                for t in range(N_BINS)
            ])
            if prev_chem is None:
                # Iter 1: no prior OH.  accel.update takes (x_k, g_x_k)
                # with r_k = g_x_k - x_k; passing (current, current) gives
                # r_k = 0 and the Picard step x + beta*0 = x.  We instead
                # pass the prescribed (grid) OH as x_k so iter 1 still
                # computes a meaningful residual under the Anderson
                # update — but more robustly, just take the fresh OH as
                # the iter-1 state (this matches Picard at alpha=1).
                oh_next_flat = oh_current_flat
                anderson_accepted = True
            else:
                oh_prev_flat = np.concatenate([
                    np.asarray(prev_chem.oxidants[t].OH).ravel()
                    for t in range(N_BINS)
                ])
                oh_next_flat, anderson_accepted = accel.apply_with_safeguard(
                    oh_prev_flat, oh_current_flat,
                )

            splice_oh_into_chem(chem, oh_next_flat, nz, ny, nx)
            rebuild_rates_from_chem(chem, grids)
            if not anderson_accepted:
                print(f"    [anderson] safeguard -> Picard fallback "
                      f"(falls={accel.n_safeguard_falls} total)")
            else:
                n_hist = len(accel.X)
                print(f"    [anderson] accepted, history_len={n_hist}")
        elif closure_alpha < 1.0:
            # Picard + alpha under-relaxation (default).
            chem = under_relax_oxidants(chem, prev_chem, alpha=closure_alpha)
        prev_chem = chem

        diag = chemistry_diagnostics(chem, grids)
        chem_summary_per_iter.append(diag)
        # Surface-mean OH per bin for quick visibility
        print("    Surface-mean OH per bin (molec/cm3):")
        for d in diag:
            print(f"      Bin {d['bin']}: OH={d['OH_surf_mean']:.2e} "
                  f"f_NO2={d['f_NO2_surf_mean']:.2f} "
                  f"k_so2(+gas)={d['k_so2_tot_surf_mean']:.2e}  "
                  f"(aq={d['k_so2_aq_surf_mean']:.2e})  "
                  f"k_nox_no3={d['k_nox_to_no3_surf_mean']:.2e}")

        # ISORROPIA closure: marginal + equilibrium partitioning.  Mutates
        # each grid's NHPartitioning / NHPartitioningEq / NO3Partitioning /
        # NO3PartitioningEq so the next operator assembly picks up the
        # closed partitioning.
        if closure_lut is not None:
            prev_iso = update_grid_partitioning(
                grids, result["orbits"], closure_lut,
                alpha=closure_alpha, prev_per_bin=prev_iso,
            )
            partitioning_summary(prev_iso)

        print(f"  [iter {it}] Reassembling + solving with DCOMP rates")
        c_warm = np.empty(N_SPECIES * N, dtype=np.float64)
        for s in range(N_SPECIES):
            mean_s = np.mean([result["orbits"][s][tau + 1] for tau in range(N_BINS)],
                              axis=0)
            c_warm[s * N:(s + 1) * N] = mean_s

        # LU / solve cache.  Species whose operator is unchanged across
        # outer iterations can reuse their LU factorisation; species whose
        # operator AND RHS are both unchanged skip the orbit solve entirely
        # and copy the prior orbit verbatim.
        #
        # Org / PM25 have no chemistry- or partitioning-dependent terms
        # AND no source couplings, so both matrix and RHS are stable
        # across iters.
        #
        # Phase 3e: O3 now has chemistry rates that depend on OH/HO2/NO/RO2
        # and a bin-varying RHS source, so it changes every outer iter
        # and is no longer in the skip-solve set.
        #
        # Under "full" closure, NH and TotalNO3 operators also change each
        # iter because their deposition blends in the LUT marginal
        # partitioning.  Under "chem-only" closure those are stable too.
        # POA belongs here with PrimaryPM25: both are chemically inert with
        # static particle deposition, so neither their operator nor their RHS
        # changes across closure iterations. Omitting POA made it re-solve
        # every iteration at the looser intermediate Krylov tolerance while
        # PrimaryPM25 stayed frozen at its tight iteration-0 answer, so the
        # two drifted apart and POA + PrimaryPM25 no longer reproduced the
        # unsplit PM2.5 (~0.5% per cell). Found by the Phase 3 split gate.
        unchanged_always = {IDX_SOA, IDX_PM25, IDX_POA}
        changed_each_iter = {IDX_SO2, IDX_NOX, IDX_O3}
        if closure_lut is not None:
            changed_each_iter |= {IDX_TOTAL_NH, IDX_TOTAL_NO3}
        unchanged_this_iter = unchanged_always | (
            set(range(N_SPECIES)) - changed_each_iter - unchanged_always
        )

        # LU cache: keep factorisations for unchanged species.  prev_cache
        # was seeded from baseline before the loop and is refreshed at the
        # bottom of every iter.
        lu_cache_in = {s: prev_cache[s] for s in unchanged_this_iter
                       if s in prev_cache}

        # Species that can also skip the solve (matrix AND RHS identical).
        # Org and PM25: no source couplings ever.  O3: no source couplings
        # wired yet.  Under chem-only closure NH is also RHS-stable
        # (no coupling into NH), but operator stability still relies on
        # partitioning not changing — which is true in chem-only.
        skip_solve_now = set(unchanged_always) & set(prev_orbits_cache.keys())

        # GMRES tolerance policy: the outer-closure gate is 2%, so
        # solving the orbit to 1e-6 on non-final iters wastes back-solve
        # work (back-solves are ~60% of wall from the 2026-04-24 profile).
        # Use krylov_tol_intermediate (default 1e-4) on non-final iters,
        # and tighten to 1e-6 on the final iter that writes the NPZ.
        # Krylov count typically drops ~25-30% at this tolerance.  See
        # notes/2026-04-24_orbit_solver_optimizations.md (Opt 3).
        # Override via --krylov-tol-intermediate 1e-6 to restore the
        # pre-Opt-3 behaviour (useful for reproducibility / ablation).
        is_final_iter = (it == chemistry_iters)
        krylov_tol = 1e-6 if is_final_iter else krylov_tol_intermediate

        # Horizontal FCT deferred correction recomputes the anti-diffusive
        # source from the current orbit every iteration, so the RHS changes for
        # every transported species: no species may be skipped, else its FCT
        # source would be frozen at a stale iterate.
        if horizontal_fct:
            skip_solve_now = set()

        prev_result = result
        result = _assemble_and_solve(
            grids, indexer, emissions_SN, N,
            umfpack_sym, perm, c_warm,
            tol=krylov_tol, maxiter=200,
            k_so2_per_bin=chem.k_so2_rate,
            k_nox_per_bin=chem.k_nox_to_no3_rate,
            k_o3_loss_per_bin=chem.k_o3_loss_rate,
            s_o3_source_per_bin=chem.s_o3_source_rate,
            k_co_loss_per_bin=chem.k_co_loss_rate,
            lu_cache=lu_cache_in,
            skip_solve_species=skip_solve_now,
            cached_orbits=prev_orbits_cache,
            return_lu_cache=True,
            keep_lu_species=unchanged_this_iter,
            fct_enabled=horizontal_fct,
        )
        # Promote the fresh LU cache for the next iter.
        prev_cache = result.get("lu_cache", {})
        prev_orbits_cache = result["orbits"]
        timings[f"orbit_solve_iter{it}"] = time.time() - t_iter

        # Iter-1 plausibility pre-flight
        if it == 1:
            plausibility_ok, plausibility_msgs = plausibility_check(
                result, baseline_result, chem, DTAU,
            )
            print("  === Iteration 1 plausibility check ===")
            for msg in plausibility_msgs:
                print(f"    {msg}")
            if not plausibility_ok:
                print("  ⚠ plausibility check flagged issues — continuing, but "
                      "inspect the diagnostics above before trusting output.")

        # Build per-bin fields for metric computation
        oh_per_bin = [chem.oxidants[tau].OH for tau in range(N_BINS)]
        o3_per_bin = [
            np.maximum(result["orbits"][IDX_O3][tau + 1], 0.0).reshape((nz, ny, nx))
            for tau in range(N_BINS)
        ]

        # O3 top-layer vs surface domain means.  With top-BC on, expect
        # top-layer to track the target (~70-100 ppbv = 140-200 ug/m3) and
        # surface to be chemistry+deposition-dominated (10-60 ppbv on land).
        # Pre-BC baseline had domain O3 ~2-3 ug/m3 everywhere — the diff
        # tells us whether the BC is meaningfully propagating.
        o3_surf_mean = float(np.mean([o[0].mean() for o in o3_per_bin]))
        o3_top_mean = float(np.mean([o[-1].mean() for o in o3_per_bin]))
        if o3_bc_target_3d is not None:
            # Show per-layer target means at the same layers printed by
            # the O3 profile (top, surface) rather than a full-3D mean
            # that would average in BL zeros.
            tgt_top = float(o3_bc_target_3d[-1].mean())
            tgt_surf = float(o3_bc_target_3d[0].mean())
            print(f"    O3 (bin-avg domain mean): surface={o3_surf_mean:.1f} "
                  f"(target@k=0: {tgt_surf:.1f}), top={o3_top_mean:.1f} "
                  f"(target@top: {tgt_top:.1f}) ug/m3")
        else:
            print(f"    O3 (bin-avg domain mean): surface={o3_surf_mean:.1f} ug/m3, "
                  f"top={o3_top_mean:.1f} ug/m3  (no BC)")
        fnh4_per_bin = [
            grids[tau].NHPartitioningEq.copy() if grids[tau].NHPartitioningEq.size > 0
            else grids[tau].NHPartitioning.copy()
            for tau in range(N_BINS)
        ]
        fno3_per_bin = [
            grids[tau].NO3PartitioningEq.copy() if grids[tau].NO3PartitioningEq.size > 0
            else (grids[tau].NO3Partitioning.copy()
                  if grids[tau].NO3Partitioning.size > 0
                  else np.zeros((nz, ny, nx)))
            for tau in range(N_BINS)
        ]
        pm25_per_bin = _pm25_per_bin(result["orbits"], grids, N, nz, ny, nx)

        # NOx and TotalNO3 negatives tracking (aloft-noise sentinel from Phase 3c)
        c_nox_per_bin = [
            result["orbits"][IDX_NOX][tau + 1].reshape((nz, ny, nx))
            for tau in range(N_BINS)
        ]
        c_no3_per_bin = [
            result["orbits"][IDX_TOTAL_NO3][tau + 1].reshape((nz, ny, nx))
            for tau in range(N_BINS)
        ]

        # Mass-conservation check (cheap; should be ~0 by construction)
        inv = check_invariants(result, chem, DTAU)

        # If OH from chem was built on orbits from iter it-1, we only have a
        # valid prev OH from iter 2 onward. For iter 1 compare against the
        # same chem field (Δ = 0 by definition). Use a zero array as prev OH
        # on iter 1 so the printed Δ reflects magnitude, not truly zero.
        if prev_oh_per_bin is None:
            # No prior OH; use this iter's OH as "prev" so Δ = 0 on iter 1.
            prev_oh_for_metric = [x.copy() for x in oh_per_bin]
        else:
            prev_oh_for_metric = prev_oh_per_bin

        metric = build_iteration_metrics(
            it,
            oh_new_per_bin=oh_per_bin, oh_prev_per_bin=prev_oh_for_metric,
            o3_new_per_bin=o3_per_bin, o3_prev_per_bin=prev_o3_per_bin,
            f_nh4_new_per_bin=fnh4_per_bin, f_nh4_prev_per_bin=prev_fnh4_per_bin,
            f_no3_new_per_bin=fno3_per_bin, f_no3_prev_per_bin=prev_fno3_per_bin,
            pm25_new_per_bin=pm25_per_bin, pm25_prev_per_bin=prev_pm25_per_bin,
            c_nox_per_bin=c_nox_per_bin, c_no3_per_bin=c_no3_per_bin,
            mass_residual_N=inv["mass_residual_N"],
            mass_residual_S=inv["mass_residual_S"],
            tol=closure_tol,
        )
        history.append(metric)
        print_metrics(metric, lat=grids[0].lat, lon=grids[0].lon)

        # Floor check — warn (don't abort) if any species breached 1e-3
        floor_warnings = []
        for k, v in inv.items():
            if k.startswith("nonneg_floor_") and v > 1e-3:
                floor_warnings.append(f"{k}={v:.2e}")
        if floor_warnings:
            print("    ⚠ non-negativity floor breached: " + ", ".join(floor_warnings))

        # Advance caches
        prev_oh_per_bin = oh_per_bin
        prev_o3_per_bin = o3_per_bin
        prev_fnh4_per_bin = fnh4_per_bin
        prev_fno3_per_bin = fno3_per_bin
        prev_pm25_per_bin = pm25_per_bin

        print(f"  [iter {it}] Iteration wall: {_fmt_time(timings[f'orbit_solve_iter{it}'])}")

        # Checkpoint after each outer iteration.  This is the insurance
        # policy: if iter k+1 crashes (physics blow-up, OOM, cluster
        # preemption) we can resume from the last good state with
        # `--resume` rather than redoing everything from baseline.
        _save_checkpoint(
            month, it, result, chem, prev_iso, history,
            closure_mode, closure_alpha, closure_tol,
            converged=history.converged(tol=closure_tol),
            grid_shape=(nz, ny, nx),
        )
        last_iter_completed = it

        # Named-cell trajectory (Delhi/Kanpur/...) at surface, noon bin —
        # prints Δ% vs the prior iter for OH, HO2, f_nh4_eq, f_no3_eq,
        # k_so2, c_pSO4, c_NH, c_NO3.  Makes Q1-style drift questions
        # directly visible from the SLURM log.
        prev_named_cell = _named_cell_report(
            chem, result, grids, prev_iso, nz, ny, nx, prev_named_cell,
            noon_tau=2,
        )

        if history.converged(tol=closure_tol) and it >= 2:
            print(f"  Converged at iter {it} (all gates < {closure_tol * 100:.1f}%)")
            break

    if chemistry_iters > 0 and not history.converged(tol=closure_tol):
        history.print_classification(tol=closure_tol)

    # After the loop, run a final ISORROPIA pass on the converged orbit so
    # the per-bin grid partitioning is consistent with the final solve —
    # important for the downstream PM2.5 mass extraction, which reads
    # grid.NHPartitioningEq / NO3PartitioningEq.
    if closure_lut is not None:
        final_iso = update_grid_partitioning(
            grids, result["orbits"], closure_lut,
            alpha=1.0, prev_per_bin=None,
            include_cross_partials=iso_cross_partials,
        )
        if iso_basin_flip is not None:
            # The α=1.0 LUT requery just overwrote grids[tau].*Partitioning*
            # at every cell with raw LUT values — including the cells we
            # froze in the iso loop. Re-apply the freeze so the saved
            # partitioning is consistent with the closure result.
            iso_basin_flip.reapply_to_grids(grids)
            for tau in range(N_BINS):
                m_no3 = iso_basin_flip.flip_no3[tau]
                m_nh4 = iso_basin_flip.flip_nh4[tau]
                if m_no3.any():
                    final_iso[tau]["f_no3_eq"][m_no3] = (
                        iso_basin_flip.frozen_f_no3_eq[tau][m_no3])
                    final_iso[tau]["f_no3_marg"][m_no3] = (
                        iso_basin_flip.frozen_f_no3_marg[tau][m_no3])
                if m_nh4.any():
                    final_iso[tau]["f_nh_eq"][m_nh4] = (
                        iso_basin_flip.frozen_f_nh4_eq[tau][m_nh4])
                    final_iso[tau]["f_nh_marg"][m_nh4] = (
                        iso_basin_flip.frozen_f_nh4_marg[tau][m_nh4])
        _RSS.set_phase("final")
        print("  Final ISORROPIA pass (consistent with converged orbit):")
        partitioning_summary(final_iso)
        if iso_basin_flip is not None:
            tot_no3, tot_nh4 = iso_basin_flip.total_frozen()
            if tot_no3 or tot_nh4:
                shape = iso_basin_flip.flip_no3.shape
                ncells = shape[0] * shape[1] * shape[2] * shape[3]
                print(f"  Basin-flip damp: {tot_no3} NO3 cells "
                      f"({100.0 * tot_no3 / ncells:.3f}%) and "
                      f"{tot_nh4} NH4 cells "
                      f"({100.0 * tot_nh4 / ncells:.3f}%) frozen at centroid")

    timings["orbit_solve"] = time.time() - t0
    print(f"  Total orbit solve: {_fmt_time(timings['orbit_solve'])}")

    orbits = result["orbits"]

    # 6. Extract prescribed PM2.5 per bin (9-species layout).
    # PM2.5 = PrimaryPM + POA + SoA + p_nh*TotalNH*N_TO_NH4
    #        + pSO4*S_TO_SO4 + p_no3*TotalNO3*N_TO_NO3
    # SoA (species 0) is now a pure-particle tracer; emissions = yield × VOC
    # added at emission time (no more p_org partitioning).
    # Uses equilibrium partitioning when available for NH/NO3 mass extraction
    # (state-variable split); marginal for operator (not visible here).
    from orbit.core.constants import N_TO_NH4, N_TO_NO3, S_TO_SO4
    # IDX_* names imported at module level — do NOT re-import here or
    # Python treats them as function-local and shadows the module names
    # throughout `process_month`, causing UnboundLocalError earlier in
    # the function.

    t0 = time.time()
    pm25_orbit = np.zeros((N_BINS, nz, ny, nx), dtype=np.float64)
    c_SN = np.empty(N_SPECIES * N, dtype=np.float64)
    for tau in range(N_BINS):
        for s in range(N_SPECIES):
            c_SN[s * N:(s + 1) * N] = orbits[s][tau + 1]

        g = grids[tau]
        # SoA = Σ_i F_p,i × C_i over the 5 VBS bins (canonical soa_mean), not
        # just the C100 bin (the old Variant-A single-tracer convention).
        c_soa  = _soa_3d_flat(lambda s: np.maximum(c_SN[s * N:(s + 1) * N], 0), g, N)
        c_pm   = np.maximum(c_SN[IDX_PM25      * N:(IDX_PM25 + 1)      * N], 0)
        c_poa  = np.maximum(c_SN[IDX_POA       * N:(IDX_POA + 1)       * N], 0)
        c_nh   = np.maximum(c_SN[IDX_TOTAL_NH  * N:(IDX_TOTAL_NH + 1)  * N], 0)
        c_pso4 = np.maximum(c_SN[IDX_PSO4      * N:(IDX_PSO4 + 1)      * N], 0)
        c_no3  = np.maximum(c_SN[IDX_TOTAL_NO3 * N:(IDX_TOTAL_NO3 + 1) * N], 0)

        # Prefer equilibrium partitioning for mass extraction.
        # SoA (species 0) enters as-is — pure-particle tracer.
        p_nh = g.NHPartitioningEq.ravel() if g.NHPartitioningEq.size > 0 else g.NHPartitioning.ravel()
        if g.NO3PartitioningEq.size > 0:
            p_no3 = g.NO3PartitioningEq.ravel()
        elif g.NO3Partitioning.size > 0:
            p_no3 = g.NO3Partitioning.ravel()
        else:
            # Archive lacks HNO3 -> NO3 partitioning unknown; assume all in
            # particle phase (upper bound). Later replaced by LUT.
            p_no3 = np.ones(N)

        pm25_flat = (c_pm + c_poa + c_soa
                     + p_nh * c_nh * N_TO_NH4
                     + c_pso4 * S_TO_SO4
                     + p_no3 * c_no3 * N_TO_NO3)
        pm25_orbit[tau] = pm25_flat.reshape((nz, ny, nx))

    pm25_mean = pm25_orbit.mean(axis=0)

    # Orbit-mean concentrations: mean of c_1..c_8
    c_orbit = np.zeros((N_SPECIES, N_BINS + 1, N), dtype=np.float64)
    c_mean = np.zeros((N_SPECIES, N), dtype=np.float64)
    for s in range(N_SPECIES):
        for step in range(N_BINS + 1):
            c_orbit[s, step] = orbits[s][step]
        c_mean[s] = np.mean(c_orbit[s, 1:], axis=0)  # mean of c_1..c_8

    timings["pm25"] = time.time() - t0

    # Headline numbers for the JSON sidecar (rendered into run.json by main).
    timings["pm25_surface_mean"] = float(pm25_mean[0].mean())
    timings["pm25_surface_max"] = float(pm25_mean[0].max())
    if pm25_mean_baseline is not None:
        timings["pm25_surface_mean_baseline"] = float(pm25_mean_baseline[0].mean())

    # Print PM2.5 diagnostics. After ISORROPIA closure has run, `pm25_mean`
    # uses the LUT equilibrium fractions on the converged orbit — that's
    # the headline answer, not "prescribed". The truly-prescribed reference
    # (baseline orbit + preproc partitioning) is in `pm25_mean_baseline`.
    if pm25_mean_baseline is not None:
        print(f"  PM2.5 [baseline, no closure] surface mean: "
              f"{pm25_mean_baseline[0].mean():.2f} ug/m3, "
              f"max: {pm25_mean_baseline[0].max():.2f} ug/m3")
        print(f"  PM2.5 [closure-converged]    surface mean: "
              f"{pm25_mean[0].mean():.2f} ug/m3, "
              f"max: {pm25_mean[0].max():.2f} ug/m3")
        print(f"  Delta (closure - baseline): "
              f"{pm25_mean[0].mean() - pm25_mean_baseline[0].mean():+.2f} ug/m3 surface mean")
    else:
        print(f"  PM2.5 surface mean (orbit avg): {pm25_mean[0].mean():.2f} ug/m3")
        print(f"  PM2.5 surface max  (orbit avg): {pm25_mean[0].max():.2f} ug/m3")
    for tau in range(N_BINS):
        print(f"    Bin {tau+1}: surface mean={pm25_orbit[tau, 0].mean():.2f}, "
              f"max={pm25_orbit[tau, 0].max():.2f}")

    # 6b. ISORROPIA PM2.5 extraction (optional)
    iso_extras = {}
    use_isorropia = (lut_path is not None
                     and lut_path.lower() != "none"
                     and os.path.exists(lut_path)
                     and grids[0].RH.size > 0)
    if use_isorropia:
        from orbit.core.isorropia_lut import IsorropiaLUT
        t0_iso = time.time()
        # Reuse the closure LUT (same file) instead of loading the 1.4 GB
        # table twice into memory.
        lut = closure_lut if closure_lut is not None else IsorropiaLUT(lut_path)
        if closure_lut is None:
            print(f"  ISORROPIA LUT loaded: {lut_path}")
        else:
            print("  ISORROPIA LUT (reusing closure LUT)")

        iso_pm25_orbit = np.zeros((N_BINS, nz, ny, nx), dtype=np.float64)
        iso_f_nh4_orbit = np.zeros((N_BINS, nz, ny, nx), dtype=np.float64)
        iso_f_no3_orbit = np.zeros((N_BINS, nz, ny, nx), dtype=np.float64)
        for tau in range(N_BINS):
            c_SN_iso = np.empty(N_SPECIES * N, dtype=np.float64)
            for s in range(N_SPECIES):
                c_SN_iso[s * N:(s + 1) * N] = orbits[s][tau + 1]
            pm25_iso, diag = extract_pm25_isorropia(c_SN_iso, grids[tau], indexer, lut)
            iso_pm25_orbit[tau] = pm25_iso
            iso_f_nh4_orbit[tau] = diag["f_nh4"]
            iso_f_no3_orbit[tau] = diag["f_no3"]

        if iso_basin_flip is not None:
            # extract_pm25_isorropia re-queries the LUT directly (bypasses
            # grid.*Partitioning*), so basin-flipping cells get raw LUT
            # values rather than the centroid we converged to. Overwrite
            # the frozen cells in iso_f_*_orbit with the centroid, and
            # take iso_pm25_orbit at frozen cells from pm25_orbit (which
            # already uses the centroid via grids[tau]).
            for tau in range(N_BINS):
                m_no3 = iso_basin_flip.flip_no3[tau]
                m_nh4 = iso_basin_flip.flip_nh4[tau]
                if m_no3.any():
                    iso_f_no3_orbit[tau][m_no3] = (
                        iso_basin_flip.frozen_f_no3_eq[tau][m_no3])
                if m_nh4.any():
                    iso_f_nh4_orbit[tau][m_nh4] = (
                        iso_basin_flip.frozen_f_nh4_eq[tau][m_nh4])
                m_any = m_no3 | m_nh4
                if m_any.any():
                    iso_pm25_orbit[tau][m_any] = pm25_orbit[tau][m_any]

        iso_pm25_mean = iso_pm25_orbit.mean(axis=0)
        iso_f_nh4_mean = iso_f_nh4_orbit.mean(axis=0)
        iso_f_no3_mean = iso_f_no3_orbit.mean(axis=0)

        print(f"  PM2.5 [ISORROPIA] surface mean (orbit avg): {iso_pm25_mean[0].mean():.2f} ug/m3")
        print(f"  PM2.5 [ISORROPIA] surface max  (orbit avg): {iso_pm25_mean[0].max():.2f} ug/m3")
        delta = iso_pm25_mean[0].mean() - pm25_mean[0].mean()
        print(f"  Delta (ISORROPIA - prescribed): {delta:+.2f} ug/m3 surface mean")
        for tau in range(N_BINS):
            # Use the pre-closure snapshot so "prescribed=" still means
            # "preproc-supplied" even after the closure has overwritten
            # grid.NHPartitioning with the LUT marginal.
            p_nh_sfc = orig_NHPartitioning_per_bin[tau][0].mean()
            p_no_sfc = orig_NOPartitioning_per_bin[tau][0].mean()
            print(f"    Bin {tau+1}: iso={iso_pm25_orbit[tau, 0].mean():.2f}, "
                  f"f_nh4={iso_f_nh4_orbit[tau, 0].mean():.3f} (prescribed={p_nh_sfc:.3f}), "
                  f"f_no3={iso_f_no3_orbit[tau, 0].mean():.3f} (prescribed={p_no_sfc:.3f})")

        # 6c. GC-concentration diagnostic: LUT queried at GEOS-Chem concentrations
        #     Isolates thermodynamic effect (same conc, different partitioning model)
        print("  Comparing prescribed vs LUT partitioning at GC concentrations:")
        gc_f_nh4_orbit = np.zeros((N_BINS, nz, ny, nx), dtype=np.float64)
        gc_f_no3_orbit = np.zeros((N_BINS, nz, ny, nx), dtype=np.float64)
        for tau in range(N_BINS):
            gc_diag = compare_partitioning_gc(bin_paths[tau], grids[tau], lut)
            gc_f_nh4_orbit[tau] = gc_diag["lut_f_nh4_at_gc"]
            gc_f_no3_orbit[tau] = gc_diag["lut_f_no3_at_gc"]
            p_nh_sfc = gc_diag["gc_p_nh"][0].mean()
            p_no_sfc = gc_diag["gc_p_no"][0].mean()
            lut_nh4_sfc = gc_diag["lut_f_nh4_at_gc"][0].mean()
            lut_no3_sfc = gc_diag["lut_f_no3_at_gc"][0].mean()
            print(f"    Bin {tau+1}: f_nh4 GC={p_nh_sfc:.3f} LUT={lut_nh4_sfc:.3f} "
                  f"(delta={lut_nh4_sfc - p_nh_sfc:+.3f}), "
                  f"f_no3 GC={p_no_sfc:.3f} LUT={lut_no3_sfc:.3f} "
                  f"(delta={lut_no3_sfc - p_no_sfc:+.3f})")
        gc_f_nh4_mean = gc_f_nh4_orbit.mean(axis=0)
        gc_f_no3_mean = gc_f_no3_orbit.mean(axis=0)

        # That was the last LUT consumer: release the ~1.8 GB of
        # interpolant tables before deposition maps + save
        # (memory-audit fix 2).
        del lut
        closure_lut = None

        timings["isorropia"] = time.time() - t0_iso
        iso_extras = {
            "iso_pm25_orbit": iso_pm25_orbit,
            "iso_pm25_mean": iso_pm25_mean,
            "iso_f_nh4_mean": iso_f_nh4_mean,
            "iso_f_no3_mean": iso_f_no3_mean,
            "gc_f_nh4_mean": gc_f_nh4_mean,
            "gc_f_no3_mean": gc_f_no3_mean,
        }
    elif lut_path and lut_path.lower() != "none":
        if not os.path.exists(lut_path):
            print(f"  ISORROPIA LUT not found: {lut_path} — skipping")
        elif grids[0].RH.size == 0:
            print("  Bin files lack RH — skipping ISORROPIA (rerun preprocessor)")

    # 7. Save orbit NPZ
    t0 = time.time()
    # Resume-critical fields (orbit_species_order / iter / converged) make
    # the final NPZ itself a valid _load_checkpoint input — no need for a
    # separate per-iter checkpoint to survive end-of-run cleanup.  The
    # species order matches the orbits dict keys, which are the contiguous
    # 0..N_SPECIES-1 indices used throughout the solver.
    save_dict = dict(
        c_orbit=c_orbit,                       # (N_SPECIES, 9, N)
        orbit_species_order=np.arange(N_SPECIES),
        iter=np.array(last_iter_completed if chemistry_iters > 0 else 0),
        converged=np.array(
            history.converged(tol=closure_tol) if chemistry_iters > 0 else False,
            dtype=bool,
        ),
        closure_mode=np.array(closure_mode),
        closure_alpha=np.array(closure_alpha),
        closure_tol=np.array(closure_tol),
        pm25_orbit=pm25_orbit,                  # (8, nz, ny, nx)
        c_mean=c_mean,                          # (N_SPECIES, N)
        pm25_mean=pm25_mean,                    # (nz, ny, nx)
        gmres_iters=result["gmres_iters"],
        gmres_resid=result["gmres_resid"],
        periodicity=result["periodicity"],
        lon=grids[0].lon,
        lat=grids[0].lat,
        p_org=grids[0].AOrgPartitioning,
        # VBS Pankow closure outputs (Phase 7). Surface fields, low-to-high
        # C* order matching IDX_VBS_BINS = (C01, C1, C10, C100, C1000):
        #
        #   vbs_c_mean    (5, ny, nx) — orbit-mean total bin mass µg/m³
        #   F_p_vbs       (5, ny, nx) — converged particle fraction
        #   M_OA_mean     (ny, nx)    — total OA mass µg/m³
        #   soa_mean      (ny, nx)    — Σ_i F_p,i × C_i (canonical SoA
        #                               that contributes to PM2.5)
        vbs_c_mean=_extract_vbs_c_mean_surface(c_mean, indexer),
        F_p_vbs=(getattr(grids[0], "F_p_vbs_surface", None)
                 if getattr(grids[0], "F_p_vbs_surface", None) is not None
                 else np.zeros((5, ny, nx))),
        M_OA_mean=(getattr(grids[0], "M_OA_surface", None)
                   if getattr(grids[0], "M_OA_surface", None) is not None
                   else np.zeros((ny, nx))),
        soa_mean=_compute_soa_mean_surface(c_mean, indexer, grids[0]),
        # Closure-converged partitioning (post-update_grid_partitioning).
        # The originals (pre-closure preproc) are saved separately below.
        p_nh=grids[0].NHPartitioning,
        p_no=grids[0].NOPartitioning,
        grid_shape=np.array([nz, ny, nx]),
        # Pre-closure baseline: orbit + PM2.5 + partitioning snapshots from
        # the ORIGINAL preproc partitioning. The headline "ISORROPIA closure
        # contribution" plot/diagnostic should compare these against the
        # closure-converged fields above.
        p_nh_preproc=orig_NHPartitioning_per_bin[0],
        p_nh_eq_preproc=orig_NHPartitioningEq_per_bin[0],
        p_no_preproc=orig_NOPartitioning_per_bin[0],
    )
    # Per-bin VBS partitioning for the linearised marginal (mirrors the iso
    # f_*_marg_3d fields below). The forward updates grid.F_p_vbs/M_OA_3d per
    # UTC bin (VBS Pankow closure loop above; M_OA varies ~10% across bins), so
    # persist all N_BINS — marginal mode then rebuilds the EXACT baseline VBS
    # deposition operator rather than an orbit-mean approximation.
    if getattr(grids[0], "F_p_vbs", None) is not None:
        save_dict["F_p_vbs_marg_3d"] = np.stack(
            [g.F_p_vbs for g in grids], axis=0).astype(np.float32)   # (N_BINS,5,nz,ny,nx)
        save_dict["M_OA_marg_3d"] = np.stack(
            [g.M_OA_3d for g in grids], axis=0).astype(np.float32)   # (N_BINS,nz,ny,nx)
    if pm25_orbit_baseline is not None:
        save_dict["pm25_orbit_baseline"] = pm25_orbit_baseline
        save_dict["pm25_mean_baseline"] = pm25_mean_baseline
    if iso_closure_history:
        save_dict["iso_closure_iters_run"] = np.array(len(iso_closure_history))
        save_dict["iso_closure_fno3_surf_mean"] = np.array(
            [h[1] for h in iso_closure_history], dtype=np.float64)
        save_dict["iso_closure_pno3_p90"] = np.array(
            [h[2] for h in iso_closure_history], dtype=np.float64)
        save_dict["iso_closure_pnh4_p90"] = np.array(
            [h[3] for h in iso_closure_history], dtype=np.float64)
    if iso_basin_flip is not None:
        # Per-bin per-cell metadata. Cells where mask=False have NaN in
        # the frozen/basin arrays — savez_compressed handles this well.
        save_dict["basin_flip_no3_mask"] = iso_basin_flip.flip_no3
        save_dict["basin_flip_nh4_mask"] = iso_basin_flip.flip_nh4
        save_dict["basin_flip_no3_frozen_eq"] = iso_basin_flip.frozen_f_no3_eq
        save_dict["basin_flip_no3_frozen_marg"] = iso_basin_flip.frozen_f_no3_marg
        save_dict["basin_flip_nh4_frozen_eq"] = iso_basin_flip.frozen_f_nh4_eq
        save_dict["basin_flip_nh4_frozen_marg"] = iso_basin_flip.frozen_f_nh4_marg
        save_dict["basin_flip_no3_low"] = iso_basin_flip.basin_low_no3
        save_dict["basin_flip_no3_high"] = iso_basin_flip.basin_high_no3
        save_dict["basin_flip_nh4_low"] = iso_basin_flip.basin_low_nh4
        save_dict["basin_flip_nh4_high"] = iso_basin_flip.basin_high_nh4
    save_dict.update(iso_extras)
    if chem_summary_per_iter:
        # Flatten per-iter diagnostics into arrays for NPZ-friendly storage.
        # Each iteration's diag is a list[dict]; convert to (n_iter, n_bins) arrays
        # keyed by field.
        keys = list(chem_summary_per_iter[0][0].keys())
        for k in keys:
            if k == "bin":
                continue
            arr = np.array([[d[k] for d in it] for it in chem_summary_per_iter],
                            dtype=np.float64)
            save_dict[f"chem_{k}"] = arr
        save_dict["chem_n_iter"] = len(chem_summary_per_iter)
        # Outer-iteration convergence trajectories
        save_dict.update(history.to_npz_dict(prefix="conv_"))
        save_dict["conv_converged"] = np.array(
            history.converged(tol=closure_tol), dtype=bool)
        # Non-convergence classification (just the last iter's state).
        classification = history.classify(tol=closure_tol)
        save_dict["conv_classification_converged"] = np.array(
            "|".join(classification["converged"]))
        save_dict["conv_classification_oscillating"] = np.array(
            "|".join(classification["oscillating"]))
        save_dict["conv_classification_drifting"] = np.array(
            "|".join(classification["drifting"]))
        # Plausibility check result (iter 1 only)
        save_dict["conv_plausibility_ok"] = np.array(plausibility_ok, dtype=bool)
        save_dict["conv_plausibility_msgs"] = np.array(plausibility_msgs)

    # 3D diagnostics from the final iteration (last chem object alive is
    # the one used to build the last operator; last iso_per_bin is the
    # post-loop final pass).  Stored full-shape: (N_BINS, nz, ny, nx).
    if chemistry_iters > 0 and chem_summary_per_iter and chem is not None:
        oh_3d = np.stack([chem.oxidants[tau].OH for tau in range(N_BINS)], axis=0)
        ho2_3d = np.stack([chem.oxidants[tau].HO2 for tau in range(N_BINS)], axis=0)
        fno2_3d = np.stack([chem.oxidants[tau].f_NO2 for tau in range(N_BINS)], axis=0)
        no3_3d = np.stack([chem.oxidants[tau].NO3 for tau in range(N_BINS)], axis=0)
        n2o5_3d = np.stack([chem.oxidants[tau].N2O5 for tau in range(N_BINS)], axis=0)
        kso2_3d = np.stack(chem.k_so2_rate, axis=0)
        knox_3d = np.stack(chem.k_nox_to_no3_rate, axis=0)
        save_dict["oh_3d"] = oh_3d.astype(np.float32)
        save_dict["ho2_3d"] = ho2_3d.astype(np.float32)
        save_dict["f_no2_3d"] = fno2_3d.astype(np.float32)
        save_dict["no3_rad_3d"] = no3_3d.astype(np.float32)
        save_dict["n2o5_3d"] = n2o5_3d.astype(np.float32)
        save_dict["k_so2_tot_3d"] = kso2_3d.astype(np.float32)
        save_dict["k_nox_to_no3_3d"] = knox_3d.astype(np.float32)
        save_dict["j_no2"] = chem.jNO2.astype(np.float32)  # (N_BINS, ny, nx)
        save_dict["j_o1d"] = chem.jO1D.astype(np.float32)
        save_dict["j_no3"] = chem.jNO3.astype(np.float32)
        save_dict["j_hono"] = chem.jHONO.astype(np.float32)

    # ISORROPIA closure 3D fields: equilibrium + marginal partitioning on
    # the final converged state.  Save whenever iso closure (--chemistry-
    # iters > 0 OR --isorropia-closure-iters > 0) has run; the per-bin
    # marginal fields are what the marginal-mode driver
    # (orbit.modes.marginal) loads as the linearisation point.
    iso_ran = (chemistry_iters > 0) or bool(iso_closure_history)
    if iso_ran:
        f_nh4_eq_3d = np.stack(
            [g.NHPartitioningEq if g.NHPartitioningEq.size > 0
             else g.NHPartitioning for g in grids], axis=0).astype(np.float32)
        f_nh4_marg_3d = np.stack([g.NHPartitioning for g in grids], axis=0).astype(np.float32)
        f_no3_eq_3d = np.stack(
            [g.NO3PartitioningEq if g.NO3PartitioningEq.size > 0
             else (g.NO3Partitioning if g.NO3Partitioning.size > 0
                   else np.zeros((nz, ny, nx))) for g in grids], axis=0,
        ).astype(np.float32)
        f_no3_marg_3d = np.stack(
            [g.NO3Partitioning if g.NO3Partitioning.size > 0
             else np.zeros((nz, ny, nx)) for g in grids], axis=0,
        ).astype(np.float32)
        save_dict["f_nh4_eq_3d"] = f_nh4_eq_3d
        save_dict["f_nh4_marg_3d"] = f_nh4_marg_3d
        save_dict["f_no3_eq_3d"] = f_no3_eq_3d
        save_dict["f_no3_marg_3d"] = f_no3_marg_3d

        # Phase 6a-rev: full ISORROPIA Jacobian block + asymmetry
        # diagnostics. Default-on per the iso-coupled-marginal addendum;
        # iso-coupled marginal mode (Phase 6c) requires the cross-
        # partials. Asymmetry fields feed the tipping-points follow-up
        # paper (cells with |f(+δ) + f(-δ) - 2·f(0)| substantial sit
        # near regime thresholds).
        if "final_iso" in locals():
            cross_keys = ("f_nh_dno3", "f_no3_dnh", "f_nh_dso4", "f_no3_dso4")
            asym_keys = ("f_nh_marg_asym", "f_no3_marg_asym")
            if iso_cross_partials and all(k in final_iso.get(0, {}) for k in cross_keys):
                for k in cross_keys:
                    save_dict[k + "_3d"] = np.stack(
                        [final_iso[tau][k] for tau in range(N_BINS)], axis=0,
                    ).astype(np.float32)
            if all(k in final_iso.get(0, {}) for k in asym_keys):
                for k in asym_keys:
                    save_dict[k + "_3d"] = np.stack(
                        [final_iso[tau][k] for tau in range(N_BINS)], axis=0,
                    ).astype(np.float32)

    # Deposition flux maps. Deposition is a diagonal sink, so the deposited
    # flux is recoverable exactly from the converged orbit -- no extra solve.
    # Orbit-mean, ug/m2/s; inorganics in element mass (N, S). See
    # orbit/core/deposition_maps.py.
    try:
        _dep = compute_deposition_maps(orbits, grids, indexer)
        save_dict["dep_dry_ug_m2_s"] = _dep["dry"]
        save_dict["dep_wet_ug_m2_s"] = _dep["wet"]
        save_dict["dep_total_ug_m2_s"] = _dep["total"]
        save_dict["dep_units"] = np.array(
            "ug/m2/s orbit-mean; x315.576 -> kg/ha/yr. Inorganics are element "
            "mass (TotalNH/TotalNO3 as N, SO2/pSO4 as S)."
        )
        print("  Deposition maps (orbit-mean):")
        print(_summarise_deposition(_dep, SPECIES_NAMES))
        if float(_dep["clipped_negative_ug_m2_s"]) < 0:
            print(f"    note: clipped {float(_dep['clipped_negative_ug_m2_s']):.2e} "
                  f"ug/m2/s of negative flux from cells with negative mass")
    except Exception as exc:  # diagnostics must never lose the solve
        print(f"  WARNING: deposition maps not computed: {exc}")

    # Phase 4b: provenance hashes. Marginal/zero-out compare these
    # against the current code/config and refuse on mismatch unless
    # --ignore-baseline-hash is set.
    save_dict["baseline_emissions_hash"] = np.array(
        _baseline_emissions_hash(_emission_basenames(), month)
    )
    save_dict["baseline_emissions_files"] = np.array(
        sorted(os.path.basename(f) for f in _emission_basenames())
    )
    save_dict["closure_settings_hash"] = np.array(
        _closure_settings_hash(
            closure_alpha, closure_tol, isorropia_anderson,
            basin_flip_damp, disable_night_nox, isorropia_closure_iters,
        )
    )
    save_dict["iso_cross_partial_definition"] = np.array(
        _ISO_CROSS_PARTIAL_DEFINITION
    )
    save_dict["code_git_sha"] = np.array(_git_sha_short())

    _RSS.set_phase("save")
    np.savez_compressed(out_path, **save_dict)
    timings["save"] = time.time() - t0
    out_mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"  Saved: {out_path} ({out_mb:.1f} MB, {timings['save']:.1f}s)")

    # 8. Free UMFPACK symbolic
    if umfpack_sym is not None:
        umfpack_free_symbolic(umfpack_sym)

    timings["peak_rss_mb"] = _peak_rss_mb()
    try:
        timings["gmres_iters_final"] = {
            SPECIES_NAMES[s]: int(result["gmres_iters"][s])
            for s in range(N_SPECIES)}
        timings["gmres_resid_final"] = {
            SPECIES_NAMES[s]: float(result["gmres_resid"][s])
            for s in range(N_SPECIES)}
    except Exception:
        pass   # diagnostics must never lose the solve
    return timings


# Backward-compat alias: external callers may still import process_month.
process_month = run_forward_month


def _print_source_inventory():
    """Print the manifest inventory with each source's species + path.

    Used by ``--list-sources``. Helps users discover the right BASENAME
    for ``--scale-source BASENAME=FACTOR``.
    """
    print(f"Baseline emissions directory: {EMISSION_DIR}")
    print(f"Number of source files: {len(_emission_basenames())}")
    print()
    print(f"{'Basename':<48}  Path")
    print("-" * 96)
    for fname in _emission_basenames():
        full = os.path.join(EMISSION_DIR, fname)
        present = "" if os.path.exists(full) else "  (missing)"
        print(f"{fname:<48}  {full}{present}")


def _dry_run_perturbation(perturbation, months, mode="marginal"):
    """Print a Perturbation summary without solving.

    Used by ``--dry-run``. Reports factors, add-sources, and per-species
    δe norms by loading the actual emission slabs (so the user catches
    typos in the YAML before submitting a multi-hour job).
    """
    print(f"=== Dry run ({mode}) ===")
    print(f"Months: {months}")
    print(f"Perturbation: {perturbation.name}")
    if perturbation.description:
        print(f"  Description: {perturbation.description}")
    if perturbation.is_empty():
        print("  (empty perturbation — δe = 0)")
        return
    if perturbation.factors:
        print("  Factors:")
        for k, v in sorted(perturbation.factors.items()):
            present = "(known)"
            if k not in {os.path.basename(f) for f in _emission_basenames()}:
                present = "(UNKNOWN — typo? use --list-sources)"
            print(f"    {k} = {v}  {present}")
    if perturbation.add_sources:
        print("  Add sources:")
        for s in perturbation.add_sources:
            sp = f" species={s.species_override}" if s.species_override else ""
            print(f"    {s.path}{sp}")
    print()
    print(
        "Dry run complete. Re-run without --dry-run to actually solve."
    )


def _resolve_baseline_path(template: str, month: int) -> str:
    """Resolve a --baseline-npz template into a per-month path.

    Supports a literal {MM} placeholder for the zero-padded month; if
    absent, the template is returned verbatim (single-month convenience).
    """
    if "{MM}" in template:
        return template.replace("{MM}", f"{month:02d}")
    if "{m}" in template:
        return template.replace("{m}", str(month))
    return template


def _check_baseline_hash(baseline_npz_path: str, month: int, args) -> None:
    """Phase 4b: refuse if the baseline doesn't match current config.

    Recomputes ``baseline_emissions_hash`` and ``closure_settings_hash``
    against the current manifest inventory + the closure args carried
    by ``args``, and raises ValueError on mismatch unless
    ``--ignore-baseline-hash`` is set.

    Pre-Phase-4b baselines lack the hash keys; we treat that as a
    soft-warn (printed) so existing baselines remain usable until they
    are re-run.
    """
    if getattr(args, "ignore_baseline_hash", False):
        return
    if not os.path.exists(baseline_npz_path):
        return  # downstream loader will produce the error
    data = np.load(baseline_npz_path, allow_pickle=False)

    expect_emis = _baseline_emissions_hash(_emission_basenames(), month)
    expect_clos = _closure_settings_hash(
        args.closure_alpha, args.closure_tol, args.isorropia_anderson,
        args.basin_flip_damp, args.disable_night_nox,
        args.isorropia_closure_iters,
    )

    if "baseline_emissions_hash" not in data.files:
        print(f"  WARN: {baseline_npz_path} pre-dates Phase-4b hashing — "
              f"cannot verify it matches current config. Re-run forward "
              f"to remove this warning, or pass --ignore-baseline-hash.")
        return

    got_emis = str(data["baseline_emissions_hash"])
    got_clos = str(data["closure_settings_hash"]) if "closure_settings_hash" in data.files else None

    mismatches = []
    if got_emis != expect_emis:
        baseline_files = (
            list(data["baseline_emissions_files"])
            if "baseline_emissions_files" in data.files else None
        )
        cur = sorted(os.path.basename(f) for f in _emission_basenames())
        added = sorted(set(cur) - set(baseline_files or []))
        removed = sorted(set(baseline_files or []) - set(cur))
        mismatches.append(
            f"emissions hash differs (baseline={got_emis} expected={expect_emis})"
            + (f" added={added}" if added else "")
            + (f" removed={removed}" if removed else "")
        )
    if got_clos is not None and got_clos != expect_clos:
        mismatches.append(
            f"closure settings hash differs (baseline={got_clos} expected={expect_clos}); "
            f"current closure args may not match the baseline run"
        )

    if mismatches:
        raise ValueError(
            "Baseline NPZ does not match current run config. "
            "Either re-run forward, fix the args to match, or pass "
            "--ignore-baseline-hash to override.\n  "
            + "\n  ".join(mismatches)
            + f"\n  baseline: {baseline_npz_path}"
        )


def _run_marginal_main(args, months, diurnal_cfg):
    """Marginal-mode dispatcher invoked from main() when --mode marginal."""
    from orbit.modes.marginal import (
        parse_cli_perturbation, run_marginal_month,
    )

    from orbit.modes.perturbation import Perturbation as _Pert

    if not args.baseline_npz:
        print("ERROR: --baseline-npz is required for --mode marginal")
        return
    if not (args.scale_source or args.add_emissions or args.perturbation):
        print("WARNING: no --scale-source / --add-emissions / --perturbation "
              "specified — δe will be zero (this is the smoke-test path)")

    # Build perturbation: YAML first (if given), then layer CLI flags on top
    # (CLI takes precedence over YAML for any duplicate keys).
    if args.perturbation:
        perturbation = _Pert.from_yaml(args.perturbation)
        cli_overlay = parse_cli_perturbation(args.scale_source, args.add_emissions)
        perturbation.factors.update(cli_overlay.factors)
        perturbation.add_sources.extend(cli_overlay.add_sources)
    else:
        perturbation = parse_cli_perturbation(args.scale_source, args.add_emissions)

    if getattr(args, "dry_run", False):
        _dry_run_perturbation(perturbation, months, mode="marginal")
        return

    output_dir = args.marginal_output_dir or os.path.join(
        os.path.dirname(OUTPUT_DIR), "marginal"
    )
    os.makedirs(output_dir, exist_ok=True)

    print("=== ORBIT SAS Marginal Solver ===")
    print(f"Months: {months}")
    print(f"Preprocessor: {PREPROC_DIR}")
    print(f"Output: {os.path.abspath(output_dir)}")
    print(f"Perturbation: {perturbation.name}")
    if perturbation.description:
        print(f"  Description: {perturbation.description}")
    if perturbation.factors:
        print(f"  Factors: {perturbation.factors}")
    if perturbation.add_sources:
        print(f"  Added sources: {[os.path.basename(s.path) for s in perturbation.add_sources]}")
    print()

    all_results = []
    for m in months:
        print(f"--- Month {m:02d} ({MONTH_ABBR[m-1]}) ---")
        t0 = time.time()
        baseline_path = _resolve_baseline_path(args.baseline_npz, m)
        _check_baseline_hash(baseline_path, m, args)
        sources = _build_emission_sources(m)
        r = run_marginal_month(
            m,
            baseline_npz_path=baseline_path,
            perturbation=perturbation,
            output_dir=output_dir,
            preproc_path_fn=_preproc_path,
            constants_path=CONSTANTS,
            baseline_sources=sources,
            diurnal_cfg=diurnal_cfg,
            output_filename=args.marginal_output_filename,
            iso_coupling=args.iso_coupling,
            horizontal_fct=getattr(args, "horizontal_fct", False),
            verbose=True,
        )
        if r is not None:
            r["wall_total"] = time.time() - t0
            all_results.append(r)
            print(f"  Wall time: {_fmt_time(r['wall_total'])}, "
                  f"max|δc|={r['delta_c_max_abs']:.3e}, "
                  f"max|δPM25|={r['delta_pm25_max_abs']:.3e}")
        else:
            print("  FAILED")
        print()

    if not all_results:
        print("No months processed.")
        return
    print("=== Marginal Summary ===")
    for r in all_results:
        print(f"  M{r['month']:02d}: wall={_fmt_time(r.get('wall_total', 0))}, "
              f"max|δc|={r['delta_c_max_abs']:.3e}, "
              f"max|δPM25|={r['delta_pm25_max_abs']:.3e}")


def _forward_kwargs_from_args(args):
    """Map an argparse Namespace into ``run_forward_month`` kwargs.

    Single source of truth for "what CLI flag drives which forward kwarg",
    used by both the forward main loop and the zero-out dispatcher. This
    prevents a class of bug where adding a new flag to argparse silently
    fails to reach a programmatic forward re-run (notably zero-out), with
    function-signature defaults of ``None`` quietly disabling features
    like ISORROPIA closure (lut_path=None).

    Excludes ``resume`` / ``warm`` / ``resume_from`` — those are
    forward-CLI-only and meaningless for a programmatic re-run.
    """
    return dict(
        lut_path=args.lut,
        chemistry_iters=args.chemistry_iters,
        photolysis_lut_path=args.photolysis_lut,
        hemco_dir=args.hemco_dir,
        closure_mode=args.closure_mode,
        closure_alpha=args.closure_alpha,
        closure_tol=args.closure_tol,
        top_bc_days=args.top_bc_days,
        top_bc_layers=args.top_bc_layers,
        top_bc_decay_factor=args.top_bc_decay_factor,
        lateral_bc_days=args.lateral_bc_days,
        lateral_bc_depth=args.lateral_bc_depth,
        krylov_tol_intermediate=args.krylov_tol_intermediate,
        closure_accel=args.closure_accel,
        anderson_m=args.anderson_m,
        disable_night_nox=args.disable_night_nox,
        isorropia_closure_iters=args.isorropia_closure_iters,
        isorropia_anderson=args.isorropia_anderson,
        basin_flip_damp=args.basin_flip_damp,
        iso_cross_partials=getattr(args, "iso_cross_partials", False),
        fold_meander_in_K=getattr(args, "fold_meander_in_K", False),
        unified_vertical_patankar=getattr(args, "unified_vertical_patankar", False),
        horizontal_fct=getattr(args, "horizontal_fct", False),
    )


def _build_perturbation(args):
    """Build a Perturbation from CLI flags + optional YAML overlay.

    YAML first, then layer CLI flags on top (CLI takes precedence over
    YAML for any duplicate keys). Shared by both marginal and zero-out
    dispatchers.
    """
    from orbit.modes.perturbation import (
        Perturbation as _Pert, parse_cli_perturbation as _parse_cli,
    )
    if args.perturbation:
        perturbation = _Pert.from_yaml(args.perturbation)
        cli_overlay = _parse_cli(args.scale_source, args.add_emissions)
        perturbation.factors.update(cli_overlay.factors)
        perturbation.add_sources.extend(cli_overlay.add_sources)
    else:
        perturbation = _parse_cli(args.scale_source, args.add_emissions)
    return perturbation


def _run_zero_out_main(args, months, diurnal_cfg):
    """Zero-out-mode dispatcher invoked from main() when --mode zero-out."""
    from orbit.modes.zero_out import run_zero_out_month

    if not args.baseline_npz:
        print("ERROR: --baseline-npz is required for --mode zero-out")
        return
    perturbation = _build_perturbation(args)
    if perturbation.is_empty():
        print("ERROR: zero-out with an empty perturbation is a no-op "
              "(δc would equal forward(baseline) - forward(baseline) ≈ 0). "
              "Specify --scale-source / --add-emissions / --perturbation.")
        return

    if getattr(args, "dry_run", False):
        _dry_run_perturbation(perturbation, months, mode="zero-out")
        return

    output_dir = args.zeroout_output_dir or os.path.join(
        os.path.dirname(OUTPUT_DIR), "zeroout"
    )
    os.makedirs(output_dir, exist_ok=True)

    forward_kwargs = _forward_kwargs_from_args(args)

    print("=== ORBIT SAS Zero-out Solver ===")
    print(f"Months: {months}")
    print(f"Preprocessor: {PREPROC_DIR}")
    print(f"Output: {os.path.abspath(output_dir)}")
    print(f"Perturbation: {perturbation.name}")
    if perturbation.description:
        print(f"  Description: {perturbation.description}")
    if perturbation.factors:
        print(f"  Factors: {perturbation.factors}")
    if perturbation.add_sources:
        print(f"  Added sources: {[os.path.basename(s.path) for s in perturbation.add_sources]}")
    print()

    all_results = []
    for m in months:
        print(f"--- Month {m:02d} ({MONTH_ABBR[m-1]}) ---")
        t0 = time.time()
        baseline_path = _resolve_baseline_path(args.baseline_npz, m)
        _check_baseline_hash(baseline_path, m, args)
        sources = _build_emission_sources(m)
        r = run_zero_out_month(
            m,
            baseline_npz_path=baseline_path,
            perturbation=perturbation,
            output_dir=output_dir,
            preproc_path_fn=_preproc_path,
            constants_path=CONSTANTS,
            baseline_sources=sources,
            forward_runner=run_forward_month,
            forward_kwargs=forward_kwargs,
            diurnal_cfg=diurnal_cfg,
            keep_perturbed_npz=args.zeroout_keep_perturbed,
            output_filename=args.zeroout_output_filename,
            verbose=True,
        )
        if r is not None:
            r["wall_total"] = time.time() - t0
            all_results.append(r)
            print(f"  Wall time: {_fmt_time(r['wall_total'])}, "
                  f"max|δc|={r['delta_c_max_abs']:.3e}, "
                  f"max|δPM25|={r['delta_pm25_max_abs']:.3e}")
        else:
            print("  FAILED")
        print()

    if not all_results:
        print("No months processed.")
        return
    print("=== Zero-out Summary ===")
    for r in all_results:
        print(f"  M{r['month']:02d}: wall={_fmt_time(r.get('wall_total', 0))}, "
              f"max|δc|={r['delta_c_max_abs']:.3e}, "
              f"max|δPM25|={r['delta_pm25_max_abs']:.3e}")


def main():
    """CLI entrypoint: parse arguments, resolve the emission manifest, and
    run the requested months.

    Dispatches to the marginal / zero-out drivers when ``--mode`` says so;
    otherwise calls :func:`run_forward_month` per month with the run log
    attached (text log + JSON sidecar), then prints the cross-month
    timing and GMRES summary tables.
    """
    parser = argparse.ArgumentParser(
        prog="orbit",
        description="Diurnal periodic orbit solver for South Asia (SAS) 2022",
    )
    parser.add_argument("--month", type=int, default=None,
                        choices=range(1, 13), metavar="M",
                        help="Single month (1-12). Default: all available.")
    parser.add_argument("--resume", action="store_true",
                        help="Skip months with existing output; for un-done months, "
                             "auto-discover latest orbit_M{MM}_iter{NN}.npz and resume. "
                             "Convenience for multi-month reruns. Use --resume-from for "
                             "explicit single-checkpoint resume.")
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Explicit resume path (any NPZ produced by this script, "
                             "including the final orbit_M{MM}.npz). Bypasses both the "
                             "skip-if-exists short-circuit and the auto-discovery scan, "
                             "so stale checkpoints from other experiments can't hijack "
                             "a run. Applies to the first --month given.")
    parser.add_argument("--warm", action="store_true",
                        help="Warm-start GMRES from previous month's orbit mean "
                             "(for inter-month chaining; the default cold start "
                             "is optimal for single-month runs)")
    parser.add_argument("--lut", type=str,
                        default=os.environ.get(
                            "ORBIT_LUT",
                            "/path/to/data/preproc/output/LUT/isorropia_lut_7d.npz"),
                        help="Path to ISORROPIA 7D LUT. Defaults to $ORBIT_LUT, "
                             "which must be set (the run refuses to start "
                             "otherwise). Set to 'none' to run without the "
                             "ISORROPIA closure.")
    parser.add_argument("--chemistry-iters", type=int, default=0,
                        help="Number of DCOMP OH/oxidant outer iterations. "
                             "0 (default) = prescribed OH/NO/NO2 (no "
                             "diagnostic-oxidant loop; O3/CO inert). N>=1 "
                             "engages DCOMP, which requires the TUV photolysis "
                             "LUT and HEMCO climatology (not included in this "
                             "release); if either is missing the run falls "
                             "back to 0. ISORROPIA partitioning closure is "
                             "controlled separately by "
                             "--isorropia-closure-iters.")
    parser.add_argument("--isorropia-closure-iters", type=int, default=6,
                        help="Max ISORROPIA partitioning iterations (separate "
                             "from --chemistry-iters). Loop is gated on "
                             "p99(|ΔPM2.5|/max(PM2.5, 1 µg/m³)) < closure_tol — "
                             "convergence on the model's actual output, not on "
                             "intermediate partitioning fractions (a clean-marine "
                             "f_NO3 swing in cells where pNO3 is 0.05 µg/m³ "
                             "shouldn't gate the answer). 6 iters covers "
                             "Picard@α=0.5 with margin; Anderson typically halves "
                             "this. Set to 0 to disable the closure entirely "
                             "(no-ISORROPIA ablation baseline).")
    parser.add_argument("--fold-meander-in-K", action="store_true",
                        help="Override grid.has_split_fluxes=False after load. "
                             "convdiff then folds K_meander into K_face and "
                             "falls back to mean-flux advection (instead of "
                             "split UAvg_plus/minus). Counterfactual test for "
                             "the Patankar exponential scheme's activation "
                             "band.")
    parser.add_argument("--unified-vertical-patankar", action="store_true",
                        help="PROTOTYPE: assemble a single unified vertical "
                             "Patankar exponential operator (omega advection + "
                             "Kzz diffusion) in place of the separate vertical "
                             "operators, removing the upwind omega scheme's "
                             "implicit numerical diffusion where physical Kzz "
                             "competes. Uses the hybrid terrain-corrected omega "
                             "(not WAvg). Off by default; research only.")
    parser.add_argument("--horizontal-fct", default=None,
                        action=argparse.BooleanOptionalAction,
                        help="Flux-corrected transport on horizontal "
                             "advection — a van Leer anti-diffusive source, "
                             "Zalesak-limited, added to the orbit RHS for every "
                             "transported species and recomputed each closure "
                             "iteration (deferred correction). Recovers ~2nd-order "
                             "transport, removing the upwind numerical diffusion "
                             "(~6e4 m²/s). Default: ON for forward and zero-out "
                             "(the production configuration); OFF for marginal, "
                             "which defaults to the monotone low-order "
                             "linearisation (the published headline choice). "
                             "Pass --horizontal-fct or --no-horizontal-fct to "
                             "override either way.")
    parser.add_argument("--isorropia-anderson", default=True,
                        action=argparse.BooleanOptionalAction,
                        help="Use Anderson(m) acceleration on the ISORROPIA "
                             "partitioning Picard loop (state vector = surface "
                             "f_NO3 across bins). Has a monotonicity safeguard "
                             "that falls back to Picard on rejection. ON by "
                             "default (production setting; roughly halves the "
                             "iter count); --no-isorropia-anderson gives plain "
                             "Picard.")
    parser.add_argument("--basin-flip-damp", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Detect per-cell basin-flipping (sign-flip of "
                             "consecutive iter deltas) in the ISORROPIA "
                             "closure and freeze affected cells at the centroid "
                             "of the bracketing basin values. Default: on. "
                             "Use --no-basin-flip-damp to disable. Saves "
                             "basin_flip_* arrays to NPZ for diagnostic "
                             "post-hoc analysis.")
    parser.add_argument("--photolysis-lut", type=str,
                        default="/path/to/data/preproc/output/LUT/photolysis_tuv_dcomp.npz",
                        help="Path to a TUV photolysis LUT (jNO2, jO1D, jNO3, "
                             "jHONO, jHCHO, jH2O2). Only used when "
                             "--chemistry-iters > 0; the LUT is not included "
                             "in this release.")
    parser.add_argument("--hemco-dir", type=str, default=None,
                        help="HEMCO climatology directory (GEOS-Chem benchmark "
                             "output). Only used when --chemistry-iters > 0; "
                             "the archive is not included in this release.")
    parser.add_argument("--closure-mode", type=str, default="full",
                        choices=("full", "chem-only"),
                        help="'full' (default) = engage ISORROPIA partitioning "
                             "closure (under --isorropia-closure-iters when "
                             "--chemistry-iters 0, or inside the chemistry loop "
                             "otherwise). 'chem-only' = no ISORROPIA pass — "
                             "TotalNO3 deposits as 100%% HNO3 (fast bound). "
                             "Use 'chem-only' only as an explicit prescribed-oxidant analogue "
                             "ablation.")
    parser.add_argument("--closure-alpha", type=float, default=0.5,
                        help="Under-relaxation coefficient for OH and partitioning "
                             "updates (0 < alpha <= 1; 1.0 = no damping). Default 0.5.")
    parser.add_argument("--closure-tol", type=float, default=DEFAULT_TOL,
                        help=f"Convergence tolerance (default {DEFAULT_TOL}). "
                             f"Gates the ISORROPIA closure on "
                             f"p99(|ΔPM2.5|/max(PM2.5, 1 µg/m³)); with "
                             f"--chemistry-iters > 0 it also gates the "
                             f"chemistry loop on |ΔOH|, |ΔO3|, |Δf_NH4|, "
                             f"|Δf_NO3|.")
    parser.add_argument("--top-bc-days", type=float, default=10.0,
                        help="Upper-boundary Newtonian nudging e-folding time "
                             "(days) at the TOP ORBIT layer.  Rate at top = "
                             "1/(N days).  0 disables the BC entirely.  "
                             "Default 10 days.  No effect unless "
                             "--chemistry-iters > 0 (requires HEMCO "
                             "climatology, not included in this release).")
    parser.add_argument("--top-bc-layers", type=int, default=4,
                        help="Number of top ORBIT layers to nudge (N, counting "
                             "down from top=nz-1).  Single-layer nudging (N=1) "
                             "was diagnosed inadequate for the SAS 15-layer "
                             "config — the UT has no local O3 chemistry source "
                             "so a single point BC exports into an empty "
                             "column.  Default 4 (top layer + 3 below) treats "
                             "the free troposphere as a volume source.  No "
                             "effect unless --chemistry-iters > 0.")
    parser.add_argument("--top-bc-decay-factor", type=float, default=3.0,
                        help="Rate decay ratio from top-nudged layer to bottom-"
                             "nudged layer.  Bottom-layer rate = top-rate / F.  "
                             "E.g. top_bc_days=10, decay=3, layers=4 gives "
                             "rates at k=[11,12,13,14] of [1/30d, 1/20.8d, "
                             "1/14.4d, 1/10d].  Reflects weakening "
                             "stratospheric influence downward through the UT. "
                             "Default 3x.  No effect unless "
                             "--chemistry-iters > 0.")
    parser.add_argument("--lateral-bc-days", type=float, default=1.0,
                        help="Lateral-boundary Newtonian nudging e-folding time "
                             "(days) at the outermost cells of each horizontal "
                             "edge (W/E/S/N).  Pins inflow air to HEMCO climatology "
                             "so advection carries realistic O3 into the interior. "
                             "Applies at all vertical layers.  0 disables lateral "
                             "nudging.  Default 1 day — fast enough to beat UT "
                             "advective export (~1 day transit time at mid-lat "
                             "winter jet).  No effect unless --chemistry-iters "
                             "> 0 (requires HEMCO climatology, not included in "
                             "this release).")
    parser.add_argument("--lateral-bc-depth", type=int, default=3,
                        help="Number of cells deep the lateral nudging extends "
                             "inward from each horizontal boundary.  Rate decays "
                             "linearly from 1/(lateral_bc_days) at d=0 (outermost) "
                             "to 1/(lateral_bc_days * M) at d=M-1 (innermost).  "
                             "Default 3 cells gives a ~1.5° (~160 km) nudged "
                             "band along each edge.  No effect unless "
                             "--chemistry-iters > 0.")
    parser.add_argument("--krylov-tol-intermediate", type=float, default=1e-4,
                        help="GMRES tolerance on non-final chemistry iters.  "
                             "Default 1e-4 (~25-30%% Krylov iter reduction).  "
                             "Final iter always uses 1e-6 for the saved NPZ.  "
                             "Set to 1e-6 to use the tight tolerance "
                             "throughout.")
    parser.add_argument("--closure-accel", type=str, default="picard",
                        choices=("picard", "anderson"),
                        help="Chemistry outer-iteration acceleration scheme.  "
                             "'picard' (default) = plain alpha under-relaxation.  "
                             "'anderson' = Anderson(m) on the OH fixed point, "
                             "with a monotonicity safeguard that falls back to "
                             "Picard on rejection.")
    parser.add_argument("--anderson-m", type=int, default=3,
                        help="Anderson memory depth (m+1 iterates retained).  "
                             "Default 3.  Used by the ISORROPIA closure when "
                             "--isorropia-anderson is on (the default) and by "
                             "the chemistry loop when --closure-accel "
                             "anderson.")
    parser.add_argument("--disable-night-nox", action="store_true",
                        help="Ablation: zero the nighttime N2O5-hydrolysis "
                             "channel of the prescribed NOx -> TotalNO3 rate. "
                             "Daytime OH+NO2 channel untouched. Quantifies "
                             "what nighttime nitrate chemistry adds to winter "
                             "IGP.")
    parser.add_argument("--diurnal-config", type=str,
                        default=_default_diurnal_config_path(),
                        help="Path to a YAML file specifying per-source "
                             "24-hour local-time emission profiles (mean=1) "
                             "and a single SAS-wide IST offset. Monthly "
                             "emission NetCDFs are sampled into N_BINS "
                             "per-UTC-bin emission vectors. Defaults to the "
                             "shipped production profiles "
                             "(orbit/data/diurnal_sas.yaml); 'none' disables "
                             "diurnal profiles (bin-flat emissions).")

    # ── Simulation modes (see MODES.md) ──────────────────────────────────
    parser.add_argument("--mode", type=str, default="forward",
                        choices=("forward", "marginal", "zero-out"),
                        help="Simulation mode. 'forward' (default) runs the "
                             "full orbit-converged simulation. 'marginal' "
                             "computes a linearised δc against a saved "
                             "baseline NPZ (sub-minute back-solve). "
                             "'zero-out' runs a second forward with the "
                             "perturbation applied and differences against "
                             "the baseline. See MODES.md.")
    parser.add_argument("--baseline-npz", type=str, default=None,
                        help="Per-month forward NPZ used as the baseline for "
                             "marginal / zero-out modes. May contain a {MM} "
                             "placeholder; otherwise the same path is used "
                             "for every month requested.")
    parser.add_argument("--scale-source", action="append", default=None,
                        metavar="BASENAME=FACTOR",
                        help="(marginal/zero-out) Scale a baseline source "
                             "by FACTOR. Multi-occurrence. E.g. "
                             "--scale-source ceds_nh3_anthro_2022_monthly.nc=0.5 "
                             "halves anthropogenic NH3.")
    parser.add_argument("--add-emissions", action="append", default=None,
                        metavar="PATH[:SPECIES]",
                        help="(marginal/zero-out) Add a new emissions NetCDF "
                             "purely additively. Multi-occurrence. SPECIES "
                             "names the variable in the NetCDF (e.g. NOx, NH3, "
                             "PM25); if omitted, the loader auto-detects.")
    parser.add_argument("--perturbation", type=str, default=None,
                        metavar="YAML",
                        help="(marginal/zero-out) Path to a perturbation YAML "
                             "(factors + add blocks). Stacks with --scale-source "
                             "and --add-emissions; CLI flags take precedence "
                             "over duplicate keys. See examples/scenarios/.")
    parser.add_argument("--marginal-output-dir", type=str, default=None,
                        help="(marginal) Output directory for marginal NPZs. "
                             "Default: outputs/sas/marginal next to OUTPUT_DIR.")
    parser.add_argument("--marginal-output-filename", type=str, default=None,
                        help="(marginal) Override the output filename. "
                             "Default: marginal_M{MM}.npz. Useful when "
                             "running multiple perturbations in parallel.")
    parser.add_argument("--zeroout-output-dir", type=str, default=None,
                        help="(zero-out) Output directory for zero-out δNPZs. "
                             "Default: outputs/sas/zeroout next to OUTPUT_DIR.")
    parser.add_argument("--zeroout-output-filename", type=str, default=None,
                        help="(zero-out) Override the output filename. "
                             "Default: zeroout_M{MM}.npz.")
    parser.add_argument("--zeroout-keep-perturbed", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="(zero-out) Keep the intermediate perturbed-run "
                             "NPZ on disk under output-dir/_perturbed/. "
                             "Default: keep (re-differencing is cheap; "
                             "regenerating the run is not).")
    parser.add_argument("--ignore-baseline-hash", action="store_true",
                        help="(marginal/zero-out) Skip the "
                             "baseline-emissions / closure-settings hash "
                             "check. Default: refuse on mismatch. Use only "
                             "when intentionally diffing across configs "
                             "(e.g. ablation studies).")
    parser.add_argument("--iso-cross-partials",
                        action=argparse.BooleanOptionalAction, default=True,
                        help="(forward) Compute and persist the full "
                             "ISORROPIA Jacobian block (4 cross-couplings: "
                             "f_nh_dno3, f_no3_dnh, f_nh_dso4, f_no3_dso4) "
                             "plus per-cell asymmetry diagnostics "
                             "(f_*_marg_asym) at the closure-converged "
                             "baseline, via symmetric ±δ finite differences. "
                             "Required by iso-coupled marginal mode. Adds "
                             "~10-20%% wall to forward. Default ON; pass "
                             "--no-iso-cross-partials to disable.")
    parser.add_argument("--iso-coupling",
                        action=argparse.BooleanOptionalAction, default=True,
                        help="(marginal) Use the full ISORROPIA-coupled "
                             "Jacobian (NH4↔NO3↔SO4 cross-blocks) in the "
                             "marginal back-solve. Default ON. Pass "
                             "--no-iso-coupling for the diagonal-only "
                             "diagnostic (ISORROPIA cross-couplings "
                             "missing). Requires a baseline NPZ produced "
                             "with --iso-cross-partials (default ON).")
    parser.add_argument("--log-file", type=str, default=None,
                        help="Write the run log here. Default: "
                             "orbit_M<MM>.log next to the output NPZ. "
                             "Existing logs are never overwritten.")
    parser.add_argument("--no-log-file", action="store_true",
                        help="Do not write a log file (terminal output only).")
    parser.add_argument("--verify-inputs", action="store_true",
                        help="Fully hash every input file for the log. Off by "
                             "default: the ISORROPIA table is 1.4 GB and the "
                             "default quick fingerprint (size + first/last MB) "
                             "already catches a wrong or truncated file.")
    parser.add_argument("--no-emission-budget", action="store_true",
                        help="Skip the per-source emission mass budget "
                             "(saves re-reading the inventory).")
    parser.add_argument("--emissions-manifest", type=str, default=None,
                        help="YAML manifest describing the baseline emission "
                             "inventory: which files, their units, stack "
                             "parameters and VOC classes. Defaults to the "
                             "shipped South Asia 2022 manifest, or "
                             "ORBIT_EMISSION_MANIFEST. Files themselves are "
                             "read from ORBIT_EMISSION_DIR.")
    parser.add_argument("--allow-missing-emissions", action="store_true",
                        help="Proceed when manifest sources marked required "
                             "are absent. Off by default: running with part "
                             "of the inventory silently missing gives a "
                             "wrong answer.")
    parser.add_argument("--list-sources", action="store_true",
                        help="Print the manifest's emission sources and "
                             "their basenames, then exit. Useful for finding "
                             "the right BASENAME for --scale-source.")
    parser.add_argument("--dry-run", action="store_true",
                        help="(marginal/zero-out) Build the Perturbation, "
                             "print a summary (factors, add sources, δe "
                             "norms by species), and exit without solving.")
    args = parser.parse_args()

    # Per-mode default for the horizontal-FCT operator: the forward (and
    # zero-out, which re-runs forward) reproduce the published baseline,
    # which used FCT; the marginal linearisation defaults to the monotone
    # low-order operator, the published headline choice for sensitivities.
    # An explicit --horizontal-fct / --no-horizontal-fct overrides both.
    if args.horizontal_fct is None:
        args.horizontal_fct = (args.mode != "marginal")

    # Resolve the emission manifest before anything reads the inventory.
    global _MANIFEST_PATH_OVERRIDE, _ALLOW_MISSING_EMISSIONS
    global _VERIFY_INPUTS, _EMISSION_BUDGET
    _RECORD.command = list(sys.argv)
    _VERIFY_INPUTS = args.verify_inputs
    _EMISSION_BUDGET = not args.no_emission_budget
    _MANIFEST_PATH_OVERRIDE = args.emissions_manifest
    _ALLOW_MISSING_EMISSIONS = args.allow_missing_emissions
    try:
        get_manifest()
    except (FileNotFoundError, ValueError) as exc:
        parser.error(f"emission manifest: {exc}")

    if args.list_sources:
        # A placeholder emissions dir would list every source as
        # "(missing)" against a fake path — refuse legibly instead.
        if EMISSION_DIR.startswith("/path/to/data"):
            raise SystemExit(
                "ORBIT_EMISSION_DIR is unset, so it fell back to the "
                "placeholder default\n"
                f"  {EMISSION_DIR}\n"
                "which is not a real path. Set ORBIT_EMISSION_DIR to the "
                "location of the\nemissions data on this machine (see the "
                "README's data section)."
            )
        _print_source_inventory()
        return

    diurnal_cfg = None
    if args.diurnal_config and args.diurnal_config.lower() != "none":
        diurnal_cfg = DiurnalConfig.from_yaml(args.diurnal_config)
        print(f"Diurnal emissions: {args.diurnal_config} "
              f"(IST offset={diurnal_cfg.ist_offset_hours}h, "
              f"{len(diurnal_cfg.profiles)} profile(s), "
              f"{len(diurnal_cfg.source_profiles)} source mapping(s))")

    # Determine which months to process
    if args.month:
        months = [args.month]
    else:
        # Auto-detect available months
        months = []
        for m in range(1, 13):
            if os.path.exists(_preproc_path(m, 1)):
                months.append(m)

    # ── Marginal mode dispatch ───────────────────────────────────────────
    # Branch off here to keep the forward path 100% unchanged. Phase 1:
    # marginal only; zero-out lands in Phase 3.
    if args.mode == "marginal":
        return _run_marginal_main(args, months, diurnal_cfg)
    if args.mode == "zero-out":
        return _run_zero_out_main(args, months, diurnal_cfg)

    _record_configuration(args, months)
    print(_RECORD.header(_VERSION))
    print()
    print(_RECORD.config_block())
    print()
    _record_inputs(args)
    print(_RECORD.inputs_block())
    print()
    _check_solver_capabilities()
    print()

    all_results = []
    _RSS.start()
    for m in months:
        print(f"--- Month {m:02d} ({MONTH_ABBR[m-1]}) ---")
        t0 = time.time()
        # --resume-from applies to the first month only.  Passing it to
        # subsequent months would be incoherent (each month has its own
        # checkpoint).  Downstream months use --resume (auto-discovery) if
        # requested, or a fresh baseline otherwise.
        rf = args.resume_from if m == months[0] else None
        _log_path, _json_path = _log_paths(m)
        if args.no_log_file:
            _log_path = None
        elif args.log_file:
            _log_path = args.log_file
        # Header first (file only), so the file stands alone as a record
        # without duplicating the banner on the console.
        _preamble = "\n\n".join([_RECORD.header(_VERSION),
                                 _RECORD.config_block(),
                                 _RECORD.inputs_block()]) + "\n\n"
        with _RECORD.attach(_log_path, preamble=_preamble):
            r = run_forward_month(
                m,
                resume=args.resume,
                warm=args.warm,
                resume_from=rf,
                diurnal_cfg=diurnal_cfg,
                **_forward_kwargs_from_args(args),
            )
            if r is not None and not r.get("skipped"):
                _RECORD.resources["peak_rss_mb"] = r.get("peak_rss_mb", 0)
                _month_res = {
                    k: float(r[k]) for k in
                    ("pm25_surface_mean", "pm25_surface_max",
                     "pm25_surface_mean_baseline", "orbit_solve",
                     "grid_load", "emissions", "save", "lu_fill_est")
                    if k in r and r[k] is not None
                }
                for k in ("gmres_iters_final", "gmres_resid_final"):
                    if k in r:
                        _month_res[k] = r[k]
                _RECORD.results[f"M{m:02d}"] = _month_res
            _RECORD.status = "completed" if r is not None else "failed"
            if _RSS.enabled:
                _RECORD.resources["peak_rss_by_phase"] = _RSS.peaks()
                print()
                print(_RSS.render())
            print()
            print(_RECORD.footer())
        if not args.no_log_file:
            try:
                _RECORD.write_json(_json_path, _VERSION)
            except Exception as exc:
                print(f"  WARNING: could not write {_json_path}: {exc}")
        if r is not None and not r.get("skipped"):
            r["wall_total"] = time.time() - t0
            all_results.append(r)
            print(f"  Wall time: {_fmt_time(r['wall_total'])}, "
                  f"peak RSS: {r['peak_rss_mb']:.0f} MB")
        elif r is not None and r.get("skipped"):
            print()
            continue
        else:
            print("  FAILED")
        print()

    if not all_results:
        print("No months processed.")
        return

    # Summary table
    print("=== Summary ===")
    header = (f"{'Month':>5s}  {'Grids':>5s}  {'Asm':>5s}  {'Orbit':>7s}  "
              f"{'Total':>7s}  {'RSS':>6s}")
    print(header)
    print("-" * len(header))
    grand_total = 0.0
    for r in all_results:
        wall = r.get("wall_total", 0)
        grand_total += wall
        print(f"{r['month']:>5d}  "
              f"{_fmt_time(r.get('grid_load', 0)):>5s}  "
              f"{_fmt_time(r.get('assembly', 0)):>5s}  "
              f"{_fmt_time(r.get('orbit_solve', 0)):>7s}  "
              f"{_fmt_time(wall):>7s}  "
              f"{r.get('peak_rss_mb', 0):>5.0f}M")
    print("-" * len(header))
    print(f"{'TOTAL':>5s}  {'':>5s}  {'':>5s}  {'':>7s}  "
          f"{_fmt_time(grand_total):>7s}")

    # GMRES summary across months
    print("\n=== GMRES Diagnostics ===")
    print(f"{'Month':>5s}  ", end="")
    for name in SPECIES_NAMES:
        print(f"{name:>7s}  ", end="")
    print()
    for r in all_results:
        # Re-load the output to get per-species info
        out_path = os.path.join(OUTPUT_DIR, f"orbit_M{r['month']:02d}.npz")
        if os.path.exists(out_path):
            d = np.load(out_path)
            iters = d["gmres_iters"]
            resid = d["gmres_resid"]
            period = d["periodicity"]
            print(f"{r['month']:>5d}  ", end="")
            for s in range(N_SPECIES):
                print(f"{iters[s]:>3d}it  ", end="")
            print()
            print(f"{'':>5s}  ", end="")
            for s in range(N_SPECIES):
                print(f"{resid[s]:.0e}  ", end="")
            print(" (residual)")
            print(f"{'':>5s}  ", end="")
            for s in range(N_SPECIES):
                print(f"{period[s]:.0e}  ", end="")
            print(" (periodicity)")

    print(f"\nOutput directory: {os.path.abspath(OUTPUT_DIR)}")
    print("Done.")


if __name__ == "__main__":
    main()
