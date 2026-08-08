# ORBIT simulation modes

ORBIT has 3 simulation types selected via `orbit --mode [mode]`
(equivalently, `python -m orbit --mode [mode]`, or
`python scripts/run_orbit.py --mode [mode]` in an uninstalled checkout),
where mode is:

- `forward`: full simulation using total emissions (with outer-loop)
- `marginal`: emissions perturbation around a baseline
- `zero-out`: the difference between 2 forward simulations

---

## When to use which

| Question | Mode | Notes |
|---|---|---|
| "What are the total PM2.5 concentrations and associated health impacts?" | forward | Produces the baseline NPZ. |
| "What are the changes in PM2.5 concentrations and health impacts from a (small) change in emissions?" | marginal | Fast; trustworthy in the linear regime. |
| "What are the changes in PM2.5 concentrations and health impacts associated with a sector (e.g. transport), or a large change in emissions?" | zero-out | Captures nonlinear responses (e.g. crossing an ISORROPIA regime boundary). |

---

## Running each mode

### Forward

An example forward simulation (the defaults are the production
configuration):

```bash
orbit --mode forward --month 1
```

### Marginal

```bash
# scale an existing source
orbit --mode marginal --month 1 \
  --baseline-npz "$ORBIT_DATA/baseline/orbit_M{MM}.npz" \
  --scale-source ceds_nh3_anthro_2022_monthly.nc=0.99

# add a new source (the :NH3 suffix names the species)
orbit --mode marginal --month 1 \
  --baseline-npz "$ORBIT_DATA/baseline/orbit_M{MM}.npz" \
  --add-emissions /path/to/new_facility.nc:NH3

# YAML scenario, custom output location
orbit --mode marginal --month 1 \
  --baseline-npz "$ORBIT_DATA/baseline/orbit_M{MM}.npz" \
  --perturbation examples/scenarios/nh3_minus_50pct.yaml \
  --marginal-output-dir outputs/scenarios/nh3_minus_50pct \
  --marginal-output-filename marginal_M01.npz
```

The baseline path may include the literal `{MM}` placeholder, which is replaced per month.

Output: `outputs/sas/marginal/marginal_M{MM}.npz` by default; the output flags above redirect it.

### Zero-out

```bash
orbit --mode zero-out --month 1 \
  --baseline-npz "$ORBIT_DATA/baseline/orbit_M{MM}.npz" \
  --perturbation examples/scenarios/nh3_minus_50pct.yaml
```

Zero-out runs a forward simulation with the perturbed emissions and
differences it against the baseline, so it needs the same inputs as forward.

Output: `outputs/sas/zeroout/zeroout_M{MM}.npz` plus `_perturbed/forward_perturbed_*_M{MM}.npz` (the perturbed-half forward NPZ; kept by default for re-differencing).

---

## Scenario configuration files

Emission scenarios are described in YAML configuration files (the same
format as the README's "Scenario configuration files" section):

```yaml
name: nh3_minus_50pct
description: 50% reduction in CEDS + GFED5 anthropogenic + biomass-burn NH3
factors:
  ceds_nh3_anthro_2022_monthly.nc: 0.50
  gfed5_nh3_bb_2022_monthly.nc: 0.50
add:
  - path: /path/to/new_facility.nc
    units: kg/m2/s
    species_override: NH3
```

- `factors`: multipliers on each emission file's total, keyed by the file's
  name (without its directory path). 0.50 halves that source's emissions,
  1.0 leaves it unchanged, and a file not listed is unchanged. In marginal
  mode the applied perturbation is the change relative to the baseline,
  `(factor - 1) × source`.
- `add`: entirely new sources (for example a single new facility).

Any flags given on the command line (`--scale-source`, `--add-emissions`)
layer on top of the configuration file and override it on duplicate keys.
See `examples/scenarios/` for the reference scenarios included in the
release.

---

## Output files

### Forward (`orbit_M{MM}.npz`)

| Key                          | Shape              | Notes                                                          |
|------------------------------|--------------------|----------------------------------------------------------------|
| `c_orbit`                    | (14, 9, N)         | Concentration along the periodic orbit, element mass; axes are (N_SPECIES, N_BINS+1, N) |
| `c_mean`                     | (14, N)            | UTC-mean concentration                                         |
| `pm25_orbit`                 | (8, nz, ny, nx)    | PM25 with the prescribed partitioning (compound mass)                            |
| `iso_pm25_orbit`             | (8, nz, ny, nx)    | ISORROPIA-closed PM25 (compound mass)                          |
| `iso_pm25_mean`              | (nz, ny, nx)       | UTC-mean of the above                                          |
| `f_nh4_marg_3d`              | (8, nz, ny, nx)    | ∂pNH4/∂c_TotalNH at baseline (linearised partitioning)         |
| `f_no3_marg_3d`              | (8, nz, ny, nx)    | ∂pNO3/∂c_TotalNO at baseline                                   |
| `f_nh4_eq_3d`                | (8, nz, ny, nx)    | Equilibrium pNH4 fraction (ISORROPIA LUT, post-closure)        |
| `f_no3_eq_3d`                | (8, nz, ny, nx)    | Equilibrium pNO3 fraction                                      |
| `baseline_emissions_hash`    | scalar str         | SHA-256 prefix of sorted emission file names + month + loader version + manifest content hash |
| `baseline_emissions_files`   | array              | Sorted emission file names                                      |
| `closure_settings_hash`      | scalar str         | SHA-256 prefix of (closure_alpha, tol, anderson, …)            |
| `code_git_sha`               | scalar str         | Short git commit identifier of the code that ran                          |

### Marginal (`marginal_M{MM}.npz`)

| Key                  | Shape           | Notes                                                                     |
|----------------------|-----------------|---------------------------------------------------------------------------|
| `mode`               | scalar str      | `"marginal"`                                                              |
| `delta_c_orbit`      | (14, 9, N)      | δc along the periodic orbit                                               |
| `delta_pm25_orbit`   | (8, nz, ny, nx) | δPM25 along the periodic orbit                                            |
| `delta_pm25_mean`    | (nz, ny, nx)    | UTC-mean δPM25                                                            |
| `solved_species`     | (k,)            | Species indices that were solved (species with zero perturbation are skipped)         |
| `sign_convention`    | scalar str      | `"perturbation_response"` (factor=0.99 → δc<0)                            |
| `perturbation_*`     |                 | Provenance: factors, add paths, scenario name + description               |

### Zero-out (`zeroout_M{MM}.npz`)

| Key                  | Shape           | Notes                                                                      |
|----------------------|-----------------|----------------------------------------------------------------------------|
| `mode`               | scalar str      | `"zero-out"`                                                               |
| `delta_c_orbit`      | (14, 9, N)      | `c_full - c_perturbed`                                                     |
| `delta_pm25_orbit`   | (8, nz, ny, nx) | `pm25_full - pm25_perturbed` (ISORROPIA-closed)                            |
| `delta_pm25_mean`    | (nz, ny, nx)    | UTC-mean                                                                   |
| `delta_f_no3_eq_3d`  | (8, nz, ny, nx) | Optional regime-shift diagnostic: Δf_no3_eq                                |
| `delta_f_nh4_eq_3d`  | (8, nz, ny, nx) | Optional regime-shift diagnostic: Δf_nh4_eq                                |
| `sign_convention`    | scalar str      | `"baseline_minus_perturbed"` (factor=0.5 → δc>0)                           |
| `perturbed_npz_path` | scalar str      | Pointer to the kept-on-disk perturbed-half forward NPZ                     |

---

## Sign conventions

The two modes save δc in **opposite** signs for the same physical change. Marginal reports the response to the perturbation, so a decrease in emissions (e.g. `factor=0.99`) yields `δc < 0`. Zero-out reports `c_full - c_perturbed`, so the same decrease in emissions yields `δc > 0`.

To compare:

```python
from orbit.modes.perturbation import unify_sign
m = unify_sign("marginal_M01.npz", target="perturbation_response")
z = unify_sign("zeroout_M01.npz", target="perturbation_response")
# Both now report a decrease in emissions as δ < 0, so you can subtract directly:
residual = m["delta_pm25_mean"] - z["delta_pm25_mean"]
```

`unify_sign` reads the NPZ's own `sign_convention` field and flips when the requested `target` differs.

---

## Run IDs

3 short SHA-256 prefixes are stored in each NPZ output file that document
details of the run configuration:

- `baseline_emissions_hash`: sorted emission file names + month + emissions-loader version + the manifest content hash (units, stack parameters, VOC classes)
- `closure_settings_hash`: `(closure_alpha, closure_tol, isorropia_anderson, basin_flip_damp, disable_night_nox, isorropia_closure_iters)` plus a tag identifying the ISORROPIA cross-partial definition
- `code_git_sha`: short git commit identifier at run-time (diagnostic only)

Marginal and zero-out report a difference against the baseline, so the model
must be identical up to the perturbation. They recompute the first 2 hashes
against the current config and refuse on mismatch:

```
ValueError: Baseline NPZ does not match current run config. …
  emissions hash differs (baseline=… expected=…) added=[…] removed=[…]
  baseline: outputs/sas/orbit/orbit_M01.npz
```

To compare runs with intentionally different configurations, skip the check:

```bash
orbit --mode zero-out --month 1 \
  --baseline-npz "$ORBIT_DATA/baseline/orbit_M{MM}.npz" \
  --perturbation examples/scenarios/nh3_minus_50pct.yaml \
  --ignore-baseline-hash
```

---

## Running many months

Months are independent, so the usual pattern is one job per month (can be
parallelised), or a single process over all months by omitting `--month`. Measured per-month cost at the
default settings: forward ~23 min and ~5.2 GB; marginal ~4 min and ~5.8 GB
(with `--horizontal-fct`, the flux-corrected linearisation, ~37 min).
