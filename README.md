<p align="center">
  <img src="docs/orbit_logo.png" alt="ORBIT" width="360">
</p>

**ORBIT** is an air quality model for predicting PM2.5 concentrations.
It's currently released for South Asia (2022) with a horizontal grid of
resolution 0.625° × 0.5°, with 15 vertical layers (up to ~65 hPa).

## How ORBIT works

ORBIT is a mechanistic, Eulerian atmospheric model with 12 transported species:
primary PM2.5, primary organic aerosol, 5 secondary organic aerosol volatility
bins, and 5 inorganic species (NH, SO₂, pSO₄, NOₓ, NO₃). ORBIT uses meteorology
data from MERRA-2 reanalyses to estimate how and where pollution moves, reacts,
and gets removed from the atmosphere.

Unlike other air quality models, ORBIT does not use time-stepping to estimate the
trajectories of pollutant concentrations. Instead, for each month, ORBIT solves
for the periodic steady state concentrations of PM2.5, using a diurnal periodic
orbit (hence the name 'ORBIT'). Essentially, this is asking *if the average day
of this calendar month repeated forever, what state would the atmosphere settle
into?* Solving this question directly estimates average PM2.5 concentrations while
retaining both a seasonal cycle (a different orbit per month) and a diurnal cycle
(eight 3-hour bins within each orbit).

To solve for each steady state, ORBIT uses 8 operators (one for each 3-hour time bin
of the day). Each bin τ has its own transport–deposition–chemistry operator `L_τ`,
which ORBIT assembles explicitly as a sparse matrix from the averaged meteorology. 
Advancing one bin is achieved through a backward-Euler step, so the propagator over τ is

```
P_τ = (I + L_τ Δτ)⁻¹
```

By composing all 8 bins, we get the monodromy operator M for one full day:

```
M = P₈ P₇ … P₁
```

ORBIT then solves for the periodic orbit as the state that returns to itself over a day, i.e.

```
(I − M) c₀ = s
```

for the concentration `c₀` at the start of the day, where `s` represents the
emissions injected throughout the day. This equation is solved by **GMRES**,
using a matrix-free composition of the eight back-solves (to avoid forming
the dense matrix `M` explicitly). Each `(I + L_τ Δτ)` is factored once per bin 
with UMFPACK (reusing the symbolic factorisation across each transported species).

## Chemistry

ORBIT transports all 12 tracers using the (linear) ORBIT solve, but is nevertheless 
able to incorporate nonlinear chemistry, via outer iteration loops.

For **secondary inorganic aerosol**, partitioning is given using a pre-compiled offline lookup table of 
ISORROPIA-II (Fountoukis and Nenes, 2007), over total SO₄, total NH, total NO₃, crustal Ca,
sea-salt Na, temperature, and relative humidity. 
After the partitioning is updated, ORBIT 
re-transports the tracers in an outer (Anderson-accelerated) iteration loop until convergence.
Because of fast equilibration between NH₃ ↔ NH₄⁺ and HNO₃ ↔ NO₃⁻, ORBIT transports these as TotalNH and
Total NO₃.

For **secondary organic aerosol**, ORBIT uses a 1-D volatility basis set (Donahue et al., 2006): 5 bins at 
C\* = {0.1, 1, 10, 100, 1000} µg m⁻³ with an OH-driven aging cascade.
Oxidant fields (OH, H₂O₂, NO₂/NO₃/N₂O₅) are prescribed from a GEOS-Chem simulation (Thakrar et al., 2022) 
and not currently fully-coupled. Gas/particle partitioning is given by the Pankow closure 
against the total absorbing organic mass (including primary organic aerosol).

## Transport and deposition

- **Horizontal advection–diffusion** uses a Patankar exponential first-order
scheme with a deferred, anti-diffusive correction (Zalesak FCT limiter) so 
transport is second-order in smooth flow and monotone at fronts.

- **Vertical advection** is given by MERRA-2 pressure velocity ω with a
  coordinate correction `ω_cross = ω − v_H · ∇_η p` to distinguish between
  upslope advection and ventilation.
- **Convection** is given by the MERRA-2 (non-local) convective updraft mass
  flux.
- **PBL mixing** follows the YSU K_zz profile with a free-tropospheric floor.
- **Dry deposition** uses Wesely (1989) resistances for gases and
  Seinfeld–Pandis impaction/interception/diffusion for accumulation-mode
  particles.
- **Wet scavenging** uses intensity-dependent in-cloud nucleation and
  EMEP-derived sub-cloud washout.
- Horizontal transport, vertical advection, vertical diffusion, and convection
  are all reconciled to a single dP·area mass measure.

## Simulation modes

- **`forward`** estimates total concentrations from total emissions.
- **`marginal`** estimates changes in concentrations from (small) changes in emissions 
  around a baseline, using the tangent-linear response δc/δe from a full
  ISORROPIA-coupled Jacobian, capturing off-diagonal chemistry (e.g. how reducing
  SO₂ emissions changes particulate nitrate).
- **`zero-out`** estimates changes in concentrations from (large) changes in emissions
  around a baseline, by finding the difference between 2 forward simulations.

See [`MODES.md`](MODES.md) for the full reference: CLI flags, sign conventions,
and the Jacobian structure.

### Health impacts (marginal deaths)

ORBIT converts marginal concentration responses into marginal **mortality**
by an adjoint solve: `scripts/compute_marginal_deaths.py` computes
∂deaths/∂emissions for every grid cell in one solve per month, using a
precomputed deaths-gradient field (population × baseline mortality rate ×
the slope of a concentration–response function, evaluated at the baseline
exposure). Gradient fields for three concentration–response functions (GEMM
five-cause, GEMM all-cause, and the GBD Integrated Exposure–Response) are
published in the companion health-impacts data deposit (link to be added).
The finished results, marginal deaths per 1000 kg emitted for every
subdistrict, pollutant, and concentration–response function, as NetCDF and
CSV, are published in the companion results deposit (link to be added), so
none of the following needs re-running to use them.

```bash
python scripts/compute_marginal_deaths.py \
    --gradient deaths_gradient_gemm5cod.nc \
    --preproc-base $ORBIT_DATA/inputs/grids_2022 \
    --orbit-dir $ORBIT_DATA/baseline \
    --constants $ORBIT_DATA/inputs/MERRA2.20150101.CN.05x0625.nc4 \
    --month 1 \
    --species PM25_primary --species NH3 --species SO2 \
    --out-dir ./deaths
```

This writes `adjoint_M01.nc` holding ∂deaths/∂emissions per species, time
bin, and grid cell. `scripts/postprocess_marginal_deaths_scenarios.py` turns
a full 12-month sweep into policy-scenario damage estimates and can
aggregate them to admin-2 subdistricts. That last step needs the `geo`
extra and the GADM 4.1 polygons (`gadm_410.gpkg`), which GADM's licence
does not allow us to redistribute; download from
[gadm.org](https://gadm.org).

## Install

ORBIT requires **Python ≥ 3.9**.

The recommended install uses conda, because ORBIT's fast solver backend
(UMFPACK, via `scikit-umfpack`) depends on the SuiteSparse C library, which
conda provides and plain pip does not:

```bash
git clone https://github.com/SumilThakr/orbit.git
cd orbit
conda create -n orbit -c conda-forge python=3.11 scikit-umfpack pymetis
conda activate orbit
pip install -e ".[fast,dev]"
```

The last command tells pip to install ORBIT from the current directory
(`-e .`) together with the optional dependency groups named `fast` (the
UMFPACK solver backend, already provided by the conda line above) and
`dev` (the test tools). The core scientific dependencies — numpy, scipy,
xarray, netCDF4, pyyaml, matplotlib, numba — are installed automatically.

**No conda?** From inside the cloned `orbit` folder, this works on any
system:

```bash
pip install -e ".[dev]"
```

ORBIT then falls back from UMFPACK to SuperLU, which is slower and needs
roughly 1.5–2× more memory (a forward month peaks around 5 GB with UMFPACK
at the default settings). To get UMFPACK without conda, first install
SuiteSparse from your system's package manager, then include `fast`:

```bash
sudo apt install libsuitesparse-dev     # Debian/Ubuntu
pip install -e ".[fast,dev]"
```

**Check that it worked:**

```bash
pytest -m "not slow"
```

Expect roughly 480 tests passing. A handful of tests skip when
`scikit-umfpack` or `pymetis` are absent — everything still runs, just
slower; ORBIT says so in a startup warning rather than failing.

The optional dependency groups, for reference: `fast` (scikit-umfpack —
the UMFPACK LU backend), `metis` (pymetis — a nested-dissection ordering
that is 1.3× faster at 25% less peak memory than the default COLAMD
ordering, with results identical to floating-point noise), `geo`
(geopandas + shapely, only needed for shapefile emissions), and `dev`
(pytest and the test dependencies). The conda command above already covers
`fast` and `metis`.

## Input data

ORBIT reads pre-built meteorology/chemistry input grids plus an emissions
directory. The South Asia 2022 inputs are archived separately:

> **Data:** [Google Drive archive](https://drive.google.com/drive/folders/1o5a3RWS9suuazPKqmJuqg-1DrExCO6PE?usp=sharing) (a Zenodo DOI will follow)

The archive contains the input grids, the MERRA-2 constants file, the
emissions directory, the ISORROPIA lookup table, and the twelve production
monthly solves (`baseline/`) — the published baseline, which the marginal
mode can linearise around directly and whose January solve doubles as the
installation-check reference.

Paths are supplied by environment variable; the Quickstart below sets
these variables:

| Variable | What it points at |
|---|---|
| `ORBIT_PREPROC_DIR` | directory of input grids, `sas_<year>_M<MM>_B<BB>.nc` (12 months × 8 bins; `sas` = the South Asia domain, fixed in this release) |
| `ORBIT_PREPROC_YEAR_TAG` | year tag in those filenames (default `2022`) |
| `ORBIT_CONSTANTS` | MERRA-2 constants file (land fraction) |
| `ORBIT_EMISSION_DIR` | directory of per-source emission NetCDFs |
| `ORBIT_LUT` | the ISORROPIA lookup table (equivalently, pass `--lut`) |
| `ORBIT_OUTPUT_DIR` | where results are written |

## Quickstart

After the install above, the CLI is available as `orbit` or `python -m orbit`;
`python scripts/run_orbit.py` is equivalent and works without installing.

**1. Point ORBIT at the unpacked data archive:**

```bash
export ORBIT_DATA=/path/to/orbit_data          # unpacked archive root
export ORBIT_PREPROC_DIR=$ORBIT_DATA/inputs/grids_2022
export ORBIT_CONSTANTS=$ORBIT_DATA/inputs/MERRA2.20150101.CN.05x0625.nc4
export ORBIT_EMISSION_DIR=$ORBIT_DATA/emissions/sas
export ORBIT_LUT=$ORBIT_DATA/inputs/isorropia_lut_7d.npz
export ORBIT_OUTPUT_DIR=./outputs
```

A wrong path fails fast at startup with a message naming the variable. Input
files are identified by a quick fingerprint (size plus first and last
megabyte), so a truncated or mismatched download is caught before the solve
starts, not twenty minutes in; `--verify-inputs` switches to full hashing.

**2. Run a marginal simulation around the published baseline.** The archive
includes the twelve production monthly solves in `baseline/`, so the marginal
mode works immediately — no forward solve needed:

```bash
orbit --mode marginal --month 1 \
    --baseline-npz $ORBIT_DATA/baseline/orbit_M01.npz \
    --perturbation examples/scenarios/nh3_minus_50pct.yaml
```

Ready-made scenarios (NH₃, NOₓ, SO₂ at several magnitudes) live in
`examples/scenarios/`; the "Scenario configuration files" section below documents the
format for writing your own. The baseline files carry configuration hashes
that are checked against your active settings, so a mismatch fails loudly
rather than silently comparing different models.

**3. Reproduce the baseline yourself.** The defaults *are* the full
production configuration (flux-corrected transport, Anderson-accelerated
inorganic closure, sector diurnal emission profiles), so no flags are
needed:

```bash
orbit --mode forward --month 1
```

Measured at the default settings: **23 minutes and 5.2 GB peak RAM** for one
month. Then check your solve against the published one:

```bash
python scripts/compare_reference_run.py \
    ./outputs/orbit_M01.npz $ORBIT_DATA/baseline/orbit_M01.npz
```

It prints a per-field verdict and an overall REPRODUCED / DIFFERS line;
REPRODUCED is the expected result on any machine.

## Running the model

### Reproduce the annual simulation

Omitting `--month` runs every month for which input grids are present. Months
are independent, so on a cluster twelve single-month jobs finish fastest —
but the sequential run is entirely practical too: about **4½–5 hours** for
the full year on an ordinary machine, a fine overnight job.

```bash
# All twelve months in one process (~4.5-5 h total)
orbit

# Or one month per job (each ~23 min, ~5 GB peak)
orbit --month $M
```

The defaults are the production configuration: flux-corrected horizontal
transport (`--no-horizontal-fct` for the monotone low-order operator),
Anderson-accelerated inorganic closure (`--no-isorropia-anderson` for plain
Picard), and the default sector diurnal emission profiles
(`--diurnal-config none` for a uniform diurnal profile).
Results land in `$ORBIT_OUTPUT_DIR` as `orbit_M<MM>.npz`, alongside per-iteration
checkpoints `orbit_M<MM>_iter<NN>.npz`.

Useful for long runs:

| Flag | Effect |
|---|---|
| `--resume` | skip months that already have output; resume unfinished ones from their last checkpoint |
| `--resume-from PATH` | resume explicitly from any NPZ this CLI produced |
| `--warm` | warm-start GMRES from the previous month's orbit mean (fewer iterations; changes the iterate path, not the converged answer) |

Each output NPZ carries the bin-resolved 3-D concentration field, the inorganic
partitioning fractions, the VBS partitioning, the orbit-mean surface PM2.5,
and per-species deposition maps.

#### Run logs

Every forward run writes a log and a machine-readable JSON record next to
the output, named for the month:

```
orbit_M01.npz        the result
orbit_M01.log        run log (also printed to the terminal)
orbit_M01.run.json   the same record, for scripts
```

Existing logs are never overwritten: a re-run writes `orbit_M01.log.2`. The
log is flushed line by line, so a job killed part-way still leaves a usable
record.

```bash
orbit --mode forward --month 1 --no-log-file        # don't write a log file
orbit --mode forward --month 1 --log-file run.log   # write it somewhere else
```

A run is reconstructible from its log alone: ORBIT version and git commit
(with a dirty flag), host, full command line, library versions, and input
fingerprints. Configuration is printed in two tables, one for settings that
change the results and one for settings that only change performance, with
each row marked `[default]` or with the flag that set it.

The log also reports a per-source emission mass budget in Tg/yr (this
month's rate scaled to a full year), so a wrong `units` declaration, a bad
regrid, or a truncated inventory shows up as a wrong total.

```bash
orbit --mode forward --month 1 --no-emission-budget   # skip the budget table
```

Input files are identified by a quick fingerprint (size plus the first and
last megabyte), which catches a wrong or truncated file without hashing the
1.4 GB lookup table on every run.

```bash
orbit --mode forward --month 1 --verify-inputs   # full SHA-256 of every input
```

Warnings appear twice: inline at the moment they are raised, and again in a
`--- Warnings ---` block at the end of the log, directly above the status
footer, so one raised early in a long run is not lost. Each entry states
what happened, its impact on the results, and a `fix:` line with the
setting that resolves it or silences it deliberately.

#### Deposition maps

ORBIT forward simulations also generate mean dry and wet deposition maps per
species as follows:

| Key | Shape | Contents |
|---|---|---|
| `dep_dry_ug_m2_s` | (14, ny, nx) | dry deposition (surface layer, `vd · c`) |
| `dep_wet_ug_m2_s` | (14, ny, nx) | wet scavenging (column integral of `wd · c · Δz`) |
| `dep_total_ug_m2_s` | (14, ny, nx) | their sum |

Units are µg m⁻² s⁻¹. The leading axis is the species index, not a vertical
axis: each species' map is 2-D, with scavenging aloft already summed into
the column integral. The indices are:

| Index | Species |
|---|---|
| 0 | VBS_C100 |
| 1 | PM2.5 (primary) |
| 2 | TotalNH |
| 3 | SO2 |
| 4 | NOx |
| 5 | pSO4 |
| 6 | TotalNO3 |
| 7 | |
| 8 | |
| 9 | VBS_C10 |
| 10 | VBS_C1 |
| 11 | VBS_C01 |
| 12 | VBS_C1000 |
| 13 | POA |

(Rows 7 and 8 are unused slots and hold zeros.)

The inorganics are reported in element mass, matching the solver's internal
convention: `TotalNH` and `TotalNO3` are N, `SO2` and `pSO4` are S. That is the
basis critical-load and ecosystem work already uses (kg N ha⁻¹ yr⁻¹), so it is
left unconverted. Primary PM2.5 and the VBS bins are aerosol/organic mass.

The SOA photolytic sink is deliberately excluded: it shares the diagonal for
numerical reasons but is a chemical loss, not deposition.

### Marginal mode

```bash
orbit --mode marginal --month 1 \
    --baseline-npz $ORBIT_DATA/baseline/orbit_M01.npz \
    --perturbation examples/scenarios/nh3_minus_50pct.yaml
```

Omitting `--month` runs every available month. Each month is linearised
around its own baseline, so for multi-month runs give `--baseline-npz` a
`{MM}` placeholder and the right file is pulled per month:

```bash
orbit --mode marginal \
    --baseline-npz "$ORBIT_DATA/baseline/orbit_M{MM}.npz" \
    --perturbation examples/scenarios/nh3_minus_50pct.yaml
```

The perturbation itself is month-independent: a rule like "NH₃ × 0.5"
applies to each solved month's own emissions.

The baseline NPZ must come from a forward run with the *same* emissions and
closure settings, for the same month. ORBIT hashes all three and refuses a
mismatched baseline rather than silently producing a wrong Jacobian. If you know why the hash differs,
the check can be overridden:

```bash
orbit --mode marginal --month 1 \
    --baseline-npz my_baseline.npz \
    --perturbation examples/scenarios/nh3_minus_50pct.yaml \
    --ignore-baseline-hash
```

Marginal gives the tangent-linear δc/δe with the full ISORROPIA-coupled
Jacobian, so it captures off-diagonal chemistry, for instance an SO₂ cut
changing particulate nitrate.

### Zero-out mode

```bash
orbit --mode zero-out --month 1 \
    --baseline-npz $ORBIT_DATA/baseline/orbit_M01.npz \
    --perturbation examples/scenarios/nh3_minus_50pct.yaml
```

Zero-out re-runs the forward solve with the perturbed emissions and differences
against the baseline. Multi-month runs work as in marginal mode, including the
`{MM}` baseline placeholder. Use zero-out instead of `marginal` for large
perturbations, where the response is not linear: a 50 % NH₃ cut, for example,
can flip the inorganic equilibrium regime (less NH₃ → less ammonium nitrate →
more HNO₃ stays in the gas phase).

### Scenario configuration files

Emission scenarios can be specified in text (YAML) files with the following
format, as seen in `examples/scenarios/nh3_minus_50pct.yaml`. Ten ready-made
scenarios are included in `examples/scenarios/`; copying one is the easiest way to
start. A scenario has two independent blocks. `factors` rescales sources
already in the baseline, keyed by file basename:

```yaml
name: nh3_minus_50pct
description: 50% cut in anthropogenic + biomass-burning NH3.

factors:
  ceds_nh3_anthro_2022_monthly.nc: 0.50
  gfed5_nh3_bb_2022_monthly.nc: 0.50
```

`add` introduces entirely new sources, and is how you bring your own
emissions into an experiment:

```yaml
add:
  - path: /data/my_new_industry_nox.nc
    format: netcdf          # netcdf | shapefile | geopackage
    units: kg/m2/s          # see the unit table below
    time_index: 0           # which timestep to take, if multi-time
    variable_mapping: {nox: NOx_emissions}   # optional; otherwise auto-detected
    bin_axis: false         # true if the file carries 8 diurnal slabs per month
```

It's possible to specify the configuration without a YAML file by using
command flags as follows:

```bash
orbit --mode zero-out --month 1 \
    --baseline-npz $ORBIT_DATA/baseline/orbit_M01.npz \
    --scale-source ceds_nox_anthro_2022_monthly_surface.nc=0.7 \
    --add-emissions /data/my_new_industry_nox.nc:nox
```

### Supplying your own emissions

Although it is possible to run ORBIT with any emissions data, the default
baseline simulation uses an emissions inventory for total PM2.5 and
precursors, compiled from various sources, and available under
`$ORBIT_EMISSION_DIR`:

| Pollutant | Data source(s) |
|---|---|
| **Anthropogenic** | |
| PM2.5 (incl. primary organic aerosol), NOx, SO2, NH3, VOC | CEDS |
| **Biomass burning** | |
| PM2.5, NOx, SO2, NH3, VOC | GFED5 |
| **Natural** | |
| NH3 | GEIA |
| Biogenic VOC | CAMS-GLOB-BIO |
| Soil NOx | CAMS-GLOB-SOIL |
| Oceanic DMS (as SO2) | CAMS-GLOB-OCE |
| Volcanic SO2 | Carn SO2 database |
| Dust and sea salt PM2.5 | MERRA-2 |

Which files make up the inventory, and how each is interpreted, is set by an
emissions configuration file, `orbit/data/emissions_sas_2022_poa.yaml`, used
by default. To change the baseline inventory (add, remove, or swap emission
files), copy that file, edit it, and point ORBIT at your copy:

```bash
orbit --mode forward --month 1 --emissions-manifest my_inventory.yaml
```

Each entry in the configuration is one emission file, with keys controlling
how it is loaded:

```yaml
sources:
  - file: my_industry_nox.nc      # a NetCDF under $ORBIT_EMISSION_DIR
    units: kg/m2/s                # see the unit table below
    stack: {height: 150, diameter: 4, temperature: 400, velocity: 15}
                                  # stack parameters give elevated injection
                                  # via plume rise; omit for surface release
  - file: my_voc.nc
    voc_class: anthro             # VOCs form no SOA without a class; see below
  - file: planned_but_not_ready.nc
    required: false               # a missing file is reported, not fatal
```

If a file is missing, ORBIT stops and names it, rather than silently running
with a partial inventory. To allow a partial run:

```bash
orbit --mode forward --month 1 --allow-missing-emissions
```

Valid `voc_class` values: `anthro`, `anthro_high_nox`, `anthro_low_nox`,
`bio_voc`, `bio_monoterpene`, `bio_isoprene`, `biomass_burning`, `ivoc`.

Your files need not be on the model grid. NetCDF inputs get coordinate
detection, longitude convention and latitude-order handling, NaN and
negative cleanup, and bilinear regridding; species are detected from
variable names (`nox`, `so2`, `nh3`, `pm25`, `poa`, `voc`, `so4`, `no3`,
`nh4`). Shapefiles and GeoPackages also work (points, polygons, and lines,
apportioned across the cells they cover; included in the recommended
install, or add the `geo` extra to a bare pip install). Supported units:

| Units | Meaning |
|---|---|
| `kg/m2/s` | flux rate (default); multiplied by cell area |
| `kg/m2` | flux total over a period; requires `averaging_period` in days |
| `kg/s`, `ug/s` | mass rate per cell |
| `kg/year`, `tons/year`, `tonnes/year` | annualised mass (`tons` = US short tons) |

Supply precursors as the compound (NH3, SO2, NOx); ORBIT converts to
element mass internally.

To scale or add emissions in an experiment around an unchanged baseline,
use a scenario file for a marginal or zero-out simulation (see previous
section). Note that a changed
emissions configuration changes the baseline hash, so marginal and zero-out
runs against a baseline built from a different inventory are refused rather
than silently compared.

### Temporal profiles

Emissions often vary over the day (e.g., traffic, residential heating and
cooking). The ORBIT default diurnal profiles are described in
`orbit/data/diurnal_sas.yaml` per source (traffic, residential, industrial,
biomass burning, biogenic, soil, flat). A different diurnal profile can be
specified by using the diurnal-config flag as follows:

```bash
orbit --month 1 --diurnal-config my_profiles.yaml   # your own profiles
orbit --month 1 --diurnal-config none               # uniform diurnal profile
```

A profile is a list of 24 hourly multipliers, averaging 1.0, that
redistributes a source's emissions within the day without changing its
daily total (multipliers that do not average exactly 1.0 are renormalised).
The hours are local time; `ist_offset_hours` shifts them to UTC when they
are sampled into the eight 3-hour bins.

```yaml
ist_offset_hours: 5.5

profiles:
  anthro_traffic:
    hours: [0.42, 0.34, ..., 0.55]    # exactly 24 values

source_profiles:
  ceds_nox_anthro_2022_monthly_surface.nc: anthro_traffic

source_profiles_by_month:      # optional, takes precedence
  11:
    gfed5_pm25_bb_2022_monthly.nc: biomass_burning
```

A mapping for a base filename also covers its
`_surface`/`_low`/`_medium`/`_high` stack-tier variants. A source with no
mapping falls back to the uniform profile with a warning (map it to `flat`
explicitly to silence it), and a mapping naming an undefined profile warns
and falls back rather than failing.

One exception: `cams_soil_nox_climatology_diurnal.nc` carries its own eight
native 3-hour values (`bin_axis: true`), so it is used directly and any YAML
profile for it is ignored.

### Configuration reference

Paths and behaviour come from environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `ORBIT_PREPROC_DIR` | placeholder | input grid directory |
| `ORBIT_PREPROC_YEAR_TAG` | `2022` | year tag in input filenames |
| `ORBIT_CONSTANTS` | placeholder | MERRA-2 constants (land fraction) |
| `ORBIT_EMISSION_DIR` | placeholder | emissions directory |
| `ORBIT_LUT` | placeholder | ISORROPIA lookup table (or pass `--lut`) |
| `ORBIT_OUTPUT_DIR` | `outputs/sas/orbit` | where results are written |
| `ORBIT_EMISSION_MANIFEST` | default configuration | emissions configuration (see above) |
| `ORBIT_EMISSION_SPIKE_FRAC` | `0` (off) | cap any emission cell holding more than this share of a file's domain total, redistributing the excess |
| `ORBIT_BIO_VOC_SPLIT` | unset | biogenic VOC speciation split |
| `ORBIT_IVOC_SCALING` | `1.5` | IVOC scaling factor |
| `ORBIT_VBS_K_AGE`, `ORBIT_VBS_FRAG`, `ORBIT_VBS_A_PHOTO` | scheme defaults | VBS aging, fragmentation, photolysis knobs |
| `ORBIT_LU_BACKEND` | `auto` | sparse LU backend; auto = UMFPACK+METIS, falling back to UMFPACK+COLAMD, then SuperLU |
| `ORBIT_SPECIES_THREADS`, `ORBIT_FACTOR_THREADS` | `1` | parallelism across species / factorizations |
| `ORBIT_PHOTOLYSIS_LUT` | unset | photolysis lookup table path (not currently in production) |

Commonly adjusted flags:

| Flag | Default | Meaning |
|---|---|---|
| `--lut PATH` | `$ORBIT_LUT` | ISORROPIA 7-D lookup table (required in practice) |
| `--emissions-manifest PATH` | default configuration | emissions configuration |
| `--diurnal-config PATH` | default profiles | diurnal emission profiles (`none` = uniform) |
| `--allow-missing-emissions` | off | proceed despite absent required sources |
| `--list-sources` | off | print the resolved inventory and exit |
| `--isorropia-anderson` | on | Anderson-accelerated inorganic closure (`--no-isorropia-anderson` for plain Picard) |
| `--horizontal-fct` | on (forward, zero-out); off (marginal) | flux-corrected horizontal transport; marginal defaults to the monotone low-order linearisation (`--horizontal-fct` / `--no-horizontal-fct` overrides) |
| `--isorropia-closure-iters N` | 6 | outer closure iterations |
| `--closure-mode {full,chem-only}` | `full` | what the outer loop refreshes |
| `--closure-alpha`, `--closure-tol` | 0.5, 0.02 | closure damping and convergence gate (2% on particulate NO3/NH4) |
| `--chemistry-iters N` | 0 | diagnostic-OH path (> 0 not currently in production) |
| `--top-bc-days`, `--lateral-bc-days` | 10, 1 | boundary-condition relaxation timescales (days) |

## Tests

```bash
pytest -m "not slow"   # ~480 tests; a few skips are normal
pytest                 # also runs tests that read the full input data
```

The tests marked `slow` read the data archive (the 1.4 GB lookup table and
input grids) and skip when it is absent.

## Citation

ORBIT was developed by Dr. Sumil K. Thakrar (2026). Manuscript in
preparation (TBD). Please email sumilthakrar@gmail.com for more
information.

## License

Released under the [MIT License](LICENSE). If you use ORBIT in academic work,
please also cite the accompanying manuscript (see Citation above).
