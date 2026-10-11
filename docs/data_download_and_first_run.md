# ORBIT data deposit: download and first real run (local workstation)

How to pull `deposit_A` from MSI tier-2 onto this machine
(`/home/sumil`, 16 cores, 31 GB RAM) and run the first real January
solve. Adapted 2026-07-31 from the ORBIT-session walkthrough, with the
transfer route rewritten for what is actually installed here: the Globus
CLI (via `uvx`) and Globus Connect Personal are set up and proven (the
airsheds ~80 GB migration ran through them today); `s3cmd` is NOT
installed and has no config. Solve-side numbers (test counts, memory,
timings, tolerances) come from the ORBIT session and are not re-verified
here.

The point of the January run: the source-receptor path has unit tests
but has never been through a real ~50-minute solve. One month exercises
logging, the budget, deposition maps, and the reproduction check in a
single go.

## What to download

Only `deposit_A` — that is everything needed to run. B (evaluation,
2.8 MB) and C (health gradients, 34 MB) are not needed yet.
`deposit_A` is 3.4 GB / 178 objects:

| Path (under `~/orbit_data/`) | What it is |
|---|---|
| `inputs/grids_2022/` | 96 files, 19 MB each (12 months × 8 bins) |
| `inputs/isorropia_lut_7d.npz` | 1.4 GB — every solve needs it |
| `inputs/MERRA2.20150101.CN.05x0625.nc4` | land fraction |
| `emissions/sas/` | **use this one** (GFED5 biomass burning, the production default since 2026-08-02) |
| `emissions/sas_finn/` | FINN v2.5 variant, same filenames — easy to point at by mistake |
| `emissions/diurnal_sas.yaml` | diurnal emission profiles by sector |
| `reference/orbit_M01.npz` | 96 MB, for the reproduction check |
| `MANIFEST.tsv` | SHA-256 per file |

## Route 1 — Globus (recommended on this machine)

Endpoints:

- **Source**: `f84c5701-cc99-418a-8afe-f026eef1dd87` — "UMN MSI Tier2"
  (S3 connector; the bucket appears as the path `/orbitrelease/`).
  There is also an HA variant `fbb22b3e-175c-4694-a6cf-8e79366f990e`
  ("UMN MSI HA Tier2") if the first misbehaves.
- **Destination**: `150105a4-8cfe-11f1-bd69-0ee7ef9370d9` — Globus
  Connect Personal "Office" on this machine. Paths are home-relative:
  `/~/orbit_data` = `/home/sumil/orbit_data`.

One-time setup (probed 2026-07-31: the Tier2 collection requires a
consent this account has not yet granted — the `globus ls` probe printed
exactly this command):

```bash
uvx --from globus-cli globus session consent \
  'urn:globus:auth:scope:transfer.api.globus.org:all[*https://auth.globus.org/scopes/f84c5701-cc99-418a-8afe-f026eef1dd87/data_access]'
```

Make sure Globus Connect Personal is running (it is NOT a service; it
was started manually for the airsheds transfer and may be down):

```bash
~/globusconnectpersonal-3.2.9/globusconnectpersonal -status \
  || ~/globusconnectpersonal-3.2.9/globusconnectpersonal -start &
```

Full deposit (3.4 GB; `--sync-level checksum` makes re-runs safe):

```bash
uvx --from globus-cli globus transfer --recursive --sync-level checksum \
  --label 'orbit deposit_A' \
  'f84c5701-cc99-418a-8afe-f026eef1dd87:/orbitrelease/v1/deposit_A/' \
  '150105a4-8cfe-11f1-bd69-0ee7ef9370d9:/~/orbit_data/'
```

Faster January-only variant (~1.7 GB; Globus batch lines take explicit
paths, no wildcards — the 8 bins are `B01`–`B08`):

```bash
cat > /tmp/orbit_jan_batch.txt <<'EOF'
inputs/isorropia_lut_7d.npz inputs/isorropia_lut_7d.npz
inputs/MERRA2.20150101.CN.05x0625.nc4 inputs/MERRA2.20150101.CN.05x0625.nc4
inputs/grids_2022/sas_2022_M01_B01.nc inputs/grids_2022/sas_2022_M01_B01.nc
inputs/grids_2022/sas_2022_M01_B02.nc inputs/grids_2022/sas_2022_M01_B02.nc
inputs/grids_2022/sas_2022_M01_B03.nc inputs/grids_2022/sas_2022_M01_B03.nc
inputs/grids_2022/sas_2022_M01_B04.nc inputs/grids_2022/sas_2022_M01_B04.nc
inputs/grids_2022/sas_2022_M01_B05.nc inputs/grids_2022/sas_2022_M01_B05.nc
inputs/grids_2022/sas_2022_M01_B06.nc inputs/grids_2022/sas_2022_M01_B06.nc
inputs/grids_2022/sas_2022_M01_B07.nc inputs/grids_2022/sas_2022_M01_B07.nc
inputs/grids_2022/sas_2022_M01_B08.nc inputs/grids_2022/sas_2022_M01_B08.nc
emissions/sas_finn/ emissions/sas_finn/ --recursive
emissions/diurnal_sas.yaml emissions/diurnal_sas.yaml
reference/ reference/ --recursive
MANIFEST.tsv MANIFEST.tsv
EOF
uvx --from globus-cli globus transfer --sync-level checksum \
  --label 'orbit deposit_A January' \
  --batch /tmp/orbit_jan_batch.txt \
  'f84c5701-cc99-418a-8afe-f026eef1dd87:/orbitrelease/v1/deposit_A' \
  '150105a4-8cfe-11f1-bd69-0ee7ef9370d9:/~/orbit_data'
```

Monitor with `uvx --from globus-cli globus task show <task-id>`. If a
grid filename 404s, `globus ls 'f84c5701-…:/orbitrelease/v1/deposit_A/inputs/grids_2022/'`
(post-consent) shows the real names — the `B01`–`B08` pattern is
inferred from the deposit README's quickstart, not verified against the
bucket. Two lessons from the airsheds transfer that apply verbatim:
hidden (dot-)paths can stall a whole MSI-side batch, and the queue does
not skip past a failing entry — neither should bite here since
`deposit_A` has no dot-paths, but if a transfer sits retrying one entry,
that is where to look.

**Status 2026-07-31**: consent granted; deposit_A transferred (task
`a9720a68-8dd2-11f1-8a84-0afff7074b21`, SUCCEEDED) into
`/home/sumil/orbit_data/` and **verified — all 178 MANIFEST.tsv SHA-256
checksums pass**.
Also fetched, at Sumil's request: `deposit_B` (evaluation; satellite +
surface comparisons) into `/home/sumil/orbit_data/deposit_B/` (task
`0b5bad96-8dd3-11f1-9fe4-0afff7074b21`, SUCCEEDED, manifest verified)
and `deposit_C` (health gradients, `deaths_gradient/`) into
`/home/sumil/orbit_data/deposit_C/` (task
`0c6706fd-8dd3-11f1-8556-02ce27bde401`, SUCCEEDED, manifest verified). B and C live in
subdirectories because each carries its own `MANIFEST.tsv`/`README.md`,
which would collide with deposit_A's at the root. Each deposit also has
a `FETCH.md` (B and C) describing what it is.

## Route 2 — s3cmd (alternative)

Not installed here and unconfigured. If preferred:
`uvx --from s3cmd s3cmd …` (or `pip install s3cmd` into some env), then
`~/.s3cfg` needs MSI's tier-2 endpoint and your keys from the MSI
portal:

```ini
host_base = s3.msi.umn.edu
host_bucket = %(bucket)s.s3.msi.umn.edu
access_key = <from MSI portal>
secret_key = <from MSI portal>
```

Then the original walkthrough commands work as written:

```bash
mkdir -p ~/orbit_data && cd ~/orbit_data
s3cmd get --recursive s3://orbitrelease/v1/deposit_A/ .
```

(January-only: the `s3cmd get` variants with
`inputs/grids_2022/sas_2022_M01_B0*.nc` etc. — s3cmd does support the
wildcard.) If `s3cmd ls s3://orbitrelease` errors, the endpoint config
above is the cause.

## Verify the transfer

```bash
cd ~/orbit_data
awk -F'\t' 'NR>1 {print $3"  "$1}' MANIFEST.tsv | sha256sum -c --quiet && echo "verified"
```

With the January-only download it will complain about the files you
skipped — expected.

## Install

Conda lives at `/home/sumil/miniforge3` on this machine:

```bash
cd /home/sumil/orbit
conda create -n orbit python=3.11 && conda activate orbit
conda install -c conda-forge scikit-umfpack     # do NOT skip — see below
pip install -e ".[fast,dev]"
```

Do not skip `scikit-umfpack`. It is the default LU backend; without it
ORBIT silently falls back to SuperLU at ~1.5–2× the memory, and a
forward month already peaks at ~17 GB. This machine has 31 GB total, so
UMFPACK is the difference between running and being OOM-killed. Check
`free -g` shows ≥ 20 GB available before starting, and do not run the
solve concurrently with heavy airsheds steps (eigensolves and residence
GMRES peak at 8–10 GB).

## Configure

```bash
export ORBIT_PREPROC_DIR=$HOME/orbit_data/inputs/grids_2022
export ORBIT_PREPROC_YEAR_TAG=2022
export ORBIT_CONSTANTS=$HOME/orbit_data/inputs/MERRA2.20150101.CN.05x0625.nc4
export ORBIT_EMISSION_DIR=$HOME/orbit_data/emissions/sas
export ORBIT_OUTPUT_DIR=$HOME/orbit_out
export ORBIT_SPECIES_THREADS=1   # 4 is ~8% faster but peaks at ~11 GB vs ~5 GB
```

Note `sas`, NOT `sas_finn` — the two directories hold identical
filenames, so pointing at the wrong one fails silently.

GFED5 became the default on 2026-08-02. The published deposit was
produced with FINN; a matched five-month comparison (January, March,
April, October, November, all with conservative regridding) found GFED5
better on spatial correlation in four of the five, decisively in March
(R² 0.113 vs 0.034) and November (0.535 vs 0.430), with a smaller worst
cell in all five. FINN scores better on normalised mean bias in every
month, but that reflects a systemic 50–65% model low bias that a fire
inventory should not be compensating. See
the development notes (FINN versus GFED5 matched comparison, 2026-08-02).

Results from this configuration will NOT match the shipped reference,
which predates the GFED5 switch, the conservative regridding fix, the
IVOC species-slot fix and the HNO3 wet-scavenging fix. Set
`ORBIT_EMISSION_DIR=$HOME/orbit_data/emissions/sas_finn` and
`ORBIT_REGRID=bilinear` to reproduce it.

## Check before committing 50 minutes

```bash
pytest tests/ -m "not slow"      # expect 373 passed, 2 skipped
orbit --list-sources             # expect 26 resolved sources
```

`--list-sources` is the real check — it resolves the manifest against
the emissions directory and fails loudly naming any missing file.

## The run

```bash
mkdir -p ~/orbit_out
orbit --month 1 --isorropia-anderson --horizontal-fct \
      --lut ~/orbit_data/inputs/isorropia_lut_7d.npz \
      > ~/orbit_out/orbit_M01_console.log 2>&1 &
```

30–50 min, ~17 GB. (Backgrounded with output to a file, per this
machine's convention — never pipe a long run through `tail`; watch the
file.) The startup banner prints the resolved LU backend; if UMFPACK is
missing there is a loud warning with the fix.

## Reproduction check

`compare_manifest_run.py` is already on this machine in the tracemodel
checkout:

```bash
python /home/sumil/tracemodel/scripts/release/compare_manifest_run.py \
    ~/orbit_out/orbit_M01.npz  ~/orbit_data/reference/orbit_M01.npz
```

Expect surface PM2.5 max relative difference ~1e-12 (the number at the
top of the output — that is the one that matters). `RESULT: DIFFERS`
from ISORROPIA cross-partials disagreeing in a handful of cells is the
known FD-near-discontinuity behaviour and is harmless.

## Afterwards

- Read `orbit_M01.log` and the `.run.json` — this run is the first
  end-to-end exercise of the new logging, which has only been
  unit-tested.
- The ORBIT session flagged one follow-up experiment:
  `ORBIT_LU_BACKEND=umfpack_metis` with a pip-installed extra. The
  walkthrough text was cut off at exactly that sentence, so confirm the
  package name and expected effect with the ORBIT session before
  running it.
