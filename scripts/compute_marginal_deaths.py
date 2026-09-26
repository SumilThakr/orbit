"""Compute marginal deaths per emission via the periodic-orbit adjoint.

Consumes the deaths-gradient NetCDF from the companion health-impacts
data deposit and solves the per-month, per-species adjoint orbit
``L^T λ = G^T S`` to get ∂J/∂e at every (emission cell, species, bin).

This script is the ORBIT-side half of the marginal-deaths integration.

Usage
-----
    python scripts/compute_marginal_deaths.py \
        --gradient /path/to/deaths_gradient_gemm5cod.nc \
        --preproc-base /path/to/data/preproc/output/SUBSET/SAS/2022 \
        --month 1 \
        --species PM25_primary \
        --out-dir /path/to/adjoint_outputs/

Or for all 12 months and all currently-supported species:

    python scripts/compute_marginal_deaths.py \
        --gradient /path/to/deaths_gradient_gemm5cod.nc \
        --preproc-base /path/to/data/preproc/output/SUBSET/SAS/2022 \
        --months 1-12 \
        --species PM25_primary \
        --out-dir /path/to/adjoint_outputs/

Subdistrict aggregation and the multi-CRF scenario rollup live in
scripts/postprocess_marginal_deaths_scenarios.py, which consumes the
per-month NetCDFs this script writes.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Import the package from the checkout this script lives in, not from
# whichever copy the environment has installed (an editable install can
# point at a different checkout).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def _parse_months(spec: str) -> list[int]:
    """'1-12', '3', '1,4,7' all valid."""
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def _preproc_path_fn(preproc_base: Path):
    """Build the per-bin path callable the adjoint driver wants.

    The production layout is:
      <preproc_base>/sas_2022_M<MM>_B<BB>.nc
    where MM is 01..12, BB is 01..08.
    """
    def fn(month: int, bin_idx: int) -> str:
        return str(preproc_base / f"sas_2022_M{month:02d}_B{bin_idx:02d}.nc")
    return fn


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--gradient", type=Path, required=True,
        help="deaths_gradient_<crf>.nc from the health-impacts data deposit.",
    )
    ap.add_argument(
        "--preproc-base", type=Path, required=True,
        help="Base directory containing sas_2022_M<MM>_B<BB>.nc bin files.",
    )
    ap.add_argument(
        "--orbit-dir", type=Path, default=None,
        help="Production orbit sweep directory (contains orbit_M<MM>.npz). "
             "Used to load iso_f_nh4/iso_f_no3 partitioning fractions for "
             "the growth Jacobian's NH/NO3 receptor terms — without this, "
             "the bin-file legacy NHPartitioning/NOPartitioning is used "
             "(wrong for ISORROPIA-based production runs).",
    )
    ap.add_argument(
        "--constants",
        default=(
            "/path/to/data/MERRA2_flat/2015/"
            "MERRA2.20150101.CN.05x0625.nc4"
        ),
        type=str,
        help="MERRA2 constants NetCDF (FRLAND).",
    )
    ap.add_argument(
        "--month", type=int, default=None,
        help="Single month 1..12. Mutually exclusive with --months.",
    )
    ap.add_argument(
        "--months", type=str, default=None,
        help="Month range/list, e.g. '1-12' or '1,4,7'. Mutually exclusive "
             "with --month.",
    )
    ap.add_argument(
        "--species", action="append", default=None,
        help="Pollutant species to compute adjoint for. Repeatable. "
             "Default: PM25_primary. See orbit.modes.adjoint.EMITTED_SPECIES "
             "for the list.",
    )
    ap.add_argument(
        "--out-dir", type=Path, required=True,
        help="Output directory; one adjoint_M<MM>.nc per month.",
    )
    ap.add_argument("--krylov-tol", type=float, default=1e-6)
    ap.add_argument("--maxiter", type=int, default=200)
    ap.add_argument("--horizontal-fct", action="store_true",
                    help="Add the frozen-coefficient FCT anti-diffusive operator "
                         "L_AD (van Leer φ + Zalesak C frozen at the baseline "
                         "orbit) to the adjoint operator, so the SR sensitivities "
                         "transpose the same ~2nd-order transport as an FCT "
                         "forward. Requires --orbit-dir pointing at FCT orbits.")
    args = ap.parse_args(argv)

    if (args.month is None) == (args.months is None):
        ap.error("specify exactly one of --month or --months.")
    if args.constants.startswith("/path/to/data"):
        ap.error(
            "--constants was not given, so it is still the placeholder "
            f"default\n  {args.constants}\nwhich is not a real path. Pass "
            "--constants pointing at the MERRA2 constants NetCDF "
            "(inputs/MERRA2.20150101.CN.05x0625.nc4 in the data archive)."
        )
    months = [args.month] if args.month is not None else _parse_months(args.months)
    species = args.species or ["PM25_primary"]

    from orbit.modes.adjoint import run_adjoint_month

    args.out_dir.mkdir(parents=True, exist_ok=True)
    preproc_fn = _preproc_path_fn(args.preproc_base)

    for month in months:
        print(f"=== Month {month:02d} ===")
        orbit_npz = None
        if args.orbit_dir is not None:
            orbit_npz = str(args.orbit_dir / f"orbit_M{month:02d}.npz")
        run_adjoint_month(
            month=month,
            gradient_path=str(args.gradient),
            output_dir=str(args.out_dir),
            preproc_path_fn=preproc_fn,
            constants_path=args.constants,
            orbit_npz_path=orbit_npz,
            species_keys=species,
            krylov_tol=args.krylov_tol,
            maxiter=args.maxiter,
            horizontal_fct=args.horizontal_fct,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
