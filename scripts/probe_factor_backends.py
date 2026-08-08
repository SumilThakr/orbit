#!/usr/bin/env python
"""Factor one bin of the real transport operator under several orderings (D2).

Prints, per ordering: UMFPACK's symbolic fill estimate, the exact numeric
factor size and factorization peak (from the Info array), and wall times.
Answers "which ordering should this machine use" in minutes, without a
solve. From the 2026-08-01 memory audit's diagnostics list.

Needs the standard ORBIT_* environment (grids, constants). Usage:

    python scripts/probe_factor_backends.py [--month 1]
"""

import argparse
import time

from orbit.cli import CONSTANTS, _preproc_path
from orbit.core.deposition import assemble_deposition
from orbit.core.grid_data import load_grid
from orbit.core.indexing import CellIndexer
from orbit.core.operator import assemble_transport_block
from orbit.core.solve import (
    _HAS_METIS, _HAS_UMFPACK, _UmfpackLU, compute_metis_ordering,
    umfpack_free_symbolic, umfpack_symbolic,
)

_ORDERING_NAMES = {0: "cholmod", 1: "amd/colamd", 2: "given", 3: "metis",
                   4: "best", 5: "none", 6: "user"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--month", type=int, default=1)
    args = ap.parse_args()

    if not _HAS_UMFPACK:
        raise SystemExit("UMFPACK not available; nothing to probe.")

    t0 = time.time()
    grid = load_grid(_preproc_path(args.month, 1), CONSTANTS)
    indexer = CellIndexer(grid.nz, grid.ny, grid.nx)
    L = (assemble_transport_block(grid, indexer, scheme="exp")
         + assemble_deposition(grid, indexer, 0)).tocsc()
    print(f"Operator: month {args.month} bin 1, n={L.shape[0]}, "
          f"nnz={L.nnz} (built in {time.time() - t0:.1f}s)")
    print()

    configs = [("umfpack default", {}),
               ("amd", {"ordering": "amd"}),
               ("metis-internal", {"ordering": "metis"})]
    if _HAS_METIS:
        t0 = time.time()
        perm = compute_metis_ordering(L, verbose=False)
        print(f"pymetis nested dissection: {time.time() - t0:.1f}s")
        configs.append(("pymetis-qinit", {"qinit": perm}))
    else:
        print("pymetis not installed; skipping the qinit config.")
    print()

    print(f"{'config':<18s} {'sym(s)':>7s} {'num(s)':>7s} {'est nnz(LU)':>12s} "
          f"{'factor MB':>10s} {'peak MB':>9s} {'strat':>6s} {'order':>10s}")
    for label, kw in configs:
        try:
            t0 = time.time()
            sym = umfpack_symbolic(L, verbose=False, **kw)
            t_sym = time.time() - t0
            t0 = time.time()
            lu = _UmfpackLU(L, sym)
            t_num = time.time() - t0
            strat = {1: "unsym", 3: "sym"}.get(sym["strategy_used"],
                                               str(sym["strategy_used"]))
            order = _ORDERING_NAMES.get(sym["ordering_used"],
                                        str(sym["ordering_used"]))
            print(f"{label:<18s} {t_sym:>7.2f} {t_num:>7.2f} "
                  f"{sym['lunz_est'] / 1e6:>11.1f}M "
                  f"{lu.numeric_bytes / 1e6:>10.1f} "
                  f"{lu.peak_bytes / 1e6:>9.1f} {strat:>6s} {order:>10s}")
            del lu
            umfpack_free_symbolic(sym)
        except Exception as exc:
            print(f"{label:<18s} FAILED: {exc}")


if __name__ == "__main__":
    main()
