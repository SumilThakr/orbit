"""The restricted grid read produces a GridData identical to a full read.

Guards _GRID_INPUT_VARS (memory-audit fix 7a, 2026-08-01): if a field is
consumed by load_grid but missing from the set, the restricted load would
silently flip an optional-field fallback (e.g. has_split_fluxes). This
test loads a real January bin file both ways and compares every attribute
bitwise, so any such omission fails loudly.
"""
import os

import numpy as np
import pytest

from orbit.core.grid_data import load_grid

_PREPROC_DIR = os.environ.get("ORBIT_PREPROC_DIR", "")
_BIN1 = os.path.join(_PREPROC_DIR, "sas_2022_M01_B01.nc")
_CONSTANTS = os.environ.get("ORBIT_CONSTANTS", "")


@pytest.mark.skipif(not os.path.exists(_BIN1),
                    reason="real preprocessor file not available")
def test_restricted_read_matches_full_read():
    constants = _CONSTANTS if os.path.exists(_CONSTANTS) else None
    g_full = load_grid(_BIN1, constants, read_all=True)
    g_restr = load_grid(_BIN1, constants)

    keys_full = set(vars(g_full))
    keys_restr = set(vars(g_restr))
    assert keys_full == keys_restr, (
        f"attribute sets differ: {keys_full ^ keys_restr}")

    for name in sorted(keys_full):
        a, b = getattr(g_full, name), getattr(g_restr, name)
        if isinstance(a, np.ndarray):
            assert a.shape == b.shape and a.dtype == b.dtype, name
            assert np.array_equal(a, b, equal_nan=True), (
                f"attribute {name} differs between full and restricted read")
        else:
            assert a == b, f"attribute {name} differs: {a!r} vs {b!r}"
