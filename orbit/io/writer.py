"""Save operators, footprints, and concentrations."""

import numpy as np
import scipy.sparse as sp


def save_operator(filepath: str, L: sp.csc_matrix, metadata: dict = None) -> None:
    """Save sparse operator to npz file.

    Parameters
    ----------
    filepath : str
    L : csc_matrix
    metadata : dict, optional
    """
    save_dict = {
        "data": L.data,
        "indices": L.indices,
        "indptr": L.indptr,
        "shape": np.array(L.shape),
    }
    if metadata:
        for k, v in metadata.items():
            save_dict[f"meta_{k}"] = np.array(v)
    np.savez_compressed(filepath, **save_dict)


def load_operator(filepath: str) -> sp.csc_matrix:
    """Load sparse operator from npz file."""
    loaded = np.load(filepath)
    shape = tuple(loaded["shape"])
    return sp.csc_matrix(
        (loaded["data"], loaded["indices"], loaded["indptr"]),
        shape=shape,
    )
