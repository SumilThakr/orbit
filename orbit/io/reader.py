"""Load preprocessor NetCDF files for ORBIT.

Lightweight port from the reference implementation. Reads the preprocessed-input
preprocessor output and returns a dict of numpy arrays.
"""

import numpy as np

try:
    import netCDF4 as nc
except ImportError:
    nc = None

try:
    import xarray as xr
except ImportError:
    xr = None


def load_preprocessor(filepath: str, variables=None) -> dict:
    """Load preprocessor NetCDF file.

    Parameters
    ----------
    filepath : str
        Path to preprocessor output NetCDF file
    variables : set of str or None
        When given, read only these variables (plus coordinates and
        global attributes). The preprocessor files carry ~84 variables
        of which GridData consumes ~59; skipping the rest saves ~170 MB
        resident across 8 bins and speeds the load.

    Returns
    -------
    dict
        Dictionary containing the requested preprocessor fields as numpy
        arrays, plus 'lon', 'lat' coordinate arrays and '_attrs' global
        attributes.
    """
    if nc is not None:
        return _load_with_netcdf4(filepath, variables=variables)
    elif xr is not None:
        return _load_with_xarray(filepath, variables=variables)
    else:
        raise ImportError("Either netCDF4 or xarray is required")


def _load_with_netcdf4(filepath: str, variables=None) -> dict:
    """Load using netCDF4. ``variables`` (a set) restricts which are read;
    coordinate variables are always included."""
    data = {}
    with nc.Dataset(filepath, "r") as f:
        for var in f.variables:
            if (variables is not None and var not in variables
                    and var not in ("lon", "lat")):
                continue
            data[var] = np.asarray(f.variables[var][:])
        data["_attrs"] = {attr: f.getncattr(attr) for attr in f.ncattrs()}
    _add_coordinates_from_attrs(data)
    return data


def _load_with_xarray(filepath: str, variables=None) -> dict:
    """Load using xarray. Same ``variables`` contract as _load_with_netcdf4."""
    ds = xr.open_dataset(filepath)
    data = {}
    for var in ds.data_vars:
        if variables is not None and var not in variables:
            continue
        data[var] = ds[var].values
    for coord in ds.coords:
        data[coord] = ds[coord].values
    data["_attrs"] = dict(ds.attrs)
    ds.close()
    _add_coordinates_from_attrs(data)
    return data


def _add_coordinates_from_attrs(data: dict) -> None:
    """Generate lon/lat arrays from global attributes if missing.

    The preprocessor stores grid spec in attributes: x0, y0, dx, dy, nx, ny.
    x0/y0 are the SW corner (cell edge), so cell centers = x0 + (i+0.5)*dx.
    """
    if "lon" in data and "lat" in data:
        return

    attrs = data.get("_attrs", {})
    x0 = attrs.get("x0")
    y0 = attrs.get("y0")
    dx = attrs.get("dx")
    dy = attrs.get("dy")
    nx = attrs.get("nx")
    ny = attrs.get("ny")

    if nx is None or ny is None:
        for var in ["Kzz", "Dz", "Temperature"]:
            if var in data and len(data[var].shape) == 3:
                _, ny, nx = data[var].shape
                break

    if all(v is not None for v in [x0, y0, dx, dy, nx, ny]):
        data["lon"] = np.asarray(x0 + (np.arange(int(nx)) + 0.5) * dx, dtype=np.float64)
        data["lat"] = np.asarray(y0 + (np.arange(int(ny)) + 0.5) * dy, dtype=np.float64)


def truncate_u(arr, nx):
    """Drop the east-most face from a staggered-U array.

    Preprocessor stores west-face velocities on staggered grid (nz, ny, nx+1).
    After truncation, UAvg[k,j,i] is the west face velocity of cell (k,j,i).
    The dropped column (index nx) is the east boundary face with no cell to its right.
    """
    if arr is not None and arr.shape[2] == nx + 1:
        return arr[:, :, :-1]
    return arr


def truncate_v(arr, ny):
    """Drop the north-most face from a staggered-V array.

    Preprocessor stores south-face velocities on staggered grid (nz, ny+1, nx).
    After truncation, VAvg[k,j,i] is the south face velocity of cell (k,j,i).
    The dropped row (index ny) is the north boundary face with no cell above it.
    """
    if arr is not None and arr.shape[1] == ny + 1:
        return arr[:, :-1, :]
    return arr


def truncate_w(arr, nz):
    """Drop the top-most face from a staggered-W array.

    Preprocessor stores bottom-face omega on staggered grid (nz+1, ny, nx).
    After truncation, omega[k,j,i] is the bottom face of layer k.
    The dropped level (index nz) is the top-of-atmosphere face.
    """
    if arr is not None and arr.shape[0] == nz + 1:
        return arr[:-1, :, :]
    return arr


def load_frland(constants_path: str, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Load FRLAND from MERRA-2 constants file and subset to grid.

    Parameters
    ----------
    constants_path : str
        Path to MERRA2 constants NetCDF
    lon, lat : ndarray
        Target grid coordinates

    Returns
    -------
    is_land : ndarray of uint8, shape (ny, nx)
        Boolean land mask (1 where FRLAND >= 0.1)
    """
    if nc is not None:
        with nc.Dataset(constants_path, "r") as f:
            frland_global = np.asarray(f.variables["FRLAND"][0, :, :])
            lat_global = np.asarray(f.variables["lat"][:])
            lon_global = np.asarray(f.variables["lon"][:])
    elif xr is not None:
        ds = xr.open_dataset(constants_path)
        frland_global = ds["FRLAND"].values[0, :, :]
        lat_global = ds["lat"].values
        lon_global = ds["lon"].values
        ds.close()
    else:
        raise ImportError("Either netCDF4 or xarray is required")

    # Find nearest indices for subsetting
    j_indices = np.array([np.argmin(np.abs(lat_global - la)) for la in lat])
    i_indices = np.array([np.argmin(np.abs(lon_global - lo)) for lo in lon])

    frland = frland_global[np.ix_(j_indices, i_indices)]
    is_land = np.asarray(frland >= 0.1, dtype=np.uint8)
    return is_land
