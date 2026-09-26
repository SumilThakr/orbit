"""HEMCO monthly climatology loader for DCOMP orbit chemistry.

Reads the 10-year GEOS-Chem v14.0.0 benchmark monthly-mean archive at
4x5 resolution and regrids key fields to the ORBIT grid.

The per-bin GC archive we ingest (`/path/to/data/AS/`) does
not include O3, CO, or CH4 as output species. This loader fills that
gap with monthly-mean climatology.

HNO3 is NOT loaded here. In the ORBIT design HNO3 is the gas-phase
component of the transported TotalNO3 species, split post-solve by
ISORROPIA; it is never read from any external archive. (HEMCO provides
no HNO3 climatology either, so a reader would have nowhere to read from
even if the design called for it.)

The climatology is bin-invariant (monthly mean, no diurnal cycle). For
O3 this is only the initial guess / free-tropospheric boundary condition;
the orbit solve computes its own per-bin O3 self-consistently. For CO
and CH4 the monthly mean is used as-is (the "B-flat" treatment of the
unified design document).

Usage
-----
    from orbit.hemco import load_hemco_climatology
    clim = load_hemco_climatology(year=2016, month=1,
                                  target_lats=grid.lat, target_lons=grid.lon)
    # clim is a dict of (nz_hemco, ny_target, nx_target) arrays in ug/m3
    #   plus pressure-level info for vertical interpolation later.

Fields returned (ug/m3, compound mass):
    O3, CO, CH4, OH, NO, NO2, SO2, HNO2, NIT, NH4, SO4, H2O2

Plus J-values in s^-1 (from the benchmark's TUV calculation):
    Jval_NO2, Jval_O3_O1D, Jval_NO3

References
----------
README at /path/to/data/GCClassic_Output/14.0.0/ — benchmark
is the 2010-2019 GC v14.0.0-rc.3 monthly-mean output.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import xarray as xr

# Default archive location (overridable).
DEFAULT_HEMCO_DIR = "/path/to/data/GCClassic_Output/14.0.0"

# Species mass (g/mol), for mol/mol -> ug/m3 conversion.
MW = {
    "O3": 48.0,
    "CO": 28.01,
    "CH4": 16.04,
    "OH": 17.01,
    "HO2": 33.01,
    "NO": 30.01,
    "NO2": 46.01,
    "SO2": 64.07,
    "HNO2": 47.01,
    "NIT": 62.0,       # NO3- (compound)
    "NH4": 18.04,
    "SO4": 96.06,
    "H2O2": 34.01,
    "N2O5": 108.01,
    "CH2O": 30.026,
}
MW_AIR = 28.97

# Species we extract from SpeciesConc (mol/mol mixing ratio).
# Stored in the output dict under the SpeciesConc_* name stripped of its prefix.
# HNO3 is NOT included: it is the gas-phase split of ORBIT's transported
# TotalNO3, produced post-solve by ISORROPIA, never read from an archive.
SPECIES_KEYS = [
    "O3", "CO", "CH4", "OH", "NO", "NO2",
    "SO2", "HNO2", "NIT", "NH4", "SO4", "H2O2",
    "CH2O",  # Formaldehyde — DCOMP noon HO2/OH source. 3h AS archive
             # lacks IJ_AVG_S__CH2O, so HCHO comes from the monthly mean;
             # jHCHO (TUV) remains bin-resolved.
]

# J-values we extract from the JValues collection (s^-1).
# Jval_HNO3 is intentionally excluded — we don't load HNO3 as a prescribed
# field so its photolysis rate isn't needed either. ORBIT's own photolysis
# comes from the TUV LUT (`orbit/core/photolysis.py`).
JVAL_KEYS = ["Jval_NO2", "JvalO3O1D", "Jval_NO3"]


@dataclass
class HemcoClimatology:
    """Regridded monthly-mean GC climatology on the ORBIT horizontal grid.

    All 3D fields are (nz_hemco, ny_target, nx_target). Pressure levels
    correspond to GC-native 72-layer grid via `lev_centers` (sigma) and
    `p_edges` (Pa, computed from Ap/Bp and a reference surface pressure).
    Vertical remapping to the ORBIT 15-layer grid happens downstream.
    """
    year: int
    month: int
    lats: np.ndarray                  # (ny,)
    lons: np.ndarray                  # (nx,)
    lev_centers: np.ndarray           # (nz_hemco,) sigma-like
    species_ugm3: dict = field(default_factory=dict)       # name -> (nz, ny, nx) ug/m3
    species_molmol: dict = field(default_factory=dict)     # name -> (nz, ny, nx) mol/mol
    jvalues: dict = field(default_factory=dict)            # name -> (nz, ny, nx) s^-1
    surface_T: Optional[np.ndarray] = None                 # (ny, nx) K if available
    surface_P: Optional[np.ndarray] = None                 # (ny, nx) Pa if available


def _blocky_upsample(data, src_lats, src_lons, tgt_lats, tgt_lons):
    """Nearest-cell resample from (..., src_ny, src_nx) -> (..., tgt_ny, tgt_nx).

    No interpolation — each target cell inherits the enclosing source cell's
    value.  Matches combine.py's `blocky_upsample` convention for consistency.
    """
    lat_idx = np.array([int(np.argmin(np.abs(src_lats - t))) for t in tgt_lats])
    lon_idx = np.array([int(np.argmin(np.abs(src_lons - t))) for t in tgt_lons])
    return data[..., lat_idx, :][..., lon_idx]


def _molmol_to_ugm3(X, species, rho_air=None):
    """Convert mol/mol volume mixing ratio to ug/m3 compound mass.

    Derivation:
        mol_species / m3 = X * (rho_air[g/m3] / MW_air[g/mol])
                        = X * rho_air[kg/m3] * 1000 / MW_air
        ug_species / m3  = mol_species/m3 * MW_species[g/mol] * 1e6 [ug/g]
                        = X * rho_air * MW_species / MW_air * 1e9

    If rho_air is None, uses a standard surface value (1.2 kg/m3).
    """
    if rho_air is None:
        rho_air = 1.2
    return X * rho_air * (MW[species] / MW_AIR) * 1e9


def density_on_levels(lev_sigma, orbit_sigma, orbit_rho):
    """Air density on the climatology's levels from the model's own column.

    For each climatology level (sigma-like coordinate) take the model layer
    whose domain-mean sigma is nearest, the same nearest-layer rule as
    ``layer_slice_by_sigma``. Returns (n_lev, ny, nx) kg/m3.
    """
    lev = np.asarray(lev_sigma, dtype=np.float64).ravel()
    sig = np.asarray(orbit_sigma, dtype=np.float64).ravel()
    rho = np.asarray(orbit_rho, dtype=np.float64)
    pick = np.array([int(np.argmin(np.abs(sig - l))) for l in lev])
    return rho[pick]


def load_hemco_climatology(year, month, target_lats, target_lons,
                           rho_air=None, hemco_dir=DEFAULT_HEMCO_DIR,
                           with_jvalues=True, rho_profile=None):
    """Load monthly-mean GC climatology regridded to ORBIT's horizontal grid.

    Parameters
    ----------
    year, month : int
        Calendar year/month.
    target_lats, target_lons : ndarray
        ORBIT grid cell centers (1D arrays in degrees).
    rho_air : ndarray or None
        If provided, shape (nz_hemco, ny, nx) or (ny, nx) in kg/m3. If None
        and ``rho_profile`` is None, a scalar 1.2 kg/m3 is used, which is
        wrong aloft by the density profile; pass ``rho_profile`` instead.
    rho_profile : (orbit_sigma, orbit_rho) or None
        The model's own column density: ``orbit_sigma`` (nz,) domain-mean
        sigma of each model layer and ``orbit_rho`` (nz, ny, nx) kg/m3.
        Mapped onto the climatology's levels by nearest sigma
        (``density_on_levels``) and used for the mol/mol to ug/m3 conversion.
    hemco_dir : str
        Root of the HEMCO archive (default: cluster path).
    with_jvalues : bool
        If True, also load the JValues collection (photolysis rates).

    Returns
    -------
    HemcoClimatology
        Monthly climatology with 3D fields on the target horizontal grid
        but HEMCO's native 72 vertical levels. Vertical remapping to ORBIT
        is a separate step (caller responsibility).
    """
    year_dir = os.path.join(hemco_dir, str(year))
    date_stamp = f"{year:04d}{month:02d}01_0000z"
    sc_path = os.path.join(year_dir, f"GEOSChem.SpeciesConc.{date_stamp}.nc4")
    jv_path = os.path.join(year_dir, f"GEOSChem.JValues.{date_stamp}.nc4")

    if not os.path.exists(sc_path):
        raise FileNotFoundError(f"HEMCO SpeciesConc missing: {sc_path}")

    ds_sc = xr.open_dataset(sc_path)
    src_lats = ds_sc.lat.values
    src_lons = ds_sc.lon.values
    lev_centers = ds_sc.lev.values  # (nz_hemco,) sigma-like

    # ORBIT grid is [0, 360) in longitude; GC benchmark is [-180, 180).
    # Map target lons to [-180, 180) for index lookup.
    tgt_lons_wrap = np.where(np.asarray(target_lons) >= 180,
                             np.asarray(target_lons) - 360,
                             np.asarray(target_lons))

    clim = HemcoClimatology(year=year, month=month,
                            lats=np.asarray(target_lats),
                            lons=np.asarray(target_lons),
                            lev_centers=lev_centers)
    if rho_air is None and rho_profile is not None:
        rho_air = density_on_levels(lev_centers, rho_profile[0], rho_profile[1])

    # Species fields
    for sp in SPECIES_KEYS:
        var = f"SpeciesConc_{sp}"
        if var not in ds_sc:
            continue
        data = ds_sc[var].isel(time=0).values  # (nz, ny_src, nx_src) mol/mol
        regrid = _blocky_upsample(data, src_lats, src_lons,
                                  clim.lats, tgt_lons_wrap)
        clim.species_molmol[sp] = regrid
        clim.species_ugm3[sp] = _molmol_to_ugm3(regrid, sp, rho_air)

    ds_sc.close()

    # J-values
    if with_jvalues and os.path.exists(jv_path):
        ds_jv = xr.open_dataset(jv_path)
        for jv in JVAL_KEYS:
            if jv not in ds_jv:
                continue
            data = ds_jv[jv].isel(time=0).values  # (nz, ny_src, nx_src) s^-1
            clim.jvalues[jv] = _blocky_upsample(data, src_lats, src_lons,
                                                clim.lats, tgt_lons_wrap)
        ds_jv.close()

    return clim


def surface_slice(clim: HemcoClimatology, species: str, units: str = "ugm3"):
    """Return HEMCO surface-level field for a species on ORBIT horizontal grid.

    HEMCO lev[0] is the surface. Useful for quick validation / diagnostics.
    """
    d = clim.species_ugm3 if units == "ugm3" else clim.species_molmol
    arr = d.get(species)
    if arr is None:
        return None
    return arr[0]  # (ny, nx)


def layer_slice_by_sigma(clim: HemcoClimatology, species: str,
                          target_sigma: float, units: str = "ugm3"):
    """Return the HEMCO layer closest to ``target_sigma`` for ``species``.

    Used to build upper-boundary targets for reduced-layer models: given
    ORBIT's top-layer sigma (P_top/P_surf ≈ 0.17 for the standard 15-layer
    SAS config), pick the HEMCO layer with the closest native sigma value
    and return its 2D (ny, nx) field.

    Nearest-layer lookup is used rather than linear interpolation in
    log(P); the gradient of O3 vs sigma in the upper troposphere is
    modest so the sub-layer bias is a few percent.  Revisit if top-BC
    validation shows a clear bias.

    Returns None if the species isn't loaded or HEMCO's vertical coord
    is unavailable.
    """
    d = clim.species_ugm3 if units == "ugm3" else clim.species_molmol
    arr = d.get(species)
    if arr is None:
        return None
    lev_centers = getattr(clim, "lev_centers", None)
    if lev_centers is None or len(lev_centers) == 0:
        return None
    lev = np.asarray(lev_centers).ravel()
    idx = int(np.argmin(np.abs(lev - target_sigma)))
    return arr[idx]  # (ny, nx)
