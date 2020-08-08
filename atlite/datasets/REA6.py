"""
Module containing specific operations for creating cutouts from the REA6 dataset.
"""

from ..gis import regrid, maybe_swap_spatial_dims
from rasterio.warp import Resampling
import os
import glob
import pandas as pd
import numpy as np
import xarray as xr
from functools import partial
from datetime import datetime

import logging
logger = logging.getLogger(__name__)

# Model, Projection and Resolution Settings
projection = 'latlong'
dx = 0.12939683446344338
dy = 0.06118376358695652
dt = '1H'

features = {
    'height': ['height'],
    'wind': [
        'wnd100m',
        'roughness'],
    'influx': [
        'influx_toa',
        'influx_direct',
        'influx_diffuse',
        'albedo'],
    'temperature': [
        'temperature',
        'soil temperature'],
    'runoff': ['runoff']}

static_features = {'height'}

def _rename_and_clean_coords(ds, add_lon_lat=True):
    """Rename 'longitude' and 'latitude' columns to 'x' and 'y'.

    Optionally (add_lon_lat, default:True) preserves latitude and longitude
    columns as 'lat' and 'lon'.
    """
    ds = ds.rename({'longitude': 'x', 'latitude': 'y'})
    ds = maybe_swap_spatial_dims(ds)
    if add_lon_lat:
        ds = ds.assign_coords(lon=ds.coords['x'], lat=ds.coords['y'])
    return ds

def get_filenames(rea_dir, coords):
    """
    Get all files in directory `rea_dir` relevent for coordinates `coords`.

    This function parses all files in the rea directory which lay in the time
    span of the coordinates.

    Parameters
    ----------
    rea_dir : str
    coords : atlite.Cutout.coords

    Returns
    -------
    pd.DataFrame with two columns `sis` and `sid` for and timeindex for all
    relevant files.
    """
    def _filenames_starting_with(name):
        pattern = os.path.join(rea_dir, "**", f"{name}*.nc")
        files = pd.Series(glob.glob(pattern, recursive=True))
        assert not files.empty, (f"No files found at {pattern}. Make sure "
                                 "rea_dir points to the correct directory!")

        files.index = pd.to_datetime(files.str.extract(r".*2D.(\d{6})",
                                                       expand=False)+"01")
        return files.sort_index()
    files = pd.concat(dict(runoff=_filenames_starting_with("RUNOFF_S"),
                           dif_rad=_filenames_starting_with("SWDIFDS_RAD"),
                           dir_rad=_filenames_starting_with("SWDIRS_RAD"),
                           t2m=_filenames_starting_with("T_2M"),
                           u_10m=_filenames_starting_with("U_10M"),
                           v_10m=_filenames_starting_with("V_10M")),
                      join="inner", axis=1)

    start = coords['time'].to_index()[0]
    end = datetime(
        coords['time'].to_index()[-1].year, 
        coords['time'].to_index()[-1].month,
        1)

    if (start < files.index[0]) or (end.date() > files.index[-1]):
        logger.error(f"Files in {rea_dir} do not cover the whole time span:"
                     f"\n\t{start} until {end}")

    return files.loc[(files.index >= start) & (files.index <= end)].sort_index()

def as_slice(zs, pad=True):
    """Convert index to slice. This speeds up the indexing."""
    if not isinstance(zs, slice):
        first, second, last = np.asarray(zs)[[0, 1, -1]]
        dz = 0.1 * (second - first) if pad else 0.
        zs = slice(first - dz, last + dz)
    return zs

def get_data(cutout, feature, tmpdir, lock=None, **creation_parameters):
    """
    Load stored REA6 data and reformat to matching the given cutout.

    This function loads and resamples the stored REA6 data for a given
    `atlite.Cutout`.

    Parameters
    ----------
    cutout : atlite.Cutout
    feature : str
        Name of the feature data to retrieve. Must be in
        `atlite.datasets.REA6.features`
    **creation_parameters :
        Mandatory arguments are:
            * 'rea_dir', str. Directory of the stored SARAH data.
        Possible arguments are:
            * 'parallel', bool. Whether to load stored files in parallel
            mode. Default is False.
            
    Returns
    -------
    xarray.Dataset
        Dataset of dask arrays of the retrieved variables.

    """
    assert cutout.dt in ('h', 'H', '1h', '1H')

    coords = cutout.coords
    chunks = cutout.chunks

    rea_dir = creation_parameters['rea_dir']
    creation_parameters.setdefault('parallel', False)

    files = get_filenames(rea_dir, coords)
    open_kwargs = dict(chunks=chunks, parallel=creation_parameters['parallel'])
    #ds_runoff = xr.open_mfdataset(files.runoff, combine='by_coords', **open_kwargs)
    ds_dif_rad = xr.open_mfdataset(files.dif_rad, combine='by_coords', **open_kwargs)
    ds_dir_rad = xr.open_mfdataset(files.dir_rad, combine='by_coords', **open_kwargs)
    ds_t2m = xr.open_mfdataset(files.t2m, combine='by_coords', **open_kwargs)
    ds_u_10m = xr.open_mfdataset(files.u_10m, combine='by_coords', **open_kwargs)
    ds_v_10m = xr.open_mfdataset(files.v_10m, combine='by_coords', **open_kwargs)
    
    ds = xr.merge([ds_dif_rad, ds_dir_rad, ds_t2m, ds_u_10m, ds_v_10m]) # ds_runoff, 
    ds = ds.sel(lon=as_slice(coords['lon']), lat=as_slice(coords['lat']))

    ds = ds.fillna(0)

    if (cutout.dx != dx) or (cutout.dy != dy):
        ds = regrid(ds, coords['lon'], coords['lat'], resampling=Resampling.average)

    #dif_attrs = dict(long_name='Surface Diffuse Shortwave Flux', units='W m-2')
    #ds['influx_diffuse'] = (ds['SIS'] - ds['SID']) .assign_attrs(**dif_attrs)
    #ds = ds.rename({'SID': 'influx_direct'}).drop_vars('SIS')
    ds = ds.assign_coords(x=ds.coords['lon'], y=ds.coords['lat'])
    ds = _rename_and_clean_coords(ds)
    return ds.swap_dims({'lon': 'x', 'lat':'y'})