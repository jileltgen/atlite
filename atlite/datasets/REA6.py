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
import dask
from dask import delayed
from dask.utils import SerializableLock

import logging
logger = logging.getLogger(__name__)

# Model, Projection and Resolution Settings
projection = 'latlong'
dx = 0.12939683446344338
dy = 0.06118376358695652
dt = '1H'

features = {
    #'height': ['height'],
    'wind': [
        'wnd10m'], # ,'roughness'
    'influx': [
        'influx_toa',
        'influx_direct',
        'influx_diffuse',
        'albedo'],
    'temperature': [
        'temperature',
        'soil temperature']}
    # ,'runoff': ['runoff']}

static_features = {'height'}

def _rename_and_clean_coords(ds, add_lon_lat=True):
    """Rename 'longitude' and 'latitude' columns to 'x' and 'y'.

    Optionally (add_lon_lat, default:True) preserves latitude and longitude
    columns as 'lat' and 'lon'.
    """
    ds = ds.rename({'lon': 'x', 'lat': 'y'})
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
                           sobs_rad=_filenames_starting_with("SOBS_RAD"),
                           t2m=_filenames_starting_with("T_2M"),
                           tsoil=_filenames_starting_with("T_SOIL"),
                           u_10m=_filenames_starting_with("U_10M"),
                           v_10m=_filenames_starting_with("V_10M")),
                      join="inner", axis=1)
    files["height"] = os.path.join(
        rea_dir, "constant", "COSMO_REA6_CONST_withOUTsponge.nc")
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

def get_data_wind(open_kwargs):
    """Get wind data for given retrieval parameters."""
    files = open_kwargs.pop("files")
    ds_u_10m = xr.open_mfdataset(files.u_10m, combine='by_coords', **open_kwargs)
    ds_v_10m = xr.open_mfdataset(files.v_10m, combine='by_coords', **open_kwargs)
    ds = xr.merge([ds_u_10m, ds_v_10m])

    ds = _rename_and_clean_coords(ds, add_lon_lat=False)

    ds['wnd10m'] = (np.sqrt(ds_u_10m['10u']**2 + ds_v_10m['10v']**2)
                     .assign_attrs(units=ds['10u'].attrs['units'],
                                   long_name="10 metre wind speed"))
    ds = ds.drop_vars(['10u', '10v'])

    return ds

def get_data_roughness(open_kwargs):
    return

def get_data_influx(open_kwargs):
    """Get influx data for given retrieval parameters."""
    files = open_kwargs.pop("files")
    ds_dif_rad = xr.open_mfdataset(files.dif_rad, combine='by_coords', **open_kwargs)
    ds_dir_rad = xr.open_mfdataset(files.dir_rad, combine='by_coords', **open_kwargs)
    ds_sobs_rad = xr.open_mfdataset(files.sobs_rad, combine='by_coords', **open_kwargs)
    ds = xr.merge([ds_dif_rad, ds_dir_rad, ds_sobs_rad])
    ds = _rename_and_clean_coords(ds)
    # Not sure if the variables are the right ones
    ds = ds.rename({'SWDIRS_RAD': 'influx_direct',
                    'SWDIFDS_RAD': 'influx_diffuse', 'var111': 'SOBS_RAD'})
    ds['influx_toa'] =  ds['influx_direct'] + ds['influx_diffuse']
    ds['albedo'] = ((ds['SOBS_RAD']/ds['influx_toa'])
                    .assign_attrs(units='(0 - 1)', long_name='Albedo'))

    ds = ds.drop_vars(['SOBS_RAD'])
    return ds

def get_data_temperature(open_kwargs):
    """Get wind temperature for given retrieval parameters."""
    files = open_kwargs.pop("files")
    ds_t2m = xr.open_mfdataset(files.t2m, combine='by_coords', **open_kwargs)
    ds_tsoil = xr.open_mfdataset(files.tsoil, combine='by_coords', **open_kwargs)
    ds = xr.merge([ds_t2m, ds_tsoil])
    ds = _rename_and_clean_coords(ds)
    ds = ds.rename({'2t': 'temperature', 'var85': 'soil temperature'})
    return ds

# def get_data_runoff(open_kwargs):
#     """Get runoff data for given retrieval parameters."""
#     ds = retrieve_data(variable=['runoff'], **open_kwargs)

#     ds = _rename_and_clean_coords(ds)
#     ds = ds.rename({'ro': 'runoff'})

#     return ds

def get_data_height(open_kwargs):
    """Get height data for given retrieval parameters."""
    rea_dir = open_kwargs.pop("rea_dir")
    file = os.path.join(rea_dir, "constant", "COSMO_REA6_CONST_withOUTsponge.nc")
    ds = xr.open_dataset(file)
    #ds = _rename_and_clean_coords(ds)
    return ds

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

    sanitize = creation_parameters.get('sanitize', True)

    rea_dir = creation_parameters['rea_dir']
    creation_parameters.setdefault('parallel', False)

    files = get_filenames(rea_dir, coords)
    open_kwargs = dict(chunks=chunks, parallel=creation_parameters['parallel'],
                       files=files)

    func = globals().get(f"get_data_{feature}")
    sanitize_func = globals().get(f"sanitize_{feature}")
    print(f"get_data_{feature}", func)
    # ds = func(open_kwargs)
    # if sanitize and sanitize_func is not None:
    #     ds = sanitize_func(ds)
    # return ds
    # def retrieve_once():
    #     ds = delayed(func)(open_kwargs)
    #     if sanitize and sanitize_func is not None:
    #         ds = delayed(sanitize_func)(ds)
    #     return ds
    # if feature in static_features:
    #     return retrieve_once()
    datasets = map(func, open_kwargs)
    return delayed(xr.concat)(datasets, dim='time')
