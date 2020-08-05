import os
import pandas as pd
import numpy as np
import xarray as xr

from dask import delayed

from tempfile import mkstemp

import cdsapi




import logging
logger = logging.getLogger(__name__)

# Model and Projection Settings
projection = 'latlong'

features = {
    'height': ['height'],
    'wind': [
        'wnd10m'],
    'influx': [
        'influx_direct',
        'influx_diffuse'],
    'temperature': [
        'temperature',
        'soil temperature'],
    'runoff': ['runoff']}

static_features = {'height'}




def _area(coords):
    # North, West, South, East. Default: global
    x0, x1 = coords['x'].min().item(), coords['x'].max().item()
    y0, y1 = coords['y'].min().item(), coords['y'].max().item()
    return [y1, x0, y0, x1]


def retrieval_times(coords):
    """
    Get list of retrieval cdsapi arguments for time dimension in coordinates.

    According to the time span in the coords argument, the entries in the list
    specify either

    * days, if number of days in coords is less or equal 10
    * months, if number of days is less or equal 90
    * years else

    Parameters
    ----------
    coords : atlite.Cutout.coords

    Returns
    -------
    list of dicts witht retrieval arguments

    """
    time = pd.Series(coords['time'])
    time_span = time.iloc[-1] - time.iloc[0]
    if len(time) == 1:
        return [{'year': str(d.year), 'month': str(d.month), 'day': str(d.day),
                 'time': d.strftime("%H:00")} for d in time]
    if time_span.days <= 10:
        return [{'year': str(d.year), 'month': str(d.month), 'day': str(d.day)}
                for d in time.dt.date.unique()]
    elif time_span.days < 90:
        return [{'year': str(year), 'month': str(month)}
                for month in time.dt.month.unique()
                for year in time.dt.year.unique()]
    else:
        return [{'year': str(year)} for year in time.dt.year.unique()]





def retrieve_data(product, chunks=None, tmpdir=None, lock=None, **updates):
    """Download data like ERA5 from the Climate Data Store (CDS)."""
    # Default request
    request = {
        'product_type': 'reanalysis',
        'format': 'netcdf',
        'day': list(range(1, 31 + 1)),
        'time': [
            '00:00', '01:00', '02:00', '03:00', '04:00', '05:00',
            '06:00', '07:00', '08:00', '09:00', '10:00', '11:00',
            '12:00', '13:00', '14:00', '15:00', '16:00', '17:00',
            '18:00', '19:00', '20:00', '21:00', '22:00', '23:00'
        ],
        'month': list(range(1, 12 + 1)),
        # 'area': [50, -1, 49, 1], # North, West, South, East. Default: global
        # 'grid': [0.25, 0.25], # Latitude/longitude grid: east-west (longitude)
        # and north-south resolution (latitude). Default: 0.25 x 0.25
    }
    request.update(updates)

    assert {'year', 'month', 'variable'}.issubset(
        request), "Need to specify at least 'variable', 'year' and 'month'"

    result = cdsapi.Client().retrieve(product, request)

    fd, target = mkstemp(suffix='.nc', dir=tmpdir)
    os.close(fd)

    try:
        if lock is not None:
            lock.acquire()
        result.download(target)
    finally:
        if lock is not None:
            lock.release()

    ds = xr.open_dataset(target, chunks=chunks or {})


    return ds


def get_data(cutout, feature, tmpdir, lock, **creation_parameters):

    ds = xr.open_mfdataset('/Users/jileltgen/Downloads/testrea6/*.nc',
                  data_vars='minimal', coords='minimal', compat='override')
    ds = ds.drop_vars({'rotated_pole', 'depth', 'depth_bnds', 'height'})
    ds = ds.rename({'rlon': 'x', 'rlat': 'y', 'height': 'oldheight', 'var90': 'runoff', 'var22': 'influx_direct', 'var23': 'influx_diffuse',
                'var85': 'soil temperature', 'var156': 'height', 'var11': 'temperature'})
    ds['wnd10m'] = np.sqrt(ds['var33']**2 + ds['var34']**2)
    ds = ds.drop_vars(['var33', 'var34'])
    ds['soil temperature'] = 273.15 - ds['soil temperature']
    ds['temperature'] = 273.15 - ds['temperature']
   

    coords = cutout.coords

    sanitize = creation_parameters.get('sanitize', True)

    retrieval_params = {'product': 'reanalysis-era5-single-levels',
                        'area': _area(coords),
                        'chunks': cutout.chunks,
                        'grid': [cutout.dx, cutout.dy],
                        'tmpdir': tmpdir,
                        'lock': lock}

    func = globals().get(f"get_data_{feature}")
    sanitize_func = globals().get(f"sanitize_{feature}")

    logger.info(f"Downloading data for feature '{feature}' to {tmpdir}.")

    def retrieve_once(time):
        ds = delayed(func)({**retrieval_params, **time})
        if sanitize and sanitize_func is not None:
            ds = delayed(sanitize_func)(ds)
        return ds

    if feature in static_features:
        return retrieve_once(retrieval_times(coords)[0])

    datasets = map(retrieve_once, retrieval_times(coords))

    return delayed(xr.concat)(datasets, dim='time')
