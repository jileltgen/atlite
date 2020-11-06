# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 10:03:13 2020

@author: arneb
"""

import os
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
import geopandas as gpd
import pandas as pd
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

import cartopy.crs as ccrs
from cartopy.crs import PlateCarree as plate
import cartopy.io.shapereader as shpreader

import xarray as xr
import atlite

import logging
import warnings

# warnings.simplefilter('always', DeprecationWarning)
# logging.captureWarnings(True)
logging.basicConfig(level=logging.INFO)

shpfilename = shpreader.natural_earth(resolution='10m',
                                      category='cultural',
                                      name='admin_0_countries')
reader = shpreader.Reader(shpfilename)
Ger = gpd.GeoSeries({r.attributes['NAME_EN']: r.geometry
                      for r in reader.records()},
                     crs={'init': 'epsg:4326'}
                     ).reindex(['Germany'])

cutout = atlite.Cutout(path="Plot-ERA5_germany_2018-1.nc",
                        module="era5",            
                        bounds=Ger.unary_union.bounds,
                        time="2018-01",
                        chunks={'time': 100}
                        )

# This is where all the work happens (this can take some time, for us it took ~15 minutes).
cutout.prepare()

projection = ccrs.Orthographic(-10, 35)
cells = gpd.GeoDataFrame({'geometry': cutout.grid_cells,
                          'capfactors': None,
                          'x': cutout.grid_coordinates()[:, 0],
                          'y': cutout.grid_coordinates()[:, 1]})

df = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
country_bound = gpd.GeoSeries(cells.unary_union)

projection = ccrs.Orthographic(-10, 35)
fig, ax = plt.subplots(subplot_kw={'projection': projection}, figsize=(6, 6))
df.plot(ax=ax, transform=plate())
country_bound.plot(ax=ax, edgecolor='orange',
                   facecolor='None', transform=plate())
fig.tight_layout()

fig = plt.figure(figsize=(12, 7))
gs = GridSpec(3, 3, figure=fig)

ax = fig.add_subplot(gs[:, 0:2], projection=projection)
plot_grid_dict = dict(alpha=0.1, edgecolor='k', zorder=4,
                      facecolor='None', transform=plate())
Ger.plot(ax=ax, zorder=1, transform=plate())
cells.plot(ax=ax, **plot_grid_dict)
country_bound.plot(ax=ax, edgecolor='orange',
                   facecolor='None', transform=plate())
ax.outline_patch.set_edgecolor('white')

ax1 = fig.add_subplot(gs[0, 2])
cutout.data.wnd100m.mean(['x', 'y']).plot(ax=ax1)
ax1.set_frame_on(False)
ax1.xaxis.set_visible(False)

ax2 = fig.add_subplot(gs[1, 2], sharex=ax1)
cutout.data.influx_direct.mean(['x', 'y']).plot()
ax2.set_frame_on(False)
ax2.xaxis.set_visible(False)

ax3 = fig.add_subplot(gs[2, 2], sharex=ax1)
cutout.data.runoff.mean(['x', 'y']).plot(ax=ax3)
ax3.set_frame_on(False)
ax3.set_xlabel(None)
fig.tight_layout()


sites = gpd.GeoDataFrame([['berlin', 13.2846501, 52.5069704, 100],
                          ['büsum', 8.8411912, 54.1371157, 100],
                          ['münchen', 11.4717963, 48.155004, 100]],
                         columns=['name', 'x', 'y', 'capacity']
                         ).set_index('name')

nearest = cutout.data.sel(
    {'x': sites.x.values, 'y': sites.y.values}, 'nearest').coords
sites['x'] = nearest.get('x').values
sites['y'] = nearest.get('y').values
cells_generation = sites.merge(
    cells, how='inner').rename(pd.Series(sites.index))

cap_factors = cutout.wind(turbine='Vestas_V112_3MW', capacity_factor=True)

fig, ax = plt.subplots(subplot_kw={'projection': projection}, figsize=(9, 7))
cap_factors.name = 'Capacity Factor Wind'
cap_factors.plot(ax=ax, transform=plate(), alpha=0.8)
cells.plot(ax=ax, **plot_grid_dict)
ax.outline_patch.set_edgecolor('white')
fig.tight_layout()

layout = xr.DataArray(cells_generation.set_index(['y', 'x']).capacity.unstack())\
                    .reindex_like(cap_factors).rename('Installed Capacity [MW]')

fig, ax = plt.subplots(subplot_kw={'projection': projection}, figsize=(9, 7))

Ger.plot(ax=ax, zorder=1, transform=plate(), alpha=0.3)
cells.plot(ax=ax, **plot_grid_dict)
layout.plot(ax=ax, transform=plate(), cmap='Reds', vmin=0,
            label='Installed Capacity [MW]')
ax.outline_patch.set_edgecolor('white')
fig.tight_layout()

fig, axes = plt.subplots(len(sites), sharex=True, figsize=(9, 4))
power_generation = cutout.wind('Vestas_V112_3MW', layout=layout,
                               shapes=cells_generation.geometry)

power_generation.to_pandas().plot(subplots=True, ax=axes)
axes[2].set_xlabel('date')
axes[1].set_ylabel('Generation [MW]')
fig.tight_layout()

cap_factors = cutout.pv(
    panel="CSi", orientation={'slope': 30., 'azimuth': 0.},
    capacity_factor=True)

fig, ax = plt.subplots(subplot_kw={'projection': projection}, figsize=(9, 7))
cap_factors.name = 'Capacity Factor Solar'
cap_factors.plot(ax=ax, transform=plate(), alpha=0.8)
cells.plot(ax=ax, **plot_grid_dict)
ax.outline_patch.set_edgecolor('white')
fig.tight_layout()

layout = xr.DataArray(cells_generation.set_index(['y', 'x']).capacity.unstack())\
                    .reindex_like(cap_factors).rename('Installed Capacity [MW]')

fig, ax = plt.subplots(subplot_kw={'projection': projection}, figsize=(9, 7))

Ger.plot(ax=ax, zorder=1, transform=plate(), alpha=0.3)
cells.plot(ax=ax, **plot_grid_dict)
layout.plot(ax=ax, transform=plate(), cmap='Reds', vmin=0,
            label='Installed Capacity [MW]')
ax.outline_patch.set_edgecolor('white')
fig.tight_layout()

fig, axes = plt.subplots(len(sites), sharex=True, figsize=(9, 4))
power_generation = cutout.pv(
    panel="CSi", orientation={'slope': 30., 'azimuth': 0.}, layout=layout,
    shapes=cells_generation.geometry)

power_generation.to_pandas().plot(subplots=True, ax=axes)
axes[2].set_xlabel('date')
axes[1].set_ylabel('Generation [MW]')
fig.tight_layout()

