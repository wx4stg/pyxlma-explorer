from pyxlma.lmalib.io import read as lma_read
from pyxlma.plot.xlma_plot_feature import color_by_time
import xarray as xr
import numpy as np
from bokeh.plotting import figure
from bokeh.models import Range1d
import pandas as pd
import hvplot.pandas
import panel as pn

import geoviews as gv
import geoviews.feature as gf
from cartopy import crs as ccrs
import shapefile
from shapely.geometry import shape

px_scale = 7
plan_edge_length = 60
hist_edge_length = 20

alt_limit = Range1d(0, 20000, bounds=(0, 100000))

ds, start_time = lma_read.dataset('LYLOUT_230615_200000_0600.dat.gz')
lon_range = Range1d(ds.network_center_longitude.data - 5, ds.network_center_longitude.data + 1)
lat_range = Range1d(ds.network_center_latitude.data - 1, ds.network_center_latitude.data + 1)

def hook_hist_src_limiter(plot, element):
    init_xmax = plot.state.x_range.end
    plot.state.x_range = Range1d(0, init_xmax, bounds=(0, init_xmax))

def hook_yalt_limiter(plot, element):
    plot.state.y_range = alt_limit

def hook_xalt_limiter(plot, element):
    plot.state.x_range = alt_limit

def hook_xlabel_rotate(plot, element):
    plot.state.xaxis.major_label_orientation = -np.pi/2

def hook_lon_linker(plot, element):
    plot.state.x_range = lon_range

def hook_lat_linker(plot, element):
    plot.state.y_range = lat_range

ds = ds.assign_coords({'number_of_events' : ds.number_of_events.values})
df = pd.DataFrame({
    'lon' : ds.event_longitude.values,
    'lat' : ds.event_latitude.values,
    'alt' : ds.event_altitude.values,
    'time' : ds.event_time.values,
    'timefloats' : color_by_time(ds.event_time.values)[-1]
})

xlim = (ds.network_center_longitude.data - 5, ds.network_center_longitude.data + 5)
ylim = (ds.network_center_latitude.data - 1, ds.network_center_latitude.data + 1)
zlim = (0, 20000)

plan_ax = df.hvplot.points(x='lon', y='lat', c='timefloats', cmap='rainbow', rasterize=True, colorbar=False, xlim=xlim, ylim=ylim, width=px_scale*plan_edge_length, height=px_scale*plan_edge_length, responsive=False, projection=ccrs.PlateCarree()).opts(hooks=[hook_lon_linker, hook_lat_linker])
counties_shp = shapefile.Reader('ne_10m_admin_2_counties.shp').shapes()
counties_shp = [shape(counties_shp[i]) for i in range(len(counties_shp))]
plan_ax = gv.Path(counties_shp).opts(color='gray') * gf.borders().opts(color='black') * gf.states().opts(color='black', line_width=2) * plan_ax
lon_alt_ax = df.hvplot.points(x='lon', y='alt', c='timefloats', cmap='rainbow', rasterize=True, colorbar=False, xlim=xlim, ylim=zlim, width=px_scale*plan_edge_length, height=px_scale*hist_edge_length, responsive=False).opts(hooks=[hook_yalt_limiter, hook_lon_linker])
lat_alt_ax = df.hvplot.points(x='alt', y='lat', c='timefloats', cmap='rainbow', rasterize=True, colorbar=False, ylim=ylim, xlim=zlim, width=px_scale*hist_edge_length, height=px_scale*plan_edge_length, responsive=False).opts(hooks=[hook_xlabel_rotate, hook_xalt_limiter, hook_lat_linker])
hist_ax = df.hvplot.hist(y='alt', bin_range=zlim, bins=20, invert=True, rasterize=True, xlabel='alt', ylabel='src', width=px_scale*hist_edge_length, height=px_scale*hist_edge_length, responsive=False).opts(hooks=[hook_xlabel_rotate, hook_hist_src_limiter, hook_yalt_limiter])
alt_time = df.hvplot.points(x='time', y='alt', c='timefloats', cmap='rainbow', rasterize=True, colorbar=False, ylim=zlim, width=px_scale*(plan_edge_length+hist_edge_length), height=px_scale*hist_edge_length, responsive=False).opts(hooks=[hook_yalt_limiter])

the_lower_part = (lon_alt_ax + hist_ax + plan_ax + lat_alt_ax).cols(2)


title = pn.pane.Markdown('## LMA Data Explorer')

pn.Column(title, alt_time, the_lower_part).servable()

