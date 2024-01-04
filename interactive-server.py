from pyxlma.lmalib.io import read as lma_read
from pyxlma.plot.xlma_plot_feature import color_by_time
from datetime import timedelta

import numpy as np
from bokeh.models import Range1d
import holoviews as hv
import panel as pn
import datashader

import geoviews as gv
import geoviews.feature as gf
from cartopy import crs as ccrs
import shapefile
from shapely.geometry import shape

px_scale = 7
plan_edge_length = 60
hist_edge_length = 20

alt_limit = Range1d(0, 20000, bounds=(0, 100000))


filename = 'LYLOUT_230615_200000_0600.dat.gz'
ds, start_time = lma_read.dataset(filename)
end_time = start_time + timedelta(seconds=int(filename.split('_')[-1].replace('.dat.gz', '')))
time_range = np.array([start_time, end_time]).astype('datetime64').astype(float)/1e3
lon_range = Range1d(ds.network_center_longitude.data - 1, ds.network_center_longitude.data + 1)
lat_range = Range1d(ds.network_center_latitude.data - 1, ds.network_center_latitude.data + 1)

def hook_hist_src_limiter(plot, element):
    init_xmax = plot.state.x_range.end
    plot.state.x_range = Range1d(0, init_xmax, bounds=(0, init_xmax))

def hook_yalt_limiter(plot, element):
    if plot.state.y_range.start < 0:
        plot.state.y_range.start = 0
    if plot.state.y_range.end > 100000:
        plot.state.y_range.end = 100000

def hook_xalt_limiter(plot, element):
    if plot.state.x_range.start < 0:
        plot.state.x_range.start = 0
    if plot.state.x_range.end > 100000:
        plot.state.x_range.end = 100000

def hook_time_limiter(plot, element):
    if type(plot.state.x_range.start) == float:
        if plot.state.x_range.start < time_range[0]:
            plot.state.x_range.start = time_range[0]
        if plot.state.x_range.end > time_range[1]:
            plot.state.x_range.end = time_range[1]

def hook_xlabel_rotate(plot, element):
    plot.state.xaxis.major_label_orientation = -np.pi/2

xlim = (ds.network_center_longitude.data - 3, ds.network_center_longitude.data + 3)
ylim = (ds.network_center_latitude.data - 3, ds.network_center_latitude.data + 3)
zlim = (0, 20000)

timefloats = color_by_time(ds.event_time.values)[-1]

plan_ax = hv.Points((ds.event_longitude.data, ds.event_latitude.data, timefloats), kdims=['lon', 'lat'], vdims=['time'])
plan_ax = hv.operation.datashader.datashade(plan_ax, aggregator=datashader.mean('time'), cmap='rainbow').opts(xlim=xlim, ylim=ylim, width=px_scale*plan_edge_length, height=px_scale*plan_edge_length, tools=['hover'])
# counties_shp = shapefile.Reader('ne_10m_admin_2_counties.shp').shapes()
# counties_shp = [shape(counties_shp[i]) for i in range(len(counties_shp))]
# plan_ax = gv.Path(counties_shp).opts(color='gray') * gf.borders().opts(color='black') * gf.states().opts(color='black', line_width=2) * plan_ax

lon_alt_ax = hv.Points((ds.event_longitude.data, ds.event_altitude.data, timefloats), kdims=['lon', 'alt'], vdims=['time'])
lon_alt_ax = hv.operation.datashader.datashade(lon_alt_ax, aggregator=datashader.mean('time'), cmap='rainbow').opts(xlim=xlim, ylim=zlim, width=px_scale*plan_edge_length, height=px_scale*hist_edge_length, hooks=[hook_yalt_limiter])

hist_ax = hv.Histogram(np.histogram(ds.event_altitude.data, bins=np.arange(0, 20001, 1000)), kdims=['alt'], vdims=['src']).opts(width=px_scale*hist_edge_length, height=px_scale*hist_edge_length, invert_axes=True).opts(hooks=[hook_xlabel_rotate, hook_hist_src_limiter, hook_yalt_limiter])

lat_alt_ax = hv.Points((ds.event_altitude.data, ds.event_latitude.data, timefloats), kdims=['alt', 'lat'], vdims=['time'])
lat_alt_ax = hv.operation.datashader.datashade(lat_alt_ax, aggregator=datashader.mean('time'), cmap='rainbow').opts(xlim=zlim, ylim=ylim, width=px_scale*hist_edge_length, height=px_scale*plan_edge_length, hooks=[hook_xlabel_rotate, hook_xalt_limiter])

alt_time = hv.Points((ds.event_time.data, ds.event_altitude.data, timefloats), kdims=['time', 'alt'], vdims=['time'])
alt_time = hv.operation.datashader.datashade(alt_time, aggregator=datashader.mean('time'), cmap='rainbow').opts(xlim=(ds.event_time.data[0], ds.event_time.data[-1]), ylim=zlim, width=px_scale*(plan_edge_length+hist_edge_length), height=px_scale*hist_edge_length, hooks=[hook_yalt_limiter, hook_time_limiter], toolbar=None)

the_lower_part = (lon_alt_ax + hist_ax + plan_ax + lat_alt_ax).cols(2)

title = pn.pane.Markdown('## LMA Data Explorer')

pn.Column(title, alt_time, the_lower_part).servable()
