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

last_mouse_coord = [0, 0, 0, 0, 0, 0, 0, 0, 0]

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

def pointer_plotter(plan_x=None, plan_y=None, lat_x=None, lat_y=None,
                    lon_x=None, lon_y=None, time_x=None, time_y=None, target=None):
    global last_mouse_coord
    last_mouse_coord[-1] += 1
    # Determine which axis the mouse is in:
    if plan_x != last_mouse_coord[0] or plan_y != last_mouse_coord[1]:
        # mouse is in planview axis
        if target == 'plan':
            crosshair =  hv.VLine(0).opts(alpha=0) * hv.HLine(0).opts(alpha=0)
        elif target == 'lon':
            crosshair =  hv.VLine(plan_x).opts(color='black') * hv.HLine(0).opts(alpha=0)
        elif target == 'lat':
            crosshair =  hv.HLine(plan_y).opts(color='black') * hv.VLine(0).opts(alpha=0)
        elif target == 'time':
            crosshair =  hv.VLine(0).opts(alpha=0) * hv.HLine(0).opts(alpha=0)
    elif lon_x != last_mouse_coord[2] or lon_y != last_mouse_coord[3]:
        # mouse is in lonalt axis
        if target == 'plan':
            crosshair = hv.VLine(lon_x).opts(color='black') * hv.HLine(0).opts(alpha=0)
        elif target == 'lon':
            crosshair = hv.VLine(0).opts(alpha=0) * hv.HLine(0).opts(alpha=0)
        elif target == 'lat':
            crosshair = hv.HLine(0).opts(alpha=0) * hv.VLine(lon_y).opts(color='black')
        elif target == 'time':
            crosshair =  hv.VLine(0).opts(alpha=0) * hv.HLine(lon_y).opts(color='black')
    elif lat_x != last_mouse_coord[4] or lat_y != last_mouse_coord[5]:
        # mouse is in latalt axis
        if target == 'plan':
            crosshair = hv.VLine(0).opts(alpha=0) * hv.HLine(lat_y).opts(color='black')
        elif target == 'lon':
            crosshair = hv.VLine(0).opts(alpha=0) * hv.HLine(lat_x).opts(color='black')
        elif target == 'lat':
            crosshair = hv.HLine(0).opts(alpha=0) * hv.VLine(0).opts(alpha=0)
        elif target == 'time':
            crosshair = hv.VLine(0).opts(alpha=0) * hv.HLine(lat_x).opts(color='black')
    elif time_x != last_mouse_coord[6] or time_y != last_mouse_coord[7]:
        # mouse is in timeseries axis
        if target == 'plan':
            crosshair = hv.VLine(0).opts(alpha=0) * hv.HLine(0).opts(alpha=0)
        elif target == 'lon':
            crosshair = hv.VLine(0).opts(alpha=0) * hv.HLine(time_y).opts(color='black')
        elif target == 'lat':
            crosshair = hv.HLine(0).opts(alpha=0) * hv.VLine(time_y).opts(color='black')
        elif target == 'time':
            crosshair = hv.VLine(0).opts(alpha=0) * hv.HLine(0).opts(alpha=0)
    else:
        crosshair =  hv.VLine(0).opts(alpha=0) * hv.HLine(0).opts(alpha=0)
    
    if last_mouse_coord[-1] == ((len(last_mouse_coord)-1)/2):
        last_mouse_coord = [plan_x, plan_y, lon_x, lon_y, lat_x, lat_y, time_x, time_y, 0]
    return crosshair


xlim = (ds.network_center_longitude.data - 3, ds.network_center_longitude.data + 3)
ylim = (ds.network_center_latitude.data - 3, ds.network_center_latitude.data + 3)
zlim = (0, 20000)

timefloats = color_by_time(ds.event_time.values)[-1]

plan_ax = hv.Points((ds.event_longitude.data, ds.event_latitude.data, timefloats), kdims=['lon', 'lat'], vdims=['time'])
plan_ax = hv.operation.datashader.datashade(plan_ax, aggregator=datashader.mean('time'), cmap='rainbow').opts(xlim=xlim, ylim=ylim, width=px_scale*plan_edge_length, height=px_scale*plan_edge_length, tools=['hover', 'lasso_select'])
# counties_shp = shapefile.Reader('ne_10m_admin_2_counties.shp').shapes()
# counties_shp = [shape(counties_shp[i]) for i in range(len(counties_shp))]
# plan_ax = gv.Path(counties_shp).opts(color='gray') * gf.borders().opts(color='black') * gf.states().opts(color='black', line_width=2) * plan_ax
plan_ax_pointer = hv.streams.PointerXY(x=0, y=0, source=plan_ax, transient=True).rename(x='plan_x', y='plan_y')

lon_alt_ax = hv.Points((ds.event_longitude.data, ds.event_altitude.data, timefloats), kdims=['lon', 'alt'], vdims=['time'])
lon_alt_ax = hv.operation.datashader.datashade(lon_alt_ax, aggregator=datashader.mean('time'), cmap='rainbow').opts(xlim=xlim, ylim=zlim, width=px_scale*plan_edge_length, height=px_scale*hist_edge_length, hooks=[hook_yalt_limiter])
lon_alt_ax_pointer = hv.streams.PointerXY(x=0, y=0, source=lon_alt_ax, transient=True).rename(x='lon_x', y='lon_y')

hist_ax = hv.Histogram(np.histogram(ds.event_altitude.data, bins=np.arange(0, 20001, 1000)), kdims=['alt'], vdims=['src']).opts(width=px_scale*hist_edge_length, height=px_scale*hist_edge_length, invert_axes=True).opts(hooks=[hook_xlabel_rotate, hook_hist_src_limiter, hook_yalt_limiter])

lat_alt_ax = hv.Points((ds.event_altitude.data, ds.event_latitude.data, timefloats), kdims=['alt', 'lat'], vdims=['time'])
lat_alt_ax = hv.operation.datashader.datashade(lat_alt_ax, aggregator=datashader.mean('time'), cmap='rainbow').opts(xlim=zlim, ylim=ylim, width=px_scale*hist_edge_length, height=px_scale*plan_edge_length, hooks=[hook_xlabel_rotate, hook_xalt_limiter])
lat_alt_ax_pointer = hv.streams.PointerXY(x=0, y=0, source=lat_alt_ax, transient=True).rename(x='lat_x', y='lat_y')

alt_time = hv.Points((ds.event_time.data, ds.event_altitude.data, timefloats), kdims=['time', 'alt'], vdims=['time'])
alt_time = hv.operation.datashader.datashade(alt_time, aggregator=datashader.mean('time'), cmap='rainbow').opts(xlim=(ds.event_time.data[0], ds.event_time.data[-1]), ylim=zlim, width=px_scale*(plan_edge_length+hist_edge_length), height=px_scale*hist_edge_length, hooks=[hook_yalt_limiter, hook_time_limiter], toolbar=None)
alt_time_pointer = hv.streams.PointerXY(x=0, y=0, source=alt_time, transient=True).rename(x='time_x', y='time_y')


plan_ax_crosshair = hv.DynamicMap(lambda plan_x, plan_y, lat_x, lat_y,
                            lon_x, lon_y, time_x, time_y: 
                            pointer_plotter(plan_x, plan_y, lat_x, lat_y,
                            lon_x, lon_y, time_x, time_y, 'plan'), streams=[plan_ax_pointer, lat_alt_ax_pointer, lon_alt_ax_pointer, alt_time_pointer])

lon_ax_crosshair = hv.DynamicMap(lambda plan_x, plan_y, lat_x, lat_y,
                            lon_x, lon_y, time_x, time_y: 
                            pointer_plotter(plan_x, plan_y, lat_x, lat_y,
                            lon_x, lon_y, time_x, time_y, 'lon'), streams=[plan_ax_pointer, lat_alt_ax_pointer, lon_alt_ax_pointer, alt_time_pointer])


lat_ax_crosshair = hv.DynamicMap(lambda plan_x, plan_y, lat_x, lat_y,
                            lon_x, lon_y, time_x, time_y: 
                            pointer_plotter(plan_x, plan_y, lat_x, lat_y,
                            lon_x, lon_y, time_x, time_y, 'lat'), streams=[plan_ax_pointer, lat_alt_ax_pointer, lon_alt_ax_pointer, alt_time_pointer])

time_ax_crosshair = hv.DynamicMap(lambda plan_x, plan_y, lat_x, lat_y,
                            lon_x, lon_y, time_x, time_y: 
                            pointer_plotter(plan_x, plan_y, lat_x, lat_y,
                            lon_x, lon_y, time_x, time_y, 'time'), streams=[plan_ax_pointer, lat_alt_ax_pointer, lon_alt_ax_pointer, alt_time_pointer])

lon_alt_ax = lon_alt_ax * lon_ax_crosshair
plan_ax = plan_ax * plan_ax_crosshair
lat_alt_ax = lat_alt_ax * lat_ax_crosshair
alt_time = alt_time * time_ax_crosshair

the_lower_part = (lon_alt_ax + hist_ax + plan_ax + lat_alt_ax).cols(2)

# the_lower_part = hv.selection.link_selections(the_lower_part)

title = pn.pane.Markdown('## LMA Data Explorer')

pn.Column(title, alt_time, the_lower_part).servable()
