from pyxlma.lmalib.io import read as lma_read
from pyxlma.plot.xlma_plot_feature import color_by_time
from datetime import timedelta

import numpy as np
import matplotlib as mpl

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

# plan view x, y; lonalt x, y; latalt x, y; time x, y; position; counter
last_mouse_coord = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
selection_geom = [np.array([]), 0]


alt_min = 0
alt_max = 100000
alt_limit = Range1d(0, 20000, bounds=(alt_min, alt_max))


filename = 'LYLOUT_230615_200000_0600.dat.gz'
ds, start_time = lma_read.dataset(filename)
end_time = start_time + timedelta(seconds=int(filename.split('_')[-1].replace('.dat.gz', '')))
time_range_dt = np.array([start_time, end_time]).astype('datetime64')
time_range = time_range_dt.astype(float)/1e3
lon_range = Range1d(ds.network_center_longitude.data - 1, ds.network_center_longitude.data + 1)
lat_range = Range1d(ds.network_center_latitude.data - 1, ds.network_center_latitude.data + 1)

def limit_to_polygon(limit_button):
    global selection_geom
    if selection_geom[0] == np.array([]):
        return
    select_path = mpl.path.Path(selection_geom[0], closed=True)
    axis = selection_geom[-1]
    if axis == 1:
        print(select_path.contains_points(np.array([ds.event_longitude.data, ds.event_latitude.data]).T).shape)
        print(ds.event_longitude.data.shape)

        

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

def highlight_plotter(plan_vertices, lat_vertices, lon_vertices, time_vertices, target):
    global selection_geom
    # Determine which axis the mouse is in:
    if last_mouse_coord[-2] == 1:
        # mouse is in planview axis
        if plan_vertices is None:
            return hv.VLine(0).opts(alpha=0) * hv.HLine(0).opts(alpha=0)
        if len(plan_vertices['xs']) > 1:
            print(plan_vertices['xs'])
            return hv.VLine(0).opts(alpha=0) * hv.HLine(0).opts(alpha=0)
        if len(plan_vertices['xs']) == 0:
            selection_geom = [np.array([]), 0]
            return hv.VLine(0).opts(alpha=0) * hv.HLine(0).opts(alpha=0)
        this_selection_geom = np.array([plan_vertices['xs'], plan_vertices['ys']]).T[:, 0, :]
        selection_geom = [this_selection_geom, 1]
        x1 = np.min(plan_vertices['xs'])
        x2 = np.max(plan_vertices['xs'])
        y1 = np.min(plan_vertices['ys'])
        y2 = np.max(plan_vertices['ys'])
        if target == 'plan':
            return hv.VLine(0).opts(alpha=0) * hv.HLine(0).opts(alpha=0)
        elif target == 'lon':
            return hv.VLine(x1).opts(color='black', line_dash='dashed') * hv.Area(([x1, x2], [alt_max, alt_max])).opts(color='black', alpha=0.3) * hv.VLine(np.max(plan_vertices['xs'])).opts(color='black', line_dash='dashed')
        elif target == 'lat':
            return hv.HLine(y1).opts(color='black', line_dash='dashed') * hv.Area(([alt_min, alt_max], [y1, y1], [y2, y2]), vdims=['latmin', 'latmax']).opts(color='black', alpha=0.3) * hv.HLine(y2).opts(color='black', line_dash='dashed')
        elif target == 'time':
            return hv.VLine(0).opts(alpha=0) * hv.HLine(0).opts(alpha=0)
    elif last_mouse_coord[-2] == 2:
        # mouse is in lonalt axis
        if len(lon_vertices['xs']) == 0:
            selection_geom = [np.array([]), 0]
            return hv.VLine(0).opts(alpha=0) * hv.HLine(0).opts(alpha=0)
        this_selection_geom = np.array([lon_vertices['xs'], lon_vertices['ys']]).T[:, 0, :]
        selection_geom = [this_selection_geom, 2]
        x1 = np.min(lon_vertices['xs'])
        x2 = np.max(lon_vertices['xs'])
        y1 = np.min(lon_vertices['ys'])
        y2 = np.max(lon_vertices['ys'])
        if target == 'plan':
            return hv.VLine(x1).opts(color='black', line_dash='dashed') * hv.Area(([x1, x2], [-90, -90], [90, 90]), vdims=['latmin', 'latmax']).opts(color='black', alpha=0.3) * hv.VLine(x2).opts(color='black', line_dash='dashed')
        elif target == 'lon':
            return hv.VLine(0).opts(alpha=0) * hv.HLine(0).opts(alpha=0)
        elif target == 'lat':
            return hv.VLine(y1).opts(color='black', line_dash='dashed') * hv.Area(([y1, y2], [-90, -90], [90, 90]), vdims=['latmin', 'latmax']).opts(color='black', alpha=0.3) * hv.VLine(y2).opts(color='black', line_dash='dashed')
        elif target == 'time':
            return hv.HLine(y1).opts(color='black', line_dash='dashed') * hv.Area(([time_range_dt[0], time_range_dt[1]], [y1, y1], [y2, y2]), vdims=['altmin', 'altmax']).opts(color='black', alpha=0.3) * hv.HLine(y2).opts(color='black', line_dash='dashed')
    elif last_mouse_coord[-2] == 3:
        # mouse is in latalt axis
        if len(lat_vertices['xs']) == 0:
            selection_geom = [np.array([]), 0]
            return hv.VLine(0).opts(alpha=0) * hv.HLine(0).opts(alpha=0)
        this_selection_geom = np.array([lon_vertices['xs'], lon_vertices['ys']]).T[:, 0, :]
        selection_geom = [this_selection_geom, 3]
        x1 = np.min(lat_vertices['xs'])
        x2 = np.max(lat_vertices['xs'])
        y1 = np.min(lat_vertices['ys'])
        y2 = np.max(lat_vertices['ys'])
        if target == 'plan':
            return hv.HLine(y1).opts(color='black', line_dash='dashed') * hv.Area(([-180, 180], [y1, y1], [y2, y2]), vdims=['latmin', 'latmax']).opts(color='black', alpha=0.3) * hv.HLine(y2).opts(color='black', line_dash='dashed')
        elif target == 'lon':
            return hv.HLine(x1).opts(color='black', line_dash='dashed') * hv.Area(([-180, 180], [x1, x1], [x2, x2]), vdims=['altmin', 'altmax']).opts(color='black', alpha=0.3) * hv.HLine(x2).opts(color='black', line_dash='dashed')
        elif target == 'lat':
            return hv.VLine(0).opts(alpha=0) * hv.HLine(0).opts(alpha=0)
        elif target == 'time':
            return hv.HLine(x1).opts(color='black', line_dash='dashed') * hv.Area(([time_range_dt[0], time_range_dt[1]], [x1, x1], [x2, x2]), vdims=['altmin', 'altmax']).opts(color='black', alpha=0.3) * hv.HLine(x2).opts(color='black', line_dash='dashed')
    elif last_mouse_coord[-2] == 4:
        # mouse is in timeseries axis
        if len(time_vertices['xs']) == 0:
            selection_geom = [np.array([]), 0]
            return hv.VLine(0).opts(alpha=0) * hv.HLine(0).opts(alpha=0)
        this_selection_geom = np.array([lon_vertices['xs'], lon_vertices['ys']]).T[:, 0, :]
        selection_geom = [this_selection_geom, 4]
        x1 = np.min(time_vertices['xs'])
        x2 = np.max(time_vertices['xs'])
        y1 = np.min(time_vertices['ys'])
        y2 = np.max(time_vertices['ys'])
        if target == 'plan':
            return hv.VLine(0).opts(alpha=0) * hv.HLine(0).opts(alpha=0)
        elif target == 'lon':
            return hv.HLine(y1).opts(color='black', line_dash='dashed') * hv.Area(([-180, 180], [y1, y1], [y2, y2]), vdims=['altmin', 'altmax']).opts(color='black', alpha=0.3) * hv.HLine(y2).opts(color='black', line_dash='dashed')
        elif target == 'lat':
            return hv.VLine(y1).opts(color='black', line_dash='dashed') * hv.Area(([y1, y2], [-90, -90], [90, 90]), vdims=['latmin', 'latmax']).opts(color='black', alpha=0.3) * hv.VLine(y2).opts(color='black', line_dash='dashed')
        elif target == 'time':
            return hv.VLine(0).opts(alpha=0) * hv.HLine(0).opts(alpha=0)
    else:
        return hv.VLine(0).opts(alpha=0) * hv.HLine(0).opts(alpha=0)

def pointer_plotter(plan_x=None, plan_y=None, lat_x=None, lat_y=None,
                    lon_x=None, lon_y=None, time_x=None, time_y=None, target=None):
    global last_mouse_coord
    last_mouse_coord[-1] += 1
    # Determine which axis the mouse is in:
    if plan_x != last_mouse_coord[0] or plan_y != last_mouse_coord[1]:
        # mouse is in planview axis
        last_mouse_coord[-2] = 1
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
        last_mouse_coord[-2] = 2
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
        last_mouse_coord[-2] = 3
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
        last_mouse_coord[-2] = 4
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
        last_mouse_coord = [plan_x, plan_y, lon_x, lon_y, lat_x, lat_y, time_x, time_y, 0, 0]
    if last_mouse_coord[-1] == ((len(last_mouse_coord))/2)-1:
        last_mouse_coord = [plan_x, plan_y, lon_x, lon_y, lat_x, lat_y, time_x, time_y, last_mouse_coord[-2], 0]
    return crosshair


xlim = (ds.network_center_longitude.data - 3, ds.network_center_longitude.data + 3)
ylim = (ds.network_center_latitude.data - 3, ds.network_center_latitude.data + 3)
zlim = (0, 20000)

timefloats = color_by_time(ds.event_time.values)[-1]

plan_points = hv.Points((ds.event_longitude.data, ds.event_latitude.data, timefloats), kdims=['lon', 'lat'], vdims=['time'])
plan_ax = hv.operation.datashader.datashade(plan_points, aggregator=datashader.max('time'), cmap='rainbow').opts(xlim=xlim, ylim=ylim, width=px_scale*plan_edge_length, height=px_scale*plan_edge_length)#, tools=['hover'])
plan_ax_polys = hv.Polygons([]).opts(hv.opts.Polygons(fill_alpha=0.3, fill_color='black'))
plan_ax_selector = hv.streams.PolyDraw(source=plan_ax_polys, drag=False, num_objects=1, show_vertices=True, vertex_style={'size': 5, 'fill_color': 'white', 'line_color' : 'black'}).rename(data='plan_vertices')
# counties_shp = shapefile.Reader('ne_10m_admin_2_counties.shp').shapes()
# counties_shp = [shape(counties_shp[i]) for i in range(len(counties_shp))]
# plan_ax = gv.Path(counties_shp).opts(color='gray') * gf.borders().opts(color='black') * gf.states().opts(color='black', line_width=2) * plan_points
plan_ax_pointer = hv.streams.PointerXY(x=0, y=0, source=plan_points).rename(x='plan_x', y='plan_y')

lon_alt_ax = hv.Points((ds.event_longitude.data, ds.event_altitude.data, timefloats), kdims=['lon', 'alt'], vdims=['time'])
lon_alt_ax = hv.operation.datashader.datashade(lon_alt_ax, aggregator=datashader.max('time'), cmap='rainbow').opts(xlim=xlim, ylim=zlim, width=px_scale*plan_edge_length, height=px_scale*hist_edge_length, hooks=[hook_yalt_limiter])#, tools=['hover'])
lon_alt_ax_polys = hv.Polygons([]).opts(hv.opts.Polygons(fill_alpha=0.3, fill_color='black'))
lon_alt_ax_selector = hv.streams.PolyDraw(source=lon_alt_ax_polys, drag=False, num_objects=1, show_vertices=True, vertex_style={'size': 5, 'fill_color': 'white', 'line_color' : 'black'}).rename(data='lon_vertices')
lon_alt_ax_pointer = hv.streams.PointerXY(x=0, y=0, source=lon_alt_ax).rename(x='lon_x', y='lon_y')


hist_ax = hv.Histogram(np.histogram(ds.event_altitude.data, bins=np.arange(0, 20001, 1000)), kdims=['alt'], vdims=['src']).opts(width=px_scale*hist_edge_length, height=px_scale*hist_edge_length, invert_axes=True).opts(hooks=[hook_xlabel_rotate, hook_hist_src_limiter, hook_yalt_limiter])#, tools=['hover'])

lat_alt_ax = hv.Points((ds.event_altitude.data, ds.event_latitude.data, timefloats), kdims=['alt', 'lat'], vdims=['time'])
lat_alt_ax = hv.operation.datashader.datashade(lat_alt_ax, aggregator=datashader.max('time'), cmap='rainbow').opts(xlim=zlim, ylim=ylim, width=px_scale*hist_edge_length, height=px_scale*plan_edge_length, hooks=[hook_xlabel_rotate, hook_xalt_limiter])#, tools=['hover'])
lat_alt_ax_polys = hv.Polygons([]).opts(hv.opts.Polygons(fill_alpha=0.3, fill_color='black'))
lat_alt_ax_selector = hv.streams.PolyDraw(source=lat_alt_ax_polys, drag=False, num_objects=1, show_vertices=True, vertex_style={'size': 5, 'fill_color': 'white', 'line_color' : 'black'}).rename(data='lat_vertices')
lat_alt_ax_pointer = hv.streams.PointerXY(x=0, y=0, source=lat_alt_ax).rename(x='lat_x', y='lat_y')


alt_time_ax = hv.Points((ds.event_time.data, ds.event_altitude.data, timefloats), kdims=['time', 'alt'], vdims=['time'])
alt_time_ax = hv.operation.datashader.datashade(alt_time_ax, aggregator=datashader.max('time'), cmap='rainbow').opts(xlim=(ds.event_time.data[0], ds.event_time.data[-1]), ylim=zlim, width=px_scale*(plan_edge_length+hist_edge_length), height=px_scale*hist_edge_length, hooks=[hook_yalt_limiter, hook_time_limiter])#, toolbar=None, tools=['hover'])
alt_time_ax_polys = hv.Polygons([]).opts(hv.opts.Polygons(fill_alpha=0.3, fill_color='black'))
alt_time_ax_selector = hv.streams.PolyDraw(source=alt_time_ax_polys, drag=False, num_objects=1, show_vertices=True, vertex_style={'size': 5, 'fill_color': 'white', 'line_color' : 'black'}).rename(data='time_vertices')
alt_time_pointer = hv.streams.PointerXY(x=0, y=0, source=alt_time_ax).rename(x='time_x', y='time_y')


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

plan_ax_highlight = hv.DynamicMap(lambda plan_vertices, lat_vertices, lon_vertices, time_vertices: 
                                 highlight_plotter(plan_vertices, lat_vertices, lon_vertices, time_vertices, 'plan'),
                                 streams=[plan_ax_selector, lat_alt_ax_selector, lon_alt_ax_selector, alt_time_ax_selector])

lon_ax_highlight = hv.DynamicMap(lambda plan_vertices, lat_vertices, lon_vertices, time_vertices: 
                                 highlight_plotter(plan_vertices, lat_vertices, lon_vertices, time_vertices, 'lon'),
                                 streams=[plan_ax_selector, lat_alt_ax_selector, lon_alt_ax_selector, alt_time_ax_selector])

lat_ax_highlight = hv.DynamicMap(lambda plan_vertices, lat_vertices, lon_vertices, time_vertices: 
                                 highlight_plotter(plan_vertices, lat_vertices, lon_vertices, time_vertices, 'lat'),
                                 streams=[plan_ax_selector, lat_alt_ax_selector, lon_alt_ax_selector, alt_time_ax_selector])

time_ax_highlight = hv.DynamicMap(lambda plan_vertices, lat_vertices, lon_vertices, time_vertices: 
                                 highlight_plotter(plan_vertices, lat_vertices, lon_vertices, time_vertices, 'time'),
                                 streams=[plan_ax_selector, lat_alt_ax_selector, lon_alt_ax_selector, alt_time_ax_selector])


lon_alt_ax = (lon_alt_ax * lon_ax_crosshair * lon_alt_ax_polys * lon_ax_highlight).opts(active_tools=['wheel_zoom', 'poly_draw'])
plan_ax = (plan_ax * plan_ax_crosshair * plan_ax_polys * plan_ax_highlight).opts(active_tools=['wheel_zoom', 'poly_draw'])
lat_alt_ax = (lat_alt_ax * lat_ax_crosshair * lat_alt_ax_polys * lat_ax_highlight).opts(active_tools=['wheel_zoom', 'poly_draw'])
alt_time_ax = (alt_time_ax * time_ax_crosshair * alt_time_ax_polys * time_ax_highlight).opts(active_tools=['wheel_zoom', 'poly_draw'])

the_lower_part = (lon_alt_ax + hist_ax + plan_ax + lat_alt_ax).cols(2)

title = pn.pane.Markdown('## LMA Data Explorer')

dataView = pn.Column(title, alt_time_ax, the_lower_part)

limit_button = pn.widgets.Button(name='Limit to Selection', button_type='primary')
pn.bind(limit_to_polygon, limit_button, watch=True)

pn.Row(dataView, limit_button).servable()
