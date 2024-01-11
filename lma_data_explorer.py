import numpy as np
from bokeh.models import Range1d
import matplotlib as mpl

from pyxlma.lmalib.io import read as lma_read
from pyxlma.plot.xlma_plot_feature import color_by_time

from datetime import datetime as dt, timedelta


import holoviews as hv
import holoviews.operation.datashader
import datashader
import panel as pn


import geoviews as gv
import geoviews.feature as gf
from cartopy import crs as ccrs
import shapefile
from shapely.geometry import shape



class LMADataExplorer:
    def __init__(self, filename):
        self.px_scale = 7

        self.plan_edge_length = 60
        self.hist_edge_length = 20

        # plan view x, y; lonalt x, y; latalt x, y; time x, y; position; counter
        self.last_mouse_coord = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.selection_geom = [np.array([]), 0]


        self.alt_min = 0
        self.alt_max = 100000
        self.alt_limit = Range1d(0, 20000, bounds=(self.alt_min, self.alt_max))

        ds, start_time = lma_read.dataset(filename)
        self.orig_dataset = ds
        self.ds = ds
        end_time = start_time + timedelta(seconds=int(filename.split('_')[-1].replace('.dat.gz', '')))
        self.time_range_dt = np.array([start_time, end_time]).astype('datetime64')
        self.time_range = self.time_range_dt.astype(float)/1e3
        self.lon_range = Range1d(self.ds.network_center_longitude.data - 1, self.ds.network_center_longitude.data + 1)
        self.lat_range = Range1d(self.ds.network_center_latitude.data - 1, self.ds.network_center_latitude.data + 1)
        self.init_plot()

    def limit_to_polygon(self, limit_button):
        global selection_geom
        if selection_geom[0] == np.array([]):
            return
        select_path = mpl.path.Path(selection_geom[0], closed=True)
        axis = selection_geom[-1]
        if axis == 1:
            print(select_path.contains_points(np.array([self.ds.event_longitude.data, self.ds.event_latitude.data]).T).shape)
            print(self.ds.event_longitude.data.shape)


    def hook_hist_src_limiter(self, plot, element):
        init_xmax = plot.state.x_range.end
        plot.state.x_range = Range1d(0, init_xmax, bounds=(0, init_xmax))

    def hook_yalt_limiter(self, plot, element):
        if plot.state.y_range.start < self.alt_min:
            plot.state.y_range.start = self.alt_min
        if plot.state.y_range.end > self.alt_max:
            plot.state.y_range.end = self.alt_max

    def hook_xalt_limiter(self, plot, element):
        if plot.state.x_range.start < self.alt_min:
            plot.state.x_range.start = self.alt_min
        if plot.state.x_range.end > self.alt_max:
            plot.state.x_range.end = self.alt_max

    def hook_time_limiter(self, plot, element):
        if type(plot.state.x_range.start) == float:
            if plot.state.x_range.start < self.time_range[0]:
                plot.state.x_range.start = self.time_range[0]
            if plot.state.x_range.end > self.time_range[1]:
                plot.state.x_range.end = self.time_range[1]

    def hook_xlabel_rotate(self, plot, element):
        plot.state.xaxis.major_label_orientation = -np.pi/2



    def pointer_plotter(self, plan_x=None, plan_y=None, lat_x=None, lat_y=None,
                    lon_x=None, lon_y=None, time_x=None, time_y=None, target=None):
        last_mouse_coord = self.last_mouse_coord
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
    

    def highlight_plotter(self, data, source=):
    



    def init_plot(self):
        xlim = (self.ds.network_center_longitude.data - 3, self.ds.network_center_longitude.data + 3)
        ylim = (self.ds.network_center_latitude.data - 3, self.ds.network_center_latitude.data + 3)
        zlim = (0, 20000)

        timefloats = color_by_time(self.ds.event_time.values)[-1]

        plan_points = hv.Points((self.ds.event_longitude.data, self.ds.event_latitude.data, timefloats), kdims=['lon', 'lat'], vdims=['time'])
        plan_ax = hv.operation.datashader.datashade(plan_points, aggregator=datashader.max('time'), cmap='rainbow').opts(xlim=xlim, ylim=ylim, width=self.px_scale*self.plan_edge_length, height=self.px_scale*self.plan_edge_length)#, tools=['hover'])
        plan_ax_polys = hv.Polygons([]).opts(hv.opts.Polygons(fill_alpha=0.3, fill_color='black'))
        plan_ax_selector = hv.streams.PolyDraw(source=plan_ax_polys, drag=False, num_objects=1, show_vertices=True, vertex_style={'size': 5, 'fill_color': 'white', 'line_color' : 'black'})
        # counties_shp = shapefile.Reader('ne_10m_admin_2_counties.shp').shapes()
        # counties_shp = [shape(counties_shp[i]) for i in range(len(counties_shp))]
        # plan_ax = gv.Path(counties_shp).opts(color='gray') * gf.borders().opts(color='black') * gf.states().opts(color='black', line_width=2) * plan_points
        plan_ax_pointer = hv.streams.PointerXY(x=0, y=0, source=plan_points).rename(x='plan_x', y='plan_y')

        lon_alt_ax = hv.Points((self.ds.event_longitude.data, self.ds.event_altitude.data, timefloats), kdims=['lon', 'alt'], vdims=['time'])
        lon_alt_ax = hv.operation.datashader.datashade(lon_alt_ax, aggregator=datashader.max('time'), cmap='rainbow').opts(xlim=xlim, ylim=zlim, width=self.px_scale*self.plan_edge_length, height=self.px_scale*self.hist_edge_length, hooks=[self.hook_yalt_limiter])#, tools=['hover'])
        lon_alt_ax_polys = hv.Polygons([]).opts(hv.opts.Polygons(fill_alpha=0.3, fill_color='black'))
        lon_alt_ax_selector = hv.streams.PolyDraw(source=lon_alt_ax_polys, drag=False, num_objects=1, show_vertices=True, vertex_style={'size': 5, 'fill_color': 'white', 'line_color' : 'black'})
        lon_alt_ax_pointer = hv.streams.PointerXY(x=0, y=0, source=lon_alt_ax).rename(x='lon_x', y='lon_y')


        hist_ax = hv.Histogram(np.histogram(self.ds.event_altitude.data, bins=np.arange(0, 20001, 1000)), kdims=['alt'], vdims=['src']).opts(width=self.px_scale*self.hist_edge_length, height=self.px_scale*self.hist_edge_length, invert_axes=True).opts(hooks=[self.hook_xlabel_rotate, self.hook_hist_src_limiter, self.hook_yalt_limiter])#, tools=['hover'])

        lat_alt_ax = hv.Points((self.ds.event_altitude.data, self.ds.event_latitude.data, timefloats), kdims=['alt', 'lat'], vdims=['time'])
        lat_alt_ax = hv.operation.datashader.datashade(lat_alt_ax, aggregator=datashader.max('time'), cmap='rainbow').opts(xlim=zlim, ylim=ylim, width=self.px_scale*self.hist_edge_length, height=self.px_scale*self.plan_edge_length, hooks=[self.hook_xlabel_rotate, self.hook_xalt_limiter])#, tools=['hover'])
        lat_alt_ax_polys = hv.Polygons([]).opts(hv.opts.Polygons(fill_alpha=0.3, fill_color='black'))
        lat_alt_ax_selector = hv.streams.PolyDraw(source=lat_alt_ax_polys, drag=False, num_objects=1, show_vertices=True, vertex_style={'size': 5, 'fill_color': 'white', 'line_color' : 'black'})
        lat_alt_ax_pointer = hv.streams.PointerXY(x=0, y=0, source=lat_alt_ax).rename(x='lat_x', y='lat_y')


        alt_time_ax = hv.Points((self.ds.event_time.data, self.ds.event_altitude.data, timefloats), kdims=['time', 'alt'], vdims=['time'])
        alt_time_ax = hv.operation.datashader.datashade(alt_time_ax, aggregator=datashader.max('time'), cmap='rainbow').opts(xlim=(self.ds.event_time.data[0], self.ds.event_time.data[-1]), ylim=zlim, width=self.px_scale*(self.plan_edge_length+self.hist_edge_length), height=self.px_scale*self.hist_edge_length, hooks=[self.hook_yalt_limiter, self.hook_time_limiter])#, toolbar=None, tools=['hover'])
        alt_time_ax_polys = hv.Polygons([]).opts(hv.opts.Polygons(fill_alpha=0.3, fill_color='black'))
        alt_time_ax_selector = hv.streams.PolyDraw(source=alt_time_ax_polys, drag=False, num_objects=1, show_vertices=True, vertex_style={'size': 5, 'fill_color': 'white', 'line_color' : 'black'})
        alt_time_pointer = hv.streams.PointerXY(x=0, y=0, source=alt_time_ax).rename(x='time_x', y='time_y')


        plan_ax_crosshair = hv.DynamicMap(lambda plan_x, plan_y, lat_x, lat_y,
                                    lon_x, lon_y, time_x, time_y: 
                                    self.pointer_plotter(plan_x, plan_y, lat_x, lat_y,
                                    lon_x, lon_y, time_x, time_y, 'plan'), streams=[plan_ax_pointer, lat_alt_ax_pointer, lon_alt_ax_pointer, alt_time_pointer])

        lon_ax_crosshair = hv.DynamicMap(lambda plan_x, plan_y, lat_x, lat_y,
                                    lon_x, lon_y, time_x, time_y: 
                                    self.pointer_plotter(plan_x, plan_y, lat_x, lat_y,
                                    lon_x, lon_y, time_x, time_y, 'lon'), streams=[plan_ax_pointer, lat_alt_ax_pointer, lon_alt_ax_pointer, alt_time_pointer])


        lat_ax_crosshair = hv.DynamicMap(lambda plan_x, plan_y, lat_x, lat_y,
                                    lon_x, lon_y, time_x, time_y: 
                                    self.pointer_plotter(plan_x, plan_y, lat_x, lat_y,
                                    lon_x, lon_y, time_x, time_y, 'lat'), streams=[plan_ax_pointer, lat_alt_ax_pointer, lon_alt_ax_pointer, alt_time_pointer])

        time_ax_crosshair = hv.DynamicMap(lambda plan_x, plan_y, lat_x, lat_y,
                                    lon_x, lon_y, time_x, time_y: 
                                    self.pointer_plotter(plan_x, plan_y, lat_x, lat_y,
                                    lon_x, lon_y, time_x, time_y, 'time'), streams=[plan_ax_pointer, lat_alt_ax_pointer, lon_alt_ax_pointer, alt_time_pointer])
        

        lon_alt_ax = (lon_alt_ax * lon_ax_crosshair * lon_alt_ax_polys).opts(active_tools=['wheel_zoom', 'poly_draw'])
        plan_ax = (plan_ax * plan_ax_crosshair * plan_ax_polys).opts(active_tools=['wheel_zoom', 'poly_draw'])
        lat_alt_ax = (lat_alt_ax * lat_ax_crosshair * lat_alt_ax_polys).opts(active_tools=['wheel_zoom', 'poly_draw'])
        alt_time_ax = (alt_time_ax * time_ax_crosshair * alt_time_ax_polys).opts(active_tools=['wheel_zoom', 'poly_draw'])

        the_lower_part = (lon_alt_ax + hist_ax + plan_ax + lat_alt_ax).cols(2)

        title = pn.pane.Markdown('## LMA Data Explorer')

        dataView = pn.Column(title, alt_time_ax, the_lower_part)

        limit_button = pn.widgets.Button(name='Limit to Selection', button_type='primary')
        pn.bind(self.limit_to_polygon, limit_button, watch=True)

        self.panelHandle = pn.Row(dataView, limit_button)


