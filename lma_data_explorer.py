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
        if self.selection_geom[0] == np.array([]):
            return
        select_path = mpl.path.Path(self.selection_geom[0], closed=True)
        axis = self.selection_geom[-1]
        if axis == 0:
            raise ValueError('Selection is in an unknown position, cannot limit to selection')
        elif axis == 1:
            # Selection in planview axis, filter by lon and lat
            points_in_selection = select_path.contains_points(np.array([self.ds.event_longitude.data, self.ds.event_latitude.data]).T)
        elif axis == 2:
            # Selection in lonalt axis, filter by lon and alt
            points_in_selection = select_path.contains_points(np.array([self.ds.event_longitude.data, self.ds.event_altitude.data]).T)
        elif axis == 3:
            # Selection in latalt axis, filter by lat and alt
            points_in_selection = select_path.contains_points(np.array([self.ds.event_altitude.data, self.ds.event_latitude.data]).T)
        elif axis == 4:
            # Selection in time axis, filter by time and alt
            select_path_arr = self.selection_geom[0]
            select_path_arr[:, 0] = select_path_arr[:, 0]*1e6
            select_path = mpl.path.Path(self.selection_geom[0], closed=True)
            points_in_selection = select_path.contains_points(np.array([self.ds.event_time.data.astype(float), self.ds.event_altitude.data]).T)
        if points_in_selection.sum() == 0:
            return
        self.ds = self.ds.isel(number_of_events=points_in_selection)
        self.clear_all_polygons()



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

    def plot_planview_points(self):
        timefloats = color_by_time(self.ds.event_time.values)[-1]
        return hv.Points((self.ds.event_longitude.data, self.ds.event_latitude.data, timefloats), kdims=['lon', 'lat'], vdims=['time'])
    
    def plot_lonalt_points(self):
        timefloats = color_by_time(self.ds.event_time.values)[-1]
        return hv.Points((self.ds.event_longitude.data, self.ds.event_altitude.data, timefloats), kdims=['lon', 'alt'], vdims=['time'])
    
    def plot_alt_hist(self):
        return hv.Histogram(np.histogram(self.ds.event_altitude.data, bins=np.arange(0, 20001, 1000)), kdims=['alt'], vdims=['src']).opts(width=self.px_scale*self.hist_edge_length, height=self.px_scale*self.hist_edge_length, invert_axes=True).opts(hooks=[self.hook_xlabel_rotate, self.hook_hist_src_limiter, self.hook_yalt_limiter])

    def plot_latalt_points(self):
        timefloats = color_by_time(self.ds.event_time.values)[-1]
        return hv.Points((self.ds.event_altitude.data, self.ds.event_latitude.data, timefloats), kdims=['alt', 'lat'], vdims=['time'])

    def plot_alttime_points(self):
        timefloats = color_by_time(self.ds.event_time.values)[-1]
        return hv.Points((self.ds.event_time.data, self.ds.event_altitude.data, timefloats), kdims=['time', 'alt'], vdims=['time'])

    def pointer_plotter(self, plan_x=None, plan_y=None, lat_x=None, lat_y=None,
                    lon_x=None, lon_y=None, time_x=None, time_y=None, target=None):
        self.last_mouse_coord[-1] += 1
        # Determine which axis the mouse is in:
        if plan_x != self.last_mouse_coord[0] or plan_y != self.last_mouse_coord[1]:
            # mouse is in planview axis
            self.last_mouse_coord[-2] = 1
            if target == 'plan':
                crosshair =  hv.VLine(0).opts(alpha=0) * hv.HLine(0).opts(alpha=0)
            elif target == 'lon':
                crosshair =  hv.VLine(plan_x).opts(color='black') * hv.HLine(0).opts(alpha=0)
            elif target == 'lat':
                crosshair =  hv.HLine(plan_y).opts(color='black') * hv.VLine(0).opts(alpha=0)
            elif target == 'time':
                crosshair =  hv.VLine(0).opts(alpha=0) * hv.HLine(0).opts(alpha=0)
        elif lon_x != self.last_mouse_coord[2] or lon_y != self.last_mouse_coord[3]:
            # mouse is in lonalt axis
            self.last_mouse_coord[-2] = 2
            if target == 'plan':
                crosshair = hv.VLine(lon_x).opts(color='black') * hv.HLine(0).opts(alpha=0)
            elif target == 'lon':
                crosshair = hv.VLine(0).opts(alpha=0) * hv.HLine(0).opts(alpha=0)
            elif target == 'lat':
                crosshair = hv.HLine(0).opts(alpha=0) * hv.VLine(lon_y).opts(color='black')
            elif target == 'time':
                crosshair =  hv.VLine(0).opts(alpha=0) * hv.HLine(lon_y).opts(color='black')
        elif lat_x != self.last_mouse_coord[4] or lat_y != self.last_mouse_coord[5]:
            # mouse is in latalt axis
            self.last_mouse_coord[-2] = 3
            if target == 'plan':
                crosshair = hv.VLine(0).opts(alpha=0) * hv.HLine(lat_y).opts(color='black')
            elif target == 'lon':
                crosshair = hv.VLine(0).opts(alpha=0) * hv.HLine(lat_x).opts(color='black')
            elif target == 'lat':
                crosshair = hv.HLine(0).opts(alpha=0) * hv.VLine(0).opts(alpha=0)
            elif target == 'time':
                crosshair = hv.VLine(0).opts(alpha=0) * hv.HLine(lat_x).opts(color='black')
        elif time_x != self.last_mouse_coord[6] or time_y != self.last_mouse_coord[7]:
            # mouse is in timeseries axis
            self.last_mouse_coord[-2] = 4
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
            self.last_mouse_coord = [plan_x, plan_y, lon_x, lon_y, lat_x, lat_y, time_x, time_y, 0, 0]
        if self.last_mouse_coord[-1] == ((len(self.last_mouse_coord))/2)-1:
            self.last_mouse_coord = [plan_x, plan_y, lon_x, lon_y, lat_x, lat_y, time_x, time_y, self.last_mouse_coord[-2], 0]
        return crosshair
    
    def handle_selection(self, data):
        if len(data['xs']) == 0 or len(data['ys']) == 0:
            self.selection_geom = [np.array([]), 0]
            return
        if self.selection_geom[-1] != 0 and self.last_mouse_coord[-2] != 0 and self.last_mouse_coord[-2] != self.selection_geom[-1]:
            # selection is in a different axis than the last selection
            # reset the polygons in the old axis
            self.clear_all_polygons()
            return
        this_selection_geom = np.array([data['xs'], data['ys']]).T[:, 0, :]
        self.selection_geom = [this_selection_geom, self.last_mouse_coord[-2]]

    def clear_all_polygons(self):
        new_polys_plan = hv.Polygons([]).opts(hv.opts.Polygons(fill_alpha=0.3, fill_color='black'))
        self.plan_ax_selector.source = new_polys_plan
        
        new_polys_lonalt = hv.Polygons([]).opts(hv.opts.Polygons(fill_alpha=0.3, fill_color='black'))
        self.lon_alt_ax_selector.source = new_polys_lonalt

        new_polys_latalt = hv.Polygons([]).opts(hv.opts.Polygons(fill_alpha=0.3, fill_color='black'))
        self.lat_alt_ax_selector.source = new_polys_latalt

        new_polys_alttime = hv.Polygons([]).opts(hv.opts.Polygons(fill_alpha=0.3, fill_color='black'))
        self.alt_time_ax_selector.source = new_polys_alttime

        new_plan_ax = (self.plan_points * self.plan_ax_crosshair * new_polys_plan * self.plan_ax_select_area).opts(xlim=self.plan_points.range('lon'), ylim=self.plan_points.range('lat'))
        new_lon_alt_ax = (self.lon_alt_points * self.lon_ax_crosshair * new_polys_lonalt * self.lon_alt_select_area).opts(xlim=self.lon_alt_points.range('lon'), ylim=self.lon_alt_points.range('alt'))
        new_lat_alt_ax = (self.lat_alt_points * self.lat_ax_crosshair * new_polys_latalt * self.lat_alt_select_area).opts(xlim=self.lat_alt_points.range('alt'), ylim=self.lat_alt_points.range('lat'))
        new_alt_time_ax = (self.alt_time_points * self.time_ax_crosshair * new_polys_alttime * self.alt_time_select_area).opts(xlim=self.alt_time_points.range('time'), ylim=self.alt_time_points.range('alt'))

        new_lower_part = (new_lon_alt_ax + self.hist_ax + new_plan_ax + new_lat_alt_ax).cols(2)

        self.panelHandle[0][2].object = new_lower_part
        self.panelHandle[0][1].object = new_alt_time_ax

        self.selection_geom = [np.array([]), 0]

        self.plan_points.range = self.panelHandle[0][2].object[2].range
        self.lon_alt_points.range = self.panelHandle[0][2].object[0].range
        self.lat_alt_points.range = self.panelHandle[0][2].object[3].range
        self.alt_time_points.range = self.panelHandle[0][1].object.range

    def plan_ax_highlighter(self):
        match self.selection_geom[-1]:
            case 0 | 1 | 4:
                # selection is in an unknown position, the planview axis or the time height axis
                # in these cases, we don't want to highlight any selection made
                return hv.VLine(0).opts(alpha=0) * hv.HLine(0).opts(alpha=0)
            case 2:
                # selection is in the lonalt axis
                # highlight longitude range of selection
                x1 = np.min(self.selection_geom[0][:, 0])
                x2 = np.max(self.selection_geom[0][:, 0])
                return (hv.Area(([x1, x2], [-90, -90], [90, 90]), vdims=['v1', 'v2']).opts(color='black', alpha=0.3) *
                        hv.VLines([x1, x2]).opts(color='black', line_dash='dashed'))
            case 3:
                # selection is in the latalt axis
                # highlight latitude range of selection
                x1 = np.min(self.selection_geom[0][:, 1])
                x2 = np.max(self.selection_geom[0][:, 1])
                return (hv.Area(([-180, 180], [x1, x1], [x2, x2]), vdims=['v1', 'v2']).opts(color='black', alpha=0.3) *
                        hv.HLines([x1, x2]).opts(color='black', line_dash='dashed'))
        
    def lon_ax_highlighter(self):
        match self.selection_geom[-1]:
            case 0 | 2:
                # selection is in an unknown position, or the lonalt axis
                # in these cases, we don't want to highlight any selection made
                return hv.VLine(0).opts(alpha=0) * hv.HLine(0).opts(alpha=0)
            case 1:
                # selection is in the planview axis
                # highlight longitude range of selection
                x1 = np.min(self.selection_geom[0][:, 0])
                x2 = np.max(self.selection_geom[0][:, 0])
                return (hv.Area(([x1, x2], [self.alt_min, self.alt_min], [self.alt_max, self.alt_max]), vdims=['v1', 'v2']).opts(color='black', alpha=0.3) *
                        hv.VLines([x1, x2]).opts(color='black', line_dash='dashed'))
            case 3:
                # selection is in the latalt axis
                # highlight altitude range of selection
                x1 = np.min(self.selection_geom[0][:, 0])
                x2 = np.max(self.selection_geom[0][:, 0])
                return (hv.Area(([-180, 180], [x1, x1], [x2, x2]), vdims=['v1', 'v2']).opts(color='black', alpha=0.3) *
                        hv.HLines([x1, x2]).opts(color='black', line_dash='dashed'))
            case 4:
                # selection is in the time axis
                # highlight altitude range of selection
                x1 = np.min(self.selection_geom[0][:, 1])
                x2 = np.max(self.selection_geom[0][:, 1])
                return (hv.Area(([-180, 180], [x1, x1], [x2, x2]), vdims=['v1', 'v2']).opts(color='black', alpha=0.3) *
                        hv.HLines([x1, x2]).opts(color='black', line_dash='dashed'))

    def lat_ax_highlighter(self):
        match self.selection_geom[-1]:
            case 0 | 3:
                # selection is in an unknown position, or the latalt axis
                # in these cases, we don't want to highlight any selection made
                return hv.VLine(0).opts(alpha=0) * hv.HLine(0).opts(alpha=0)
            case 1:
                # selection is in the planview axis
                # highlight latitude range of selection
                x1 = np.min(self.selection_geom[0][:, 1])
                x2 = np.max(self.selection_geom[0][:, 1])
                return (hv.Area(([self.alt_min, self.alt_max], [x1, x1], [x2, x2]), vdims=['v1', 'v2']).opts(color='black', alpha=0.3) *
                        hv.HLines([x1, x2]).opts(color='black', line_dash='dashed'))
            case 2:
                # selection is in the lonalt axis
                # highlight altitude range of selection
                x1 = np.min(self.selection_geom[0][:, 1])
                x2 = np.max(self.selection_geom[0][:, 1])
                return (hv.Area(([x1, x2], [-90, -90], [90, 90]), vdims=['v1', 'v2']).opts(color='black', alpha=0.3) *
                        hv.VLines([x1, x2]).opts(color='black', line_dash='dashed'))
            case 4:
                # selection is in the time axis
                # highlight altitude range of selection
                x1 = np.min(self.selection_geom[0][:, 1])
                x2 = np.max(self.selection_geom[0][:, 1])
                return (hv.Area(([x1, x2], [-90, -90], [90, 90]), vdims=['v1', 'v2']).opts(color='black', alpha=0.3) *
                        hv.VLines([x1, x2]).opts(color='black', line_dash='dashed'))

    def time_ax_highlighter(self):
        match self.selection_geom[-1]:
            case 0 | 1 | 4:
                # selection is in an unknown position, the planview axis or the time height axis
                # in these cases, we don't want to highlight any selection made
                return hv.VLine(0).opts(alpha=0) * hv.HLine(0).opts(alpha=0)
            case 2:
                # selection is in the lonalt axis
                # highlight altitude range of selection
                x1 = np.min(self.selection_geom[0][:, 1])
                x2 = np.max(self.selection_geom[0][:, 1])
                return (hv.Area(([self.time_range_dt[0], self.time_range_dt[1]], [x1, x1], [x2, x2]), vdims=['v1', 'v2']).opts(color='black', alpha=0.3) *
                        hv.HLines([x1, x2]).opts(color='black', line_dash='dashed'))
            case 3:
                # selection is in the latalt axis
                # highlight altitude range of selection
                x1 = np.min(self.selection_geom[0][:, 0])
                x2 = np.max(self.selection_geom[0][:, 0])
                return (hv.Area(([self.time_range_dt[0], self.time_range_dt[1]], [x1, x1], [x2, x2]), vdims=['v1', 'v2']).opts(color='black', alpha=0.3) *
                        hv.HLines([x1, x2]).opts(color='black', line_dash='dashed'))


    def init_plot(self):
        xlim = (self.ds.network_center_longitude.data - 3, self.ds.network_center_longitude.data + 3)
        ylim = (self.ds.network_center_latitude.data - 3, self.ds.network_center_latitude.data + 3)
        zlim = (0, 20000)

        plan_points = hv.DynamicMap(self.plot_planview_points)
        plan_points = hv.operation.datashader.datashade(plan_points, aggregator=datashader.max('time'), cmap='rainbow').opts(xlim=xlim, ylim=ylim, width=self.px_scale*self.plan_edge_length, height=self.px_scale*self.plan_edge_length)
        plan_ax_polys = hv.Polygons([]).opts(hv.opts.Polygons(fill_alpha=0.3, fill_color='black'))
        plan_ax_selector = hv.streams.PolyDraw(source=plan_ax_polys, drag=False, num_objects=1, show_vertices=True, vertex_style={'size': 5, 'fill_color': 'white', 'line_color' : 'black'})
        plan_ax_selector.add_subscriber(self.handle_selection)
        plan_ax_select_area = hv.DynamicMap(self.plan_ax_highlighter)
        # counties_shp = shapefile.Reader('ne_10m_admin_2_counties.shp').shapes()
        # counties_shp = [shape(counties_shp[i]) for i in range(len(counties_shp))]
        # plan_ax = gv.Path(counties_shp).opts(color='gray') * gf.borders().opts(color='black') * gf.states().opts(color='black', line_width=2) * plan_points
        plan_ax_pointer = hv.streams.PointerXY(x=0, y=0, source=plan_points).rename(x='plan_x', y='plan_y')

        lon_alt_points = hv.DynamicMap(self.plot_lonalt_points)
        lon_alt_points = hv.operation.datashader.datashade(lon_alt_points, aggregator=datashader.max('time'), cmap='rainbow').opts(xlim=xlim, ylim=zlim, width=self.px_scale*self.plan_edge_length, height=self.px_scale*self.hist_edge_length, hooks=[self.hook_yalt_limiter])
        lon_alt_ax_polys = hv.Polygons([]).opts(hv.opts.Polygons(fill_alpha=0.3, fill_color='black'))
        lon_alt_ax_selector = hv.streams.PolyDraw(source=lon_alt_ax_polys, drag=False, num_objects=1, show_vertices=True, vertex_style={'size': 5, 'fill_color': 'white', 'line_color' : 'black'})
        lon_alt_ax_selector.add_subscriber(self.handle_selection)
        lon_alt_select_area = hv.DynamicMap(self.lon_ax_highlighter)
        lon_alt_ax_pointer = hv.streams.PointerXY(x=0, y=0, source=lon_alt_points).rename(x='lon_x', y='lon_y')


        hist_ax = hv.DynamicMap(self.plot_alt_hist)

        lat_alt_points = hv.DynamicMap(self.plot_latalt_points)
        lat_alt_points = hv.operation.datashader.datashade(lat_alt_points, aggregator=datashader.max('time'), cmap='rainbow').opts(xlim=zlim, ylim=ylim, width=self.px_scale*self.hist_edge_length, height=self.px_scale*self.plan_edge_length, hooks=[self.hook_xlabel_rotate, self.hook_xalt_limiter])
        lat_alt_ax_polys = hv.Polygons([]).opts(hv.opts.Polygons(fill_alpha=0.3, fill_color='black'))
        lat_alt_ax_selector = hv.streams.PolyDraw(source=lat_alt_ax_polys, drag=False, num_objects=1, show_vertices=True, vertex_style={'size': 5, 'fill_color': 'white', 'line_color' : 'black'})
        lat_alt_ax_selector.add_subscriber(self.handle_selection)
        lat_alt_select_area = hv.DynamicMap(self.lat_ax_highlighter)
        lat_alt_ax_pointer = hv.streams.PointerXY(x=0, y=0, source=lat_alt_points).rename(x='lat_x', y='lat_y')


        alt_time_points = hv.DynamicMap(self.plot_alttime_points)
        alt_time_points = hv.operation.datashader.datashade(alt_time_points, aggregator=datashader.max('time'), cmap='rainbow').opts(xlim=(self.ds.event_time.data[0], self.ds.event_time.data[-1]), ylim=zlim, width=self.px_scale*(self.plan_edge_length+self.hist_edge_length), height=self.px_scale*self.hist_edge_length, hooks=[self.hook_yalt_limiter, self.hook_time_limiter])
        alt_time_ax_polys = hv.Polygons([]).opts(hv.opts.Polygons(fill_alpha=0.3, fill_color='black'))
        alt_time_ax_selector = hv.streams.PolyDraw(source=alt_time_ax_polys, drag=False, num_objects=1, show_vertices=True, vertex_style={'size': 5, 'fill_color': 'white', 'line_color' : 'black'})
        alt_time_ax_selector.add_subscriber(self.handle_selection)
        alt_time_select_area = hv.DynamicMap(self.time_ax_highlighter)
        alt_time_pointer = hv.streams.PointerXY(x=0, y=0, source=alt_time_points).rename(x='time_x', y='time_y')


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
        
        self.plan_points = plan_points
        self.lon_alt_points = lon_alt_points
        self.lat_alt_points = lat_alt_points
        self.alt_time_points = alt_time_points
        
        self.plan_ax_crosshair = plan_ax_crosshair
        self.lon_ax_crosshair = lon_ax_crosshair
        self.lat_ax_crosshair = lat_ax_crosshair
        self.time_ax_crosshair = time_ax_crosshair

        self.plan_ax_select_area = plan_ax_select_area
        self.lon_alt_select_area = lon_alt_select_area
        self.lat_alt_select_area = lat_alt_select_area
        self.alt_time_select_area = alt_time_select_area

        self.plan_ax_selector = plan_ax_selector
        self.lon_alt_ax_selector = lon_alt_ax_selector
        self.lat_alt_ax_selector = lat_alt_ax_selector
        self.alt_time_ax_selector = alt_time_ax_selector

        self.hist_ax = hist_ax

        lon_alt_ax = (lon_alt_points * lon_ax_crosshair * lon_alt_ax_polys * lon_alt_select_area)
        plan_ax = (plan_points * plan_ax_crosshair * plan_ax_polys * plan_ax_select_area)
        lat_alt_ax = (lat_alt_points * lat_ax_crosshair * lat_alt_ax_polys * lat_alt_select_area)
        alt_time_ax = pn.pane.HoloViews(alt_time_points * time_ax_crosshair * alt_time_ax_polys * alt_time_select_area)

        the_lower_part = (lon_alt_ax + hist_ax + plan_ax + lat_alt_ax).cols(2)
        the_lower_part = pn.pane.HoloViews(the_lower_part)

        title = pn.pane.Markdown('## LMA Data Explorer')

        dataView = pn.Column(title, alt_time_ax, the_lower_part)

        limit_button = pn.widgets.Button(name='Limit to Selection', button_type='primary')
        pn.bind(self.limit_to_polygon, limit_button, watch=True)
        self.panelHandle = pn.Row(dataView, limit_button)

