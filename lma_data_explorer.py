import numpy as np
import xarray as xr
from bokeh.models import Range1d, WheelZoomTool
import matplotlib as mpl
from os import path

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

from functools import partial, reduce



class LMADataExplorer:
    def __init__(self, filenames, px_scale=7, color_by_dropdown=None, datashade_switch=None,
                 datashade_label=None, event_filter_controls=None, event_filter_table=None, html_pane=None):
        self.filenames = filenames
        self.px_scale = px_scale

        self.plan_edge_length = 60
        self.hist_edge_length = 20

        # plan view x, y; lonalt x, y; latalt x, y; time x, y; position; counter
        self.last_mouse_coord = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.selection_geom = [np.array([]), 0]


        self.alt_min = 0
        self.alt_max = 100000
        self.alt_limit = Range1d(0, 20000, bounds=(self.alt_min, self.alt_max))

        ds, start_time = lma_read.dataset(filenames)
        if 'event_assigned_charge' not in ds.data_vars:
            ds = ds.set_coords('number_of_stations')
            ds['event_assigned_charge'] = xr.zeros_like(ds['number_of_events'], dtype=np.int8)
            ds = ds.reset_coords('number_of_stations')
        self.orig_dataset = ds
        self.ds = ds
        if type(filenames) != str:
            filename = filenames[0]
        end_time = start_time + timedelta(seconds=int(filename.split('_')[-1].replace('.dat.gz', '')))
        self.time_range_py_dt = [start_time, end_time]
        self.time_range_dt = np.array(self.time_range_py_dt).astype('datetime64')
        self.time_range = self.time_range_dt.astype(float)/1e3
        self.lon_range = Range1d(self.ds.network_center_longitude.data - 1, self.ds.network_center_longitude.data + 1)
        self.lat_range = Range1d(self.ds.network_center_latitude.data - 1, self.ds.network_center_latitude.data + 1)
        self.datashade_label = datashade_label
        self.datashade_label.value = f'Enable Datashader? ({self.ds.number_of_events.data.shape[0]} src)'
        self.filter_history = np.ones_like(self.ds.number_of_events.data).reshape(1, -1)
        self.filter_history_pretty = []
        self.bad_selection_flag = False
        self.event_filter_controls = event_filter_controls
        self.event_filter_type_selector = event_filter_controls[0]
        self.event_filter_op_selector = event_filter_controls[1]
        self.event_filter_value_input = event_filter_controls[2]
        options = {}
        for var in self.ds.data_vars:
            if 'number_of_events' in ds[var].dims and len(ds[var].dims) == 1:
                pretty_text = var.replace('_', ' ').title().replace('Chi2', 'χ²').replace('Id', 'ID')
                options[pretty_text] = var
        self.event_filter_type_selector.options = options
        if 'event_chi2' in self.event_filter_type_selector.options.values():
            self.event_filter_type_selector.value = 'event_chi2'
            self.event_filter_value_input.value = 1.0
        self.event_filter_table = event_filter_table
        self.html_pane = html_pane
        self.html_pane.object = self.ds
        self.init_plot(color_by_dropdown, datashade_switch)

    def limit_to_polygon(self, _):
        select_path_array = self.selection_geom[0]
        select_path = mpl.path.Path(self.selection_geom[0], closed=True)
        axis = self.selection_geom[-1]
        self.selection_geom = [np.array([]), 0]
        if select_path == np.array([]) or self.bad_selection_flag or axis == 0:
            return
        elif axis == 1:
            # Selection in planview axis, filter by lon and lat
            points_in_selection = select_path.contains_points(np.array([self.orig_dataset.event_longitude.data, self.orig_dataset.event_latitude.data]).T)
        elif axis == 2:
            # Selection in lonalt axis, filter by lon and alt
            points_in_selection = select_path.contains_points(np.array([self.orig_dataset.event_longitude.data, self.orig_dataset.event_altitude.data]).T)
        elif axis == 3:
            # Selection in latalt axis, filter by lat and alt
            points_in_selection = select_path.contains_points(np.array([self.orig_dataset.event_altitude.data, self.orig_dataset.event_latitude.data]).T)
        elif axis == 4:
            # Selection in time axis, filter by time and alt
            select_path_array[:, 0] = select_path_array[:, 0]*1e6
            select_path = mpl.path.Path(select_path_array, closed=True)
            points_in_selection = select_path.contains_points(np.array([self.orig_dataset.event_time.data.astype(float), self.orig_dataset.event_altitude.data]).T)
        if points_in_selection.sum() == 0:
            print('no points in selection')
            return
        points_in_selection = points_in_selection.reshape(1, -1)
        new_filter_history = np.append(self.filter_history, points_in_selection, axis=0)
        values_to_plot = np.prod(new_filter_history, axis=0)
        new_ds = self.orig_dataset.isel(number_of_events=values_to_plot.astype(bool))
        if new_ds.number_of_events.data.shape[0] == 0:
            print('no events in selection')
            return
        self.ds = new_ds
        self.html_pane.object = self.ds
        self.filter_history = new_filter_history
        self.datashade_label.value = f'Enable Datashader? ({self.ds.number_of_events.data.shape[0]} src)'
        poly_num = 0
        for pretty_str in self.filter_history_pretty:
            if 'Polygon' in pretty_str:
                poly_num = int(pretty_str.replace('Polygon #', ''))
        pretty_string = f'Polygon #{poly_num+1}'
        self.filter_history_pretty.append(pretty_string)
        self.event_filter_table[0].append(pn.pane.HTML(f'<h4>{pretty_string}</h4>', height=int(self.px_scale*6)))
        remove_button = pn.widgets.Button(icon='square-x-filled', button_type='danger', button_style='outline', width=int(self.px_scale*6), height=int(self.px_scale*6))
        this_remove = partial(self.remove_filter, value_to_remove=pretty_string)
        pn.bind(this_remove, unused=remove_button, watch=True)
        self.event_filter_table[1].append(remove_button)


    def mark_polygon(self, mark, unused):
        select_path_array = self.selection_geom[0]
        select_path = mpl.path.Path(self.selection_geom[0], closed=True)
        axis = self.selection_geom[-1]
        self.selection_geom = [np.array([]), 0]
        if select_path == np.array([]) or self.bad_selection_flag or axis == 0:
            return
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
            select_path_array[:, 0] = select_path_array[:, 0]*1e6
            select_path = mpl.path.Path(select_path_array, closed=True)
            points_in_selection = select_path.contains_points(np.array([self.ds.event_time.data.astype(float), self.ds.event_altitude.data]).T)
        if points_in_selection.sum() == 0:
            print('no points in selection')
            return
        points_to_mark = self.ds.isel(number_of_events=points_in_selection.astype(bool)).event_id.data
        self.ds['event_assigned_charge'].data[np.isin(self.ds['event_id'].data, points_to_mark)] = mark
        self.orig_dataset['event_assigned_charge'].data[np.isin(self.orig_dataset['event_id'].data, points_to_mark)] = mark


    def limit_to_filter(self, _):
        filter_var = self.event_filter_type_selector.value
        filter_op_str = self.event_filter_op_selector.value
        filter_val = self.event_filter_value_input.value
        variable_filtered_pretty = list(self.event_filter_type_selector.options.keys())[self.event_filter_type_selector.values.index(filter_var)]
        pretty_string = f'{variable_filtered_pretty} {filter_op_str} {filter_val}'
        if pretty_string in self.filter_history_pretty:
            print('already filtered by this')
            return
        if filter_op_str == '==':
            filter_op = np.equal
        elif filter_op_str == '>':
            filter_op = np.greater
        elif filter_op_str == '<':
            filter_op = np.less
        elif filter_op_str == '>=':
            filter_op = np.greater_equal
        elif filter_op_str == '<=':
            filter_op = np.less_equal
        elif filter_op_str == '!=':
            filter_op = np.not_equal
        filter_vals = self.orig_dataset[filter_var].data
        filter_result = filter_op(filter_vals, filter_val)
        new_filter_history = np.append(self.filter_history, filter_result.reshape(1, -1), axis=0)
        values_to_plot = np.prod(new_filter_history, axis=0)
        new_ds = self.orig_dataset.isel(number_of_events=values_to_plot.astype(bool))
        if new_ds.number_of_events.data.shape[0] == 0:
            print('no events in selection')
            return
        self.ds = new_ds
        self.html_pane.object = self.ds
        self.filter_history = new_filter_history
        self.datashade_label.value = f'Enable Datashader? ({self.ds.number_of_events.data.shape[0]} src)'
        self.filter_history_pretty.append(pretty_string)
        self.event_filter_table[0].append(pn.pane.HTML(f'<h4>{pretty_string}</h4>', height=int(self.px_scale*6)))
        remove_button = pn.widgets.Button(icon='square-x-filled', button_type='danger', button_style='outline', width=int(self.px_scale*6), height=int(self.px_scale*6))
        this_remove = partial(self.remove_filter, value_to_remove=pretty_string)
        pn.bind(this_remove, unused=remove_button, watch=True)
        self.event_filter_table[1].append(remove_button)


    def remove_filter(self, value_to_remove, unused):
        idx_to_remove = self.filter_history_pretty.index(value_to_remove)
        self.filter_history_pretty.pop(idx_to_remove)
        self.event_filter_table[0].pop(idx_to_remove)
        self.event_filter_table[1].pop(idx_to_remove)
        new_filter_history = np.delete(self.filter_history, idx_to_remove+1, axis=0)
        values_to_plot = np.prod(new_filter_history, axis=0)
        new_ds = self.orig_dataset.isel(number_of_events=values_to_plot.astype(bool))
        self.ds = new_ds
        self.html_pane.object = self.ds
        self.filter_history = new_filter_history
        self.datashade_label.value = f'Enable Datashader? ({self.ds.number_of_events.data.shape[0]} src)'


    def hook_hist_src_limiter(self, plot, element):
        init_xmax = plot.state.x_range.end
        plot.state.x_range = Range1d(0, init_xmax, bounds=(0, init_xmax))

    def hook_yalt_limiter(self, plot, element):
        plot.state.select_one(WheelZoomTool).maintain_focus = False
        plot.state.y_range = self.init_alt_range

    def hook_xalt_limiter(self, plot, element):
        plot.state.select_one(WheelZoomTool).maintain_focus = False
        plot.state.x_range = self.init_alt_range

    def hook_time_limiter(self, plot, element):
        if type(plot.state.x_range.start) == float:
            if plot.state.x_range.start < self.time_range[0]:
                plot.state.x_range.start = self.time_range[0]
            if plot.state.x_range.end > self.time_range[1]:
                plot.state.x_range.end = self.time_range[1]

    def hook_xlabel_rotate(self, plot, element):
        plot.state.xaxis.major_label_orientation = -np.pi/2


    def plan_range_handle(self, x_range, y_range):
        self.xlim = x_range
        self.ylim = y_range

    
    def lonalt_range_handle(self, x_range, y_range):
        self.xlim = x_range
        self.zlim = y_range

    
    def latalt_range_handle(self, x_range, y_range):
        self.zlim = x_range
        self.ylim = y_range


    def should_show_datashaded_points(self, color_by_match, color_by, should_datashade):
        if should_datashade:
            if color_by == color_by_match:
                return True
        return False


    def things_to_plot(self, should_timefloat, ax):
        if ax == 'plan':
            data = [self.ds.event_longitude, self.ds.event_latitude, self.ds.event_altitude, self.ds.event_time, self.ds.event_power, self.ds.event_assigned_charge]
            kdims = ['Longitude', 'Latitude']
            vdims = ['Altitude', 'Time', 'Power', 'Charge (User Assigned)']
        elif ax == 'lonalt':
            data = [self.ds.event_longitude, self.ds.event_altitude, self.ds.event_latitude, self.ds.event_time, self.ds.event_power, self.ds.event_assigned_charge]
            kdims = ['Longitude', 'Altitude']
            vdims = ['Latitude', 'Time', 'Power', 'Charge (User Assigned)']
        elif ax == 'latalt':
            data = [self.ds.event_altitude, self.ds.event_latitude, self.ds.event_longitude, self.ds.event_time, self.ds.event_power, self.ds.event_assigned_charge]
            kdims = ['Altitude', 'Latitude']
            vdims = ['Longitude', 'Time', 'Power', 'Charge (User Assigned)']
        elif ax == 'alttime':
            data = [self.ds.event_time, self.ds.event_altitude, self.ds.event_longitude, self.ds.event_latitude, self.ds.event_power, self.ds.event_assigned_charge]
            kdims = ['Time', 'Altitude']
            vdims = ['Longitude', 'Latitude', 'Power', 'Charge (User Assigned)']
        if should_timefloat:
            timefloats = color_by_time(self.ds.event_time.values, tlim=self.time_range_dt)[-1]
            data.append(timefloats)
            vdims.append('Seconds since start')
        return tuple(data), kdims, vdims


    def plot_points_datashaded(self, ax, color_by):
        should_i_timefloat = False
        if color_by == 'Time':
            should_i_timefloat = True
            agg = datashader.max('Seconds since start')
            this_cmap = 'rainbow'
            this_cnorm = 'linear'
        elif color_by == 'Charge (User Assigned)':
            agg = datashader.by('Charge (User Assigned)', datashader.max('Charge (User Assigned)'))
            this_cmap = ['blue', 'green', 'red']
            this_cnorm = 'linear'
        elif color_by == 'Charge (chargepol)':
            pass
        elif color_by == 'Power (dBW)':
            agg = datashader.max('Power')
            this_cmap = 'rainbow'
            this_cnorm = 'linear'
        elif color_by == 'Event Density':
            agg = datashader.count()
            this_cmap = 'rainbow'
            this_cnorm = 'linear'
        elif color_by == 'Log Event Density':
            agg = datashader.count()
            this_cmap = 'rainbow'
            this_cnorm = 'log'
        elif color_by == 'Altitude':
            agg = datashader.max('Altitude')
            this_cmap = 'rainbow'
            this_cnorm = 'linear'
        data, kdims, vdims = self.things_to_plot(should_i_timefloat, ax)
        points = hv.Points(data, kdims=kdims, vdims=vdims)
        shaded = hv.operation.datashader.datashade(points, aggregator=agg, cmap=this_cmap, cnorm=this_cnorm, dynamic=True).opts(tools=['hover'])
        if ax == 'plan':
            this_proj = ccrs.PlateCarree()
            this_xlim = self.xlim
            this_ylim = self.ylim
            this_width = self.px_scale*self.plan_edge_length
            this_height = self.px_scale*self.plan_edge_length
            this_hooks = []
        elif ax == 'lonalt':
            this_proj = None
            this_xlim = self.xlim
            this_ylim = self.zlim
            this_width = self.px_scale*self.plan_edge_length
            this_height = self.px_scale*self.hist_edge_length
            this_hooks = [self.hook_yalt_limiter]
        elif ax == 'latalt':
            this_proj = None
            this_xlim = self.zlim
            this_ylim = self.ylim
            this_width = self.px_scale*self.hist_edge_length
            this_height = self.px_scale*self.plan_edge_length
            this_hooks = [self.hook_xlabel_rotate, self.hook_xalt_limiter]
        elif ax == 'alttime':
            this_proj = None
            this_xlim = (self.ds.event_time.data[0], self.ds.event_time.data[-1])
            this_ylim = self.zlim
            this_width = self.px_scale*(self.plan_edge_length+self.hist_edge_length)
            this_height = self.px_scale*self.hist_edge_length
            this_hooks = [self.hook_yalt_limiter, self.hook_time_limiter]
        shaded = shaded.opts(projection=this_proj, xlim=this_xlim, ylim=this_ylim, width=this_width, height=this_height, hooks=this_hooks)
        return shaded

    def plot_planview_points(self, color_by, should_datashade, filter_history):
        should_i_timefloat = False
        if color_by == 'Time':
            should_i_timefloat = True
            agg = 'Seconds since start'
            cmap = 'rainbow'
            clim = (np.nan, np.nan)
        elif color_by == 'Charge (User Assigned)':
            agg = 'Charge (User Assigned)'
            cmap = ['blue', 'green', 'red']
            clim = (-1, 1)
        elif color_by == 'Charge (chargepol)':
            pass
        elif color_by == 'Power (dBW)':
            agg = 'Power'
            cmap = 'rainbow'
            clim = (np.nan, np.nan)
        elif color_by == 'Altitude':
            agg = 'Altitude'
            cmap = 'rainbow'
            clim = (np.nan, np.nan)
        elif 'Event Density' in color_by:
            agg = None
            cmap = 'rainbow'
            clim = (np.nan, np.nan)
            latlonbinwidth = 0.01
            latmin = (np.min(self.ds.event_latitude.data) // latlonbinwidth )*latlonbinwidth
            latmax = (np.max(self.ds.event_latitude.data) // latlonbinwidth + 1)*latlonbinwidth
            lonmin = (np.min(self.ds.event_longitude.data) // latlonbinwidth )*latlonbinwidth
            lonmax = (np.max(self.ds.event_longitude.data) // latlonbinwidth + 1)*latlonbinwidth
            latbins = np.arange(latmin, latmax, latlonbinwidth)
            lonbins = np.arange(lonmin, lonmax, latlonbinwidth)
            hist, _, _ = np.histogram2d(self.ds.event_longitude.data, self.ds.event_latitude.data, bins=[lonbins, latbins])
            hist[hist == 0] = np.nan
            if 'Log' in color_by:
                this_cnorm = 'log'
            else:
                this_cnorm = 'linear'
            img = hv.Image((lonbins[:-1]+latlonbinwidth/2, latbins[:-1]+latlonbinwidth/2, hist.T), kdims=['Longitude', 'Latitude'], vdims=['Event Density']
                                ).opts(hv.opts.Image(cmap=cmap, cnorm=this_cnorm, clim=clim)).opts(projection=ccrs.PlateCarree(), xlim=self.xlim, ylim=self.ylim,
                                                                                             width=self.px_scale*self.plan_edge_length, height=self.px_scale*self.plan_edge_length, tools=['hover'], visible=not should_datashade)
        if agg is None:
            points = hv.Points(([0], [0], [0]), kdims=['Longitude', 'Latitude'], vdims=['Event Density']).opts(visible=False)
        else:
            data, kdims, vdims = self.things_to_plot(should_i_timefloat, 'plan')
            points = hv.Points(data, kdims=kdims, vdims=vdims)
            points = points.opts(hv.opts.Points(color=agg, cmap=cmap, clim=clim, size=5)).opts(projection=ccrs.PlateCarree(), xlim=self.xlim, ylim=self.ylim, width=self.px_scale*self.plan_edge_length, height=self.px_scale*self.plan_edge_length, tools=['hover'], line_color='black', visible=not should_datashade)
            img = hv.Image((np.array([0, 1]), np.array([0, 1]), np.array([[0,0],[0,0]])), kdims=kdims, vdims=[agg]).opts(visible=False)
        return points * img

    def plot_lonalt_points(self, color_by, should_datashade, filter_history):
        should_i_timefloat = False
        if color_by == 'Time':
            should_i_timefloat = True
            agg = 'Seconds since start'
            cmap = 'rainbow'
            clim = (np.nan, np.nan)
        elif color_by == 'Charge (User Assigned)':
            agg = 'Charge (User Assigned)'
            cmap = ['blue', 'green', 'red']
            clim = (-1, 1)
        elif color_by == 'Charge (chargepol)':
            pass
        elif color_by == 'Power (dBW)':
            agg = 'Power'
            cmap = 'rainbow'
            clim = (np.nan, np.nan)
        elif color_by == 'Altitude':
            agg = 'Altitude'
            cmap = 'rainbow'
            clim = (np.nan, np.nan)
        elif 'Event Density' in color_by:
            agg = None
            cmap = 'rainbow'
            clim = (np.nan, np.nan)
            latlonbinwidth = 0.01
            lonmin = (np.min(self.ds.event_longitude.data) // latlonbinwidth )*latlonbinwidth
            lonmax = (np.max(self.ds.event_longitude.data) // latlonbinwidth + 1)*latlonbinwidth
            lonbins = np.arange(lonmin, lonmax, latlonbinwidth)
            altbinwidth = 100
            altmin = 0
            altmax = (np.max(self.ds.event_altitude.data) // altbinwidth + 1)*altbinwidth
            altbins = np.arange(altmin, altmax, altbinwidth)
            hist, _, _ = np.histogram2d(self.ds.event_longitude.data, self.ds.event_altitude.data, bins=[lonbins, altbins])
            hist[hist == 0] = np.nan
            if 'Log' in color_by:
                this_cnorm = 'log'
            else:
                this_cnorm = 'linear'
            img = hv.Image((lonbins[:-1]+latlonbinwidth/2, altbins[:-1]+altbinwidth/2, hist.T), kdims=['Longitude', 'Altitude'], vdims=['Event Density']
                                ).opts(hv.opts.Image(cmap=cmap, cnorm=this_cnorm, clim=clim)).opts(xlim=self.xlim, ylim=self.zlim, width=self.px_scale*self.plan_edge_length, height=self.px_scale*self.hist_edge_length, tools=['hover'], hooks=[self.hook_yalt_limiter], visible=not should_datashade)
        if agg is None:
            points = hv.Points(([0], [0], [0]), kdims=['Longitude', 'Altitude'], vdims=['Event Density']).opts(visible=False)
        else:
            data, kdims, vdims = self.things_to_plot(should_i_timefloat, 'lonalt')
            points = hv.Points(data, kdims=kdims, vdims=vdims)
            points = points.opts(hv.opts.Points(color=agg, cmap=cmap, clim=clim, size=5)).opts(xlim=self.xlim, ylim=self.zlim, width=self.px_scale*self.plan_edge_length, height=self.px_scale*self.hist_edge_length, tools=['hover'], line_color='black', hooks=[self.hook_yalt_limiter], visible=not should_datashade)
            img = hv.Image((np.array([0, 1]), np.array([0, 1]), np.array([[0,0],[0,0]])), kdims=kdims, vdims=[agg]).opts(visible=False)
        return points * img
    

    def plot_alt_hist(self):
        return hv.Histogram(np.histogram(self.ds.event_altitude.data, bins=np.arange(0, 20001, 1000)), kdims=['Altitude'], vdims=['src']).opts(width=self.px_scale*self.hist_edge_length, height=self.px_scale*self.hist_edge_length, invert_axes=True).opts(hooks=[self.hook_xlabel_rotate, self.hook_hist_src_limiter, self.hook_yalt_limiter])


    def plot_latalt_points(self, color_by, should_datashade, filter_history):
        should_i_timefloat = False
        if color_by == 'Time':
            should_i_timefloat = True
            agg = 'Seconds since start'
            cmap = 'rainbow'
            clim = (np.nan, np.nan)
        elif color_by == 'Charge (User Assigned)':
            agg = 'Charge (User Assigned)'
            cmap = ['blue', 'green', 'red']
            clim = (-1, 1)
        elif color_by == 'Charge (chargepol)':
            pass
        elif color_by == 'Power (dBW)':
            agg = 'Power'
            cmap = 'rainbow'
            clim = (np.nan, np.nan)
        elif color_by == 'Altitude':
            agg = 'Altitude'
            cmap = 'rainbow'
            clim = (np.nan, np.nan)
        elif 'Event Density' in color_by:
            agg = None
            cmap = 'rainbow'
            clim = (np.nan, np.nan)
            latlonbinwidth = 0.01
            latmin = (np.min(self.ds.event_latitude.data) // latlonbinwidth )*latlonbinwidth
            latmax = (np.max(self.ds.event_latitude.data) // latlonbinwidth + 1)*latlonbinwidth
            latbins = np.arange(latmin, latmax, latlonbinwidth)
            altbinwidth = 100
            altmin = 0
            altmax = (np.max(self.ds.event_altitude.data) // altbinwidth + 1)*altbinwidth
            altbins = np.arange(altmin, altmax, altbinwidth)
            hist, _, _ = np.histogram2d(self.ds.event_altitude.data, self.ds.event_latitude.data, bins=[altbins, latbins])
            hist[hist == 0] = np.nan
            if 'Log' in color_by:
                this_cnorm = 'log'
            else:
                this_cnorm = 'linear'
            img = hv.Image((altbins[:-1]+altbinwidth/2, latbins[:-1]+latlonbinwidth/2, hist.T), kdims=['Altitude', 'Latitude'], vdims=['Event Density']
                                ).opts(hv.opts.Image(cmap=cmap, cnorm=this_cnorm, clim=clim)).opts(xlim=self.zlim, ylim=self.ylim, width=self.px_scale*self.hist_edge_length, height=self.px_scale*self.plan_edge_length, tools=['hover'], hooks=[self.hook_xlabel_rotate, self.hook_xalt_limiter], visible=not should_datashade)
        if agg is None:
            points = hv.Points(([0], [0], [0]), kdims=['Altitude', 'Latitude'], vdims=['Event Density']).opts(visible=False)
        else:
            data, kdims, vdims = self.things_to_plot(should_i_timefloat, 'latalt')
            points = hv.Points(data, kdims=kdims, vdims=vdims)
            points = points.opts(hv.opts.Points(color=agg, cmap=cmap, clim=clim, size=5)).opts(xlim=self.zlim, ylim=self.ylim, width=self.px_scale*self.hist_edge_length, height=self.px_scale*self.plan_edge_length, tools=['hover'], line_color='black', hooks=[self.hook_xlabel_rotate, self.hook_xalt_limiter], visible=not should_datashade)
            img = hv.Image((np.array([0, 1]), np.array([0, 1]), np.array([[0,0],[0,0]])), kdims=kdims, vdims=[agg]).opts(visible=False)
        return points * img


    def plot_alttime_points(self, color_by, should_datashade, filter_history):
        should_i_timefloat = False
        if color_by == 'Time':
            should_i_timefloat = True
            agg = 'Seconds since start'
            cmap = 'rainbow'
            clim = (np.nan, np.nan)
        elif color_by == 'Charge (User Assigned)':
            agg = 'Charge (User Assigned)'
            cmap = ['blue', 'green', 'red']
            clim = (-1, 1)
        elif color_by == 'Charge (chargepol)':
            pass
        elif color_by == 'Power (dBW)':
            agg = 'Power'
            cmap = 'rainbow'
            clim = (np.nan, np.nan)
        elif color_by == 'Altitude':
            agg = 'Altitude'
            cmap = 'rainbow'
            clim = (np.nan, np.nan)
        elif 'Event Density' in color_by:
            agg = None
            cmap = 'rainbow'
            clim = (np.nan, np.nan)
            timebinwidth = 1e9
            tmin = (np.min(self.ds.event_time.data.astype(float)) // timebinwidth )*timebinwidth
            tmax = (np.max(self.ds.event_time.data.astype(float)) // timebinwidth + 1)*timebinwidth
            timebins = np.arange(tmin, tmax, timebinwidth)
            timebins_ctr = timebins[:-1] + timebinwidth/2
            timebins_ctr_dt = timebins_ctr.astype('datetime64[ns]')
            altbinwidth = 100
            altmin = 0
            altmax = (np.max(self.ds.event_altitude.data) // altbinwidth + 1)*altbinwidth
            altbins = np.arange(altmin, altmax, altbinwidth)
            hist, _, _ = np.histogram2d(self.ds.event_time.data.astype(float), self.ds.event_altitude.data, bins=[timebins, altbins])
            hist[hist == 0] = np.nan
            if 'Log' in color_by:
                this_cnorm = 'log'
            else:
                this_cnorm = 'linear'
            img = hv.Image((timebins_ctr_dt, altbins[:-1]+altbinwidth/2, hist.T), kdims=['Time', 'Altitude'], vdims=['Event Density']
                                ).opts(hv.opts.Image(cmap=cmap, cnorm=this_cnorm, clim=clim)).opts(xlim=(self.ds.event_time.data[0], self.ds.event_time.data[-1]), ylim=self.zlim, width=self.px_scale*(self.plan_edge_length+self.hist_edge_length), height=self.px_scale*self.hist_edge_length, tools=['hover'], hooks=[self.hook_yalt_limiter, self.hook_time_limiter], visible=not should_datashade)
        if agg is None:
            points = hv.Points(([self.ds.event_time.data[0]], [0], [0]), kdims=['Time', 'Altitude'], vdims=['Event Density']).opts(visible=False)
        else:
            data, kdims, vdims = self.things_to_plot(should_i_timefloat, 'alttime')
            points = hv.Points(data, kdims=kdims, vdims=vdims)
            points = points.opts(hv.opts.Points(color=agg, cmap=cmap, clim=clim, size=5)).opts(xlim=(self.ds.event_time.data[0], self.ds.event_time.data[-1]), ylim=self.zlim, width=self.px_scale*(self.plan_edge_length+self.hist_edge_length), height=self.px_scale*self.hist_edge_length, tools=['hover'], line_color='black', hooks=[self.hook_yalt_limiter, self.hook_time_limiter], visible=not should_datashade)
            img = hv.Image((np.array([self.ds.event_time.data[0], self.ds.event_time.data[-1]]), np.array([0, 1]), np.array([[0,0],[0,0]])), kdims=kdims, vdims=[agg]).opts(visible=False)
        return points * img
    

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
            self.bad_selection_flag = False
            return
        if self.selection_geom[-1] != 0 and self.last_mouse_coord[-2] != 0 and self.last_mouse_coord[-2] != self.selection_geom[-1]:
            # selection is in a different axis than the last selection
            self.bad_selection_flag = True
            return
        this_selection_geom = np.array([data['xs'], data['ys']]).T[:, 0, :]
        self.selection_geom = [this_selection_geom, self.last_mouse_coord[-2]]

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


    def init_plot(self, color_by_dropdown=None, datashade_switch=None):
        self.xlim = (self.ds.network_center_longitude.data - 3, self.ds.network_center_longitude.data + 3)
        self.ylim = (self.ds.network_center_latitude.data - 3, self.ds.network_center_latitude.data + 3)
        self.zlim = (0, 20000)

        self.init_lon_range = Range1d(self.xlim[0], self.xlim[1])
        self.init_lat_range = Range1d(self.ylim[0], self.ylim[1])
        self.init_alt_range = Range1d(self.zlim[0], self.zlim[1], bounds=(0, 100000))

        plan_points = hv.DynamicMap(pn.bind(self.plot_planview_points, color_by=color_by_dropdown, should_datashade=datashade_switch, filter_history=self.filter_history, watch=True))
        plan_ax_polys = hv.Polygons([]).opts(hv.opts.Polygons(fill_alpha=0.3, fill_color='black'))
        plan_ax_selector = hv.streams.PolyDraw(source=plan_ax_polys, drag=False, num_objects=1, show_vertices=True, vertex_style={'size': 5, 'fill_color': 'white', 'line_color' : 'black'})
        plan_ax_selector.add_subscriber(self.handle_selection)
        plan_ax_select_area = hv.DynamicMap(self.plan_ax_highlighter)
        self.plan_range_stream = hv.streams.RangeXY(source=plan_points)
        self.plan_range_stream.add_subscriber(self.plan_range_handle)
        plan_pointer_src = hv.Points(([0], [0], [0]), kdims=['Longitude', 'Latitude'], vdims=['Point Source']).opts(visible=False)
        plan_ax_pointer = hv.streams.PointerXY(x=0, y=0, source=plan_pointer_src).rename(x='plan_x', y='plan_y')

        lon_alt_points = hv.DynamicMap(pn.bind(self.plot_lonalt_points, color_by=color_by_dropdown, should_datashade=datashade_switch, filter_history=self.filter_history, watch=True))
        lon_alt_ax_polys = hv.Polygons([]).opts(hv.opts.Polygons(fill_alpha=0.3, fill_color='black'))
        lon_alt_ax_selector = hv.streams.PolyDraw(source=lon_alt_ax_polys, drag=False, num_objects=1, show_vertices=True, vertex_style={'size': 5, 'fill_color': 'white', 'line_color' : 'black'})
        lon_alt_ax_selector.add_subscriber(self.handle_selection)
        lon_alt_select_area = hv.DynamicMap(self.lon_ax_highlighter)
        self.lon_alt_range_stream = hv.streams.RangeXY(source=lon_alt_points)
        self.lon_alt_range_stream.add_subscriber(self.lonalt_range_handle)
        lon_pointer_src = hv.Points(([0], [0], [0]), kdims=['Longitude', 'Altitude'], vdims=['Point Source']).opts(visible=False)
        lon_alt_ax_pointer = hv.streams.PointerXY(x=0, y=0, source=lon_pointer_src).rename(x='lon_x', y='lon_y')


        hist_ax = hv.DynamicMap(self.plot_alt_hist)

        lat_alt_points = hv.DynamicMap(pn.bind(self.plot_latalt_points, color_by=color_by_dropdown, should_datashade=datashade_switch, filter_history=self.filter_history, watch=True))
        lat_alt_ax_polys = hv.Polygons([]).opts(hv.opts.Polygons(fill_alpha=0.3, fill_color='black'))
        lat_alt_ax_selector = hv.streams.PolyDraw(source=lat_alt_ax_polys, drag=False, num_objects=1, show_vertices=True, vertex_style={'size': 5, 'fill_color': 'white', 'line_color' : 'black'})
        lat_alt_ax_selector.add_subscriber(self.handle_selection)
        lat_alt_select_area = hv.DynamicMap(self.lat_ax_highlighter)
        self.lat_alt_range_stream = hv.streams.RangeXY(source=lat_alt_points)
        self.lat_alt_range_stream.add_subscriber(self.latalt_range_handle)
        lat_pointer_src = hv.Points(([0], [0], [0]), kdims=['Altitude', 'Latitude'], vdims=['Point Source']).opts(visible=False)
        lat_alt_ax_pointer = hv.streams.PointerXY(x=0, y=0, source=lat_pointer_src).rename(x='lat_x', y='lat_y')


        alt_time_points = hv.DynamicMap(pn.bind(self.plot_alttime_points, color_by=color_by_dropdown, should_datashade=datashade_switch, filter_history=self.filter_history, watch=True))
        alt_time_ax_polys = hv.Polygons([]).opts(hv.opts.Polygons(fill_alpha=0.3, fill_color='black'))
        alt_time_ax_selector = hv.streams.PolyDraw(source=alt_time_ax_polys, drag=False, num_objects=1, show_vertices=True, vertex_style={'size': 5, 'fill_color': 'white', 'line_color' : 'black'})
        alt_time_ax_selector.add_subscriber(self.handle_selection)
        alt_time_select_area = hv.DynamicMap(self.time_ax_highlighter)
        alt_time_pointer_src = hv.Points(([self.ds.event_time.data[0]], [0], [0]), kdims=['Time', 'Altitude'], vdims=['Point Source']).opts(visible=False)
        alt_time_pointer = hv.streams.PointerXY(x=0, y=0, source=alt_time_pointer_src).rename(x='time_x', y='time_y')

        points_shaded = []
        for ax in ['plan', 'lonalt', 'latalt', 'alttime']:
            this_ax_points_shaded = []
            for colorshade in color_by_dropdown.options:
                if colorshade in ['Charge (User Assigned)', 'Charge (chargepol)']:
                    continue
                points = self.plot_points_datashaded(ax, colorshade)
                if colorshade != color_by_dropdown.value:
                    points.opts(visible=False)
                should_this_display = pn.bind(self.should_show_datashaded_points, color_by_match=colorshade, color_by=color_by_dropdown, should_datashade=datashade_switch, watch=True)
                pn.bind(points.opts, visible=should_this_display, watch=True)
                this_ax_points_shaded.append(points)
            this_ax_points_shaded = reduce(lambda x, y: x*y, this_ax_points_shaded)
            points_shaded.append(this_ax_points_shaded)

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
        
        self.plan_ax_pointer = plan_ax_pointer
        self.lon_alt_ax_pointer = lon_alt_ax_pointer
        self.lat_alt_ax_pointer = lat_alt_ax_pointer
        self.alt_time_pointer = alt_time_pointer

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

        counties_shp = shapefile.Reader('ne_10m_admin_2_counties.shp').shapes()
        counties_shp = [shape(counties_shp[i]) for i in range(len(counties_shp))]
        self.plan_ax_bg = gv.Path(counties_shp).opts(color='gray') * gf.borders().opts(color='black') * gf.states().opts(color='black', line_width=2)

        lon_alt_ax = (lon_pointer_src * lon_alt_points * points_shaded[1] * lon_ax_crosshair * lon_alt_ax_polys * lon_alt_select_area)
        plan_ax = (plan_pointer_src * plan_points * points_shaded[0] * plan_ax_crosshair * plan_ax_polys * plan_ax_select_area * self.plan_ax_bg)
        lat_alt_ax = (lat_pointer_src * lat_alt_points * points_shaded[2] * lat_ax_crosshair * lat_alt_ax_polys * lat_alt_select_area)
        alt_time_ax = pn.pane.HoloViews(alt_time_pointer_src * alt_time_points * points_shaded[3] * time_ax_crosshair * alt_time_ax_polys * alt_time_select_area)

        the_lower_part = (lon_alt_ax + hist_ax + plan_ax + lat_alt_ax).cols(2)
        the_lower_part = pn.pane.HoloViews(the_lower_part)

        netw_name = 'LYLOUT'
        if 'station_network' in self.ds.keys():
            netw_name = [s for s in self.ds.station_network.data[0].decode('utf-8') if s.isalpha()]
            netw_name = ''.join(netw_name)
        else:
            filename = self.filenames
            if type(self.filenames) != str:
                filename = filename[0]
            filename = path.basename(filename)
            splitfile = filename.split('_')
            if len(splitfile) >= 2:
                netw_name = splitfile[0]
        for i in range(1, len(netw_name)):
            if netw_name[i-1].islower() and netw_name[i].isupper():
                netw_name = netw_name[:i] + ' ' + netw_name[i:]

        netw_name = netw_name.replace('LMA', '').replace('_', ' ').replace('  ', ' ')


        title = pn.pane.HTML(f'<h2>{netw_name} LMA on {self.time_range_py_dt[0].strftime("%d %b %Y")}</h2>')#, styles={'text-align': 'center'})

        self.panelHandle  = pn.Column(title, alt_time_ax, the_lower_part)


