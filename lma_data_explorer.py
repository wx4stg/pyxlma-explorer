import numpy as np
from bokeh.models import Range1d, WheelZoomTool
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

from functools import reduce



class LMADataExplorer:
    def __init__(self, filename, color_by_dropdown=None, datashade_switch=None):
        self.filename = filename
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
        self.time_range_py_dt = [start_time, end_time]
        self.time_range_dt = np.array(self.time_range_py_dt).astype('datetime64')
        self.time_range = self.time_range_dt.astype(float)/1e3
        self.lon_range = Range1d(self.ds.network_center_longitude.data - 1, self.ds.network_center_longitude.data + 1)
        self.lat_range = Range1d(self.ds.network_center_latitude.data - 1, self.ds.network_center_latitude.data + 1)
        self.init_plot(color_by_dropdown, datashade_switch)

    def limit_to_polygon(self):
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
        self.rerender()


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

    def plot_planview_points_datashaded(self, color_by):
        if color_by == 'Time':
            timefloats = color_by_time(self.ds.event_time.values, tlim=self.time_range_dt)[-1]
            points = hv.Points((self.ds.event_longitude.data, self.ds.event_latitude.data, self.ds.event_altitude.data, self.ds.event_time.data, timefloats, self.ds.event_power), kdims=['Longitude', 'Latitude'], vdims=['Altitude', 'Time', 'Seconds since start', 'Power'])
            shaded = hv.operation.datashader.datashade(points, aggregator=datashader.max('Seconds since start'), cmap='rainbow', dynamic=True).opts(projection=ccrs.PlateCarree(), xlim=self.xlim, ylim=self.ylim, width=self.px_scale*self.plan_edge_length, height=self.px_scale*self.plan_edge_length, tools=['hover'])
        elif color_by == 'Charge (User Assigned)':
            pass
        elif color_by == 'Charge (chargepol)':
            pass
        elif color_by == 'Power (dBW)':
            points = hv.Points((self.ds.event_longitude.data, self.ds.event_latitude.data, self.ds.event_altitude.data, self.ds.event_time.data, self.ds.event_power), kdims=['Longitude', 'Latitude'], vdims=['Altitude', 'Time', 'Power'])
            shaded = hv.operation.datashader.datashade(points, aggregator=datashader.max('Power'), cmap='rainbow', dynamic=True).opts(projection=ccrs.PlateCarree(), xlim=self.xlim, ylim=self.ylim, width=self.px_scale*self.plan_edge_length, height=self.px_scale*self.plan_edge_length, tools=['hover'])
        elif color_by == 'Event Density':
            points = hv.Points((self.ds.event_longitude.data, self.ds.event_latitude.data, self.ds.event_altitude.data, self.ds.event_time.data, self.ds.event_power), kdims=['Longitude', 'Latitude'], vdims=['Altitude', 'Time', 'Power'])
            shaded = hv.operation.datashader.datashade(points, aggregator=datashader.count(), cmap='rainbow', dynamic=True).opts(projection=ccrs.PlateCarree(), xlim=self.xlim, ylim=self.ylim, width=self.px_scale*self.plan_edge_length, height=self.px_scale*self.plan_edge_length, tools=['hover'])
        elif color_by == 'Log Event Density':
            points = hv.Points((self.ds.event_longitude.data, self.ds.event_latitude.data, self.ds.event_altitude.data, self.ds.event_time.data, self.ds.event_power), kdims=['Longitude', 'Latitude'], vdims=['Altitude', 'Time', 'Power'])
            shaded = hv.operation.datashader.datashade(points, aggregator=datashader.count(), cmap='rainbow', cnorm='log', dynamic=True).opts(projection=ccrs.PlateCarree(), xlim=self.xlim, ylim=self.ylim, width=self.px_scale*self.plan_edge_length, height=self.px_scale*self.plan_edge_length, tools=['hover'])
        elif color_by == 'Altitude':
            points = hv.Points((self.ds.event_longitude.data, self.ds.event_latitude.data, self.ds.event_altitude.data, self.ds.event_time.data, self.ds.event_power), kdims=['Longitude', 'Latitude'], vdims=['Altitude', 'Time', 'Power'])
            shaded = hv.operation.datashader.datashade(points, aggregator=datashader.max('Altitude'), cmap='rainbow', dynamic=True).opts(projection=ccrs.PlateCarree(), xlim=self.xlim, ylim=self.ylim, width=self.px_scale*self.plan_edge_length, height=self.px_scale*self.plan_edge_length, tools=['hover'])
        return shaded

    def plot_planview_points(self, color_by, should_datashade):
        if color_by == 'Time':
            timefloats = color_by_time(self.ds.event_time.values, tlim=self.time_range_dt)[-1]
            points = hv.Points((self.ds.event_longitude.data, self.ds.event_latitude.data, self.ds.event_altitude.data, self.ds.event_time.data, timefloats, self.ds.event_power), kdims=['Longitude', 'Latitude'], vdims=['Altitude', 'Time', 'Seconds since start', 'Power'])
            points = points.opts(hv.opts.Points(color='Seconds since start', cmap='rainbow', size=5)).opts(projection=ccrs.PlateCarree(), xlim=self.xlim, ylim=self.ylim, width=self.px_scale*self.plan_edge_length, height=self.px_scale*self.plan_edge_length, tools=['hover'], line_color='black', visible=not should_datashade)
        elif color_by == 'Charge (User Assigned)':
            pass
        elif color_by == 'Charge (chargepol)':
            pass
        elif color_by == 'Power (dBW)':
            points = hv.Points((self.ds.event_longitude.data, self.ds.event_latitude.data, self.ds.event_altitude.data, self.ds.event_time.data, self.ds.event_power), kdims=['Longitude', 'Latitude'], vdims=['Altitude', 'Time', 'Power'])
            points = points.opts(hv.opts.Points(color='Power', cmap='rainbow', size=5)).opts(projection=ccrs.PlateCarree(), xlim=self.xlim, ylim=self.ylim, width=self.px_scale*self.plan_edge_length, height=self.px_scale*self.plan_edge_length, tools=['hover'], line_color='black', visible=not should_datashade)
        elif color_by == 'Event Density':
            latlonbinwidth = 0.01
            latmin = (np.min(self.ds.event_latitude.data) // latlonbinwidth )*latlonbinwidth
            latmax = (np.max(self.ds.event_latitude.data) // latlonbinwidth + 1)*latlonbinwidth
            lonmin = (np.min(self.ds.event_longitude.data) // latlonbinwidth )*latlonbinwidth
            lonmax = (np.max(self.ds.event_longitude.data) // latlonbinwidth + 1)*latlonbinwidth
            latbins = np.arange(latmin, latmax, latlonbinwidth)
            lonbins = np.arange(lonmin, lonmax, latlonbinwidth)
            hist, _, _ = np.histogram2d(self.ds.event_longitude.data, self.ds.event_latitude.data, bins=[lonbins, latbins])
            hist[hist == 0] = np.nan
            points = hv.Image((lonbins[:-1]+latlonbinwidth/2, latbins[:-1]+latlonbinwidth/2, hist.T), kdims=['Longitude', 'Latitude'], vdims=['Event Density']
                                ).opts(hv.opts.Image(cmap='rainbow')).opts(projection=ccrs.PlateCarree(), xlim=self.xlim, ylim=self.ylim, width=self.px_scale*self.plan_edge_length, height=self.px_scale*self.plan_edge_length, tools=['hover'], visible=not should_datashade)
        elif color_by == 'Log Event Density':
            latlonbinwidth = 0.01
            latmin = (np.min(self.ds.event_latitude.data) // latlonbinwidth )*latlonbinwidth
            latmax = (np.max(self.ds.event_latitude.data) // latlonbinwidth + 1)*latlonbinwidth
            lonmin = (np.min(self.ds.event_longitude.data) // latlonbinwidth )*latlonbinwidth
            lonmax = (np.max(self.ds.event_longitude.data) // latlonbinwidth + 1)*latlonbinwidth
            latbins = np.arange(latmin, latmax, latlonbinwidth)
            lonbins = np.arange(lonmin, lonmax, latlonbinwidth)
            hist, _, _ = np.histogram2d(self.ds.event_longitude.data, self.ds.event_latitude.data, bins=[lonbins, latbins])
            hist[hist == 0] = np.nan
            points = hv.Image((lonbins[:-1]+latlonbinwidth/2, latbins[:-1]+latlonbinwidth/2, hist.T), kdims=['Longitude', 'Latitude'], vdims=['Event Density']
                                ).opts(hv.opts.Image(cmap='rainbow', cnorm='log')).opts(projection=ccrs.PlateCarree(), xlim=self.xlim, ylim=self.ylim, width=self.px_scale*self.plan_edge_length, height=self.px_scale*self.plan_edge_length, tools=['hover'], visible=not should_datashade)
        elif color_by == 'Altitude':
            points = hv.Points((self.ds.event_longitude.data, self.ds.event_latitude.data, self.ds.event_altitude.data, self.ds.event_time.data, self.ds.event_power), kdims=['Longitude', 'Latitude'], vdims=['Altitude', 'Time', 'Power'])
            points = points.opts(hv.opts.Points(color='Altitude', cmap='rainbow', size=5)).opts(projection=ccrs.PlateCarree(), xlim=self.xlim, ylim=self.ylim, width=self.px_scale*self.plan_edge_length, height=self.px_scale*self.plan_edge_length, tools=['hover'], line_color='black', visible=not should_datashade)
        return points
    

    def plot_lonalt_points_datashaded(self, color_by):
        if color_by == 'Time':
            timefloats = color_by_time(self.ds.event_time.values, tlim=self.time_range_dt)[-1]
            points = hv.Points((self.ds.event_longitude.data, self.ds.event_altitude.data, self.ds.event_time.data, timefloats, self.ds.event_power), kdims=['Longitude', 'Altitude'], vdims=['Time', 'Seconds since start', 'Power'])
            shaded = hv.operation.datashader.datashade(points, aggregator=datashader.max('Seconds since start'), cmap='rainbow').opts(xlim=self.xlim, ylim=self.zlim, width=self.px_scale*self.plan_edge_length, height=self.px_scale*self.hist_edge_length, tools=['hover'], hooks=[self.hook_yalt_limiter])
        elif color_by == 'Charge (User Assigned)':
            pass
        elif color_by == 'Charge (chargepol)':
            pass
        elif color_by == 'Power (dBW)':
            points = hv.Points((self.ds.event_longitude.data, self.ds.event_altitude.data, self.ds.event_time.data, self.ds.event_power), kdims=['Longitude', 'Altitude'], vdims=['Time', 'Power'])
            shaded = hv.operation.datashader.datashade(points, aggregator=datashader.max('Power'), cmap='rainbow').opts(xlim=self.xlim, ylim=self.zlim, width=self.px_scale*self.plan_edge_length, height=self.px_scale*self.hist_edge_length, tools=['hover'], hooks=[self.hook_yalt_limiter])
        elif color_by == 'Event Density':
            points = hv.Points((self.ds.event_longitude.data, self.ds.event_altitude.data, self.ds.event_time.data, self.ds.event_power), kdims=['Longitude', 'Altitude'], vdims=['Time', 'Power'])
            shaded = hv.operation.datashader.datashade(points, aggregator=datashader.count(), cmap='rainbow').opts(xlim=self.xlim, ylim=self.zlim, width=self.px_scale*self.plan_edge_length, height=self.px_scale*self.hist_edge_length, tools=['hover'], hooks=[self.hook_yalt_limiter])
        elif color_by == 'Log Event Density':
            points = hv.Points((self.ds.event_longitude.data, self.ds.event_altitude.data, self.ds.event_time.data, self.ds.event_power), kdims=['Longitude', 'Altitude'], vdims=['Time', 'Power'])
            shaded = hv.operation.datashader.datashade(points, aggregator=datashader.count(), cmap='rainbow', cnorm='log').opts(xlim=self.xlim, ylim=self.zlim, width=self.px_scale*self.plan_edge_length, height=self.px_scale*self.hist_edge_length, tools=['hover'], hooks=[self.hook_yalt_limiter])
        elif color_by == 'Altitude':
            points = hv.Points((self.ds.event_longitude.data, self.ds.event_altitude.data, self.ds.event_time.data, self.ds.event_power), kdims=['Longitude', 'Altitude'], vdims=['Time', 'Power'])
            shaded = hv.operation.datashader.datashade(points, aggregator=datashader.max('Altitude'), cmap='rainbow').opts(xlim=self.xlim, ylim=self.zlim, width=self.px_scale*self.plan_edge_length, height=self.px_scale*self.hist_edge_length, tools=['hover'], hooks=[self.hook_yalt_limiter])
        return shaded
    

    def plot_lonalt_points(self, color_by, should_datashade):
        if color_by == 'Time':
            timefloats = color_by_time(self.ds.event_time.values, tlim=self.time_range_dt)[-1]
            points = hv.Points((self.ds.event_longitude.data, self.ds.event_altitude.data, self.ds.event_time.data, timefloats, self.ds.event_power), kdims=['Longitude', 'Altitude'], vdims=['Time', 'Seconds since start', 'Power'])
            points = points.opts(hv.opts.Points(color='Seconds since start', cmap='rainbow', size=5)).opts(xlim=self.xlim, ylim=self.zlim, width=self.px_scale*self.plan_edge_length, height=self.px_scale*self.hist_edge_length, tools=['hover'], line_color='black', hooks=[self.hook_yalt_limiter], visible=not should_datashade)
        elif color_by == 'Charge (User Assigned)':
            pass
        elif color_by == 'Charge (chargepol)':
            pass
        elif color_by == 'Power (dBW)':
            points = hv.Points((self.ds.event_longitude.data, self.ds.event_altitude.data, self.ds.event_time.data, self.ds.event_power), kdims=['Longitude', 'Altitude'], vdims=['Time', 'Power'])
            points = points.opts(hv.opts.Points(color='Power', cmap='rainbow', size=5)).opts(xlim=self.xlim, ylim=self.zlim, width=self.px_scale*self.plan_edge_length, height=self.px_scale*self.hist_edge_length, tools=['hover'], line_color='black', hooks=[self.hook_yalt_limiter], visible=not should_datashade)
        elif color_by == 'Event Density':
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
            points = hv.Image((lonbins[:-1]+latlonbinwidth/2, altbins[:-1]+altbinwidth/2, hist.T), kdims=['Longitude', 'Altitude'], vdims=['Event Density']
                                ).opts(hv.opts.Image(cmap='rainbow')).opts(xlim=self.xlim, ylim=self.zlim, width=self.px_scale*self.plan_edge_length, height=self.px_scale*self.hist_edge_length, tools=['hover'], hooks=[self.hook_yalt_limiter], visible=not should_datashade)
        elif color_by == 'Log Event Density':
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
            points = hv.Image((lonbins[:-1]+latlonbinwidth/2, altbins[:-1]+altbinwidth/2, hist.T), kdims=['Longitude', 'Altitude'], vdims=['Event Density']
                                ).opts(hv.opts.Image(cmap='rainbow', cnorm='log')).opts(xlim=self.xlim, ylim=self.zlim, width=self.px_scale*self.plan_edge_length, height=self.px_scale*self.hist_edge_length, tools=['hover'], hooks=[self.hook_yalt_limiter], visible=not should_datashade)
        elif color_by == 'Altitude':
            points = hv.Points((self.ds.event_longitude.data, self.ds.event_altitude.data, self.ds.event_time.data, self.ds.event_power), kdims=['Longitude', 'Altitude'], vdims=['Time', 'Power'])
            points = points.opts(hv.opts.Points(color='Altitude', cmap='rainbow', size=5)).opts(xlim=self.xlim, ylim=self.zlim, width=self.px_scale*self.plan_edge_length, height=self.px_scale*self.hist_edge_length, tools=['hover'], line_color='black', hooks=[self.hook_yalt_limiter], visible=not should_datashade)
        return points
    

    def plot_alt_hist(self):
        return hv.Histogram(np.histogram(self.ds.event_altitude.data, bins=np.arange(0, 20001, 1000)), kdims=['Altitude'], vdims=['src']).opts(width=self.px_scale*self.hist_edge_length, height=self.px_scale*self.hist_edge_length, invert_axes=True).opts(hooks=[self.hook_xlabel_rotate, self.hook_hist_src_limiter, self.hook_yalt_limiter])


    def plot_latalt_points_datashaded(self, color_by):
        if color_by == 'Time':
            timefloats = color_by_time(self.ds.event_time.values, tlim=self.time_range_dt)[-1]
            points = hv.Points((self.ds.event_altitude.data, self.ds.event_latitude.data, self.ds.event_time.data, timefloats, self.ds.event_power), kdims=['Altitude', 'Latitude'], vdims=['Time', 'Seconds since start', 'Power'])
            shaded = hv.operation.datashader.datashade(points, aggregator=datashader.max('Seconds since start'), cmap='rainbow').opts(xlim=self.zlim, ylim=self.ylim, width=self.px_scale*self.hist_edge_length, height=self.px_scale*self.plan_edge_length, tools=['hover'], hooks=[self.hook_xlabel_rotate, self.hook_xalt_limiter])
        elif color_by == 'Charge (User Assigned)':
            pass
        elif color_by == 'Charge (chargepol)':
            pass
        elif color_by == 'Power (dBW)':
            points = hv.Points((self.ds.event_altitude.data, self.ds.event_latitude.data, self.ds.event_time.data, self.ds.event_power), kdims=['Altitude', 'Latitude'], vdims=['Time', 'Power'])
            shaded = hv.operation.datashader.datashade(points, aggregator=datashader.max('Power'), cmap='rainbow').opts(xlim=self.zlim, ylim=self.ylim, width=self.px_scale*self.hist_edge_length, height=self.px_scale*self.plan_edge_length, tools=['hover'], hooks=[self.hook_xlabel_rotate, self.hook_xalt_limiter])
        elif color_by == 'Event Density':
            points = hv.Points((self.ds.event_altitude.data, self.ds.event_latitude.data, self.ds.event_time.data, self.ds.event_power), kdims=['Altitude', 'Latitude'], vdims=['Time', 'Power'])
            shaded = hv.operation.datashader.datashade(points, aggregator=datashader.count(), cmap='rainbow').opts(xlim=self.zlim, ylim=self.ylim, width=self.px_scale*self.hist_edge_length, height=self.px_scale*self.plan_edge_length, tools=['hover'], hooks=[self.hook_xlabel_rotate, self.hook_xalt_limiter])
        elif color_by == 'Log Event Density':
            points = hv.Points((self.ds.event_altitude.data, self.ds.event_latitude.data, self.ds.event_time.data, self.ds.event_power), kdims=['Altitude', 'Latitude'], vdims=['Time', 'Power'])
            shaded = hv.operation.datashader.datashade(points, aggregator=datashader.count(), cmap='rainbow', cnorm='log').opts(xlim=self.zlim, ylim=self.ylim, width=self.px_scale*self.hist_edge_length, height=self.px_scale*self.plan_edge_length, tools=['hover'], hooks=[self.hook_xlabel_rotate, self.hook_xalt_limiter])
        elif color_by == 'Altitude':
            points = hv.Points((self.ds.event_altitude.data, self.ds.event_latitude.data, self.ds.event_time.data, self.ds.event_power), kdims=['Altitude', 'Latitude'], vdims=['Time', 'Power'])
            shaded = hv.operation.datashader.datashade(points, aggregator=datashader.max('Altitude'), cmap='rainbow').opts(xlim=self.zlim, ylim=self.ylim, width=self.px_scale*self.hist_edge_length, height=self.px_scale*self.plan_edge_length, tools=['hover'], hooks=[self.hook_xlabel_rotate, self.hook_xalt_limiter])
        return shaded


    def plot_latalt_points(self, color_by, should_datashade):
        if color_by == 'Time':
            timefloats = color_by_time(self.ds.event_time.values, tlim=self.time_range_dt)[-1]
            points = hv.Points((self.ds.event_altitude.data, self.ds.event_latitude.data, self.ds.event_time.data, timefloats, self.ds.event_power), kdims=['Altitude', 'Latitude'], vdims=['Time', 'Seconds since start', 'Power'])
            points = points.opts(hv.opts.Points(color='Seconds since start', cmap='rainbow', size=5)).opts(xlim=self.zlim, ylim=self.ylim, width=self.px_scale*self.hist_edge_length, height=self.px_scale*self.plan_edge_length, tools=['hover'], line_color='black', hooks=[self.hook_xlabel_rotate, self.hook_xalt_limiter], visible=not should_datashade)
        elif color_by == 'Charge (User Assigned)':
            pass
        elif color_by == 'Charge (chargepol)':
            pass
        elif color_by == 'Power (dBW)':
            points = hv.Points((self.ds.event_altitude.data, self.ds.event_latitude.data, self.ds.event_time.data, self.ds.event_power), kdims=['Altitude', 'Latitude'], vdims=['Time', 'Power'])
            points = points.opts(hv.opts.Points(color='Power', cmap='rainbow', size=5)).opts(xlim=self.zlim, ylim=self.ylim, width=self.px_scale*self.hist_edge_length, height=self.px_scale*self.plan_edge_length, tools=['hover'], line_color='black', hooks=[self.hook_xlabel_rotate, self.hook_xalt_limiter], visible=not should_datashade)
        elif color_by == 'Event Density':
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
            points = hv.Image((altbins[:-1]+altbinwidth/2, latbins[:-1]+latlonbinwidth/2, hist.T), kdims=['Altitude', 'Latitude'], vdims=['Event Density']
                                ).opts(hv.opts.Image(cmap='rainbow')).opts(xlim=self.zlim, ylim=self.ylim, width=self.px_scale*self.hist_edge_length, height=self.px_scale*self.plan_edge_length, tools=['hover'], hooks=[self.hook_xlabel_rotate, self.hook_xalt_limiter], visible=not should_datashade)
        elif color_by == 'Log Event Density':
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
            points = hv.Image((altbins[:-1]+altbinwidth/2, latbins[:-1]+latlonbinwidth/2, hist.T), kdims=['Altitude', 'Latitude'], vdims=['Event Density']
                            ).opts(hv.opts.Image(cmap='rainbow', cnorm='log')).opts(xlim=self.zlim, ylim=self.ylim, width=self.px_scale*self.hist_edge_length, height=self.px_scale*self.plan_edge_length, tools=['hover'], hooks=[self.hook_xlabel_rotate, self.hook_xalt_limiter], visible=not should_datashade)
        elif color_by == 'Altitude':
            points = hv.Points((self.ds.event_altitude.data, self.ds.event_latitude.data, self.ds.event_time.data, self.ds.event_power), kdims=['Altitude', 'Latitude'], vdims=['Time', 'Power'])
            points = points.opts(hv.opts.Points(color='Altitude', cmap='rainbow', size=5)).opts(xlim=self.zlim, ylim=self.ylim, width=self.px_scale*self.hist_edge_length, height=self.px_scale*self.plan_edge_length, tools=['hover'], line_color='black', hooks=[self.hook_xlabel_rotate, self.hook_xalt_limiter], visible=not should_datashade)
        return points
    

    def plot_alttime_points_datashaded(self, color_by):
        if color_by == 'Time':
            timefloats = color_by_time(self.ds.event_time.values, tlim=self.time_range_dt)[-1]
            points = hv.Points((self.ds.event_time.data, self.ds.event_altitude.data, timefloats, self.ds.event_power), kdims=['Time', 'Altitude'], vdims=['Seconds since start', 'Power'])
            shaded = hv.operation.datashader.datashade(points, aggregator=datashader.max('Seconds since start'), cmap='rainbow').opts(xlim=(self.ds.event_time.data[0], self.ds.event_time.data[-1]), ylim=self.zlim, width=self.px_scale*(self.plan_edge_length+self.hist_edge_length), height=self.px_scale*self.hist_edge_length, tools=['hover'], hooks=[self.hook_yalt_limiter, self.hook_time_limiter])
        elif color_by == 'Charge (User Assigned)':
            pass
        elif color_by == 'Charge (chargepol)':
            pass
        elif color_by == 'Power (dBW)':
            points = hv.Points((self.ds.event_time.data, self.ds.event_altitude.data, self.ds.event_power), kdims=['Time', 'Altitude'], vdims=['Power'])
            shaded = hv.operation.datashader.datashade(points, aggregator=datashader.max('Power'), cmap='rainbow').opts(xlim=(self.ds.event_time.data[0], self.ds.event_time.data[-1]), ylim=self.zlim, width=self.px_scale*(self.plan_edge_length+self.hist_edge_length), height=self.px_scale*self.hist_edge_length, tools=['hover'], hooks=[self.hook_yalt_limiter, self.hook_time_limiter])
        elif color_by == 'Event Density':
            points = hv.Points((self.ds.event_time.data, self.ds.event_altitude.data, self.ds.event_power), kdims=['Time', 'Altitude'], vdims=['Power'])
            shaded = hv.operation.datashader.datashade(points, aggregator=datashader.count(), cmap='rainbow').opts(xlim=(self.ds.event_time.data[0], self.ds.event_time.data[-1]), ylim=self.zlim, width=self.px_scale*(self.plan_edge_length+self.hist_edge_length), height=self.px_scale*self.hist_edge_length, tools=['hover'], hooks=[self.hook_yalt_limiter, self.hook_time_limiter])
        elif color_by == 'Log Event Density':
            points = hv.Points((self.ds.event_time.data, self.ds.event_altitude.data, self.ds.event_power), kdims=['Time', 'Altitude'], vdims=['Power'])
            shaded = hv.operation.datashader.datashade(points, aggregator=datashader.count(), cmap='rainbow', cnorm='log').opts(xlim=(self.ds.event_time.data[0], self.ds.event_time.data[-1]), ylim=self.zlim, width=self.px_scale*(self.plan_edge_length+self.hist_edge_length), height=self.px_scale*self.hist_edge_length, tools=['hover'], hooks=[self.hook_yalt_limiter, self.hook_time_limiter])
        elif color_by == 'Altitude':
            points = hv.Points((self.ds.event_time.data, self.ds.event_altitude.data, self.ds.event_power), kdims=['Time', 'Altitude'], vdims=['Power'])
            shaded = hv.operation.datashader.datashade(points, aggregator=datashader.max('Altitude'), cmap='rainbow').opts(xlim=(self.ds.event_time.data[0], self.ds.event_time.data[-1]), ylim=self.zlim, width=self.px_scale*(self.plan_edge_length+self.hist_edge_length), height=self.px_scale*self.hist_edge_length, tools=['hover'], hooks=[self.hook_yalt_limiter, self.hook_time_limiter])
        return shaded


    def plot_alttime_points(self, color_by, should_datashade):
        if color_by == 'Time':
            timefloats = color_by_time(self.ds.event_time.values, tlim=self.time_range_dt)[-1]
            points = hv.Points((self.ds.event_time.data, self.ds.event_altitude.data, timefloats, self.ds.event_power), kdims=['Time', 'Altitude'], vdims=['Seconds since start', 'Power'])
            points = points.opts(hv.opts.Points(color='Seconds since start', cmap='rainbow', size=5)).opts(xlim=(self.ds.event_time.data[0], self.ds.event_time.data[-1]), ylim=self.zlim, width=self.px_scale*(self.plan_edge_length+self.hist_edge_length), height=self.px_scale*self.hist_edge_length, tools=['hover'], line_color='black', hooks=[self.hook_yalt_limiter, self.hook_time_limiter], visible=not should_datashade)
        elif color_by == 'Charge (User Assigned)':
            pass
        elif color_by == 'Charge (chargepol)':
            pass
        elif color_by == 'Power (dBW)':
            points = hv.Points((self.ds.event_time.data, self.ds.event_altitude.data, self.ds.event_power), kdims=['Time', 'Altitude'], vdims=['Power'])
            points = points.opts(hv.opts.Points(color='Power', cmap='rainbow', size=5)).opts(xlim=(self.ds.event_time.data[0], self.ds.event_time.data[-1]), ylim=self.zlim, width=self.px_scale*(self.plan_edge_length+self.hist_edge_length), height=self.px_scale*self.hist_edge_length, tools=['hover'], line_color='black', hooks=[self.hook_yalt_limiter, self.hook_time_limiter], visible=not should_datashade)
        elif color_by == 'Event Density':
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
            points = hv.Image((timebins_ctr_dt, altbins[:-1]+altbinwidth/2, hist.T), kdims=['Time', 'Altitude'], vdims=['Event Density']
                                ).opts(hv.opts.Image(cmap='rainbow')).opts(xlim=(self.ds.event_time.data[0], self.ds.event_time.data[-1]), ylim=self.zlim, width=self.px_scale*(self.plan_edge_length+self.hist_edge_length), height=self.px_scale*self.hist_edge_length, tools=['hover'], hooks=[self.hook_yalt_limiter, self.hook_time_limiter], visible=not should_datashade)
        elif color_by == 'Log Event Density':
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
            points = hv.Image((timebins_ctr_dt, altbins[:-1]+altbinwidth/2, hist.T), kdims=['Time', 'Altitude'], vdims=['Event Density']
                                ).opts(hv.opts.Image(cmap='rainbow', cnorm='log')).opts(xlim=(self.ds.event_time.data[0], self.ds.event_time.data[-1]), ylim=self.zlim, width=self.px_scale*(self.plan_edge_length+self.hist_edge_length), height=self.px_scale*self.hist_edge_length, tools=['hover'], hooks=[self.hook_yalt_limiter, self.hook_time_limiter], visible=not should_datashade)
        elif color_by == 'Altitude':
            points = hv.Points((self.ds.event_time.data, self.ds.event_altitude.data, self.ds.event_power), kdims=['Time', 'Altitude'], vdims=['Power'])
            points = points.opts(hv.opts.Points(color='Altitude', cmap='rainbow', size=5)).opts(xlim=(self.ds.event_time.data[0], self.ds.event_time.data[-1]), ylim=self.zlim, width=self.px_scale*(self.plan_edge_length+self.hist_edge_length), height=self.px_scale*self.hist_edge_length, tools=['hover'], line_color='black', hooks=[self.hook_yalt_limiter, self.hook_time_limiter], visible=not should_datashade)
        return points
    

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
            self.rerender()
            return
        this_selection_geom = np.array([data['xs'], data['ys']]).T[:, 0, :]
        self.selection_geom = [this_selection_geom, self.last_mouse_coord[-2]]

    def rerender(self):
        new_plan_points = self.plot_planview_points()
        self.plan_ax_pointer.source = new_plan_points
        self.plan_range_stream.source = new_plan_points
        new_polys_plan = hv.Polygons([]).opts(hv.opts.Polygons(fill_alpha=0.3, fill_color='black'))
        self.plan_ax_selector.source = new_polys_plan
        
        new_lon_alt_points = self.plot_lonalt_points()
        self.lon_alt_ax_pointer.source = new_lon_alt_points
        self.lon_alt_range_stream.source = new_lon_alt_points
        new_polys_lonalt = hv.Polygons([]).opts(hv.opts.Polygons(fill_alpha=0.3, fill_color='black'))
        self.lon_alt_ax_selector.source = new_polys_lonalt

        new_lat_alt_points = self.plot_latalt_points()
        self.lat_alt_ax_pointer.source = new_lat_alt_points
        self.lat_alt_range_stream.source = new_lat_alt_points
        new_polys_latalt = hv.Polygons([]).opts(hv.opts.Polygons(fill_alpha=0.3, fill_color='black'))
        self.lat_alt_ax_selector.source = new_polys_latalt

        new_alt_time_points = self.plot_alttime_points()
        self.alt_time_pointer.source = new_alt_time_points
        new_polys_alttime = hv.Polygons([]).opts(hv.opts.Polygons(fill_alpha=0.3, fill_color='black'))
        self.alt_time_ax_selector.source = new_polys_alttime

        new_plan_ax = (new_plan_points * self.plan_ax_crosshair * new_polys_plan * self.plan_ax_select_area * self.plan_ax_bg).opts(xlim=self.xlim, ylim=self.ylim)
        new_lon_alt_ax = (new_lon_alt_points * self.lon_ax_crosshair * new_polys_lonalt * self.lon_alt_select_area).opts(xlim=self.xlim, ylim=self.zlim)
        new_lat_alt_ax = (new_lat_alt_points * self.lat_ax_crosshair * new_polys_latalt * self.lat_alt_select_area).opts(xlim=self.zlim, ylim=self.ylim)
        new_hist_ax = hv.DynamicMap(self.plot_alt_hist)
        self.hist_ax = new_hist_ax
        new_alt_time_ax = (new_alt_time_points * self.time_ax_crosshair * new_polys_alttime * self.alt_time_select_area).opts(xlim=(self.ds.event_time.data[0], self.ds.event_time.data[-1]), ylim=self.zlim)

        new_lower_part = (new_lon_alt_ax + self.hist_ax + new_plan_ax + new_lat_alt_ax).cols(2)

        self.panelHandle[2].object = new_lower_part
        self.panelHandle[1].object = new_alt_time_ax

        self.selection_geom = [np.array([]), 0]

        self.plan_points = new_plan_points
        self.lon_alt_points = new_lon_alt_points
        self.lat_alt_points = new_lat_alt_points
        self.alt_time_points = new_alt_time_points


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

        plan_points = hv.DynamicMap(pn.bind(self.plot_planview_points, color_by=color_by_dropdown, should_datashade=datashade_switch, watch=True))
        plan_ax_polys = hv.Polygons([]).opts(hv.opts.Polygons(fill_alpha=0.3, fill_color='black'))
        plan_ax_selector = hv.streams.PolyDraw(source=plan_ax_polys, drag=False, num_objects=1, show_vertices=True, vertex_style={'size': 5, 'fill_color': 'white', 'line_color' : 'black'})
        plan_ax_selector.add_subscriber(self.handle_selection)
        plan_ax_select_area = hv.DynamicMap(self.plan_ax_highlighter)
        self.plan_range_stream = hv.streams.RangeXY(source=plan_points)
        self.plan_range_stream.add_subscriber(self.plan_range_handle)
        plan_ax_pointer = hv.streams.PointerXY(x=0, y=0, source=plan_points).rename(x='plan_x', y='plan_y')

        lon_alt_points = hv.DynamicMap(pn.bind(self.plot_lonalt_points, color_by=color_by_dropdown, should_datashade=datashade_switch, watch=True))
        lon_alt_ax_polys = hv.Polygons([]).opts(hv.opts.Polygons(fill_alpha=0.3, fill_color='black'))
        lon_alt_ax_selector = hv.streams.PolyDraw(source=lon_alt_ax_polys, drag=False, num_objects=1, show_vertices=True, vertex_style={'size': 5, 'fill_color': 'white', 'line_color' : 'black'})
        lon_alt_ax_selector.add_subscriber(self.handle_selection)
        lon_alt_select_area = hv.DynamicMap(self.lon_ax_highlighter)
        self.lon_alt_range_stream = hv.streams.RangeXY(source=lon_alt_points)
        self.lon_alt_range_stream.add_subscriber(self.lonalt_range_handle)
        lon_alt_ax_pointer = hv.streams.PointerXY(x=0, y=0, source=lon_alt_points).rename(x='lon_x', y='lon_y')


        hist_ax = hv.DynamicMap(self.plot_alt_hist)

        lat_alt_points = hv.DynamicMap(pn.bind(self.plot_latalt_points, color_by=color_by_dropdown, should_datashade=datashade_switch, watch=True))
        lat_alt_ax_polys = hv.Polygons([]).opts(hv.opts.Polygons(fill_alpha=0.3, fill_color='black'))
        lat_alt_ax_selector = hv.streams.PolyDraw(source=lat_alt_ax_polys, drag=False, num_objects=1, show_vertices=True, vertex_style={'size': 5, 'fill_color': 'white', 'line_color' : 'black'})
        lat_alt_ax_selector.add_subscriber(self.handle_selection)
        lat_alt_select_area = hv.DynamicMap(self.lat_ax_highlighter)
        self.lat_alt_range_stream = hv.streams.RangeXY(source=lat_alt_points)
        self.lat_alt_range_stream.add_subscriber(self.latalt_range_handle)
        lat_alt_ax_pointer = hv.streams.PointerXY(x=0, y=0, source=lat_alt_points).rename(x='lat_x', y='lat_y')


        alt_time_points = hv.DynamicMap(pn.bind(self.plot_alttime_points, color_by=color_by_dropdown, should_datashade=datashade_switch, watch=True))
        alt_time_ax_polys = hv.Polygons([]).opts(hv.opts.Polygons(fill_alpha=0.3, fill_color='black'))
        alt_time_ax_selector = hv.streams.PolyDraw(source=alt_time_ax_polys, drag=False, num_objects=1, show_vertices=True, vertex_style={'size': 5, 'fill_color': 'white', 'line_color' : 'black'})
        alt_time_ax_selector.add_subscriber(self.handle_selection)
        alt_time_select_area = hv.DynamicMap(self.time_ax_highlighter)
        alt_time_pointer = hv.streams.PointerXY(x=0, y=0, source=alt_time_points).rename(x='time_x', y='time_y')

        points_shaded = []
        for func_to_plot in [self.plot_planview_points_datashaded, self.plot_lonalt_points_datashaded, self.plot_latalt_points_datashaded, self.plot_alttime_points_datashaded]:
            this_ax_points_shaded = []
            for colorshade in color_by_dropdown.options:
                if colorshade in ['Charge (User Assigned)', 'Charge (chargepol)']:
                    continue
                points = func_to_plot(color_by=colorshade)
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

        lon_alt_ax = (lon_alt_points * points_shaded[1] * lon_ax_crosshair * lon_alt_ax_polys * lon_alt_select_area)
        plan_ax = (plan_points * points_shaded[0] * plan_ax_crosshair * plan_ax_polys * plan_ax_select_area * self.plan_ax_bg)
        lat_alt_ax = (lat_alt_points * points_shaded[2] * lat_ax_crosshair * lat_alt_ax_polys * lat_alt_select_area)
        alt_time_ax = pn.pane.HoloViews(alt_time_points * points_shaded[3] * time_ax_crosshair * alt_time_ax_polys * alt_time_select_area)

        the_lower_part = (lon_alt_ax + hist_ax + plan_ax + lat_alt_ax).cols(2)
        the_lower_part = pn.pane.HoloViews(the_lower_part)

        netw_name = 'LYLOUT'
        filename = self.filename
        if type(self.filename) != str:
            filename = filename[0]
        splitfile = filename.split('_')
        if len(splitfile) >= 2:
            netw_name = splitfile[0]
        if netw_name == 'LYLOUT':
            if 'station_network' in self.ds.keys():
                netw_name = [s for s in self.ds.station_network.data[0].decode('utf-8') if s.isalpha()]
                netw_name = ''.join(netw_name)
        netw_name.replace('LMA', '')


        title = pn.pane.Markdown(f'## {netw_name} LMA on {self.time_range_py_dt[0].strftime("%d %b %Y")}', styles={'text-align': 'center'})

        self.panelHandle  = pn.Column(title, alt_time_ax, the_lower_part)


