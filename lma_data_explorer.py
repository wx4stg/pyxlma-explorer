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

        self.should_datashade = True

        self.color_by = 'Time'

        self.init_plot()

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


    def change_colorby(self, color_by_selector):
        self.color_by = color_by_selector
        self.rerender()

    def plot_planview_points(self):
        if self.color_by == 'Time':
            timefloats = color_by_time(self.ds.event_time.values)[-1]
            if self.should_datashade:
                points = hv.Points((self.ds.event_longitude.data, self.ds.event_latitude.data, self.ds.event_altitude.data, self.ds.event_time.data, timefloats, self.ds.event_power), kdims=['Longitude', 'Latitude'], vdims=['Altitude', 'Time', 'time_color', 'Power'])
                shaded = hv.operation.datashader.datashade(points, aggregator=datashader.max('time_color'), cmap='rainbow').opts(xlim=self.xlim, ylim=self.ylim, width=self.px_scale*self.plan_edge_length, height=self.px_scale*self.plan_edge_length, tools=['hover'])
            else:
                shaded = hv.Points((self.ds.event_longitude.data, self.ds.event_latitude.data, self.ds.event_altitude.data, self.ds.event_time.data, timefloats, self.ds.event_power), kdims=['Longitude', 'Latitude'], vdims=['Altitude', 'Time', 'time_color', 'Power']
                                       ).opts(hv.opts.Points(color='time_color', cmap='rainbow', size=5)).opts(xlim=self.xlim, ylim=self.ylim, width=self.px_scale*self.plan_edge_length, height=self.px_scale*self.plan_edge_length, tools=['hover'], line_color='black')
        elif self.color_by == 'Charge (User Assigned)':
            pass
        elif self.color_by == 'Charge (chargepol)':
            pass
        elif self.color_by == 'Power (dBW)':
            if self.should_datashade:
                points = hv.Points((self.ds.event_longitude.data, self.ds.event_latitude.data, self.ds.event_altitude.data, self.ds.event_time.data, self.ds.event_power), kdims=['Longitude', 'Latitude'], vdims=['Altitude', 'Time', 'Power'])
                shaded = hv.operation.datashader.datashade(points, aggregator=datashader.max('Power'), cmap='rainbow').opts(xlim=self.xlim, ylim=self.ylim, width=self.px_scale*self.plan_edge_length, height=self.px_scale*self.plan_edge_length, tools=['hover'])
            else:
                shaded = hv.Points((self.ds.event_longitude.data, self.ds.event_latitude.data, self.ds.event_altitude.data, self.ds.event_time.data, self.ds.event_power), kdims=['Longitude', 'Latitude'], vdims=['Altitude', 'Time', 'Power']
                                   ).opts(hv.opts.Points(color='Power', cmap='rainbow', size=5)).opts(xlim=self.xlim, ylim=self.ylim, width=self.px_scale*self.plan_edge_length, height=self.px_scale*self.plan_edge_length, tools=['hover'], line_color='black')
        elif self.color_by == 'Event Density':
            if self.should_datashade:
                points = hv.Points((self.ds.event_longitude.data, self.ds.event_latitude.data, self.ds.event_altitude.data, self.ds.event_time.data, self.ds.event_power), kdims=['Longitude', 'Latitude'], vdims=['Altitude', 'Time', 'Power'])
                shaded = hv.operation.datashader.datashade(points, aggregator=datashader.count(), cmap='rainbow').opts(xlim=self.xlim, ylim=self.ylim, width=self.px_scale*self.plan_edge_length, height=self.px_scale*self.plan_edge_length, tools=['hover'])
            else:
                latlonbinwidth = 0.01
                latmin = (np.min(self.ds.event_latitude.data) // latlonbinwidth )*latlonbinwidth
                latmax = (np.max(self.ds.event_latitude.data) // latlonbinwidth + 1)*latlonbinwidth
                lonmin = (np.min(self.ds.event_longitude.data) // latlonbinwidth )*latlonbinwidth
                lonmax = (np.max(self.ds.event_longitude.data) // latlonbinwidth + 1)*latlonbinwidth
                latbins = np.arange(latmin, latmax, latlonbinwidth)
                lonbins = np.arange(lonmin, lonmax, latlonbinwidth)
                hist, _, _ = np.histogram2d(self.ds.event_longitude.data, self.ds.event_latitude.data, bins=[lonbins, latbins])
                hist[hist == 0] = np.nan
                shaded = hv.Image((lonbins[:-1]+latlonbinwidth/2, latbins[:-1]+latlonbinwidth/2, hist.T), kdims=['Longitude', 'Latitude']
                                  ).opts(hv.opts.Image(cmap='rainbow')).opts(xlim=self.xlim, ylim=self.ylim, width=self.px_scale*self.plan_edge_length, height=self.px_scale*self.plan_edge_length, tools=['hover'])
        elif self.color_by == 'Log Event Density':
            if self.should_datashade:
                points = hv.Points((self.ds.event_longitude.data, self.ds.event_latitude.data, self.ds.event_altitude.data, self.ds.event_time.data, self.ds.event_power), kdims=['Longitude', 'Latitude'], vdims=['Altitude', 'Time', 'Power'])
                shaded = hv.operation.datashader.datashade(points, aggregator=datashader.count(), cmap='rainbow', cnorm='log').opts(xlim=self.xlim, ylim=self.ylim, width=self.px_scale*self.plan_edge_length, height=self.px_scale*self.plan_edge_length, tools=['hover'])
            else:
                latlonbinwidth = 0.01
                latmin = (np.min(self.ds.event_latitude.data) // latlonbinwidth )*latlonbinwidth
                latmax = (np.max(self.ds.event_latitude.data) // latlonbinwidth + 1)*latlonbinwidth
                lonmin = (np.min(self.ds.event_longitude.data) // latlonbinwidth )*latlonbinwidth
                lonmax = (np.max(self.ds.event_longitude.data) // latlonbinwidth + 1)*latlonbinwidth
                latbins = np.arange(latmin, latmax, latlonbinwidth)
                lonbins = np.arange(lonmin, lonmax, latlonbinwidth)
                hist, _, _ = np.histogram2d(self.ds.event_longitude.data, self.ds.event_latitude.data, bins=[lonbins, latbins])
                hist[hist == 0] = np.nan
                shaded = hv.Image((lonbins[:-1]+latlonbinwidth/2, latbins[:-1]+latlonbinwidth/2, hist.T), kdims=['Longitude', 'Latitude']
                                  ).opts(hv.opts.Image(cmap='rainbow', cnorm='log')).opts(xlim=self.xlim, ylim=self.ylim, width=self.px_scale*self.plan_edge_length, height=self.px_scale*self.plan_edge_length, tools=['hover'])
        elif self.color_by == 'Altitude':
            if self.should_datashade:
                points = hv.Points((self.ds.event_longitude.data, self.ds.event_latitude.data, self.ds.event_altitude.data, self.ds.event_time.data, self.ds.event_power), kdims=['Longitude', 'Latitude'], vdims=['Altitude', 'Time', 'Power'])
                shaded = hv.operation.datashader.datashade(points, aggregator=datashader.max('Altitude'), cmap='rainbow').opts(xlim=self.xlim, ylim=self.ylim, width=self.px_scale*self.plan_edge_length, height=self.px_scale*self.plan_edge_length, tools=['hover'])
            else:
                shaded = hv.Points((self.ds.event_longitude.data, self.ds.event_latitude.data, self.ds.event_altitude.data, self.ds.event_time.data, self.ds.event_power), kdims=['Longitude', 'Latitude'], vdims=['Altitude', 'Time', 'Power']
                                   ).opts(hv.opts.Points(color='Altitude', cmap='rainbow', size=5)).opts(xlim=self.xlim, ylim=self.ylim, width=self.px_scale*self.plan_edge_length, height=self.px_scale*self.plan_edge_length, tools=['hover'], line_color='black')
        # counties_shp = shapefile.Reader('ne_10m_admin_2_counties.shp').shapes()
        # counties_shp = [shape(counties_shp[i]) for i in range(len(counties_shp))]
        # shaded = shaded * gv.Path(counties_shp).opts(color='gray') * gf.borders().opts(color='black') * gf.states().opts(color='black', line_width=2)
        return shaded
    
    def plot_lonalt_points(self):
        if self.color_by == 'Time':
            timefloats = color_by_time(self.ds.event_time.values)[-1]
            if self.should_datashade:
                points = hv.Points((self.ds.event_longitude.data, self.ds.event_altitude.data, self.ds.event_time.data, timefloats, self.ds.event_power), kdims=['Longitude', 'Altitude'], vdims=['Time', 'time_color', 'Power'])
                shaded = hv.operation.datashader.datashade(points, aggregator=datashader.max('time_color'), cmap='rainbow').opts(xlim=self.xlim, ylim=self.zlim, width=self.px_scale*self.plan_edge_length, height=self.px_scale*self.hist_edge_length, tools=['hover'], hooks=[self.hook_yalt_limiter])
            else:
                shaded = hv.Points((self.ds.event_longitude.data, self.ds.event_altitude.data, self.ds.event_time.data, timefloats, self.ds.event_power), kdims=['Longitude', 'Altitude'], vdims=['Time', 'time_color', 'Power']
                                   ).opts(hv.opts.Points(color='time_color', cmap='rainbow', size=5)).opts(xlim=self.xlim, ylim=self.zlim, width=self.px_scale*self.plan_edge_length, height=self.px_scale*self.hist_edge_length, tools=['hover'], line_color='black', hooks=[self.hook_yalt_limiter])
        elif self.color_by == 'Charge (User Assigned)':
            pass
        elif self.color_by == 'Charge (chargepol)':
            pass
        elif self.color_by == 'Power (dBW)':
            if self.should_datashade:
                points = hv.Points((self.ds.event_longitude.data, self.ds.event_altitude.data, self.ds.event_time.data, self.ds.event_power), kdims=['Longitude', 'Altitude'], vdims=['Time', 'Power'])
                shaded = hv.operation.datashader.datashade(points, aggregator=datashader.max('Power'), cmap='rainbow').opts(xlim=self.xlim, ylim=self.zlim, width=self.px_scale*self.plan_edge_length, height=self.px_scale*self.hist_edge_length, tools=['hover'], hooks=[self.hook_yalt_limiter])
            else:
                shaded = hv.Points((self.ds.event_longitude.data, self.ds.event_altitude.data, self.ds.event_time.data, self.ds.event_power), kdims=['Longitude', 'Altitude'], vdims=['Time', 'Power']
                                   ).opts(hv.opts.Points(color='Power', cmap='rainbow', size=5)).opts(xlim=self.xlim, ylim=self.zlim, width=self.px_scale*self.plan_edge_length, height=self.px_scale*self.hist_edge_length, tools=['hover'], line_color='black', hooks=[self.hook_yalt_limiter])
        elif self.color_by == 'Event Density':
            if self.should_datashade:
                points = hv.Points((self.ds.event_longitude.data, self.ds.event_altitude.data, self.ds.event_time.data, self.ds.event_power), kdims=['Longitude', 'Altitude'], vdims=['Time', 'Power'])
                shaded = hv.operation.datashader.datashade(points, aggregator=datashader.count(), cmap='rainbow').opts(xlim=self.xlim, ylim=self.zlim, width=self.px_scale*self.plan_edge_length, height=self.px_scale*self.hist_edge_length, tools=['hover'], hooks=[self.hook_yalt_limiter])
            else:
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
                shaded = hv.Image((lonbins[:-1]+latlonbinwidth/2, altbins[:-1]+altbinwidth/2, hist.T), kdims=['Longitude', 'Altitude'], vdims=['Event Density']
                                  ).opts(hv.opts.Image(cmap='rainbow')).opts(xlim=self.xlim, ylim=self.zlim, width=self.px_scale*self.plan_edge_length, height=self.px_scale*self.hist_edge_length, tools=['hover'], hooks=[self.hook_yalt_limiter])
        elif self.color_by == 'Log Event Density':
            if self.should_datashade:
                points = hv.Points((self.ds.event_longitude.data, self.ds.event_altitude.data, self.ds.event_time.data, self.ds.event_power), kdims=['Longitude', 'Altitude'], vdims=['Time', 'Power'])
                shaded = hv.operation.datashader.datashade(points, aggregator=datashader.count(), cmap='rainbow', cnorm='log').opts(xlim=self.xlim, ylim=self.zlim, width=self.px_scale*self.plan_edge_length, height=self.px_scale*self.hist_edge_length, tools=['hover'], hooks=[self.hook_yalt_limiter])
            else:
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
                shaded = hv.Image((lonbins[:-1]+latlonbinwidth/2, altbins[:-1]+altbinwidth/2, hist.T), kdims=['Longitude', 'Altitude'], vdims=['Log Event Density']
                                  ).opts(hv.opts.Image(cmap='rainbow', cnorm='log')).opts(xlim=self.xlim, ylim=self.zlim, width=self.px_scale*self.plan_edge_length, height=self.px_scale*self.hist_edge_length, tools=['hover'], hooks=[self.hook_yalt_limiter])
        elif self.color_by == 'Altitude':
            if self.should_datashade:
                points = hv.Points((self.ds.event_longitude.data, self.ds.event_altitude.data, self.ds.event_time.data, self.ds.event_power), kdims=['Longitude', 'Altitude'], vdims=['Time', 'Power'])
                shaded = hv.operation.datashader.datashade(points, aggregator=datashader.max('Altitude'), cmap='rainbow').opts(xlim=self.xlim, ylim=self.zlim, width=self.px_scale*self.plan_edge_length, height=self.px_scale*self.hist_edge_length, tools=['hover'], hooks=[self.hook_yalt_limiter])
            else:
                shaded = hv.Points((self.ds.event_longitude.data, self.ds.event_altitude.data, self.ds.event_time.data, self.ds.event_power), kdims=['Longitude', 'Altitude'], vdims=['Time', 'Power']
                                   ).opts(hv.opts.Points(color='Altitude', cmap='rainbow', size=5)).opts(xlim=self.xlim, ylim=self.zlim, width=self.px_scale*self.plan_edge_length, height=self.px_scale*self.hist_edge_length, tools=['hover'], line_color='black', hooks=[self.hook_yalt_limiter])
        
        return shaded
    
    def plot_alt_hist(self):
        return hv.Histogram(np.histogram(self.ds.event_altitude.data, bins=np.arange(0, 20001, 1000)), kdims=['Altitude'], vdims=['src']).opts(width=self.px_scale*self.hist_edge_length, height=self.px_scale*self.hist_edge_length, invert_axes=True).opts(hooks=[self.hook_xlabel_rotate, self.hook_hist_src_limiter, self.hook_yalt_limiter])

    def plot_latalt_points(self):
        if self.color_by == 'Time':
            timefloats = color_by_time(self.ds.event_time.values)[-1]
            if self.should_datashade:
                points = hv.Points((self.ds.event_altitude.data, self.ds.event_latitude.data, self.ds.event_time.data, timefloats, self.ds.event_power), kdims=['Altitude', 'Latitude'], vdims=['Time', 'time_color', 'Power'])
                shaded = hv.operation.datashader.datashade(points, aggregator=datashader.max('time_color'), cmap='rainbow').opts(xlim=self.zlim, ylim=self.ylim, width=self.px_scale*self.hist_edge_length, height=self.px_scale*self.plan_edge_length, tools=['hover'], hooks=[self.hook_xlabel_rotate, self.hook_xalt_limiter])
            else:
                shaded = hv.Points((self.ds.event_altitude.data, self.ds.event_latitude.data, self.ds.event_time.data, timefloats, self.ds.event_power), kdims=['Altitude', 'Latitude'], vdims=['Time', 'time_color', 'Power']
                                   ).opts(hv.opts.Points(color='time_color', cmap='rainbow', size=5)).opts(xlim=self.zlim, ylim=self.ylim, width=self.px_scale*self.hist_edge_length, height=self.px_scale*self.plan_edge_length, tools=['hover'], line_color='black', hooks=[self.hook_xlabel_rotate, self.hook_xalt_limiter])
        elif self.color_by == 'Charge (User Assigned)':
            pass
        elif self.color_by == 'Charge (chargepol)':
            pass
        elif self.color_by == 'Power (dBW)':
            if self.should_datashade:
                points = hv.Points((self.ds.event_altitude.data, self.ds.event_latitude.data, self.ds.event_time.data, self.ds.event_power), kdims=['Altitude', 'Latitude'], vdims=['Time', 'Power'])
                shaded = hv.operation.datashader.datashade(points, aggregator=datashader.max('Power'), cmap='rainbow').opts(xlim=self.zlim, ylim=self.ylim, width=self.px_scale*self.hist_edge_length, height=self.px_scale*self.plan_edge_length, tools=['hover'], hooks=[self.hook_xlabel_rotate, self.hook_xalt_limiter])
            else:
                shaded = hv.Points((self.ds.event_altitude.data, self.ds.event_latitude.data, self.ds.event_time.data, self.ds.event_power), kdims=['Altitude', 'Latitude'], vdims=['Time', 'Power']
                          ).opts(hv.opts.Points(color='Power', cmap='rainbow', size=5)).opts(xlim=self.zlim, ylim=self.ylim, width=self.px_scale*self.hist_edge_length, height=self.px_scale*self.plan_edge_length, tools=['hover'], line_color='black', hooks=[self.hook_xlabel_rotate, self.hook_xalt_limiter])
        elif self.color_by == 'Event Density':
            if self.should_datashade:
                points = hv.Points((self.ds.event_altitude.data, self.ds.event_latitude.data, self.ds.event_time.data, self.ds.event_power), kdims=['Altitude', 'Latitude'], vdims=['Time', 'Power'])
                shaded = hv.operation.datashader.datashade(points, aggregator=datashader.count(), cmap='rainbow').opts(xlim=self.zlim, ylim=self.ylim, width=self.px_scale*self.hist_edge_length, height=self.px_scale*self.plan_edge_length, tools=['hover'], hooks=[self.hook_xlabel_rotate, self.hook_xalt_limiter])
            else:
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
                shaded = hv.Image((altbins[:-1]+altbinwidth/2, latbins[:-1]+latlonbinwidth/2, hist.T), kdims=['Altitude', 'Latitude'], vdims=['Event Density']
                                  ).opts(hv.opts.Image(cmap='rainbow')).opts(xlim=self.zlim, ylim=self.ylim, width=self.px_scale*self.hist_edge_length, height=self.px_scale*self.plan_edge_length, tools=['hover'], hooks=[self.hook_xlabel_rotate, self.hook_xalt_limiter])
        elif self.color_by == 'Log Event Density':
            if self.should_datashade:
                points = hv.Points((self.ds.event_altitude.data, self.ds.event_latitude.data, self.ds.event_time.data, self.ds.event_power), kdims=['Altitude', 'Latitude'], vdims=['Time', 'Power'])
                shaded = hv.operation.datashader.datashade(points, aggregator=datashader.count(), cmap='rainbow', cnorm='log').opts(xlim=self.zlim, ylim=self.ylim, width=self.px_scale*self.hist_edge_length, height=self.px_scale*self.plan_edge_length, tools=['hover'], hooks=[self.hook_xlabel_rotate, self.hook_xalt_limiter])
            else:
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
                shaded = hv.Image((altbins[:-1]+altbinwidth/2, latbins[:-1]+latlonbinwidth/2, hist.T), kdims=['Altitude', 'Latitude'], vdims=['Log Event Density']
                                ).opts(hv.opts.Image(cmap='rainbow', cnorm='log')).opts(xlim=self.zlim, ylim=self.ylim, width=self.px_scale*self.hist_edge_length, height=self.px_scale*self.plan_edge_length, tools=['hover'], hooks=[self.hook_xlabel_rotate, self.hook_xalt_limiter])
        elif self.color_by == 'Altitude':
            if self.should_datashade:
                points = hv.Points((self.ds.event_altitude.data, self.ds.event_latitude.data, self.ds.event_time.data, self.ds.event_power), kdims=['Altitude', 'Latitude'], vdims=['Time', 'Power'])
                shaded = hv.operation.datashader.datashade(points, aggregator=datashader.max('Altitude'), cmap='rainbow').opts(xlim=self.zlim, ylim=self.ylim, width=self.px_scale*self.hist_edge_length, height=self.px_scale*self.plan_edge_length, tools=['hover'], hooks=[self.hook_xlabel_rotate, self.hook_xalt_limiter])
            else:
                shaded = hv.Points((self.ds.event_altitude.data, self.ds.event_latitude.data, self.ds.event_time.data, self.ds.event_power), kdims=['Altitude', 'Latitude'], vdims=['Time', 'Power']
                          ).opts(hv.opts.Points(color='Altitude', cmap='rainbow', size=5)).opts(xlim=self.zlim, ylim=self.ylim, width=self.px_scale*self.hist_edge_length, height=self.px_scale*self.plan_edge_length, tools=['hover'], line_color='black', hooks=[self.hook_xlabel_rotate, self.hook_xalt_limiter])
        
        return shaded

    def plot_alttime_points(self):
        if self.color_by == 'Time':
            timefloats = color_by_time(self.ds.event_time.values)[-1]
            if self.should_datashade:
                points = hv.Points((self.ds.event_time.data, self.ds.event_altitude.data, timefloats, self.ds.event_power), kdims=['Time', 'Altitude'], vdims=['time_color', 'Power'])
                shaded = hv.operation.datashader.datashade(points, aggregator=datashader.max('time_color'), cmap='rainbow').opts(xlim=(self.ds.event_time.data[0], self.ds.event_time.data[-1]), ylim=self.zlim, width=self.px_scale*(self.plan_edge_length+self.hist_edge_length), height=self.px_scale*self.hist_edge_length, tools=['hover'], hooks=[self.hook_yalt_limiter, self.hook_time_limiter])
            else:
                shaded = hv.Points((self.ds.event_time.data, self.ds.event_altitude.data, timefloats, self.ds.event_power), kdims=['Time', 'Altitude'], vdims=['time_color', 'Power']
                                   ).opts(hv.opts.Points(color='time_color', cmap='rainbow', size=5)).opts(xlim=(self.ds.event_time.data[0], self.ds.event_time.data[-1]), ylim=self.zlim, width=self.px_scale*(self.plan_edge_length+self.hist_edge_length), height=self.px_scale*self.hist_edge_length, tools=['hover'], line_color='black', hooks=[self.hook_yalt_limiter, self.hook_time_limiter])
        elif self.color_by == 'Charge (User Assigned)':
            pass
        elif self.color_by == 'Charge (chargepol)':
            pass
        elif self.color_by == 'Power (dBW)':
            if self.should_datashade:
                points = hv.Points((self.ds.event_time.data, self.ds.event_altitude.data, self.ds.event_power), kdims=['Time', 'Altitude'], vdims=['Power'])
                shaded = hv.operation.datashader.datashade(points, aggregator=datashader.max('Power'), cmap='rainbow').opts(xlim=(self.ds.event_time.data[0], self.ds.event_time.data[-1]), ylim=self.zlim, width=self.px_scale*(self.plan_edge_length+self.hist_edge_length), height=self.px_scale*self.hist_edge_length, tools=['hover'], hooks=[self.hook_yalt_limiter, self.hook_time_limiter])
            else:
                shaded = hv.Points((self.ds.event_time.data, self.ds.event_altitude.data, self.ds.event_power), kdims=['Time', 'Altitude'], vdims=['Power']
                                   ).opts(hv.opts.Points(color='Power', cmap='rainbow', size=5)).opts(xlim=(self.ds.event_time.data[0], self.ds.event_time.data[-1]), ylim=self.zlim, width=self.px_scale*(self.plan_edge_length+self.hist_edge_length), height=self.px_scale*self.hist_edge_length, tools=['hover'], line_color='black', hooks=[self.hook_yalt_limiter, self.hook_time_limiter])
        elif self.color_by == 'Event Density':
            if self.should_datashade:
                points = hv.Points((self.ds.event_time.data, self.ds.event_altitude.data, self.ds.event_power), kdims=['Time', 'Altitude'], vdims=['Power'])
                shaded = hv.operation.datashader.datashade(points, aggregator=datashader.count(), cmap='rainbow').opts(xlim=(self.ds.event_time.data[0], self.ds.event_time.data[-1]), ylim=self.zlim, width=self.px_scale*(self.plan_edge_length+self.hist_edge_length), height=self.px_scale*self.hist_edge_length, tools=['hover'], hooks=[self.hook_yalt_limiter, self.hook_time_limiter])
            else:
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
                shaded = hv.Image((timebins_ctr_dt, altbins[:-1]+altbinwidth/2, hist.T), kdims=['Time', 'Altitude'], vdims=['Event Density']
                                  ).opts(hv.opts.Image(cmap='rainbow')).opts(xlim=(self.ds.event_time.data[0], self.ds.event_time.data[-1]), ylim=self.zlim, width=self.px_scale*(self.plan_edge_length+self.hist_edge_length), height=self.px_scale*self.hist_edge_length, tools=['hover'], hooks=[self.hook_yalt_limiter, self.hook_time_limiter])
        elif self.color_by == 'Log Event Density':
            if self.should_datashade:
                points = hv.Points((self.ds.event_time.data, self.ds.event_altitude.data, self.ds.event_power), kdims=['Time', 'Altitude'], vdims=['Power'])
                shaded = hv.operation.datashader.datashade(points, aggregator=datashader.count(), cmap='rainbow', cnorm='log').opts(xlim=(self.ds.event_time.data[0], self.ds.event_time.data[-1]), ylim=self.zlim, width=self.px_scale*(self.plan_edge_length+self.hist_edge_length), height=self.px_scale*self.hist_edge_length, tools=['hover'], hooks=[self.hook_yalt_limiter, self.hook_time_limiter])
            else:
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
                shaded = hv.Image((timebins_ctr_dt, altbins[:-1]+altbinwidth/2, hist.T), kdims=['Time', 'Altitude'], vdims=['Log Event Density']
                                  ).opts(hv.opts.Image(cmap='rainbow', cnorm='log')).opts(xlim=(self.ds.event_time.data[0], self.ds.event_time.data[-1]), ylim=self.zlim, width=self.px_scale*(self.plan_edge_length+self.hist_edge_length), height=self.px_scale*self.hist_edge_length, tools=['hover'], hooks=[self.hook_yalt_limiter, self.hook_time_limiter])
        elif self.color_by == 'Altitude':
            if self.should_datashade:
                points = hv.Points((self.ds.event_time.data, self.ds.event_altitude.data, self.ds.event_power), kdims=['Time', 'Altitude'], vdims=['Power'])
                shaded = hv.operation.datashader.datashade(points, aggregator=datashader.max('Altitude'), cmap='rainbow').opts(xlim=(self.ds.event_time.data[0], self.ds.event_time.data[-1]), ylim=self.zlim, width=self.px_scale*(self.plan_edge_length+self.hist_edge_length), height=self.px_scale*self.hist_edge_length, tools=['hover'], hooks=[self.hook_yalt_limiter, self.hook_time_limiter])
            else:
                shaded = hv.Points((self.ds.event_time.data, self.ds.event_altitude.data, self.ds.event_power), kdims=['Time', 'Altitude'], vdims=['Power']
                                   ).opts(hv.opts.Points(color='Altitude', cmap='rainbow', size=5)).opts(xlim=(self.ds.event_time.data[0], self.ds.event_time.data[-1]), ylim=self.zlim, width=self.px_scale*(self.plan_edge_length+self.hist_edge_length), height=self.px_scale*self.hist_edge_length, tools=['hover'], line_color='black', hooks=[self.hook_yalt_limiter, self.hook_time_limiter])
        
        return shaded

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
        new_polys_plan = hv.Polygons([]).opts(hv.opts.Polygons(fill_alpha=0.3, fill_color='black'))
        self.plan_ax_selector.source = new_polys_plan
        
        new_lon_alt_points = self.plot_lonalt_points()
        self.lon_alt_ax_pointer.source = new_lon_alt_points
        new_polys_lonalt = hv.Polygons([]).opts(hv.opts.Polygons(fill_alpha=0.3, fill_color='black'))
        self.lon_alt_ax_selector.source = new_polys_lonalt

        new_lat_alt_points = self.plot_latalt_points()
        self.lat_alt_ax_pointer.source = new_lat_alt_points
        new_polys_latalt = hv.Polygons([]).opts(hv.opts.Polygons(fill_alpha=0.3, fill_color='black'))
        self.lat_alt_ax_selector.source = new_polys_latalt

        new_alt_time_points = self.plot_alttime_points()
        self.alt_time_pointer.source = new_alt_time_points
        new_polys_alttime = hv.Polygons([]).opts(hv.opts.Polygons(fill_alpha=0.3, fill_color='black'))
        self.alt_time_ax_selector.source = new_polys_alttime

        new_plan_ax = (new_plan_points * self.plan_ax_crosshair * new_polys_plan * self.plan_ax_select_area).opts(xlim=self.plan_points.range('Longitude'), ylim=self.plan_points.range('Latitude'))
        new_lon_alt_ax = (new_lon_alt_points * self.lon_ax_crosshair * new_polys_lonalt * self.lon_alt_select_area).opts(xlim=self.lon_alt_points.range('Longitude'), ylim=self.lon_alt_points.range('Altitude'))
        new_lat_alt_ax = (new_lat_alt_points * self.lat_ax_crosshair * new_polys_latalt * self.lat_alt_select_area).opts(xlim=self.lat_alt_points.range('Altitude'), ylim=self.lat_alt_points.range('Latitude'))
        new_hist_ax = hv.DynamicMap(self.plot_alt_hist)
        self.hist_ax = new_hist_ax
        new_alt_time_ax = (new_alt_time_points * self.time_ax_crosshair * new_polys_alttime * self.alt_time_select_area).opts(xlim=self.alt_time_points.range('Time'), ylim=self.alt_time_points.range('Altitude'))

        new_lower_part = (new_lon_alt_ax + self.hist_ax + new_plan_ax + new_lat_alt_ax).cols(2)

        self.panelHandle[2].object = new_lower_part
        self.panelHandle[1].object = new_alt_time_ax

        self.selection_geom = [np.array([]), 0]

        self.plan_points.range = self.panelHandle[2].object[2].range
        self.lon_alt_points.range = self.panelHandle[2].object[0].range
        self.lat_alt_points.range = self.panelHandle[2].object[3].range
        self.alt_time_points.range = self.panelHandle[1].object.range

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


    def init_plot(self):
        self.xlim = (self.ds.network_center_longitude.data - 3, self.ds.network_center_longitude.data + 3)
        self.ylim = (self.ds.network_center_latitude.data - 3, self.ds.network_center_latitude.data + 3)
        self.zlim = (0, 20000)

        plan_points = self.plot_planview_points()
        plan_ax_polys = hv.Polygons([]).opts(hv.opts.Polygons(fill_alpha=0.3, fill_color='black'))
        plan_ax_selector = hv.streams.PolyDraw(source=plan_ax_polys, drag=False, num_objects=1, show_vertices=True, vertex_style={'size': 5, 'fill_color': 'white', 'line_color' : 'black'})
        plan_ax_selector.add_subscriber(self.handle_selection)
        plan_ax_select_area = hv.DynamicMap(self.plan_ax_highlighter)
        plan_ax_pointer = hv.streams.PointerXY(x=0, y=0, source=plan_points).rename(x='plan_x', y='plan_y')

        lon_alt_points = self.plot_lonalt_points()
        lon_alt_ax_polys = hv.Polygons([]).opts(hv.opts.Polygons(fill_alpha=0.3, fill_color='black'))
        lon_alt_ax_selector = hv.streams.PolyDraw(source=lon_alt_ax_polys, drag=False, num_objects=1, show_vertices=True, vertex_style={'size': 5, 'fill_color': 'white', 'line_color' : 'black'})
        lon_alt_ax_selector.add_subscriber(self.handle_selection)
        lon_alt_select_area = hv.DynamicMap(self.lon_ax_highlighter)
        lon_alt_ax_pointer = hv.streams.PointerXY(x=0, y=0, source=lon_alt_points).rename(x='lon_x', y='lon_y')


        hist_ax = hv.DynamicMap(self.plot_alt_hist)

        lat_alt_points = self.plot_latalt_points()
        lat_alt_ax_polys = hv.Polygons([]).opts(hv.opts.Polygons(fill_alpha=0.3, fill_color='black'))
        lat_alt_ax_selector = hv.streams.PolyDraw(source=lat_alt_ax_polys, drag=False, num_objects=1, show_vertices=True, vertex_style={'size': 5, 'fill_color': 'white', 'line_color' : 'black'})
        lat_alt_ax_selector.add_subscriber(self.handle_selection)
        lat_alt_select_area = hv.DynamicMap(self.lat_ax_highlighter)
        lat_alt_ax_pointer = hv.streams.PointerXY(x=0, y=0, source=lat_alt_points).rename(x='lat_x', y='lat_y')


        alt_time_points = self.plot_alttime_points()
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

        lon_alt_ax = (lon_alt_points * lon_ax_crosshair * lon_alt_ax_polys * lon_alt_select_area)
        plan_ax = (plan_points * plan_ax_crosshair * plan_ax_polys * plan_ax_select_area)
        lat_alt_ax = (lat_alt_points * lat_ax_crosshair * lat_alt_ax_polys * lat_alt_select_area)
        alt_time_ax = pn.pane.HoloViews(alt_time_points * time_ax_crosshair * alt_time_ax_polys * alt_time_select_area)

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


