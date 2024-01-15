from lma_data_explorer import LMADataExplorer
import panel as pn

filename = 'LYLOUT_230615_200000_0600.dat.gz'

lmae = LMADataExplorer(filename)

limit_button = pn.widgets.Button(name='Limit to Selection', button_type='primary')
pn.bind(lmae.limit_to_polygon, limit_button, watch=True)


# TODO: what is 'by points', 'charge density'?, 'pts above threshold'?
color_by_selector = pn.widgets.Select(name='Color By', options=['Time', 'Charge (User Assigned)', 'Charge (chargepol)',
                                                                'Power (dBW)', 'Event Density', 'Log Event Density', 'Altitude'], value=lmae.color_by)
pn.bind(lmae.change_colorby, color_by_selector, watch=True)

left_controls = pn.Column(color_by_selector)
right_controls = pn.Column(limit_button)


pn.Row(left_controls, lmae.panelHandle, right_controls).servable()