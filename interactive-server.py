from lma_data_explorer import LMADataExplorer
import panel as pn

## init lma explorer object
filename = 'LYLOUT_230615_200000_0600.dat.gz'

## left controls

datashader_label = pn.widgets.StaticText(value=f'Enable Datashader? ( src)')
datashader_switch = pn.widgets.Switch(value=True)
datashader_switch_row = pn.Row(datashader_label, datashader_switch)
# TODO: what is 'by points', 'charge density'?, 'pts above threshold'?
color_by_selector = pn.widgets.Select(name='Color By', options=['Time', 'Charge (User Assigned)', 'Charge (chargepol)',
                                                                'Power (dBW)', 'Event Density', 'Log Event Density', 'Altitude'], value='Time')

## Right controls
limit_button = pn.widgets.Button(name='Limit to Selection', button_type='primary')

lmae = LMADataExplorer(filename, color_by_selector, datashader_switch, datashader_label)


# Bind widgets to callbacks
pn.bind(lmae.limit_to_polygon, limit_button, watch=True)


## Assemble layout
left_controls = pn.Column(datashader_switch_row, color_by_selector)
right_controls = pn.Column(limit_button)
the_layout = pn.Row(left_controls, lmae.panelHandle, right_controls)

## Start server
pn.serve(the_layout)
