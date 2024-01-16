from lma_data_explorer import LMADataExplorer
import panel as pn

## init lma explorer object
filename = 'LYLOUT_230615_200000_0600.dat.gz'
lmae = LMADataExplorer(filename)

## left controls

num_src = lmae.ds.number_of_events.data.shape[0]
datashader_label = pn.widgets.StaticText(value=f'Enable Datashader? ({num_src} src)')
datashader_switch = pn.widgets.Switch(value=True)
datashader_switch_row = pn.Row(datashader_label, datashader_switch)
# TODO: what is 'by points', 'charge density'?, 'pts above threshold'?
color_by_selector = pn.widgets.Select(name='Color By', options=['Time', 'Charge (User Assigned)', 'Charge (chargepol)',
                                                                'Power (dBW)', 'Event Density', 'Log Event Density', 'Altitude'], value=lmae.color_by)

## Right controls
limit_button = pn.widgets.Button(name='Limit to Selection', button_type='primary')




## Define callbacks
def datashader_switch_callback(datashader_switch_val):
    lmae.should_datashade = datashader_switch_val
    lmae.rerender()

def polygon_limit_btn_callback(limit_button):
    print('limiting to polygon')
    lmae.limit_to_polygon()
    global datashader_label
    datashader_label.value = f'Enable Datashader? ({lmae.ds.number_of_events.data.shape[0]} src)'





# Bind widgets to callbacks
pn.bind (datashader_switch_callback, datashader_switch, watch=True)
pn.bind(lmae.change_colorby, color_by_selector, watch=True)
pn.bind(polygon_limit_btn_callback, limit_button, watch=True)


## Assemble layout
left_controls = pn.Column(datashader_switch_row, color_by_selector)
right_controls = pn.Column(limit_button)
the_layout = pn.Row(left_controls, lmae.panelHandle, right_controls)

## Start server
the_layout.servable()
