from lma_data_explorer import LMADataExplorer
import panel as pn
from os import path, listdir

filenames = [path.join('data', f) for f in listdir('data')]

px_scale = 7
width_of_major = 60*px_scale

## left controls
datashader_label = pn.widgets.StaticText(value=f'Enable Datashader? ( src)', width=(width_of_major*3)//4)
datashader_switch = pn.widgets.Switch(value=True)
datashader_switch_row = pn.Row(datashader_label, datashader_switch)
# TODO: what is 'by points', 'charge density'?, 'pts above threshold'?
color_by_selector = pn.widgets.Select(name='Color By', options=['Time', 'Charge (User Assigned)', 'Charge (chargepol)',
                                                                'Power (dBW)', 'Event Density', 'Log Event Density', 'Altitude'], value='Time')
event_filter_type_selector = pn.widgets.Select(options=[], value=None, width=width_of_major//4)
event_filter_operation_selector = pn.widgets.Select(name='', options=['==', '>', '<', '>=', '<=', '!='], value='<=', width=int(width_of_major*.15))
event_filter_value_selector = pn.widgets.FloatInput(value=0.0, width=width_of_major//5)
event_filter_add = pn.widgets.Button(name='Apply', button_type='success', width=width_of_major//10) # 
event_filter_controls = pn.Row(event_filter_type_selector, event_filter_operation_selector, event_filter_value_selector, event_filter_add)
event_filter_history = pn.Row(pn.Column(width=width_of_major//2), pn.Column(width=width_of_major//2), width=width_of_major)
dataset_html = pn.pane.HTML()

## Right controls
limit_button = pn.widgets.Button(name='Limit to Selection', button_type='primary', width=width_of_major//5)
mark_minus_button = pn.widgets.Button(icon='minus', button_type='primary', width=width_of_major//5)
mark_unassigned_button = pn.widgets.Button(icon='circle-off', button_type='success', width=width_of_major//5)
mark_plus_button = pn.widgets.Button(icon='plus', button_type='danger', width=width_of_major//5)
charge_buttons_row = pn.Row(mark_minus_button, mark_unassigned_button, mark_plus_button, width=width_of_major)
cluster_button = pn.widgets.Button(name='Cluster Flashes', button_type='primary', width=width_of_major//5)

flash_id_selector = pn.widgets.IntInput()
flash_id_button = pn.widgets.Button(name='Go', button_type='primary', width=width_of_major//10)
prev_flash_step = pn.widgets.Button(name='Previous', button_type='primary', width=width_of_major//10)
next_flash_step = pn.widgets.Button(name='Next', button_type='primary', width=width_of_major//10)
flash_id_controls = pn.Column(pn.Row(flash_id_selector, flash_id_button), pn.Row(prev_flash_step, next_flash_step))

flash_filter_type_selector = pn.widgets.Select(options=[], value=None, width=width_of_major//4)
flash_filter_operation_selector = pn.widgets.Select(name='', options=['==', '>', '<', '>=', '<=', '!='], value='<=', width=int(width_of_major*.15))
flash_filter_value_selector = pn.widgets.FloatInput(value=0.0, width=width_of_major//5)
flash_filter_add = pn.widgets.Button(name='Apply', button_type='success', width=width_of_major//10)
flash_filter_controls = pn.Row(flash_filter_type_selector, flash_filter_operation_selector, flash_filter_value_selector, flash_filter_add)
flash_filter_history = pn.Row(pn.Column(width=width_of_major//2), pn.Column(width=width_of_major//2), width=width_of_major)


## init lma explorer object
lmae = LMADataExplorer(filenames, 7, color_by_selector, datashader_switch, datashader_label, event_filter_controls, event_filter_history, dataset_html, flash_id_selector, flash_filter_controls, flash_filter_history)


# Bind widgets to callbacks
pn.bind(lmae.limit_to_polygon, limit_button, watch=True)
pn.bind(lmae.limit_to_filter, event_filter_add, watch=True)
pn.bind(lmae.mark_polygon, mark=1, unused=mark_plus_button, watch=True)
pn.bind(lmae.mark_polygon, mark=-1, unused=mark_minus_button, watch=True)
pn.bind(lmae.mark_polygon, mark=0, unused=mark_unassigned_button, watch=True)
pn.bind(lmae.cluster_flashes, cluster_button, watch=True)
pn.bind(lmae.view_flash_id, flash_id_button, watch=True)


## Assemble layout
left_controls = pn.Column(datashader_switch_row, color_by_selector, pn.pane.HTML('<h4>Event Filters:</h4>'), event_filter_controls, event_filter_history, pn.pane.HTML('<h4>Dataset Live View:</h4>'), dataset_html, width=width_of_major, height=900)
right_controls = pn.Column(limit_button, pn.pane.HTML('<h4>Charge Assignment:</h4>'), charge_buttons_row, pn.pane.HTML('<h4>Flash Tools:</h4>'), cluster_button, pn.pane.HTML('<h4>View Flash ID:</h4>'), flash_id_controls, pn.pane.HTML('<h4>Flash Filters:</h4>'), flash_filter_controls, flash_filter_history, width=width_of_major)
the_layout = pn.Row(left_controls, lmae.panelHandle, right_controls)

## Start server
pn.serve(the_layout)
