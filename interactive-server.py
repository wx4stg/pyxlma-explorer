from lma_data_explorer import LMADataExplorer
import panel as pn
from os import path, listdir

filenames = [path.join('data', f) for f in listdir('data')]

px_scale = 7
width_of_major = 60*px_scale

## init lma explorer object
lmae = LMADataExplorer(filenames=filenames)

## left controls
datashader_label = pn.widgets.StaticText.from_param(lmae.param.datashade_label, name='', width=(width_of_major*3)//4)
datashader_switch = pn.widgets.Switch.from_param(lmae.param.should_datashade)
datashader_switch_row = pn.Row(datashader_label, datashader_switch)
# TODO: what is 'by points', 'charge density'?, 'pts above threshold'?
color_by_selector = pn.widgets.Select.from_param(lmae.param.color_by, name='Color By:')
event_filter_type_selector = pn.widgets.Select.from_param(lmae.param.event_filter_type_selector, name='', width=width_of_major//4)
event_filter_operation_selector = pn.widgets.Select.from_param(lmae.param.event_filter_op_selector, name='', width=int(width_of_major*.15))
event_filter_value_input = pn.widgets.FloatInput.from_param(lmae.param.event_filter_value_input, name='', width=width_of_major//5)
event_filter_add = pn.widgets.Button.from_param(lmae.param.event_filter_add, name='Apply', button_type='success', width=width_of_major//10)
event_filter_controls = pn.Row(event_filter_type_selector, event_filter_operation_selector, event_filter_value_input, event_filter_add)
lmae.event_filter_table.width = width_of_major

## Right controls
limit_button = pn.widgets.Button.from_param(lmae.param.limit_button, name='Limit to Selection', button_type='primary', width=width_of_major//5)
mark_minus_button = pn.widgets.Button.from_param(lmae.param.mark_minus_button, name='', icon='minus', button_type='primary', width=width_of_major//5)
mark_unassigned_button = pn.widgets.Button.from_param(lmae.param.mark_unassigned_button, name='', icon='circle-off', button_type='success', width=width_of_major//5)
mark_plus_button = pn.widgets.Button.from_param(lmae.param.mark_plus_button, name='', icon='plus', button_type='danger', width=width_of_major//5)
charge_buttons_row = pn.Row(mark_minus_button, mark_unassigned_button, mark_plus_button, width=width_of_major)
cluster_button = pn.widgets.Button.from_param(lmae.param.cluster_button, name='Cluster Flashes', button_type='primary', width=width_of_major//5)

flash_id_selector = pn.widgets.IntInput.from_param(lmae.param.flash_id_selector, name='')
flash_id_button = pn.widgets.Button.from_param(lmae.param.flash_id_button, name='Go', button_type='primary', width=width_of_major//10)
prev_flash_step = pn.widgets.Button.from_param(lmae.param.prev_flash_step, name='Previous', button_type='primary', width=width_of_major//10)
next_flash_step = pn.widgets.Button.from_param(lmae.param.next_flash_step, name='Next', button_type='primary', width=width_of_major//10)
flash_id_controls = pn.Column(pn.Row(flash_id_selector, flash_id_button), pn.Row(prev_flash_step, next_flash_step))

flash_filter_type_selector = pn.widgets.Select.from_param(lmae.param.flash_filter_type_selector, name='', width=width_of_major//4)
flash_filter_operation_selector = pn.widgets.Select.from_param(lmae.param.flash_filter_op_selector, name='', width=int(width_of_major*.15))
flash_filter_value_input = pn.widgets.FloatInput.from_param(lmae.param.flash_filter_value_input, name='', width=width_of_major//5)
# TODO: flash_filter_add = pn.widgets.Button.from_param(lmae.param.flash_filter_add, name='Apply', button_type='success', width=width_of_major//10)
flash_filter_controls = pn.Row(flash_filter_type_selector, flash_filter_operation_selector, flash_filter_value_input)#, flash_filter_add)

## Assemble layout
left_controls = pn.Column(datashader_switch_row, color_by_selector, pn.pane.HTML('<h4>Event Filters:</h4>'), event_filter_controls, lmae.event_filter_table, pn.pane.HTML('<h4>Dataset Live View:</h4>'), lmae.dataset_html, width=width_of_major, height=900)
right_controls = pn.Column(limit_button, pn.pane.HTML('<h4>Charge Assignment:</h4>'), charge_buttons_row, pn.pane.HTML('<h4>Flash Tools:</h4>'), cluster_button, pn.pane.HTML('<h4>View Flash ID:</h4>'), flash_id_controls, pn.pane.HTML('<h4>Flash Filters:</h4>'), flash_filter_controls, lmae.flash_filter_table, width=width_of_major)
the_layout = pn.Row(left_controls, lmae.panel_handle, right_controls)

## Start server
the_layout.servable()
