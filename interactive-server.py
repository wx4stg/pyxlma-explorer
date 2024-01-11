from lma_data_explorer import LMADataExplorer

filename = 'LYLOUT_230615_200000_0600.dat.gz'

lma = LMADataExplorer(filename)
lma.panelHandle.servable()
