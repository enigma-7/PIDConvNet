import numpy as np
from TOOLS import DATA, MODELS, LOG, ML, PLOT, DEFAULTS

for i in range(1,4):
dataname = 'DS4/'
ocdbdir = DEFAULTS.ocdbdir
datadir = DEFAULTS.datadir + dataname
plotdir = DEFAULTS.plotdir

dataset = np.load(datadir + '0_tracks.npy')
infoset = np.load(datadir + '0_info_set.npy')
print("Loaded: %s \n" % datadir )

dataset, infoset, coordinates = DATA.process_tracklet_(dataset, infoset)
#infoset = DATA.ocdb_expand_(infoset, ocdbdir)

np.save(datadir + 'tracklet_dataset.npy', dataset)
np.save(datadir + 'tracklet_infoset.npy', infoset)
print(dataset.shape)
