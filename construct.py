import numpy as np
from TOOLS import DATA, MODELS, LOG, ML, PLOT, DEFAULTS

# for i in range(1,5):
#     dataname = 'DS%d/'%i
#     ocdbdir = DEFAULTS.ocdbdir
#     datadir = DEFAULTS.datadir + dataname
#     plotdir = DEFAULTS.plotdir
#
#     dataset = np.load(datadir + '0_tracks.npy')
#     infoset = np.load(datadir + '0_info_set.npy')
#     print("Loaded: %s \n" % datadir )
#
#     mask = infoset[:,9] != 0
#     print(mask.sum())
#
#     np.save(datadir + '0_tracks.npy', dataset[mask])
#     np.save(datadir + '0_info_set.npy', infoset[mask])
#     print(dataset.shape)

# dataname = 'all_tracks_6_tracklets_even_chamber_calib/'
# dataname  = 'DS1/'

for i in range(1,5):
    dataname = 'DS%d/'%i
    ocdbdir = DEFAULTS.ocdbdir
    datadir = DEFAULTS.datadir + dataname
    plotdir = DEFAULTS.plotdir

    dataset = np.load(datadir + '0_tracks.npy')
    infoset = np.load(datadir + '0_info_set.npy')
    # dataset, infoset = DATA.shuffle_(dataset, infoset)
    print("Loaded: %s \n" % datadir )

    dataset, infoset = DATA.process_tracklet_(dataset, infoset)
    # dataset, infoset = DATA.ocdb_tracklet_(dataset, infoset, ocdbdir)
    infoset = DATA.ocdb_expand_(infoset, ocdbdir)

    np.save(datadir + 'tracklet_dataset.npy', dataset)
    np.save(datadir + 'tracklet_infoset.npy', infoset)
    print(dataset.shape)

# for i in range(1,5):
#     dataname = 'DS%d_pad/'%i
#     ocdbdir = DEFAULTS.ocdbdir
#     datadir = DEFAULTS.datadir + dataname
#     plotdir = DEFAULTS.plotdir
#
#     dataset = np.load(datadir + '0_tracks.npy')
#     infoset = np.load(datadir + '0_info_set.npy')
#     dataset, infoset = DATA.shuffle_(dataset, infoset)
#     print("Loaded: %s \n" % datadir )
#
#     dataset, infoset, coordinates = DATA.process_track_(dataset, infoset)
#
#     np.save(datadir + 'track_dataset.npy', dataset)
#     np.save(datadir + 'track_infoset.npy', infoset)
#     print(dataset.shape)
