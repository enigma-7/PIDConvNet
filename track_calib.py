import numpy as np
import pandas as pd
from TOOLS import DEFAULTS, DATA

datadir = DEFAULTS.datadir + 'DS1/'#'000%d/all/'%run_no
ocdbdir = DEFAULTS.ocdbdir
raw_data = np.load(datadir + '0_tracks.npy')
raw_info = np.load(datadir + '0_info_set.npy') # det ()14:20) row (20:26) col (26:32) presence (32:38)

dataset, infoset, coordinates = DATA.process_track_(raw_data, raw_info)
dataset, infoset, ocdbinfo = ocdb_track_(dataset, infoset, coordinates, DEFAULTS.ocdbdir)

info_cols = ["label", "nsigmae", "nsigmap", "PT", "{dE}/{dx}", "Momenta [GeV]", "$\\eta$", "$\\theta$", "$\\phi$","run_no", "event", "V0trackID",  "track"]


# def calib_track_(dataset, infoset, )
# gainglob = np.zeros((dataset.shape[0],17,24))                   #gain for chambers
# gainpads = np.zeros((dataset.shape[0],17,24))           #gain for individual pads
# ocdbinfo = np.zeros((dataset.shape[0], chamber.values.shape[1]))
