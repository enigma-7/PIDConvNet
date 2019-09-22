import numpy as np
import pandas as pd
from TOOLS import DEFAULTS, DATA

run_no = 265378

ocdbdir = DEFAULTS.ocdbdir
datadir = DEFAULTS.datadir + 'DS2/'#'000%d/all/'%run_no

raw_data = np.load(datadir + '0_tracks.npy')
raw_info = np.load(datadir + '0_info_set.npy') # det ()14:20) row (20:26) col (26:32) presence (32:38)

dataset, infoset, coordinates = DATA.process_tracklet_(raw_data, raw_info)
dataset, infoset = DATA.ocdb_tracklet_(dataset, infoset, coordinates, DEFAULTS.ocdbdir)

ocdb_cols = ["Anode Voltage", "Drift Voltage", "Drift Velocity", "ExB"]
info_cols = ["label", "nsigmae", "nsigmap", "PT", "{dE}/{dx}", "Momenta [GeV]", "$\\eta$", "$\\theta$", "$\\phi$",
    "run_no", "event", "V0trackID",  "track"]

info_cols.extend(ocdb_cols)
infoframe = pd.DataFrame(data=extend, columns= info_cols)
infoframe.head(20)

infoframe[ocdb_cols].describe()

# infoframe[chamber.columns.values].describe()
#
# ranges = infoframe["ExB"].describe().values[3:]
# [elec_dataset, elec_infoset], [pion_dataset, pion_infoset] = DATA.elec_pion_split_(dataset, infoset)
# class_data = [pion_dataset, elec_dataset]
# class_info = [pion_infoset, elec_infoset]
#
# avetracklets = np.zeros((4,17,24))
# for i in range(4):
#     avetracklets[i] = (elec_dataset[np.logical_and((elec_infoset[:,16] < ranges[i+1]),
#         (elec_infoset[:,16] > ranges[i]))].mean(axis=(0,3)))
#
# import matplotlib.pyplot as plt
#
# plt.imshow(avetracklets[0])
#
# plt.imshow(avetracklets[1])
#
# plt.imshow(avetracklets[2])
#
# plt.imshow(avetracklets[3])
