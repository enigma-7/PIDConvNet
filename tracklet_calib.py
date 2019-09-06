import numpy as np
import pandas as pd
from TOOLS import DEFAULTS, DATA


runs = [265378, 265381, 265383, 265385, 265420]
run_no = 265378

ocdbdir = DEFAULTS.ocdbdir
datadir = DEFAULTS.datadir + 'DS1/'#'000%d/all/'%run_no

raw_data = np.load(datadir + '0_tracks.npy')
raw_info = np.load(datadir + '0_info_set.npy') # det ()14:20) row (20:26) col (26:32) presence (32:38)

dataset, infoset, coordinates = DATA.process_tracklet_(raw_data, raw_info)
calib = DATA.calib_track_(dataset, infoset, coordinates, DEFAULTS.ocdbdir)

def calib_tracklet_(dataset, infoset, coordinates, ocdbdir):
    R = infoset[:,9].astype('int')
    runs = set(R)

    for run in runs:
        print(run)
        gainglob = pd.read_csv(ocdbdir + 'chamber_info_2016_%d.txt'%run, header = None).values[:,3]
        gainlocl = pd.read_csv(ocdbdir + 'local_gains_2016_%d.txt'%run,
            header = None).values.reshape((540, 16,-1))[:,:,2:]           #(detector, row, column)

        gainG = np.ones(dataset.shape)                   #gain for chambers
        gainP = np.ones(dataset.shape)
        mask = np.where(R==run, range(dataset.shape[0]), -1)

        for i, [d, r, c] in enumerate(coordinates):
            if i == mask[i]:
                gainP[i,:,:,0] = np.tile(gainlocl[d, r, c-8:c+9],(24,1)).T
                gainG[i,:,:,0] = np.tile(gainglob[d],(17,24))

        dataset = np.multiply(np.multiply(dataset, gainP),gainG)
    return dataset

gainG = np.ones(dataset.shape)                   #gain for chambers
gainP = np.ones(dataset.shape)
for i, [d, r, c] in enumerate(coordinates[i]):
    gainP[i,:,:,0] = np.tile(gainlocl[d, r, c-8:c+9],(24,1)).T
    gainG[i,:,:,0] = np.tile(gainglob[d],(17,24))
calib = calib_tracklet_(dataset, infoset, coordinates, ocdbdir)

calib_cols = ["Detector", "Anode Voltage", "Drift Voltage", "Gain", "Drift Velocity", "ExB"]
info_cols = ["label", "nsigmae", "nsigmap", "PT", "{dE}/{dx}", "Momenta [GeV]", "$\\eta$", "$\\theta$", "$\\phi$",
    "run_no", "event", "V0trackID",  "track"]

chamber = pd.read_csv(ocdbdir + 'chamber_info_2016_%d.txt'%run_no, header = None)
chamber.columns = calib_cols

globalgains = chamber["Gain"].values
localgains = pd.read_csv(ocdbdir + 'local_gains_2016_%d.txt'%run_no, header = None)
localgains = localgains.values.reshape((540, 16,-1))[:,:,2:]            #(detector, row, pads)

chamber.drop(columns=["Detector","Gain"], inplace=True)
chamber.describe()

gainglob = np.zeros((dataset.shape[0],17,24))                   #gain for chambers
gainpads = np.zeros((dataset.shape[0],17,24))           #gain for individual pads
ocdbinfo = np.zeros((dataset.shape[0], chamber.values.shape[1]))

for i, [d, r, c] in enumerate(coordinates):
    gainglob[i] = np.tile(globalgains[d],(17,24))
    gainpads[i] = np.tile(localgains[d, r, c-8:c+9],(24,1)).T
    ocdbinfo[i] = chamber.values[d]

gainglob = gainglob.reshape(-1,17,24,1)
gainpads = gainpads.reshape(-1,17,24,1)

Xcalib = np.multiply(np.multiply(dataset, gainpads),gainglob)
#
# info_cols.extend(chamber.columns.values)
# infoset = np.append(infoset, ocdbinfo, axis=1)
# infoframe = pd.DataFrame(data=infoset, columns= info_cols)
# infoframe.head(5)
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
