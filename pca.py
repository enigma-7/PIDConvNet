import numpy as np
import matplotlib.pyplot as plt
from TOOLS import DATA, MODELS, LOG, METRICS
import random, matplotlib, datetime, os

run_no = '000265378/'
dataname = 'even/'
directory = 'data/output/' + run_no + dataname
raw_data = np.load(directory + '0_tracks.npy')
raw_info = np.load(directory + '0_info_set.npy')
print("Loaded: %s" % directory)

raw_data, raw_info = DATA.shuffle_(raw_data, raw_info)
columns = ["label", "nsigmae", "nsigmap", "PT", "${dE}/{dx}$", "Momenta [GeV]", "eta", "theta", "phi", "event", "V0trackID",  "track"]

params = raw_info[:,3:9]
parammean = params.sum(axis=0)/params.shape[0]
paramstdv = np.sqrt(np.square((params - parammean)).sum(axis=0)/(params.shape[0]-1))
P = (params - parammean)
Pcovariance = np.matmul(P , P.T)/P.shape[0]

s = 80
n = 5
raw_info[s:s+n,0]
print(np.around(Pcovariance[s:s+n,s:s+n],2))
