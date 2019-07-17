import numpy as np
import matplotlib.pyplot as plt
from TOOLS import DATA, MODELS, LOG, METRICS, PLOT
import random, matplotlib, datetime, os

run_no = '000265378/'
dataname = 'all/'
directory = 'data/output/' + run_no + dataname
raw_data = np.load(directory + '0_tracks.npy')
raw_info = np.load(directory + '0_info_set.npy')
print('Loaded: %s' % directory)

dataset, infoset = DATA.process_1(raw_data, raw_info)

LQ1arr = DATA.bin_time_(dataset).sum(axis=(1,2,3))
(elec_LQ1arr, elec_y), (pion_LQ1arr, pion_y) = DATA.elec_pion_split_(LQ1arr,infoset)

plt.hist(elec_LQ1arr, alpha=0.5, label = 'positive', normed=True)
plt.hist(pion_LQ1arr, alpha=0.5, label = 'negative', normed=True)
#plt.axvline(x=cutoff, label='Threshold')
plt.legend(loc=9)
plt.xscale('log')
plt.yscale('log')
plt.show()
