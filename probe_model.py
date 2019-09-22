import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import random, matplotlib
from TOOLS import PLOT, METRICS, DATA, DEFAULTS
matplotlib.rcParams.update({'font.size': 16})
matplotlib.rcParams['text.usetex'] = True

from keras.models import load_model
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform
from scipy.signal import convolve2d

directory = DEFAULTS.datadir + 'DS2/'
raw_data = np.load(directory + '0_tracks.npy')
raw_info = np.load(directory + '0_info_set.npy')
print("Loaded: %s" % directory)

sumtracks = [np.sum(raw_info[:,6] == i) for i in range(7)]               #Sanity check
dataset, infoset, coord = DATA.process_tracklet_(raw_data, raw_info)

X, y = DATA.shuffle_(dataset/1024, infoset[:,0])
plt.imshow(X[0][:,:,0])

modldir = "saved_models/"
conv_sizes1 = [16]
conv_sizes2 = [64]
dense_sizes1 = [1024]
dense_sizes2 = [256]

for conv_size1 in conv_sizes1:
    for conv_size2 in conv_sizes2:
        for dense_size1 in dense_sizes1:
            for dense_size2 in dense_sizes2:
                mname = "conv-%d-%d-filters-dense-%d-%d-nodes-"%(conv_size1,
                    conv_size2, dense_size1, dense_size2)
                with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
                    model = load_model(modldir + mname + ".h5", custom_objects={'F1': METRICS.F1})

conv_1 = model.get_weights()
num = random.randint(1,dataset.shape[0])
tracklet = dataset[num,:,:,0]
trackids = infoset[num][-3:].astype(int)

plt.imshow(tracklet)
plt.title("Displaying tracklet %s:"%"/".join([str(i) for i in trackids]))
output = np.array([convolve2d(tracklet, conv_1[:,:,0,i], mode='same') for i in range(16)])
output = output.swapaxes(0,1).swapaxes(1,2)
output = output.reshape(output.shape[0],output.shape[1],1,output.shape[2])

shapearr =  [model.get_weights()[i].shape for i in range(len(model.get_weights()))]
shapearr
#PLOT.tileplot_(conv_1)
#PLOT.tileplot_(output, title="Convolved tracklets %s:"%"/".join([str(i) for i in trackids]))

model

input_aux = Input(shape=infoarray.shape[1:], name="kinematics")
x = concatenate([input_aux, flattened])
x = Dense(256)(x)
x = Dense(64)(x)
output_aux = Dense(1, activation='sigmoid', name="output_aux")(x)
y = Dense(256)(flattened)
y = Dense(64)(y)
output_non = Dense(1, activation='sigmoid', name="output_non")(y)
