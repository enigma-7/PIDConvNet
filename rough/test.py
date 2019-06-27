import numpy as np
import matplotlib.pyplot as plt
import matplotlib, DATA, MODELS, time, datetime, yaml, json, yaml, random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.callbacks import TensorBoard
from yaml import Loader, Dumper

run_no = '000265378/'
dataname = 'test/'
directory = 'data/output/' + run_no + dataname
raw_data = np.load(directory + '0_tracks.npy')
raw_info = np.load(directory + '0_info_set.npy')
print("Loaded: %s" % directory)

sumtracks = [np.sum(raw_info[:,6] == i) for i in range(7)]               #Sanity check
dataset, infoset = DATA.process_1(raw_data, raw_info)

X, y = DATA.shuffle_(dataset/1024, infoset[:,0])

print("Electron occurence: %.2f" % (100*sum(y)/len(y)))
