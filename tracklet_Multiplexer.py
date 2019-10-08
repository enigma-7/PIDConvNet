import numpy as np
from TOOLS import DATA, MODELS, LOG, ML, PLOT, DEFAULTS, custom_models
import random, datetime, os, matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn import preprocessing
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPool2D, concatenate, GaussianNoise
matplotlib.rcParams.update({'font.size': 16})
matplotlib.rcParams['text.usetex'] = True

dataname = 'all_tracks_6_tracklets_even_chamber_calib/'
ocdbdir = DEFAULTS.ocdbdir
datadir = DEFAULTS.datadir + dataname
plotdir = DEFAULTS.plotdir

dataset = np.load(datadir + 'track_dataset.npy').reshape(-1,17,24,1)
infoset = np.repeat(np.load(datadir + 'track_infoset.npy'), 6, axis=0)
print("Loaded: %s \n" % datadir )

columns = DEFAULTS.info_cols_tracklet_ + DEFAULTS.ocdb_cols1 + DEFAULTS.ocdb_cols2

(X, infoset), (Xv, valid_infoset), (Xt, test_infoset) = DATA.TVT_split_(dataset, infoset)
T  = infoset[:,0].astype(int)
Tv = valid_infoset[:,0].astype(int)
Tt = test_infoset[:,0].astype(int)

tracklet_pid_model = custom_models.ComplexConvTrackletPID()

tracklet_pid_model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
				  loss='binary_crossentropy',
				  metrics=[ML.pion_con],
				  )

history = tracklet_pid_model.fit(X,
			  T,
			  batch_size=1024,
			  epochs=100,
			  validation_split=0.2,)
