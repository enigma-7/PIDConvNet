import numpy as np
from TOOLS import DATA, MODELS, LOG, ML, PLOT, DEFAULTS
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

infoset = np.load(DEFAULTS.datadir + 'DS4/' + 'tracklet_infoset.npy')
dataset = np.load(DEFAULTS.modldir + 'P-DS4-tracklet_V_U_wbce5.npy')

(I[:,13] ==5.0).sum()
mask = I[:,13] == 6.0
I = I[mask]
X = X[mask]

(X, I), (Xv, Iv), (Xt, It)= DATA.TVT_split_(X, I)
T  = I[:,0].astype(int)
Tv = Iv[:,0].astype(int)
Tt = It[:,0].astype(int)
A  = I[:,14:]
Av = Iv[:,14:]
At = It[:,14:]

input_X = Input(shape=X.shape[1:], name="info")
x = Dense(64)(input_X)
x = Dense(8)(x)
output_aux = Dense(1, activation='sigmoid')(x)

model = Model(inputs=input_X, outputs=output_aux)
#model.summary()
model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=1e-3), loss=ML.WBCE,metrics=[ML.pion_con])
model.fit(X, T, batch_size=2**8, epochs=100, validation_data=(Xv,Tv), )
# model.summary()
P = model.predict(Xt)
PLOT.classification_(P,Tt)
