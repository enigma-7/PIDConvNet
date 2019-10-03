import numpy as np
from TOOLS import DATA, MODELS, LOG, ML,PLOT, DEFAULTS
import random, datetime, os
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPool2D, concatenate, GaussianNoise
matplotlib.rcParams.update({'font.size': 16})
matplotlib.rcParams['text.usetex'] = True

dataname = 'DS2/'
ocdbdir = DEFAULTS.ocdbdir
datadir = DEFAULTS.datadir + dataname
plotdir = DEFAULTS.plotdir

dataset = np.load(datadir + 'tracklet_dataset.npy')
infoset = np.load(datadir + 'tracklet_infoset.npy')

print("Loaded: %s \n" % datadir )

(X, infoset), (Xv, valid_infoset), (Xt, test_infoset) = DATA.TVT_split_(dataset/1024, infoset)
T  = infoset[:,0].astype(int)
Tv = valid_infoset[:,0].astype(int)
Tt = test_infoset[:,0].astype(int)
I  = infoset
Iv = valid_infoset
It = test_infoset# I  = infoset[:,nx:ny]

(cs_1, cs_2, d1_1, d1_2) = (8, 16, 128, 64)
mname = "Utracklet_" + "C-%d-%d-D-%d-%d"%(cs_1, cs_2, d1_1, d1_2)
# stamp = datetime.datetime.now().strftime("%d-%m-%H%M%S") + "_"
# tensorboard, csvlogger = LOG.logger_(dataname, stamp, mname)

input = Input(shape=X.shape[1:],)# name="X-in")
x = Conv2D(cs_1, [2,3], activation='relu', padding ='same')(input)
x = MaxPool2D([2,2], 2, padding='valid')(x)
x = Conv2D(cs_2, [2,3], activation='relu', padding='same')(x)
x = MaxPool2D([2,2], 2, padding='valid')(x)
x = Flatten()(x)
x = Dense(d1_1)(x)
x = Dense(d1_2)(x)
output = Dense(1, activation='sigmoid',)(x)# name="X-out")(x)

model = Model(inputs=input, outputs=output)
model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=1e-3), loss='binary_crossentropy', metrics=[ML.pion_con])
model.fit(x=X, y=T, batch_size = 2**9, epochs=10, validation_data=(Xv,Tv))#, callbacks=[tensorboard, csvlogger])

"""## learning_rate = 1e-3, batch_size = 2**8 ##
frame, names = LOG.import_(DEFAULTS.tracklet_uncalib)

array = frame.values.reshape(-1,20,5)
colours = ['red', 'goldenrod', 'green', 'blue', 'purple']

fig, axes = plt.subplots(1, 2, figsize=(12,6))
axes[0].plot(array[0,:,2].cumsum(), array[0,:,0], color='black', label="Training")
axes[0].plot(array[0,:,2].cumsum(), array[0,:,3], color='black', label="Validation", linestyle='--')
axes[0].set_ylabel("Loss")
axes[0].set_title("(a)")
axes[1].plot(array[0,:,2].cumsum(), array[0,:,1], color='black', label="Training")
axes[1].plot(array[0,:,2].cumsum(), array[0,:,4], color='black', label="Validation", linestyle='--')
axes[1].set_ylabel("Pion Contamination")
axes[1].set_title("(b)")
for i in range(2):
    axes[i].grid()
    axes[i].set_xlabel("Training Time [$s$]")
plt.legend()
# plt.savefig(plotdir + names[0] + ".png")
"""

P = model.predict(Xt).reshape(-1)
PLOT.classification_(P,Tt)

columns = DEFAULTS.info_cols_tracklet_ + DATA.ocdb_cols1

Ltrack, Ttrack = DATA.likelihood_(P, It)
PLOT.classification_(Ltrack, Ttrack)
