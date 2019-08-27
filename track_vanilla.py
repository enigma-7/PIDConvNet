import numpy as np
from TOOLS import DATA, MODELS, LOG, METRICS, PLOT, DEFAULTS
import random, datetime, os, matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPool2D, concatenate, GaussianNoise
import tensorflow as tf
matplotlib.rcParams.update({'font.size': 18})
matplotlib.rcParams['text.usetex'] = True

run_no = '000265378/'
dataname = 'all/'
directory =  DEFAULTS.datadir + 'all1/'
raw_data = np.load(directory + '0_tracks.npy')
raw_info = np.load(directory + '0_info_set.npy')
print("Loaded: %s" % directory)

dataset, infoset = DATA.process_track_(raw_data, raw_info)
X , infoset = DATA.shuffle_(dataset/1024, infoset)

(X, infoset), (Xv, valid_infoset), (Xt, test_infoset) = DATA.TVT_split_(X, infoset)
T  = infoset[:,0]
Tv = valid_infoset[:,0]
Tt = test_infoset[:,0]
# I  = infoset[:,nx:ny]
# Iv = valid_infoset[:,nx:ny]
# It = test_infoset[:,nx:ny]# I  = infoset[:,nx:ny]

(cs_1, cs_2, d1_1, d1_2) = (8, 16, 128, 64)

stamp = datetime.datetime.now().strftime("%d-%m-%H%M%S") + "_"
mname = "conv-%d-%d-dense-%d-%d"%(cs_1, cs_2, d1_1, d1_2)
#tensorboard, csvlogger = LOG.logger_(run_no, dataname, stamp, mname)

input = Input(shape=X.shape[1:], name="X-in")
x = Conv2D(cs_1, [2,3], activation='relu', padding ='same')(input)
x = MaxPool2D([2,2], 2, padding='valid')(x)
x = Conv2D(cs_2, [2,3], activation='relu', padding='same')(x)
x = MaxPool2D([2,2], 2, padding='valid')(x)
x = Flatten()(x)
x = GaussianNoise(10/1024)(x)
x = Dense(d1_1)(x)
x = Dense(d1_2)(x)
x = Dropout(0.3)(x)
output = Dense(1, activation='sigmoid', name="X-out")(x)

model = Model(inputs=input, outputs=output)
model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=1e-3), loss='binary_crossentropy', metrics=[METRICS.pion_con])
model.fit(x=X, y=T, batch_size = 2**8, epochs=10, validation_data=(Xv,Tv))#), callbacks=[tensorboard, csvlogger])

P = model.predict(Xt).reshape(Xt.shape[0])
PLOT.classification_(P, Tt, DEFAULTS.plotdir + mname + "_class.png", save = False)

e_pred = P[Tt==1]
p_pred = P[Tt==0]
argsort = e_pred.argsort()

thresholds=np.linspace(0,1,1000)

TP = np.array([e_pred[e_pred>threshold].sum() for threshold in thresholds])
FN = np.array([e_pred[e_pred<threshold].sum() for threshold in thresholds])
FP = np.array([p_pred[p_pred>threshold].sum() for threshold in thresholds])
TN = np.array([p_pred[p_pred<threshold].sum() for threshold in thresholds])

TPR = TP/(FN+TP)            #True Positive Rate / Recall
FPR = FP/(TN+FP)            #False Positive Rate
PPV = TP/(TP+FP+1e-5)            #Positive Predictive Value / Precision
PPV

thresholds[TPR<0.905][0]

pioncon = FPR[TPR>0.9][-1]          #estimate pion contamination
decbound = e_pred[argsort[np.multiply(argsort.shape[-1],(1-90/100)).astype(int)]]   #estimate decision boundary
print(decbound)
AUC = np.sum(np.abs(np.diff(FPR)*TPR[1:]))          #estimate area under curve
