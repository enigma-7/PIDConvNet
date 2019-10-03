import numpy as np
from TOOLS import DATA, MODELS, LOG, ML,PLOT, DEFAULTS
import random, datetime, os
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPool2D, concatenate, GaussianNoise
matplotlib.rcParams.update({'font.size': 16})
matplotlib.rcParams['text.usetex'] = True

dataname = 'DS3/'
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
P = np.load(DEFAULTS.modldir + "P-tracklet_V_U_ambient.npy")

PLOT.classification_(P,Tt)

Ltrack, Ttrack, layers = DATA.likelihood_(P, It)
[(layers==i).sum() for i in range(1,7)]

thresholds=np.linspace(0,1,100)
pioncon = np.zeros(6)
eleceff = np.zeros(6)
decbound = np.zeros(6)
for i in range(1,7):
    mask = layers == i
    L = Ltrack[mask]
    T = Ttrack[mask]
    e_pred = L[T==1]
    p_pred = L[T==0]
    argsort = e_pred.argsort()

    TP = np.array([(e_pred>threshold).sum() for threshold in thresholds])
    FN = np.array([(e_pred<threshold).sum() for threshold in thresholds])
    FP = np.array([(p_pred>threshold).sum() for threshold in thresholds])
    TN = np.array([(p_pred<threshold).sum() for threshold in thresholds])

    TPR = TP/(FN+TP)            #True Positive Rate / Recall
    FPR = FP/(TN+FP)            #False Positive Rate

    pioncon[i-1] = FPR[TPR<0.905][0]          #estimate pion contamination
    eleceff[i-1] = TPR[TPR<0.905][0]
    decbound[i-1] = thresholds[TPR<0.905][0]

    # PLOT.classification_(L, T)
plt.figure(figsize=(8,6))
plt.plot(range(1,7), pioncon, '*')
plt.ylabel("$\\pi$-contamination")
plt.xlabel("No. of layers in track")
plt.grid(which='both')
# plt.yscale('log')
