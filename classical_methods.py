import numpy as np
from TOOLS import DATA, MODELS, LOG, ML, PLOT, DEFAULTS
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

dataname = 'DS1/'
ocdbdir = DEFAULTS.ocdbdir
datadir = DEFAULTS.datadir + dataname
plotdir = DEFAULTS.plotdir

raw_data = np.load(datadir + '0_tracks.npy')
raw_info = np.load(datadir + '0_info_set.npy')
print("Loaded: %s \n" % datadir )

dataset, infoset = DATA.process_tracklet_(raw_data, raw_info)
dataset.shape
dataset = dataset.sum(axis=(1,3))
X, I = DATA.shuffle_(dataset, infoset)
T = I[:,0]

def classification_(S, T, cnames = ["$\\pi$","$e$"], colour = ['r', 'g']):
    Qp = S[T==0]
    Qe = S[T==1]
    thresholds = np.linspace(0, S.max(), 1000)
    argsort = Qe.argsort()

    TP = np.array([Qe[Qe>threshold].shape[0] for threshold in thresholds])
    FN = np.array([Qe[Qe<threshold].shape[0] for threshold in thresholds])
    FP = np.array([Qp[Qp>threshold].shape[0] for threshold in thresholds])
    TN = np.array([Qp[Qp<threshold].shape[0] for threshold in thresholds])

    TPR = TP/(FN+TP)
    FPR = FP/(TN+FP)

    pioncon = FPR[TPR<0.905][0]          #estimate pion contamination
    eleceff = TPR[TPR<0.905][0]
    decbound = thresholds[TPR<0.905][0]

    fig, axes = plt.subplots(1, 2, figsize=(15,5))
    axes[1].plot(FPR,TPR, 'gray')
    axes[1].vlines(pioncon, 0, eleceff, 'k', '--', label = "$\\epsilon_{\\pi}|_{%.2f}$ = %.2f"%(eleceff, pioncon))
    axes[1].hlines(eleceff, 0, pioncon, 'k', '--')
    axes[1].set_ylabel("$e$-efficiency")
    axes[1].set_xlabel("$\\pi$-contamination")
    axes[1].legend()
    # axes[1].text(pioncon+0.05, 0.4, "$\\varepsilon_\\pi$ = "+ str(np.round(pioncon, 3)), fontsize=18)
    axes[1].grid()

    cp, b, p = axes[0].hist(Qp, color = colour[0], label=cnames[0], bins = 50, histtype='step', linewidth=2.0)
    ce, b, p = axes[0].hist(Qe, color = colour[1], label=cnames[1], bins = 50, histtype='step', linewidth=2.0)
    axes[0].set_yscale('log')
    axes[0].vlines(decbound, 0, max(cp), 'k', label="Decision Boundary")
    axes[0].set_xlabel("$\\sigma$")
    axes[0].set_ylabel("Counts")
    axes[0].legend()
    axes[0].grid()
###     Mean ADC classification     ###

S = X.sum(axis=1)           #Sum of ADC value per tracklet
S = S/(17*24)               #Mean ADC per tracklet

classification_(S, T)

###     Truncated Mean classification       ###
S = np.array([X[i][X[i].argsort()][:int(X[i].shape[-1]*0.6)].sum() for i in range(X.shape[0])])
S = S/(17*24)

classification_(S, T )

###     likelihood method       ###
S = np.array([X[i][X[i].argsort()][:int(X[i].shape[-1]*0.6)].sum() for i in range(X.shape[0])])
S = S/(17*24)

Qp = S[T==0]
Qe = S[T==1]

b = np.linspace(0, max(S), 100)
p_pdf, bp = np.histogram(Qp, density=True, bins = b)
e_pdf, be = np.histogram(Qe, density=True, bins = b)
p_pdf
plt.plot(bp[1:],p_pdf)
plt.plot(bp[1:],e_pdf)
