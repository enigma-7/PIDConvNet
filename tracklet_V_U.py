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

X = np.load(datadir + 'tracklet_dataset.npy')/1024
infoset = np.load(datadir + 'tracklet_infoset.npy')

print("Loaded: %s \n" % datadir )

(X, infoset), (Xv, valid_infoset) = DATA.TV_split_(X, infoset)
T  = infoset[:,0].astype(int)
Tv = valid_infoset[:,0].astype(int)
I  = infoset
Iv = valid_infoset

(cs_1, cs_2, d1_1, d1_2) = (8, 16, 128, 64)
mname = "tracklet_V_U"
stamp = datetime.datetime.now().strftime("%d-%m-%H%M%S") + "_"
tensorboard, csvlogger = LOG.logger_(dataname, stamp, mname)

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
model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=1e-3), loss=ML.WBCE, metrics=[ML.pion_con])
model.fit(x=X, y=T, batch_size = 2**9, epochs=20, validation_data=(Xv,Tv), callbacks=[tensorboard, csvlogger])

"""
## learning_rate = 1e-3, batch_size = 2**8 ##
frame, names = LOG.import_(tracklet_V_U5)

array = frame.values.reshape(-1,20,5)
colours = ['red', 'goldenrod', 'green', 'blue', 'purple']

fig, axes = plt.subplots(1, 2, figsize=(12,6))
axes[0].plot(array[0,:,2].cumsum(), array[0,:,0], color='black', label="Training")
axes[0].plot(array[0,:,2].cumsum(), array[0,:,3], color='black', label="Validation", linestyle=':')
axes[0].set_ylabel("Loss")
axes[0].set_title("(a)")
axes[1].plot(array[0,:,2].cumsum(), array[0,:,1], color='black', label="Training")
axes[1].plot(array[0,:,2].cumsum(), array[0,:,4], color='black', label="Validation", linestyle=':')
axes[1].set_ylabel("Pion Contamination")
axes[1].set_title("(b)")
for i in range(2):
    axes[i].grid()
    axes[i].set_xlabel("Training Time [$s$]")
plt.legend()
# plt.savefig(DEFAULTS.plotdir + mname + ".png")
"""


def classification_(predict, targets):
    thresholds=np.linspace(0,1,100)
    cnames = ["$\\pi$","$e$"]
    colour = ['r', 'g']
    styles = ['--','-.']
    scale='log'
    b = np.linspace(0,1,50)
    pdf = True
    matplotlib.rcParams.update({'font.size': 16})
    matplotlib.rcParams['text.usetex'] = True
    e_pred = predict[targets==1]
    p_pred = predict[targets==0]
    argsort = e_pred.argsort()

    TP = np.array([(e_pred>threshold).sum() for threshold in thresholds])
    FN = np.array([(e_pred<threshold).sum() for threshold in thresholds])
    FP = np.array([(p_pred>threshold).sum() for threshold in thresholds])
    TN = np.array([(p_pred<threshold).sum() for threshold in thresholds])

    TPR = TP/(FN+TP)            #True Positive Rate / Recall
    FPR = FP/(TN+FP)            #False Positive Rate
    PPV = TP/(TP+FP)            #Positive Predictive Value / Precision
    uTPR = np.sqrt(TPR*(1-TPR)/(TP + FN))
    uFPR = np.sqrt(FPR*(1-FPR)/(TN + FP))

    for k, val in enumerate(PPV):
        if np.isnan(val):
            PPV[k] = PPV[k-1]
        else:
            continue

    mask = TPR<0.905
    pioncon = FPR[mask][0]          #estimate pion contamination
    eleceff = TPR[mask][0]
    Upioncon = uFPR[mask][0]
    Ueleceff = uTPR[mask][0]
    decbound = thresholds[mask][0]

    fig, axes = plt.subplots(1, 2, figsize=(15,5))
    axes[1].plot(FPR,TPR, 'gray')
    axes[1].vlines(pioncon, 0, eleceff, 'k', '--',
        label = "$\\epsilon_{\\pi}|_{%.2f} = (%.2f \\pm %.2f)\\%%$"%(eleceff, pioncon*100,Upioncon*100))
    axes[1].hlines(eleceff, 0, pioncon, 'k', '--')
    axes[1].set_ylabel("$e$-efficiency")
    axes[1].set_xlabel("$\\pi$-contamination")
    axes[1].legend()
    axes[1].grid()

    cp, b, p = axes[0].hist(p_pred, color = colour[0], label=cnames[0],
        bins = b, histtype='step', linewidth=2.0, density = pdf)
    ce, b, p = axes[0].hist(e_pred, color = colour[1], label=cnames[1],
        bins = b, histtype='step', linewidth=2.0, density = pdf)
    axes[0].set_yscale(scale)
    axes[0].vlines(decbound, 0, max(cp), 'k', label=str(decbound.round(2)))
    axes[0].set_xlabel("$\\sigma$")
    axes[0].set_ylabel("Counts")
    axes[0].legend(loc=8)
    axes[0].grid()
    filename= mname
    # fig.savefig(plotdir + filename + 'Lclass.png')

It = np.load(DEFAULTS.datadir + 'DS4/' + 'tracklet_infoset.npy')
Xt = np.load(DEFAULTS.datadir + 'DS4/' + 'tracklet_dataset.npy')/1024

P = model.predict(Xt).reshape(-1)

PLOT.classification_(P, It[:,0])
np.save(DEFAULTS.modldir + 'P-DS4-' + mname + '_wbce5-1', P)
for i in set(It[:,13]):
    mask = It[:,13] == i
    PLOT.classification_(P[mask],It[:,0][mask])

columns = DEFAULTS.info_cols_tracklet_ + DEFAULTS.ocdb_cols1

Ltrack, Ttrack, layers = DATA.likelihood_(P, It)
classification_(Ltrack, Ttrack)

thresholds=np.linspace(0,1,100)
pioncon = np.zeros(6)
eleceff = np.zeros(6)
decbound = np.zeros(6)
Ueleceff = np.zeros(6)
Upioncon = np.zeros(6)
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
    uTPR = np.sqrt(TPR*(1-TPR)/(TP + FN))
    uFPR = np.sqrt(FPR*(1-FPR)/(TN + FP))

    mask = TPR<0.905
    pioncon[i-1] = FPR[mask][0]          #estimate pion contamination
    eleceff[i-1] = TPR[mask][0]
    Upioncon[i-1] = uFPR[mask][0]
    Ueleceff[i-1] = uTPR[mask][0]
    decbound[i-1] = thresholds[mask][0]
    # PLOT.classification_(L, T)
plt.figure(figsize=(8,6))
plt.plot(range(1,7), pioncon, '*')
plt.ylabel("$\\pi$-contamination")
plt.xlabel("No. of layers in track")
plt.grid(which='both')
plt.savefig(DEFAULTS.plotdir + 'likelihoodvslayers.png')
