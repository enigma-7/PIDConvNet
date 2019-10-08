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

dataname = 'DS2/'
ocdbdir = DEFAULTS.ocdbdir
datadir = DEFAULTS.datadir + dataname
plotdir = DEFAULTS.plotdir

raw_data = np.load(datadir + '0_tracks.npy')
raw_info = np.load(datadir + '0_info_set.npy')
print("Loaded: %s \n" % datadir )

dataset, infoset = DATA.process_tracklet_(raw_data, raw_info)

(X, infoset), (Xv, valid_infoset), (Xt, test_infoset) = DATA.TVT_split_(dataset/1024, infoset)
T  = infoset[:,0].astype(int)
Tv = valid_infoset[:,0].astype(int)
Tt = test_infoset[:,0].astype(int)
I  = infoset
Iv = valid_infoset
It = test_infoset# I  = infoset[:,nx:ny]

(cs_1, cs_2, d1_1, d1_2) = (8, 16, 128, 64)

weights = [1/10, 1.0, 10.0]
stamp = datetime.datetime.now().strftime("%d-%m-%H%M%S") + "_"
thresholds=np.linspace(0,1,1000)

cols = ['black', 'indigo', 'teal']
scale='log'

def logit(p):
    return np.log(p) - np.log(1 - p)

Parray = np.zeros((len(weights), Xt.shape[0]))
for i, w in enumerate(weights):
    mname = "Utracklet_" + "C-%d-%d-D-%d-%d_W%.1f-trial"%(cs_1, cs_2, d1_1, d1_2, w)
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

    def WBCE(y_true, y_pred, weight = w, from_logits=False):
        y_pred = tf.cast(y_pred, dtype='float32')
        y_true = tf.cast(y_true, y_pred.dtype)
        return K.mean(ML.weighted_binary_crossentropy(y_true, y_pred, weight=weight, from_logits=from_logits), axis=-1)

    model = Model(inputs=input, outputs=output)
    model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=1e-3), loss=WBCE, metrics=[ML.pion_con])
    model.fit(x=X, y=T, batch_size = 2**8, epochs=20, validation_data=(Xv,Tv))#, callbacks=[tensorboard, csvlogger])

    Parray[i] = model.predict(Xt).reshape(-1)

fig1, axes1 = plt.subplots(2, 3, figsize=(16,8), sharex=True,)
fig2, axes2 = plt.subplots(2, 3, figsize=(16,8), sharex=True,)
for i, w in enumerate(weights):
    P = Parray[i]
    e_pred = P[Tt==1]
    p_pred = P[Tt==0]
    argsort = e_pred.argsort()

    TP = np.array([(e_pred>threshold).sum() for threshold in thresholds])
    FN = np.array([(e_pred<threshold).sum() for threshold in thresholds])
    FP = np.array([(p_pred>threshold).sum() for threshold in thresholds])
    TN = np.array([(p_pred<threshold).sum() for threshold in thresholds])

    TPR = TP/(FN+TP)
    FPR = FP/(TN+FP)
    pioncon = FPR[TPR<0.905][0]          #estimate pion contamination
    eleceff = TPR[TPR<0.905][0]
    decbound = thresholds[TPR<0.905][0]

    b = np.linspace(0, 1, DEFAULTS.bin_no)
    cp, b, p = axes1[0,i].hist(p_pred, color = DEFAULTS.colour[0],
        label=DEFAULTS.cnames[0], bins = b, histtype='step', linewidth=2.0, density=True)
    ce, b, p = axes1[0,i].hist(e_pred, color = DEFAULTS.colour[1],
        label=DEFAULTS.cnames[1], bins = b, histtype='step', linewidth=2.0, density=True)
    axes1[0,i].set_yscale(scale)
    axes1[0,i].vlines(decbound, 0, max(cp), 'k', label="Decision Boundary")
    axes1[0,i].set_xlabel("$\\sigma$")
    axes1[0,0].set_ylabel("Counts")
    # axes1[0,i].legend()
    axes1[0,i].grid()
    axes1[0,i].set_title(str(DEFAULTS.letter[i]) + " %.1f"%w)

    axes1[1,i].plot(FPR,TPR, 'gray')
    axes1[1,i].vlines(pioncon, 0, eleceff, 'k', '--')
    axes1[1,i].hlines(eleceff, 0, pioncon, 'k', '--',label = "$\\epsilon_{\\pi}|_{%.2f}$ = %.2f"%(eleceff, pioncon))
    axes1[1,0].set_ylabel("$e$-efficiency")
    axes1[1,i].set_xlabel("$\\pi$-contamination")
    axes1[1,i].legend()
    axes1[1,i].grid()

    Ltrack, Ttrack = DATA.likelihood_(P, It)
    e_pred_track = Ltrack[Ttrack==1]
    p_pred_track = Ltrack[Ttrack==0]
    argsort = e_pred_track.argsort()

    TP = np.array([(e_pred_track>threshold).sum() for threshold in thresholds])
    FN = np.array([(e_pred_track<threshold).sum() for threshold in thresholds])
    FP = np.array([(p_pred_track>threshold).sum() for threshold in thresholds])
    TN = np.array([(p_pred_track<threshold).sum() for threshold in thresholds])
    TP
    TPR = TP/(TP+FN)
    FPR = FP/(TN+FP)
    pioncon = FPR[TPR<0.905][0]          #estimate pion contamination
    eleceff = TPR[TPR<0.905][0]
    decbound = thresholds[TPR<0.905][0]

    cp, b, p = axes2[0,i].hist(p_pred_track, color = DEFAULTS.colour[0],
        label=DEFAULTS.cnames[0], bins = b, histtype='step', linewidth=2.0, density=True)
    ce, b, p = axes2[0,i].hist(e_pred_track, color = DEFAULTS.colour[1],
        label=DEFAULTS.cnames[1], bins = b, histtype='step', linewidth=2.0, density=True)
    axes2[0,i].set_yscale(scale)
    axes2[0,i].vlines(decbound, 0, max(cp), 'k', label="Decision Boundary")
    axes2[0,i].set_xlabel("$\\sigma$")
    axes2[0,0].set_ylabel("Counts")
    # axes2[0,i].legend()
    axes2[0,i].grid()
    axes2[0,i].set_title(str(DEFAULTS.letter[i]) + " %.2f"%w)

    axes2[1,i].plot(FPR,TPR, 'gray')
    axes2[1,i].vlines(pioncon, 0, eleceff, 'k', '--')
    axes2[1,i].hlines(eleceff, 0, pioncon, 'k', '--',label = "$\\epsilon_{\\pi}|_{%.2f}$ = %.2f"%(eleceff, pioncon))
    axes2[1,0].set_ylabel("$e$-efficiency")
    axes2[1,i].set_xlabel("$\\pi$-contamination")
    axes2[1,i].legend()
    axes2[1,i].grid()

# fig1.savefig(plotdir + mname + "_tracklet_class_weight_iter.png")
# fig2.savefig(plotdir + mname + "_track_class_weight_iter.png")
