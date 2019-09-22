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
X, infoset = DATA.shuffle_(dataset/1024, infoset)

(X, infoset), (Xv, valid_infoset), (Xt, test_infoset) = DATA.TVT_split_(X, infoset)
T  = infoset[:,0].astype(int)
Tv = valid_infoset[:,0].astype(int)
Tt = test_infoset[:,0].astype(int)
I  = infoset
Iv = valid_infoset
It = test_infoset# I  = infoset[:,nx:ny]

(cs_1, cs_2, d1_1, d1_2) = (8, 16, 128, 64)

weights = [1/8, 1.0, 8.0]
stamp = datetime.datetime.now().strftime("%d-%m-%H%M%S") + "_"
thresholds=np.linspace(0,1,1000)

cols = ['black', 'indigo', 'teal']
scale='log'

fig1, axes1 = plt.subplots(2, 3, figsize=(16,8))
fig2, axes2 = plt.subplots(2, 3, figsize=(16,8))
for i, w in enumerate(weights):
    # mname = "Utracklet_" + "C-%d-%d-D-%d-%d_W%.1f"%(cs_1, cs_2, d1_1, d1_2,w)
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
    model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=1e-3), loss=WBCE, metrics=[ML.pion_con, ML.prec])
    model.fit(x=X, y=T, batch_size = 2**8, epochs=5, validation_data=(Xv,Tv))#, callbacks=[tensorboard, csvlogger])

    P = model.predict(Xt).reshape(-1)
    e_pred = P[Tt==1]
    p_pred = P[Tt==0]
    argsort = e_pred.argsort()

    TP = np.array([e_pred[e_pred>threshold].sum() for threshold in thresholds])
    FN = np.array([e_pred[e_pred<threshold].sum() for threshold in thresholds])
    FP = np.array([p_pred[p_pred>threshold].sum() for threshold in thresholds])
    TN = np.array([p_pred[p_pred<threshold].sum() for threshold in thresholds])

    TPR = TP/(FN+TP)
    FPR = FP/(TN+FP)
    PiC = FPR[TPR>0.9][-1]
    DBD = e_pred[argsort[np.multiply(argsort.shape[-1],(1-90/100)).astype(int)]]
    AUC = np.sum(np.abs(np.diff(FPR)*TPR[1:]))

    b = np.linspace(0,1,DEFAULTS.bin_no)
    cp, b, p = axes1[0,i].hist(p_pred, color = DEFAULTS.colour[0],
        label=DEFAULTS.cnames[0], bins = b, histtype='step', linewidth=2.0, density=True)
    ce, b, p = axes1[0,i].hist(e_pred, color = DEFAULTS.colour[1],
        label=DEFAULTS.cnames[1], bins = b, histtype='step', linewidth=2.0, density=True)
    axes1[0,i].set_yscale(scale)
    axes1[0,i].vlines(thresholds[TPR<0.9][0], 0, max(cp), 'k', label="Decision Boundary")
    axes1[0,i].set_xlabel("Output from $\\sigma$")
    axes1[0,0].set_ylabel("Counts")
    axes1[0,i].legend()
    axes1[0,i].grid()
    axes1[0,i].set_title(str(DEFAULTS.letter[i]) + " %.2f"%w)

    axes1[1,i].plot(FPR,TPR, 'gray')
    axes1[1,i].vlines(PiC, 0, 0.9, 'k', '--')
    axes1[1,i].hlines(0.9, 0, PiC, 'k', '--')
    axes1[1,0].set_ylabel("$e$-efficiency")
    axes1[1,i].set_xlabel("$\\pi$-contamination")
    axes1[1,i].text(PiC+0.05, 0.4, "$\\varepsilon_\\pi$ = "+ str(np.round(PiC, 3)), fontsize=18)
    axes1[1,i].text(PiC+0.05, 0.2, "AUC = "+ str(np.round(AUC, 2)), fontsize=18)
    axes1[1,i].grid()

    e_prob = np.zeros(len(P))
    p_prob = np.zeros(len(P))
    p_pdf = (cp/len(p_pred))
    e_pdf = (ce/len(e_pred))

    for j, t in enumerate(Tt):
        e_prob[j] = e_pdf[np.logical_and(P[j] >= b[:-1], P[j] < b[1:])]
        p_prob[j] = p_pdf[np.logical_and(P[j] >= b[:-1], P[j] < b[1:])]

    trackID = np.array(list(set([(r,e,v,t) for [r,e,v,t] in It[:,9:13].astype(int)])))
    print("\n %d tracklets from %d unique tracks \n"%(It.shape[0],trackID.shape[0]))

    Ltrack = np.zeros(trackID.shape[0])
    Ttrack = np.zeros(trackID.shape[0])

    for k, [r,e,v,t] in enumerate(trackID):
        rmask = It[:,9]  == r
        emask = It[:,10] == e
        vmask = It[:,11] == v
        tmask = It[:,12] == t
        mask = np.logical_and(np.logical_and(np.logical_and(rmask, emask), vmask), tmask)
        Ltrack[k] = np.prod(e_prob[mask])/(np.prod(e_prob[mask]) + np.prod(p_prob[mask]))
        Ttrack[k] = Tt[mask][0]
        if len(set(Tt[mask])) != 1:
            print('somethings gone fucky at %d'%k)

    e_pred_track = Ltrack[Ttrack==1]
    p_pred_track = Ltrack[Ttrack==0]
    argsort = e_pred_track.argsort()

    TP = np.array([e_pred_track[e_pred_track>threshold].sum() for threshold in thresholds])
    FN = np.array([e_pred_track[e_pred_track<threshold].sum() for threshold in thresholds])
    FP = np.array([p_pred_track[p_pred_track>threshold].sum() for threshold in thresholds])
    TN = np.array([p_pred_track[p_pred_track<threshold].sum() for threshold in thresholds])

    TPR = TP/(FN+TP)
    FPR = FP/(TN+FP)
    PiC = (FPR[TPR>0.9][-1] + FPR[TPR<0.9][0])/2
    DBD = e_pred_track[argsort[np.multiply(argsort.shape[-1],(1-90/100)).astype(int)]]
    AUC = np.sum(np.abs(np.diff(FPR)*TPR[1:]))

    cp, b, p = axes2[0,i].hist(p_pred_track, color = DEFAULTS.colour[0],
        label=DEFAULTS.cnames[0], bins = b, histtype='step', linewidth=2.0, density = True)
    ce, b, p = axes2[0,i].hist(e_pred_track, color = DEFAULTS.colour[1],
        label=DEFAULTS.cnames[1], bins = b, histtype='step', linewidth=2.0, density = True)
    axes2[0,i].set_yscale(scale)
    axes2[0,i].vlines(thresholds[TPR<0.9][0], 0, max(cp), 'k', label="Decision Boundary")
    axes2[0,i].set_xlabel("$\\sigma$")
    axes2[0,0].set_ylabel("Counts")
    # axes2[0,i].legend()
    axes2[0,i].grid()
    axes2[0,i].set_title(str(DEFAULTS.letter[i]) + " %.2f"%w)

    axes2[1,i].plot(FPR,TPR, 'gray')
    axes2[1,i].vlines(PiC, 0, 0.9, 'k', '--')
    axes2[1,i].hlines(0.9, 0, PiC, 'k', '--')
    axes2[1,0].set_ylabel("$e$-efficiency")
    axes2[1,i].set_xlabel("$\\pi$-contamination")
    axes2[1,i].text(PiC+0.05, 0.4, "$\\varepsilon_\\pi$ = "+ str(np.round(PiC, 3)), fontsize=18)
    axes2[1,i].text(PiC+0.05, 0.2, "AUC = "+ str(np.round(AUC, 2)), fontsize=18)
    axes2[1,i].grid()

# fig1.savefig(plotdir + mname + "_tracklet_class_weight_iter.png")
# fig2.savefig(plotdir + mname + "_track_class_weight_iter.png")

info_cols = DEFAULTS.info_cols_tracklet_
info_cols[9:13]
