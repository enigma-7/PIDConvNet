import numpy as np
from TOOLS import DATA, MODELS, LOG, METRICS,PLOT, DEFAULTS
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
model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=1e-3), loss='binary_crossentropy', metrics=[METRICS.pion_con])
model.fit(x=X, y=T, batch_size = 2**8, epochs=20, validation_data=(Xv,Tv))#, callbacks=[tensorboard, csvlogger])
X = 0
T = 0
Xv = 0
Tv = 0
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

thresholds=np.linspace(0,1,1000)
cnames = ["$\\pi$","$e$"]
colour = ['r', 'g']
cols = ['black', 'indigo', 'teal']
styles = ['--','-.']
scale='log'

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

fig, axes = plt.subplots(1, 2, figsize=(15,5))
axes[1].plot(FPR,TPR, 'gray')
axes[1].vlines(PiC, 0, 0.9, 'k', '--')
axes[1].hlines(0.9, 0, PiC, 'k', '--')
axes[1].set_ylabel("$e$-efficiency")
axes[1].set_xlabel("$\\pi$-contamination")
axes[1].text(PiC+0.05, 0.4, "$\\varepsilon_\\pi$ = "+ str(np.round(PiC, 3)), fontsize=18)
axes[1].text(PiC+0.05, 0.2, "AUC = "+ str(np.round(AUC, 2)), fontsize=18)
axes[1].grid()

b = np.linspace(0,1,40)
cp, b, p = axes[0].hist(p_pred, color = 'r', label=cnames[0], bins = b, histtype='step', linewidth=2.0)
ce, b, p = axes[0].hist(e_pred, color = 'g', label=cnames[1], bins = b, histtype='step', linewidth=2.0)
axes[0].set_yscale(scale)
axes[0].vlines(thresholds[TPR<0.9][0], 0, max(cp), 'k', label="Decision Boundary")
axes[0].set_xlabel("Output from $\\sigma$")
axes[0].set_ylabel("Counts")
axes[0].legend()
axes[0].grid()
plt.savefig(plotdir + mname + "_tracklet_class.png")
#PLOT.classification_(P, Tt, plotdir + mname + "_class.png", save = False)

e_prob = np.zeros(len(P))
p_prob = np.zeros(len(P))
p_pdf = (cp/len(p_pred))
e_pdf = (ce/len(e_pred))
for i, t in enumerate(Tt):
    e_prob[i] = e_pdf[np.logical_and(P[i] >= b[:-1], P[i] < b[1:])]
    p_prob[i] = p_pdf[np.logical_and(P[i] >= b[:-1], P[i] < b[1:])]

info_cols = DEFAULTS.info_cols_tracklet_
info_cols[9:13]
trackID = np.array(list(set([(r,e,v,t) for [r,e,v,t] in It[:,9:13].astype(int)])))

print("\n %d tracklets from %d unique tracks \n"%(It.shape[0],trackID.shape[0]))

Ltrack = np.zeros(trackID.shape[0])
Ttrack = np.zeros(trackID.shape[0])

for i, [r,e,v,t] in enumerate(trackID):
    rmask = It[:,9]  == r
    emask = It[:,10] == e
    vmask = It[:,11] == v
    tmask = It[:,12] == t
    mask = np.logical_and(np.logical_and(np.logical_and(rmask, emask), vmask), tmask)
    Ltrack[i] = np.prod(e_prob[mask])/(np.prod(e_prob[mask]) + np.prod(p_prob[mask]))
    Ttrack[i] = Tt[mask][0]
    if len(set(Tt[mask])) != 1:
        print('somethings gone fucky at %d'%i)


# [r,e,v,t] = trackID[5833]
# rmask = It[:,9]  == r
# emask = It[:,10] == e
# vmask = It[:,11] == v
# tmask = It[:,12] == t
# mask = np.logical_and(np.logical_and(np.logical_and(rmask, emask), vmask), tmask)
# pd.DataFrame(It[mask])

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

fig, axes = plt.subplots(1, 2, figsize=(15,5))
axes[1].plot(FPR,TPR, 'navy')
axes[1].vlines(PiC, 0, 0.9, 'k', '--')
axes[1].hlines(0.9, 0, PiC, 'k', '--')
axes[1].set_ylabel("$e$-efficiency")
axes[1].set_xlabel("$\\pi$-contamination")
axes[1].text(PiC+0.05, 0.4, "$\\varepsilon_\\pi$ = "+ str(np.round(PiC, 3)), fontsize=18)
axes[1].text(PiC+0.05, 0.2, "AUC = "+ str(np.round(AUC, 2)), fontsize=18)
axes[1].grid()

cp, b, p = axes[0].hist(p_pred_track, color = 'r', label=cnames[0], bins = b, histtype='step', linewidth=2.0)
ce, b, p = axes[0].hist(e_pred_track, color = 'g', label=cnames[1], bins = b, histtype='step', linewidth=2.0)
axes[0].set_yscale('log')
axes[0].vlines(thresholds[TPR<0.9][0], 0, max(cp), 'k', label="Decision Boundary")
axes[0].set_xlabel("$L(e|\\sigma)$")
axes[0].set_ylabel("Counts")
axes[0].legend()
axes[0].grid()
plt.savefig(plotdir + mname + "_track_class.png")
