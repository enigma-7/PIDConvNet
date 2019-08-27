import numpy as np
from TOOLS import DATA, MODELS, LOG, METRICS,PLOT, DEFAULTS
import random, datetime, os
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPool2D, concatenate, GaussianNoise
matplotlib.rcParams.update({'font.size': 18})
matplotlib.rcParams['text.usetex'] = True

run_no = '000265378/'
dataname = 'all/'
directory = DEFAULTS.datadir + run_no + dataname
raw_data = np.load(directory + '0_tracks.npy')
raw_info = np.load(directory + '0_info_set.npy')
print("Loaded: %s" % directory)

dataset, infoset = DATA.process_tracklet_(raw_data, raw_info)
X, infoset = DATA.shuffle_(dataset, infoset)

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

input = Input(shape=X.shape[1:],)# name="X-in")
x = Conv2D(cs_1, [3,3], activation='relu', padding ='same')(input)
x = MaxPool2D([2,2], 2, padding='valid')(x)
x = Conv2D(cs_2, [3,3], activation='relu', padding='same')(x)
x = MaxPool2D([2,2], 2, padding='valid')(x)
x = Flatten()(x)
x = Dense(d1_1)(x)
x = Dense(d1_2)(x)
output = Dense(1, activation='sigmoid',)(x)# name="X-out")(x)

model = Model(inputs=input, outputs=output)
model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=1e-3), loss='binary_crossentropy', metrics=[METRICS.pion_con])
model.fit(x=X, y=T, batch_size = 100, epochs=5, validation_data=(Xv,Tv))#), callbacks=[tensorboard, csvlogger])

# net1.compile(optimizer=tf.train.AdamOptimizer(learning_rate=1e-3), loss='binary_crossentropy', metrics=[METRICS.pion_con])
# net1.fit(x=X, y=T, batch_size = 1000, epochs=20, validation_data=(Xv,Tv))#), callbacks=[tensorboard, csvlogger])
# net1.summary()
#
# plotdir = '/home/jibran/Desktop/neuralnet/plots/'
# directory = "logs-CSV/" + run_no + dataname
#
# frame = pd.read_csv(directory + stamp + mname, index_col=None, header=0)
# frame['model_name'] = mname
# frame = frame.set_index(['model_name', 'epoch'])
# frame.head()
# array = frame.values.reshape(-1,20,5)
# colours = ['red', 'goldenrod', 'green', 'blue', 'purple']
#
# fig, axes = plt.subplots(1, 2, figsize=(12,6))
# axes[0].plot(array[0,:,2].cumsum(), array[0,:,0], color='black', label="Training")
# axes[0].plot(array[0,:,2].cumsum(), array[0,:,3], color='black', label="Validation", linestyle='--')
# axes[0].set_ylabel("Loss")
# axes[1].plot(array[0,:,2].cumsum(), array[0,:,1], color='black', label="Training")
# axes[1].plot(array[0,:,2].cumsum(), array[0,:,4], color='black', label="Validation", linestyle='--')
# axes[1].set_ylabel("Pion Contamination")
# for i in range(2):
#     axes[i].grid()
#     axes[i].set_xlabel("Training Time [$s$]")
# plt.legend()
# # plt.savefig(plotdir + mname + ".png")
#
# P = net1.predict(Xt)
# thresholds=np.linspace(0,1,1000)
# cnames = ["$\\pi$","$e$"]
# colour = ['r', 'g']
# cols = ['black', 'indigo', 'teal']
# styles = ['--','-.']
# scale='log'
#
# e_pred = P[Tt==1]
# p_pred = P[Tt==0]
# argsort = e_pred.argsort()
#
# TP = np.array([e_pred[e_pred>threshold].sum() for threshold in thresholds])
# FN = np.array([e_pred[e_pred<threshold].sum() for threshold in thresholds])
# FP = np.array([p_pred[p_pred>threshold].sum() for threshold in thresholds])
# TN = np.array([p_pred[p_pred<threshold].sum() for threshold in thresholds])
#
# TPR = TP/(FN+TP)
# FPR = FP/(TN+FP)
# PiC = FPR[TPR>0.9][-1]
# DBD = e_pred[argsort[np.multiply(argsort.shape[-1],(1-90/100)).astype(int)]]
# AUC = np.sum(np.abs(np.diff(FPR)*TPR[1:]))
#
# fig, axes = plt.subplots(1, 2, figsize=(15,5))
# axes[0].plot(FPR,TPR, 'orangered')
# axes[0].vlines(PiC, 0, 0.9, 'k', '--')
# axes[0].hlines(0.9, 0, PiC, 'k', '--')
# axes[0].set_ylabel("$e$-efficiency")
# axes[0].set_xlabel("$\\pi$-contamination")
# axes[0].text(PiC+0.05, 0.4, "$\\varepsilon_\\pi$ = "+ str(np.round(PiC, 3)), fontsize=18)
# axes[0].text(PiC+0.05, 0.2, "AUC = "+ str(np.round(AUC, 2)), fontsize=18)
# axes[0].grid()
#
# c, b, p = axes[1].hist(p_pred ,label=cnames[0], color=colour[0],bins = 40, alpha = 0.7)
# axes[1].hist(e_pred ,label=cnames[1], color=colour[1], bins = b, alpha = 0.7)
# axes[1].set_yscale(scale)
# axes[1].vlines(thresholds[TPR<0.9][0], 0, max(c), 'k', label="Decision Boundary")
# axes[1].legend()
# axes[1].grid()
# # plt.savefig(plotdir + mname + "_class.png")
# PLOT.classification_(P, Tt, plotdir + mname + "_class.png", save = False)
