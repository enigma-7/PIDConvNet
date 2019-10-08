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

ocdbdir = DEFAULTS.ocdbdir
datadir = DEFAULTS.datadir
plotdir = DEFAULTS.plotdir

# X = np.append(np.load(datadir + 'DS3/' + 'track_dataset.npy'),
#     np.load(datadir + 'DS4/' + 'track_dataset.npy'), axis = 0)
#
# infoset = np.append(np.load(datadir + 'DS3/' + 'track_infoset.npy'),
#     np.load(datadir + 'DS4/' + 'track_infoset.npy'), axis = 0)
#
# print("Loaded: %s \n" % datadir )
#
# (X, infoset), (Xv, valid_infoset), (Xt, test_infoset) = DATA.TVT_split_(X/1024, infoset)
# T  = infoset[:,0].astype(int)
# Tv = valid_infoset[:,0].astype(int)
# Tt = test_infoset[:,0].astype(int)
# I  = infoset
# Iv = valid_infoset
# It = test_infoset
#
# (cs_1, cs_2, d1_1, d1_2) = (8, 16, 128, 64)
#
# dropouts = [0.0, 0.15, 0.30, 0.45]
# stamp = datetime.datetime.now().strftime("%d-%m-%H%M%S") + "_"
# for dropout in dropouts:
#     mname = "track_V_U_dropout_%.2f"%dropout
#     tensorboard, csvlogger = LOG.logger_('DS5/', stamp, mname)
#
#     input = Input(shape=X.shape[1:], name="X-in")
#     x = Conv2D(cs_1, [2,3], activation='relu', padding ='same')(input)
#     x = MaxPool2D([2,2], 2, padding='valid')(x)
#     x = Conv2D(cs_2, [2,3], activation='relu', padding='same')(x)
#     x = MaxPool2D([2,2], 2, padding='valid')(x)
#     x = Flatten()(x)
#     x = Dropout(dropout)(x)
#     x = Dense(d1_1)(x)
#     x = Dense(d1_2)(x)
#     output = Dense(1, activation='sigmoid', name="X-out")(x)
#
#     model = Model(inputs=input, outputs=output)
#     model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=1e-3), loss=ML.WBCE, metrics=[ML.pion_con])
#     model.fit(x=X, y=T, batch_size = 2**10, epochs=200, validation_data=(Xv,Tv), callbacks=[tensorboard, csvlogger])


# mname = "track_%.1f_"%dropout + "C-%d-%d-D-%d-%d"%(cs_1, cs_2, d1_1, d1_2)
    # tensorboard, csvlogger = LOG.logger_(dataname, stamp, mname)
#
# input = Input(shape=X.shape[1:], name="X-in")
# x = Conv2D(cs_1, [2,3], activation='relu', padding ='same')(input)
# x = MaxPool2D([2,2], 2, padding='valid')(x)
# x = Conv2D(cs_2, [2,3], activation='relu', padding='same')(x)
# x = MaxPool2D([2,2], 2, padding='valid')(x)
# x = Flatten()(x)
# x = Dense(d1_1)(x)
# x = Dense(d1_2)(x)
# output = Dense(1, activation='sigmoid', name="X-out")(x)
#
# model = Model(inputs=input, outputs=output)
# model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=1e-3), loss='binary_crossentropy', metrics=[ML.pion_con])
# model.fit(x=X, y=T, batch_size = 2**9, epochs=20, validation_data=(Xv,Tv))#, callbacks=[tensorboard, csvlogger])
# X = 0
# infoset = 0

#
# Iter2 = DEFAULTS.track_V_U_dropout_iter2
# Iter2.sort()
#
# frame, nm = LOG.import_(Iter2, position=1)
# frame.head()
#
# array = frame.values.reshape(-1,100,5)
#
# Iter3 = DEFAULTS.track_V_U_dropout_iter3
# Iter3.sort()
#
# frame2, nm2 = LOG.import_(Iter3, position=1)
# frame2.head()
#
# array2 = frame2.values.reshape(-1,100,5)
# array[1] = array2[0]
# array[3] = array2[1]
# nm[1] = nm2[0]
# nm[3] = nm2[1]
# array.shape

from scipy.signal import savgol_filter
import glob

Iter6 = DEFAULTS.track_V_U_dropout_iter6
Iter6.sort()

frame, nm = LOG.import_(Iter6, position=1)
frame.head()
array = frame.values.reshape(-1,200,5)
colours = ['steelblue', 'teal', 'lightseagreen', 'cyan', 'purple']
letters = ['(a)', '(b)', '(c)', '(d)']
epochtimes = array[:,:,2].mean(axis=1)

figpcon, pcon = plt.subplots(1, 4, figsize=(20,6), sharey=True, sharex=True)
for j in range(4):
    pcon[j].plot(array[j,:,2].cumsum(), savgol_filter(array[j,:,1], 11, 1), color=colours[j])
    pcon[j].plot(array[j,:,2].cumsum(), savgol_filter(array[j,:,4], 11, 1), color=colours[j], linestyle=':')
    pcon[j].grid()
    pcon[j].set_xlabel("Training Time [s]")
    pcon[j].set_title(letters[j] + " Dropout = " +  nm[j])
pcon[0].set_ylabel("$\\pi$-contamination")

figloss, loss = plt.subplots(1, 4, figsize=(20,6), sharey=True, sharex=True)
for j in range(4):
    loss[j].plot(array[j,:,2].cumsum(), savgol_filter(array[j,:,0], 11, 1), color=colours[j])
    loss[j].plot(array[j,:,2].cumsum(), savgol_filter(array[j,:,3], 11, 1), color=colours[j], linestyle=':')
    loss[j].grid()
    loss[j].set_xlabel("Training Time [s]")
    loss[j].set_title(letters[j] + " Dropout = " +  nm[j])
loss[0].set_ylabel("Loss")
figloss.savefig(plotdir + "track_V_U_dropout_loss" + ".png")

figdiff, diff = plt.subplots(1, 2, figsize=(14,5), sharex=True)
title = [" - Loss", " - $\\pi$-contamination"]
for j in range(4):
    time = array[j,:,2].cumsum()
    mask = np.logical_and(time > 100, time < 30000)
    T_loss = savgol_filter(array[j][:,0],21,2)
    V_loss = savgol_filter(array[j][:,3],21,2)
    T_pcon = savgol_filter(array[j][:,1],21,2)
    V_pcon = savgol_filter(array[j][:,4],21,2)
    diff[0].plot(time[mask], (V_loss - T_loss)[mask], label = nm[j])
    diff[1].plot(time[mask], (V_pcon - T_pcon)[mask])
for i in range(2):
    diff[i].legend()
    diff[i].grid()
    diff[i].set_xlabel("Training Time [s]")
    diff[i].set_title(DEFAULTS.letter[i] + title[i])
diff[0].set_ylabel("$\\Delta L_{7.0}$")
diff[1].set_ylabel("$\\Delta \\epsilon_{\\pi}|_{0.9}$")
figdiff.savefig(plotdir + "track_V_U_dropout_diff" + ".png")
    # diff[i].set_yscale('log')

# train[0].legend()
# P = model.predict(Xt).reshape(-1)
# PLOT.classification_(P, Tt)

# e_pred = P[Tt==1]
# p_pred = P[Tt==0]
# argsort = e_pred.argsort()
#
# thresholds=np.linspace(0,1,1000)
#
# TP = np.array([e_pred[e_pred>threshold].sum() for threshold in thresholds])
# FN = np.array([e_pred[e_pred<threshold].sum() for threshold in thresholds])
# FP = np.array([p_pred[p_pred>threshold].sum() for threshold in thresholds])
# TN = np.array([p_pred[p_pred<threshold].sum() for threshold in thresholds])
#
# TPR = TP/(FN+TP)            #True Positive Rate / Recall
# FPR = FP/(TN+FP)            #False Positive Rate
# PPV = TP/(TP+FP)            #Positive Predictive Value / Precision
# plt.plot(TPR, PPV)
# PPV
# thresholds[TPR<0.905][0]
#
# pioncon = FPR[TPR>0.9][-1]          #estimate pion contamination
# decbound = e_pred[argsort[np.multiply(argsort.shape[-1],(1-90/100)).astype(int)]]   #estimate decision boundary
# print(decbound)
# AUC = np.sum(np.abs(np.diff(FPR)*TPR[1:]))          #estimate area under curve
