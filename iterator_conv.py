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

dataname = 'DS2/'
ocdbdir = DEFAULTS.ocdbdir
datadir = DEFAULTS.datadir + dataname
plotdir = DEFAULTS.plotdir

dataset = np.load(datadir + 'tracklet_dataset.npy')
infoset = np.load(datadir + 'tracklet_infoset.npy')

print("Loaded: %s \n" % datadir )
(X, infoset), (Xv, valid_infoset), (Xt, test_infoset) = DATA.TVT_split_(dataset/1024, infoset)
T  = infoset[:,0]
Tv = valid_infoset[:,0]
Tt = test_infoset[:,0]

##      MODEL       ##

stamp = datetime.datetime.now().strftime("%d-%m-%H%M%S")

###     Iteration 1     ###
# conv_sizes = [(4, 8), (8, 16), (16, 32), (32, 64), (64, 128)]
#
# for i, conv_size in enumerate(conv_sizes):
#     mname = "conv-%d-%d-dense-1024-"%(conv_size)
#     tensorboard, csvlogger = LOG.logger_(run_no, dataname, stamp, mname)
#     model = MODELS.blank_2_1_(conv_size[0], conv_size[1], 1024)
#     model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=1e-3), loss='binary_crossentropy', metrics=[METRICS.pion_con]) #Change loss function
#     model.fit(x=X, y=T, batch_size = 100, epochs=20, validation_data=(Xv,Tv), callbacks=[tensorboard, csvlogger])
    # model.summary()

###     Iteration 2     ###
# conv_sizes = [(4, 8), (8, 16), (16, 32), (32, 64), (64, 128)]
#
# def WBCE(y_true, y_pred, weight = 8.0, from_logits=False):
#     y_pred = tf.cast(y_pred, dtype='float32')
#     y_true = tf.cast(y_true, y_pred.dtype)
#     return K.mean(ML.weighted_binary_crossentropy(y_true, y_pred, weight=weight, from_logits=from_logits), axis=-1)
#
# arr = np.array([199105, 396929,  794881, 1600001, 3247105])
# epochs = (5*arr[-1]/arr).astype(int)
#
# for i, conv_size in enumerate(conv_sizes):
#     mname = "C-%d-%d-D-1024-"%(conv_size)
#     tensorboard, csvlogger = LOG.logger_(dataname, stamp, mname)
#     model = MODELS.blank_2_1_(conv_size[0], conv_size[1], 1024)
#     model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=1e-3), loss=WBCE, metrics=[ML.pion_con]) #Change loss function
#     model.fit(x=X, y=T, batch_size = 2**9, epochs=epochs[i], validation_data=(Xv,Tv), callbacks=[tensorboard, csvlogger])
#     # model.summary()

###     Iteration 3     ###

# conv_sizes = [(8, 16), (16, 32), (32, 64)]
#
# def WBCE(y_true, y_pred, weight = 8.0, from_logits=False):
#     y_pred = tf.cast(y_pred, dtype='float32')
#     y_true = tf.cast(y_true, y_pred.dtype)
#     return K.mean(ML.weighted_binary_crossentropy(y_true, y_pred, weight=weight, from_logits=from_logits), axis=-1)
#
# epochs = [100, 42, 24]
#
# for i, conv_size in enumerate(conv_sizes):
#     mname = "C-%d-%d-D-1024-"%(conv_size)
#     tensorboard, csvlogger = LOG.logger_(dataname, stamp, mname)
#     model = MODELS.blank_2_1_(conv_size[0], conv_size[1], 1024)
#     model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=1e-3), loss=WBCE, metrics=[ML.pion_con]) #Change loss function
#     model.fit(x=X, y=T, batch_size = 2**9, epochs=epochs[i], validation_data=(Xv,Tv), callbacks=[tensorboard, csvlogger])
#     # model.summary()

###     Iteration 4     ###
#
# conv_sizes = [(64, 128)]
#
# def WBCE(y_true, y_pred, weight = 8.0, from_logits=False):
#     y_pred = tf.cast(y_pred, dtype='float32')
#     y_true = tf.cast(y_true, y_pred.dtype)
#     return K.mean(ML.weighted_binary_crossentropy(y_true, y_pred, weight=weight, from_logits=from_logits), axis=-1)
#
#
# for i, conv_size in enumerate(conv_sizes):
#     mname = "_C-%d-%d-D-1024"%(conv_size)
#     tensorboard, csvlogger = LOG.logger_(dataname, stamp, mname)
#     model = MODELS.blank_2_1_(conv_size[0], conv_size[1], 1024)
#     model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=1e-3), loss=WBCE, metrics=[ML.pion_con]) #Change loss function
#     model.fit(x=X, y=T, batch_size = 2**9, epochs=10, validation_data=(Xv,Tv), callbacks=[tensorboard, csvlogger])
#     # model.summary()

###     Iteration 4     ###

conv_sizes = [(4, 8)]

def WBCE(y_true, y_pred, weight = 8.0, from_logits=False):
    y_pred = tf.cast(y_pred, dtype='float32')
    y_true = tf.cast(y_true, y_pred.dtype)
    return K.mean(ML.weighted_binary_crossentropy(y_true, y_pred, weight=weight, from_logits=from_logits), axis=-1)


for i, conv_size in enumerate(conv_sizes):
    mname = "_C-%d-%d-D-1024"%(conv_size)
    tensorboard, csvlogger = LOG.logger_(dataname, stamp, mname)
    model = MODELS.blank_2_1_(conv_size[0], conv_size[1], 1024)
    model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=1e-3), loss=WBCE, metrics=[ML.pion_con]) #Change loss function
    model.fit(x=X, y=T, batch_size = 2**9, epochs=150, validation_data=(Xv,Tv), callbacks=[tensorboard, csvlogger])
    # model.summary()

#
# directory = "logs-CSV/" + run_no + dataname
# glob.glob(directory + "08-08-164626_*")
# filenames = [directory + "08-08-164626_" + "conv-%d-%d-dense-1024"%(cs) for cs in conv_sizes]
# plotdir = '/home/jibran/Desktop/neuralnet/plots/'
#
# li = []
# nm = []
# # filenames
# # fils = sorted(filenames, key=lambda item: (int(item.partition('conv-')[-1].partition('-')[0])
# #     if item.isdigit() else float('inf'), item))
# #
# # item = '4 sheets'
# # int(item.partition(' ')[0])
# # filename = 'logs-CSV/000265378/all/08-08-164626_conv-64-128-dense-1024'
# # int(filename.partition('conv-')[-1].partition('-')[0])
# #
# # fils
#
# # for i, filename in enumerate(filenames):
# #     nm.append(filename.split('_')[-1])
# #     df = pd.read_csv(filename, index_col=None, header=0)
# #     df['model_name'] = ["%s"%filename.split('_')[-1]]*20
# #     li.append(df)
# #
# # frame = pd.concat(li, axis=0, names=nm)
# # frame = frame.set_index(['model_name', 'epoch'])
# # frame.head()
# # frame.sum(level='model_name').sort_values(by='train_time', inplace=True)
# # array = frame.values.reshape(-1,20,5)
# # # np.save(directory + "Iter1_08-08-164626_.npy",array)
# # colours = ['red', 'goldenrod', 'green', 'blue', 'purple']
# # epochtimes = array[:,:,2].mean(axis=1)
# # fig, axes = plt.subplots(1, 3, figsize=(17,6))
# # for j in range(array.shape[-1]):
# #     axes[0].plot(range(1,21), array[j,:,0], label=nm[j], color=colours[j])
# #     axes[1].plot(array[j,:,2].cumsum(), array[j,:,0], label=nm[j], color=colours[j])
# #     axes[2].bar(j+1, epochtimes[j], color = colours[j], width = 0.4)
# # axes[0].set_ylabel("Loss")
# # axes[0].grid()
# # axes[0].set_xlabel("Epoch")
# # axes[1].set_ylabel("Loss")
# # axes[1].grid()
# # axes[1].legend()
# # axes[1].set_xlabel("Training Time [$s$]")
# # axes[2].set_xlabel("Model no.")
# # axes[2].set_ylabel("Average Training Epoch [$s$]")
# # axes[2].grid()
# #
# # # plt.savefig(plotdir + 'Iter1.png')
# #
# # for m in [1,4]:
# #     fig, axes = plt.subplots(1, 2, figsize=(12,6))
# #     axes[0].plot(array[m,:,2].cumsum(), array[m,:,0], color=colours[m], label="Training")
# #     axes[0].plot(array[m,:,2].cumsum(), array[m,:,3], color=colours[m], label="Validation", linestyle='--')
# #     axes[0].set_ylabel("Loss")
# #     axes[1].plot(array[m,:,2].cumsum(), array[m,:,1], color=colours[m], label="Training")
# #     axes[1].plot(array[m,:,2].cumsum(), array[m,:,4], color=colours[m], label="Validation", linestyle='--')
# #     axes[1].set_ylabel("Pion Contamination")
# #     for i in range(2):
# #         axes[i].grid()
# #         axes[i].set_xlabel("Training Time [$s$]")
# #     plt.legend()
# #     # plt.savefig(plotdir + nm[m] + ".png")

from scipy.signal import savgol_filter
import glob
filenames = DEFAULTS.tracklet_conv_iter2
filenames.sort()

Iter2 = []
for i, filename in enumerate(filenames):
    df = pd.read_csv(filename, index_col=None, header=0)
    Iter2.append(df.values)

filenames = DEFAULTS.tracklet_conv_iter3
filenames.sort()
Iter3 = []
for i, filename in enumerate(filenames):
    df = pd.read_csv(filename, index_col=None, header=0)
    Iter3.append(df.values)
Iter2[1:4] = Iter3

filenames = DEFAULTS.tracklet_conv_iter4
Iter4 = []
for i, filename in enumerate(filenames):
    df = pd.read_csv(filename, index_col=None, header=0)
    Iter4.append(df.values)
Iter2[-1] = Iter4[0]

columns = df.columns
colours = ['red', 'goldenrod', 'green', 'blue', 'purple']
conv_sizes = [(4, 8), (8, 16), (16, 32), (32, 64), (64, 128)]
names = ["$V_{(%d,%d)}^{(1024)}$"%tuple for tuple in conv_sizes]
epochtimes = [Iter2[i][:,3].mean() for i in range(len(Iter2))]
traintimes = [Iter2[j][:,3].cumsum() for j in range(len(Iter2))]

figloss, loss = plt.subplots(1, 2, figsize=(12,5), sharey = True)
figpcon, pcon = plt.subplots(1, 2, figsize=(12,5), sharey = True)
loss[0].set_ylabel("Loss")
pcon[0].set_ylabel("$\\pi$-contimination")
title = [" - Training", " - Validation"]
for i in range(2):
    loss[i].grid()
    pcon[i].grid()
    loss[i].set_title(DEFAULTS.letter[i] + title[i])
    pcon[i].set_title(DEFAULTS.letter[i] + title[i])
    pcon[i].set_xlabel("Training Time [s]")
    loss[i].set_xlabel("Training Time [s]")
for j in range(len(Iter2)):
    T_loss = savgol_filter(Iter2[j][:,1],3, 1)
    V_loss = savgol_filter(Iter2[j][:,4],3, 1)
    T_pcon = savgol_filter(Iter2[j][:,2],3, 1)
    V_pcon = savgol_filter(Iter2[j][:,5],3, 1)
    T = traintimes[j]
    mask1 = T < 2800
    # mask2 = np.logical_and(T>700, mask1)
    loss[0].plot(T[mask1], T_loss[mask1], color=colours[j],)
    loss[1].plot(T[mask1], V_loss[mask1], linestyle = '-', color=colours[j], label=names[j])
    pcon[0].plot(T[mask1], T_pcon[mask1], color=colours[j],)
    pcon[1].plot(T[mask1], V_pcon[mask1], linestyle = '-', color=colours[j], label=names[j])
loss[1].legend()
pcon[1].legend()

figloss.savefig(DEFAULTS.plotdir + "tracklet_conv_iter-loss.png")
figpcon.savefig(DEFAULTS.plotdir + "tracklet_conv_iter-pcon.png")

# axes[2].set_xlabel("Model no.")
# axes[2].set_ylabel("Average Training Epoch [$s$]")
# for k in range(2):
#     axes[k].legend()
#     axes[k].grid()
