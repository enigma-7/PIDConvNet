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

# X = np.append(np.load(datadir + 'DS3_pad/' + 'track_dataset.npy'),
#     np.load(datadir + 'DS4_pad/' + 'track_dataset.npy'), axis = 0)
#
# infoset = np.append(np.load(datadir + 'DS3_pad/' + 'track_infoset.npy'),
#     np.load(datadir + 'DS4_pad/' + 'track_infoset.npy'), axis = 0)

dataname = 'DS3/'
ocdbdir = DEFAULTS.ocdbdir
datadir = DEFAULTS.datadir + dataname
plotdir = DEFAULTS.plotdir

X = np.load(datadir + 'tracklet_dataset.npy')/1024
infoset = np.load(datadir + 'tracklet_infoset.npy')
print("Loaded: %s \n" % datadir )

(X, infoset), (Xv, valid_infoset), (Xt, test_infoset) = DATA.TVT_split_(X, infoset)
T  = infoset[:,0].astype(int)
Tv = valid_infoset[:,0].astype(int)
Tt = test_infoset[:,0].astype(int)
I  = infoset
Iv = valid_infoset
It = test_infoset

(cs_1, cs_2, d1_1, d1_2) = (8, 16, 256, 128)
dropout = 0.45

stamp = datetime.datetime.now().strftime("%d-%m-%H%M%S") + "_"
mname = "track_V_C_chamber_dropout%.2f"%dropout
tensorboard, csvlogger = LOG.logger_(dataname, stamp, mname)

input = Input(shape=X.shape[1:], name="X-in")
x = Conv2D(cs_1, [3,4], bias_initializar='normal', activation='relu', padding ='same')(input)
x = MaxPool2D([2,2], 2, padding='valid')(x)
x = Conv2D(cs_2, [2,3], bias_initializar='normal', activation='relu', padding='same')(x)
x = MaxPool2D([2,2], 2, padding='valid')(x)
x = Flatten()(x)
x = Dropout(dropout)(x)
x = Dense(d1_1)(x)
x = Dense(d1_2)(x)
output = Dense(1, activation='sigmoid', name="X-out")(x)

model = Model(inputs=input, outputs=output)
model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=1e-3), loss='binary_crossentropy', metrics=[ML.pion_con])
model.fit(x=X, y=T, batch_size = 2**10, epochs=200, validation_data=(Xv,Tv), callbacks=[tensorboard, csvlogger])
# from scipy.signal import savgol_filter
# import glob
#
# filenames = glob.glob('logs-CSV/DS5/04-10-124841_track_V_U_dropout_*')
# filenames.sort()
#
# frame, nm = LOG.import_(filenames, position=1)
# frame.head()
# array = frame.values.reshape(-1,100,5)
# array[0]
# colours = ['steelblue', 'teal', 'lightseagreen', 'cyan', 'purple']
# letters = ['(a)', '(b)', '(c)', '(d)']
# epochtimes = array[:,:,2].mean(axis=1)
# fig, axes = plt.subplots(1, 4, figsize=(20,6), sharey=True)
#
# for j in range(array.shape[0]):
#     axes[j].plot(array[j,:,2].cumsum(), savgol_filter(array[j,:,1], 11, 1), color=colours[j])
#     axes[j].plot(array[j,:,2].cumsum(), savgol_filter(array[j,:,4], 11, 1), color=colours[j], linestyle=':')
#     axes[j].grid()
#     axes[j].set_xlabel("Training Time [s]")
#     axes[j].set_title(letters[j] + " Dropout = " +  nm[j])
# axes[0].set_ylabel("Loss")
# plt.savefig(plotdir + mname + ".png")

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
    axes[0].legend(loc=9)
    axes[0].grid()
    filename= mname
    fig.savefig(plotdir + filename + 'class.png')


P = model.predict(Xt).reshape(-1)
classification_(P, Tt)
