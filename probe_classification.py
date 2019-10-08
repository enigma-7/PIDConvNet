import numpy as np
from TOOLS import DATA, MODELS, LOG, ML, PLOT, DEFAULTS
import random, datetime, os, matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn import preprocessing
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPool2D, concatenate, GaussianNoise
matplotlib.rcParams.update({'font.size': 16})
matplotlib.rcParams['text.usetex'] = True

dataname = 'DS4/'
ocdbdir = DEFAULTS.ocdbdir
datadir = DEFAULTS.datadir + dataname
plotdir = DEFAULTS.plotdir

infoset = np.load(datadir + 'tracklet_infoset.npy')
dataset = np.load(DEFAULTS.modldir + 'P-DS4-tracklet_V_U_wbce5-1.npy')            #Predictions from ConvNets
dataset = np.append(dataset.reshape(-1,1), infoset[:,14:], axis = 1)              #All tracklet specific information
print("Loaded: %s \n" % datadir )

set(infoset[:,14])
cols = DEFAULTS.info_cols_tracklet_ + DEFAULTS.ocdb_cols1 + DEFAULTS.ocdb_cols2
cols2 = ['sigmoid'] + DEFAULTS.info_cols_tracklet_[14:] + DEFAULTS.ocdb_cols1 + DEFAULTS.ocdb_cols2

thresholds=np.linspace(0,1,100)
pioncon1 = np.zeros(6)
pioncon2 = np.zeros(6)
eleceff = np.zeros(6)
Ueleceff = np.zeros(6)
decbound = np.zeros(6)

X_ = {}                 #Track level data
I_ = {}                 #Track level info

for l in range(1,7):
    mask = infoset[:,13] == l
    I_[str(l)] = infoset[mask][::l,:14]
    X_[str(l)]  = dataset[:,0][mask].reshape(-1,l)

P_ = {}
Xt_ = {}
It_ = {}
epoch = [100]*6
batch = [2**10]*6
pioncon = np.zeros(6)
Upioncon = np.zeros(6)
for l in range(1,7):
    # X_[str(l)], I_[str(l)] = DATA.shuffle_(X_[str(l)], I_[str(l)])
    (X, I), (Xv, Iv), (Xt, It) = DATA.TVT_split_(X_[str(l)], I_[str(l)], test_split=0.35, valid_split=0.15)
    T  = I[:,0].astype(int)
    Tv = Iv[:,0].astype(int)
    Tt = It[:,0].astype(int)
    print(X.shape)

    input = Input(shape=X.shape[1:], name="info")
    x = Dense(36, activation = 'relu')(input)
    x = Dropout(0.5)(x)
    x = Dense(216, activation = 'relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(36, activation = 'relu')(x)
    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=input, outputs=output)
    model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=1e-3), loss=ML.WBCE, metrics=['accuracy'])
    history = model.fit(X, T, batch_size=batch[l-1], epochs=epoch[l-1], validation_data=(Xv,Tv), )

    P = model.predict(Xt).reshape(-1)
    P_[str(l)] = P
    Xt_[str(l)] = Xt
    It_[str(l)] = It

    e_pred = P[Tt==1]
    p_pred = P[Tt==0]
    argsort = e_pred.argsort()

    TP = np.array([(e_pred>threshold).sum() for threshold in thresholds])
    FN = np.array([(e_pred<threshold).sum() for threshold in thresholds])
    FP = np.array([(p_pred>threshold).sum() for threshold in thresholds])
    TN = np.array([(p_pred<threshold).sum() for threshold in thresholds])

    TPR = TP/(FN+TP)            #True Positive Rate / Recall
    FPR = FP/(TN+FP)            #False Positive Rate
    uTPR = np.sqrt(TPR*(1-TPR)/(TP + FN))
    uFPR = np.sqrt(FPR*(1-FPR)/(TN + FP))

    pioncon[l-1]  =  FPR[TPR<0.905][0]
    Upioncon[l-1] = uFPR[TPR<0.905][0]

L_ = {}
TT_ = {}
pioncon1 = np.zeros(6)
Upioncon1 = np.zeros(6)
for l in range(1,7):
    b = np.linspace(0,1,100)
    Xt = Xt_[str(l)].reshape(-1)
    It = np.repeat(It_[str(l)], l, axis=0)
    Tt = It[:,0]

    L, TT, lY = DATA.likelihood_(Xt, It)
    print(set(lY))
    e_pred = L[TT==1]
    p_pred = L[TT==0]
    argsort = e_pred.argsort()

    TP = np.array([(e_pred>threshold).sum() for threshold in thresholds])
    FN = np.array([(e_pred<threshold).sum() for threshold in thresholds])
    FP = np.array([(p_pred>threshold).sum() for threshold in thresholds])
    TN = np.array([(p_pred<threshold).sum() for threshold in thresholds])

    TPR = TP/(FN+TP)            #True Positive Rate / Recall
    FPR = FP/(TN+FP)            #False Positive Rate
    uTPR = np.sqrt(TPR*(1-TPR)/(TP + FN))
    uFPR = np.sqrt(FPR*(1-FPR)/(TN + FP))

    pioncon1[l-1]  =  FPR[TPR<0.905][0]
    Upioncon1[l-1] = uFPR[TPR<0.905][0]
plt.figure(figsize=(8,6))
plt.errorbar(range(1,7), pioncon1, Upioncon1, 0, 'k.', label = "Likelihood")
plt.errorbar(range(1,7), pioncon, Upioncon, 0, 'b.', label = "ANN")
plt.legend()
plt.xlabel("Layers")
plt.ylabel("$\\pi$-contamination")
plt.grid()
# for layers in [4,5,6]:
#     layers = int(layers)
#     mask = infoset[:,13] == layers
#     I = infoset[mask][::layers]
#     X = dataset[:,0][mask].reshape(-1,layers)
#
#
#
#     (X, I), (Xv, Iv), (Xt, It) = DATA.TVT_split_(X, I, test_split=0.3, valid_split=0.2)
#     T  = I[:,0].astype(int)
#     Tv = Iv[:,0].astype(int)
#     Tt = It[:,0].astype(int)
#
#
#
#     model = Model(inputs=input_X, outputs=output_aux)
#     model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=1e-4), loss=ML.WBCE, metrics=[ML.pion_con])
#     history = model.fit(X, T, batch_size=2**10, epochs=100, validation_data=(Xv,Tv), )
#
#     P = model.predict(Xt)
#
#     e_pred = P[Tt==1]
#     p_pred = P[Tt==0]
#     argsort = e_pred.argsort()
#
#     TP = np.array([(e_pred>threshold).sum() for threshold in thresholds])
#     FN = np.array([(e_pred<threshold).sum() for threshold in thresholds])
#     FP = np.array([(p_pred>threshold).sum() for threshold in thresholds])
#     TN = np.array([(p_pred<threshold).sum() for threshold in thresholds])
#
#     TPR = TP/(FN+TP)            #True Positive Rate / Recall
#     FPR = FP/(TN+FP)            #False Positive Rate
#
#     pioncon1[layers-1] = FPR[TPR<0.905][0]          #estimate pion contamination
#     eleceff[layers-1] = TPR[TPR<0.905][0]
#     decbound[layers-1] = thresholds[TPR<0.905][0]
#
#     L, TT, layerz = DATA.likelihood_(
#         Xt.reshape(-1),
#         np.repeat(It, layers, axis=0))
#     e_pred = L[TT==1]
#     p_pred = L[TT==0]
#     argsort = e_pred.argsort()
#
#     TP = np.array([(e_pred>threshold).sum() for threshold in thresholds])
#     FN = np.array([(e_pred<threshold).sum() for threshold in thresholds])
#     FP = np.array([(p_pred>threshold).sum() for threshold in thresholds])
#     TN = np.array([(p_pred<threshold).sum() for threshold in thresholds])
#
#     TPR = TP/(FN+TP)            #True Positive Rate / Recall
#     FPR = FP/(TN+FP)            #False Positive Rate
#
#     pioncon2[layers-1] = FPR[TPR<0.905][0]          #estimate pion contamination??
    # PLOT.classification_(P, Tt)
# for layers in [2,3,4,5,6]:
#     layers = int(layers)
#     mask = infoset[:,13] == layers
#     I_ = infoset[mask]
#     X  = dataset[:,0][mask].reshape(-1,layers)
#
#     # X, I_ = DATA.shuffle_(X,I_)
#     (X, I), (Xv, Iv), (Xt, It) = DATA.TVT_split_(X, I_[::layers])
#     T  = I[:,0].astype(int)
#     Tv = Iv[:,0].astype(int)
#     Tt = It[:,0].astype(int)
#
#     input_X = Input(shape=X.shape[1:], name="info")
#     x = Dense(layers**2)(input_X)
#     x = Dense(layers)(x)
#     output_aux = Dense(1, activation='sigmoid')(x)
#
#     model = Model(inputs=input_X, outputs=output_aux)
#     model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=1e-4), loss=ML.WBCE, metrics=[ML.pion_con])
#     model.fit(X, T, batch_size=2**9, epochs=100, validation_data=(Xv,Tv), )


# L, TT, layerz = DATA.likelihood_(dataset[:,0], infoset)
#
# thresholds=np.linspace(0,1,100)
# pioncon = np.zeros(6)
# eleceff = np.zeros(6)
# decbound = np.zeros(6)
# for i in range(1,7):
#     mask = layers == i
#     L = Ltrack[mask]
#     T = Ttrack[mask]
#     e_pred = L[T==1]
#     p_pred = L[T==0]
#     argsort = e_pred.argsort()
#
#     TP = np.array([(e_pred>threshold).sum() for threshold in thresholds])
#     FN = np.array([(e_pred<threshold).sum() for threshold in thresholds])
#     FP = np.array([(p_pred>threshold).sum() for threshold in thresholds])
#     TN = np.array([(p_pred<threshold).sum() for threshold in thresholds])
#
#     TPR = TP/(FN+TP)            #True Positive Rate / Recall
#     FPR = FP/(TN+FP)            #False Positive Rate
#
#     pioncon1[i-1] = FPR[TPR<0.905][0]          #estimate pion contamination
#     eleceff[i-1] = TPR[TPR<0.905][0]
#     decbound[i-1] = thresholds[TPR<0.905][0]
#
#     # PLOT.classification_(L, T)
# plt.figure(figsize=(8,6))
# plt.plot(range(1,7), pioncon1, '*')
# plt.plot(range(2,7), pioncon[1:])
# plt.ylabel("$\\pi$-contamination")
# plt.xlabel("No. of layers in track")
# plt.grid(which='both')
# plt.yscale('log')
#
# from scipy.optimize import curve_fit
#
# # n = 17    ## 1900 - 2300
# # plt.hist(infoset[:,n][np.logical_and(infoset[:,n] > truncL[0], infoset[:,n] < truncU[0])])
# def gaussian(x, mu, sig):
#     return (1/(sig*np.sqrt(2*np.pi))*np.exp(-0.5*((x-mu)**2/sig**2)))
#
# index = [18, 19, 20]            #infoset index
# bin_no = [11, 15, 18]           #No. of bins
# truncL = [0.38, 1.44, 0.126]    #Lower bound
# truncU = [0.60, 1.53, 0.140]    #Upper bound
#
# params = np.zeros((len(index),2))
# boundL = np.zeros((len(index),2))
# boundU = np.zeros((len(index),2))
#
# array = [0,1,2]
#
# fig, axes = plt.subplots(1, len(array), figsize=(18,6))
# # params = np.array([index, bin_no, truncL,truncU])
# for j, i in enumerate(array):
#     n = index[i]
#     mask = np.logical_and(infoset[:,n] > truncL[i], infoset[:,n] < truncU[i])
#     y, b = np.histogram(infoset[:,n][mask], bins=bin_no[i], density=True)
#     x = (b[:-1] + b[1:])/2
#     u = np.sqrt(y)
#
#     p0  = np.array([(truncU[i] + truncL[i])/2, (truncU[i] - truncL[i])/5])
#
#     name = ["sigma", "mu"]
#
#     popt, pcov = curve_fit(gaussian, x, y, p0, sigma=u, absolute_sigma=True)
#     dymin = (y - gaussian(x,*popt))/u       #vectorised again
#     min_chisq = sum(dymin**2)
#     dof = len(x) - len(popt)                #number of degrees of freedom
#
#     mult = 2
#     mult1 = 1.0
#     mult2 = 2.0
#
#     tmodel = np.linspace(popt[0] - mult*p0[1],popt[0] + mult*p0[1],1000)
#     lower = tmodel[np.logical_and(tmodel > popt[0]-mult2*popt[1], tmodel < popt[0]-mult1*popt[1])]
#     upper = tmodel[np.logical_and(tmodel > popt[0]+mult1*popt[1], tmodel < popt[0]+mult2*popt[1])]
#
#     params[i] = popt
#     boundL[i] = [lower[0], lower[-1]]
#     boundU[i] = [upper[0], upper[-1]]
#
#     lzero = np.zeros(lower.shape[0])
#     uzero = np.zeros(upper.shape[0])
#
#     """
#     print("Chi square: %.2f"%min_chisq)
#     print("Number of degrees of freedom %d"%dof)
#     print("Chi square per degree of freedom: %.2f"%(min_chisq/dof), "\n")
#
#     for i,pmin in enumerate(popt):
#         print('%2i %-10s %12f +/- %10f'%(i+1, name[i] ,pmin, np.sqrt(pcov[i,i])*np.sqrt(min_chisq/dof)),'\n')
#
#     perr = np.sqrt(np.diag(pcov))
#     print(perr, '\n')
#     """
#
#     axes[j].errorbar(x, y, u, 0, 'k.', label = "%d Datapoints"%x.shape[0])
#     axes[j].fill_between(lower, lzero, gaussian(lower, *popt), where=gaussian(lower, *popt) >= lzero,
#         facecolor='cyan', interpolate=True, alpha=0.4, label= "$R_1$")# = [%f,%f]$"%(boundL[i,0],boundL[i,1]))
#     axes[j].fill_between(upper, uzero, gaussian(upper, *popt), where=gaussian(upper, *popt) >= uzero,
#         facecolor='darkcyan', interpolate=True, alpha=0.4, label= "$R_2$")#" = [%.2f\\times 10^{-1},%.2f\\times 10^{-1}]$"%(boundU[i,0]*10,boundU[i,1]*10))
#     #plt.plot(tmodel, gaussian(tmodel,*p0), label="guess")
#     axes[j].plot(tmodel, gaussian(tmodel, *popt), '-r', label="$P_{normal}(x;%.2f,%.2f)$"%(popt[0], popt[1]))
#     axes[j].set_xlabel(columns[n])
#     axes[j].set_ylabel("Normalized Counts")
#     axes[j].legend()
#     axes[j].grid()
#
# # fig.savefig(plotdir + "ambient_conditions.png")
# for j,i in enumerate(array):
#     Lmask = np.logical_and(It[:,index[i]] > boundL[i,0], It[:,index[i]] < boundL[i,1])
#     Umask = np.logical_and(It[:,index[i]] > boundU[i,0], It[:,index[i]] < boundU[i,1])
#     PLOT.classification_(P[Lmask],Tt[Lmask])
#     PLOT.classification_(P[Umask],Tt[Umask])


# trackID = np.array(list(set([(r,e,v,t) for [r,e,v,t] in infoset[:,9:13].astype(int)])))
#
# Ltrack = np.zeros((trackID.shape[0], 6))
# Ttrack = np.zeros(trackID.shape[0])
# layers = np.zeros(trackID.shape[0])
# anomalies = []
# for i, [r,e,v,t] in enumerate(trackID):
#     rmask = infoset[:,9]  == r
#     emask = infoset[:,10] == e
#     vmask = infoset[:,11] == v
#     tmask = infoset[:,12] == t
#     mask = np.logical_and(np.logical_and(np.logical_and(rmask, emask), vmask), tmask)
#     if len(set(infoset[:,0][mask])) != 1:
#         print('somethings gone funny at %d'%i)
#         anomalies.append(i)
#         continue
#     if mask.sum() > 6.0:
#         print(i)
#         anomalies.append(i)
#         continue
#     layers[i] = mask.sum()
#     Ltrack[i] = np.pad(dataset[mask], (0, 6 - mask.sum()), 'constant', constant_values=(0, 1))
