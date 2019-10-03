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

dataset = np.load(datadir + 'tracklet_dataset.npy')
infoset = np.load(datadir + 'tracklet_infoset.npy')

print("Loaded: %s \n" % datadir )

(X, infoset), (Xv, valid_infoset), (Xt, test_infoset) = DATA.TVT_split_(dataset/1024, infoset)
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

# input = Input(shape=X.shape[1:],)# name="X-in")
# x = Conv2D(cs_1, [2,3], activation='relu', padding ='same')(input)
# x = MaxPool2D([2,2], 2, padding='valid')(x)
# x = Conv2D(cs_2, [2,3], activation='relu', padding='same')(x)
# x = MaxPool2D([2,2], 2, padding='valid')(x)
# x = Flatten()(x)
# x = Dense(d1_1)(x)
# x = Dense(d1_2)(x)
# output = Dense(1, activation='sigmoid',)(x)# name="X-out")(x)
#
# def WBCE(y_true, y_pred, weight = 10.0, from_logits=False):
#     y_pred = tf.cast(y_pred, dtype='float32')
#     y_true = tf.cast(y_true, y_pred.dtype)
#     return K.mean(ML.weighted_binary_crossentropy(y_true, y_pred, weight=weight, from_logits=from_logits), axis=-1)
#
# model = Model(inputs=input, outputs=output)
# model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=1e-3), loss=WBCE, metrics=[ML.pion_con])
# model.fit(x=X, y=T, batch_size = 2**9, epochs=10, validation_data=(Xv,Tv))#, callbacks=[tensorboard, csvlogger])

# P = model.predict(Xt).reshape(-1)
modldir = "saved_models/"
# np.save(modldir + "P-tracklet_V_U_ambient.npy", P)
P = np.load(modldir + "P-tracklet_V_U_ambient.npy")

PLOT.classification_(P,Tt)

columns = DEFAULTS.info_cols_tracklet_ + DEFAULTS.ocdb_cols1

from scipy.optimize import curve_fit

# n = 17    ## 1900 - 2300
# plt.hist(infoset[:,n][np.logical_and(infoset[:,n] > truncL[0], infoset[:,n] < truncU[0])])
def gaussian(x, mu, sig):
    return (1/(sig*np.sqrt(2*np.pi))*np.exp(-0.5*((x-mu)**2/sig**2)))

index = [18, 19, 20]            #infoset index
bin_no = [11, 15, 18]           #No. of bins
truncL = [0.38, 1.44, 0.126]    #Lower bound
truncU = [0.60, 1.53, 0.140]    #Upper bound

params = np.zeros((len(index),2))
boundL = np.zeros((len(index),2))
boundU = np.zeros((len(index),2))

array = [0,1,2]

fig, axes = plt.subplots(1, len(array), figsize=(18,6))
# params = np.array([index, bin_no, truncL,truncU])
for j, i in enumerate(array):
    n = index[i]
    mask = np.logical_and(infoset[:,n] > truncL[i], infoset[:,n] < truncU[i])
    y, b = np.histogram(infoset[:,n][mask], bins=bin_no[i], density=True)
    x = (b[:-1] + b[1:])/2
    u = np.sqrt(y)

    p0  = np.array([(truncU[i] + truncL[i])/2, (truncU[i] - truncL[i])/5])

    name = ["sigma", "mu"]

    popt, pcov = curve_fit(gaussian, x, y, p0, sigma=u, absolute_sigma=True)
    dymin = (y - gaussian(x,*popt))/u       #vectorised again
    min_chisq = sum(dymin**2)
    dof = len(x) - len(popt)                #number of degrees of freedom

    mult = 2
    mult1 = 1.0
    mult2 = 2.0

    tmodel = np.linspace(popt[0] - mult*p0[1],popt[0] + mult*p0[1],1000)
    lower = tmodel[np.logical_and(tmodel > popt[0]-mult2*popt[1], tmodel < popt[0]-mult1*popt[1])]
    upper = tmodel[np.logical_and(tmodel > popt[0]+mult1*popt[1], tmodel < popt[0]+mult2*popt[1])]

    params[i] = popt
    boundL[i] = [lower[0], lower[-1]]
    boundU[i] = [upper[0], upper[-1]]

    lzero = np.zeros(lower.shape[0])
    uzero = np.zeros(upper.shape[0])

    """
    print("Chi square: %.2f"%min_chisq)
    print("Number of degrees of freedom %d"%dof)
    print("Chi square per degree of freedom: %.2f"%(min_chisq/dof), "\n")

    for i,pmin in enumerate(popt):
        print('%2i %-10s %12f +/- %10f'%(i+1, name[i] ,pmin, np.sqrt(pcov[i,i])*np.sqrt(min_chisq/dof)),'\n')

    perr = np.sqrt(np.diag(pcov))
    print(perr, '\n')
    """

    axes[j].errorbar(x, y, u, 0, 'k.', label = "%d Datapoints"%x.shape[0])
    axes[j].fill_between(lower, lzero, gaussian(lower, *popt), where=gaussian(lower, *popt) >= lzero,
        facecolor='cyan', interpolate=True, alpha=0.4, label= "$R_1$")# = [%f,%f]$"%(boundL[i,0],boundL[i,1]))
    axes[j].fill_between(upper, uzero, gaussian(upper, *popt), where=gaussian(upper, *popt) >= uzero,
        facecolor='darkcyan', interpolate=True, alpha=0.4, label= "$R_2$")#" = [%.2f\\times 10^{-1},%.2f\\times 10^{-1}]$"%(boundU[i,0]*10,boundU[i,1]*10))
    #plt.plot(tmodel, gaussian(tmodel,*p0), label="guess")
    axes[j].plot(tmodel, gaussian(tmodel, *popt), '-r', label="$P_{normal}(x;%.2f,%.2f)$"%(popt[0], popt[1]))
    axes[j].set_xlabel(columns[n])
    axes[j].set_ylabel("Normalized Counts")
    axes[j].legend()
    axes[j].grid()

# fig.savefig(plotdir + "ambient_conditions.png")
for j,i in enumerate(array):
    Lmask = np.logical_and(It[:,index[i]] > boundL[i,0], It[:,index[i]] < boundL[i,1])
    Umask = np.logical_and(It[:,index[i]] > boundU[i,0], It[:,index[i]] < boundU[i,1])
    PLOT.classification_(P[Lmask],Tt[Lmask])
    PLOT.classification_(P[Umask],Tt[Umask])
