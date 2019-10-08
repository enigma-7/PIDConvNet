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

dataname = 'DS2/'
ocdbdir = DEFAULTS.ocdbdir
datadir = DEFAULTS.datadir + dataname
plotdir = DEFAULTS.plotdir

dataset = np.load(datadir + 'tracklet_dataset.npy')
infoset = np.load(datadir + 'tracklet_infoset.npy')

print("Loaded: %s \n" % datadir )

columns = DEFAULTS.info_cols_tracklet_ + DEFAULTS.ocdb_cols1 + DEFAULTS.ocdb_cols2

ni = 18
nf = None
params = infoset[:,ni:nf]
print(columns[ni:nf])

scaler = preprocessing.StandardScaler()
params = scaler.fit_transform(params)
infoset[:,ni:nf] = params
infoframe = pd.DataFrame(data=infoset, columns=columns)
infoframe.head()

(X, infoset), (Xv, valid_infoset), (Xt, test_infoset) = DATA.TVT_split_(dataset/1024, infoset)
T  = infoset[:,0].astype(int)
Tv = valid_infoset[:,0].astype(int)
Tt = test_infoset[:,0].astype(int)
I  = infoset[:,ni:nf]
Iv = valid_infoset[:,ni:nf]
It = test_infoset[:,ni:nf]

(cs_1, cs_2, d1_1, d1_2, d2_1, d2_2) = (8, 16, 128, 64, 32, 16)
stamp = datetime.datetime.now().strftime("%d-%m-%H%M%S") + "_"
mname = "C-%d-%d-D-%d-%d-D6-%d-%d"%(cs_1,cs_2, d1_1, d1_2, d2_1, d2_2)
#tensorboard, csvlogger = LOG.logger_(run_no, dataname, stamp, mname)

input_main = Input(shape=X.shape[1:], name="X-in")
x = Conv2D(cs_1, [2,3], activation='relu', padding ='same')(input_main)
x = MaxPool2D([2,2], 2, padding='valid')(x)
x = Conv2D(cs_2, [2,3], activation='relu', padding='same')(x)
x = MaxPool2D([2,2], 2, padding='valid')(x)
x = Flatten()(x)
x = Dense(d1_1)(x)
x = Dense(d1_2)(x)
output_aux = Dense(1, activation='sigmoid', name="X-out")(x)
input_aux = Input(shape=I.shape[1:], name="info")
x = concatenate([input_aux, output_aux])
x = Dense(d2_1)(x)
x = Dense(d2_2)(x)
output_main = Dense(1, activation='sigmoid', name="I-out")(x)

def WBCE(y_true, y_pred, weight = 10.0, from_logits=False):
    y_pred = tf.cast(y_pred, dtype='float32')
    y_true = tf.cast(y_true, y_pred.dtype)
    return K.mean(ML.weighted_binary_crossentropy(y_true, y_pred, weight=weight, from_logits=from_logits), axis=-1)

model = Model(inputs=[input_main,input_aux], outputs=[output_main, output_aux])
# model.summary()
model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=1e-3),
    loss=WBCE, metrics=[ML.pion_con])
model.fit([X,I], [T,T], batch_size=2**9, epochs=10, validation_data=([Xv,Iv],[Tv,Tv]),)
    #callbacks=[tensorboard, csvlogger])

model.summary()

"""

input_main = Input(shape=X.shape[1:], name="X-in")
x = Conv2D(cs_1, [2,3], activation='relu', padding ='same')(input_main)
x = MaxPool2D([2,2], 2, padding='valid')(x)
x = Conv2D(cs_2, [2,3], activation='relu', padding='same')(x)
x = MaxPool2D([2,2], 2, padding='valid')(x)
x = Flatten()(x)
input_aux = Input(shape=I.shape[1:], name="gain")
x = concatenate([input_aux, x])
x = Dense(d1_1)(x)
x = Dense(d1_2)(x)
output_main = Dense(1, activation='sigmoid', name="X-out")(x)

model = Model(inputs=[input_main,input_aux], outputs=output_main)
# model.summary()
model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=1e-3),
    loss='binary_crossentropy',metrics=[ML.pion_con])
model.fit([X,I], T, batch_size=2**8, epochs=10, validation_data=([Xv,Iv],Tv),)
    #callbacks=[tensorboard, csvlogger])
"""

# plotdir = '/home/jibran/Desktop/neuralnet/plots/'
# directory = "logs-CSV/" + run_no + dataname
#
# frame = pd.read_csv(directory + stamp + mname, index_col=None, header=0)
# frame['model_name'] = mname
# frame = frame.set_index(['model_name', 'epoch'])
# frame.sort_index(axis=1,ascending=True)
# frame.head()
# array = frame.values.reshape(-1,20,11)
# colours = ['red', 'goldenrod', 'green', 'blue', 'purple']
#
# fig, axes = plt.subplots(1, 3, figsize=(18,6))
# timearray = array[0,:,5].cumsum()
# axes[0].plot(timearray, array[0,:,4], color='black', label="Total")
# axes[0].plot(timearray, array[0,:,10], color='black', linestyle='--')
# axes[0].plot(timearray, array[0,:,2], color='indigo', label="Output 1")
# axes[0].plot(timearray, array[0,:,8], color='indigo', linestyle='--')
# axes[0].plot(timearray, array[0,:,0], color='teal', label="Output 2")
# axes[0].plot(timearray, array[0,:,6], color='teal', linestyle='--')
# axes[0].set_ylabel("Loss")
# axes[1].plot(timearray, array[0,:,3], color='indigo', label="Training")
# axes[1].plot(timearray, array[0,:,9], color='indigo', label="Validation", linestyle='--')
# axes[1].set_title("Output 1")
# axes[1].set_ylabel("Pion Contamination")
# axes[2].plot(timearray, array[0,:,1], color='teal', label="Training")
# axes[2].plot(timearray, array[0,:,7], color='teal', label="Validation", linestyle='--')
# axes[2].set_ylabel("Pion Contamination")
# axes[2].set_title("Output 2")
# for i in range(3):
#     axes[i].grid()
#     axes[i].legend()
#     axes[i].set_xlabel("Training Time [$s$]")
# # plt.savefig(plotdir + mname + ".png")
#
# P = model.predict([Xt,It])
# thresholds=np.linspace(0,1,1000)
# cnames = ["$\\pi$","$e$"]
# colour = ['r', 'g']
# cols = ['black', 'indigo', 'teal']
# styles = ['--','-.']
# scale='log'
#
# for i in range(2):
#     e_pred = P[i][Tt==1]
#     p_pred = P[i][Tt==0]
#     argsort = e_pred.argsort()
#
#     TP = np.array([e_pred[e_pred>threshold].sum() for threshold in thresholds])
#     FN = np.array([e_pred[e_pred<threshold].sum() for threshold in thresholds])
#     FP = np.array([p_pred[p_pred>threshold].sum() for threshold in thresholds])
#     TN = np.array([p_pred[p_pred<threshold].sum() for threshold in thresholds])
#
#     TPR = TP/(FN+TP)
#     FPR = FP/(TN+FP)
#     PiC = FPR[TPR>0.9][-1]
#     DBD = e_pred[argsort[np.multiply(argsort.shape[-1],(1-90/100)).astype(int)]]
#     AUC = np.sum(np.abs(np.diff(FPR)*TPR[1:]))
#
#     fig, axes = plt.subplots(1, 2, figsize=(15,5))
#     axes[0].plot(FPR,TPR, cols[i+1])
#     axes[0].vlines(PiC, 0, 0.9, 'k', '--')
#     axes[0].hlines(0.9, 0, PiC, 'k', '--')
#     axes[0].set_ylabel("$e$-efficiency")
#     axes[0].set_xlabel("$\\pi$-contamination")
#     axes[0].text(PiC+0.05, 0.4, "$\\varepsilon_\\pi$ = "+ str(np.round(PiC, 3)), fontsize=18)
#     axes[0].text(PiC+0.05, 0.2, "AUC = "+ str(np.round(AUC, 2)), fontsize=18)
#     axes[0].grid()
#
#     c, b, p = axes[1].hist(p_pred ,label=cnames[0], color=colour[0],bins = 40, alpha = 0.7)
#     axes[1].hist(e_pred ,label=cnames[1], color=colour[1], bins = b, alpha = 0.7)
#     axes[1].set_yscale(scale)
#     axes[1].vlines(thresholds[TPR<0.9][0], 0, max(c), 'k', label="Decision Boundary")
#     axes[1].legend()
#     axes[1].grid()
#     # plt.savefig(plotdir + mname + "_class" + str(i) + ".png")
