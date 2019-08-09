import numpy as np
from TOOLS import DATA, MODELS, LOG, METRICS, PLOT
import random, matplotlib, datetime, os
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPool2D, concatenate
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn import preprocessing
import pandas as pd
from tensorflow.keras.callbacks import TensorBoard, Callback
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'font.size': 18})
matplotlib.rcParams['text.usetex'] = True

run_no = '000265378/'
dataname = 'all/'
directory = 'data/output/' + run_no + dataname
raw_data = np.load(directory + '0_tracks.npy')
raw_info = np.load(directory + '0_info_set.npy')
print('Loaded: %s' % directory)

dataset, infoset = DATA.process_1(raw_data, raw_info)
dataset, infoset = DATA.shuffle_(dataset, infoset)

columns = ["label", "nsigmae", "nsigmap", "PT", "${dE}/{dx}$", "Momenta [GeV]", "eta", "theta", "phi", "event", "V0trackID",  "track"]

nx = 3
ny = 9
params = infoset[:,nx:ny]
print(columns[nx:ny])

scaler = preprocessing.StandardScaler()
params = scaler.fit_transform(params)
infoset[:,nx:ny] = params
# infoframe = pd.DataFrame(data=infoset, columns=columns)
# infoframe.head()

(X, infoset), (Xv, valid_infoset), (Xt, test_infoset) = DATA.TVT_split_(dataset/1024, infoset)

T  = infoset[:,0]
I  = infoset[:,nx:ny]
Tv = valid_infoset[:,0]
Iv = valid_infoset[:,nx:ny]
Tt = test_infoset[:,0]
It = test_infoset[:,nx:ny]

cs_1 = 8
cs_2 = 16
d1_1 = 256
d1_2 = 64
d2_1 = 32
d2_2 = 16

input_main = Input(shape=X.shape[1:], name="tracklet")
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

stamp = datetime.datetime.now().strftime("%d-%m-%H%M%S") + "_"
mname = "C-%d-%d-D-%d-%d-D6-%d-%d"%(cs_1,cs_2, d1_1, d1_2, d2_1, d2_2)
tensorboard, csvlogger = LOG.logger_(run_no, dataname, stamp, mname)

model = Model(inputs=[input_main,input_aux], outputs=[output_aux, output_main])
model.summary()
model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=1e-3),
    loss='binary_crossentropy',metrics=[METRICS.pion_con])
model.fit([X,I], [T,T], batch_size=100, epochs=20, validation_data=([Xv,Iv],[Tv, Tv]),
    callbacks=[tensorboard, csvlogger])

plotdir = '/home/jibran/Desktop/neuralnet/plots/'
directory = "logs-CSV/" + run_no + dataname

frame = pd.read_csv(directory + stamp + mname, index_col=None, header=0)
frame['model_name'] = mname
frame = frame.set_index(['model_name', 'epoch'])
frame.sort_index(axis=1,ascending=True)
frame.head()
array = frame.values.reshape(-1,20,11)
colours = ['red', 'goldenrod', 'green', 'blue', 'purple']

fig, axes = plt.subplots(1, 3, figsize=(18,6))
timearray = array[0,:,5].cumsum()
axes[0].plot(timearray, array[0,:,4], color='black', label="Total")
axes[0].plot(timearray, array[0,:,10], color='black', linestyle='--')
axes[0].plot(timearray, array[0,:,2], color='indigo', label="Output 1")
axes[0].plot(timearray, array[0,:,8], color='indigo', linestyle='--')
axes[0].plot(timearray, array[0,:,0], color='teal', label="Output 2")
axes[0].plot(timearray, array[0,:,6], color='teal', linestyle='--')
axes[0].set_ylabel("Loss")
axes[1].plot(timearray, array[0,:,3], color='indigo', label="Training")
axes[1].plot(timearray, array[0,:,9], color='indigo', label="Validation", linestyle='--')
axes[1].set_title("Output 1")
axes[1].set_ylabel("Pion Contamination")
axes[2].plot(timearray, array[0,:,1], color='teal', label="Training")
axes[2].plot(timearray, array[0,:,7], color='teal', label="Validation", linestyle='--')
axes[2].set_ylabel("Pion Contamination")
axes[2].set_title("Output 2")
for i in range(3):
    axes[i].grid()
    axes[i].legend()
    axes[i].set_xlabel("Training Time [$s$]")
# plt.savefig(plotdir + mname + ".png")

P = model.predict([Xt,It])
thresholds=np.linspace(0,1,1000)
cnames = ["$\\pi$","$e$"]
colour = ['r', 'g']
cols = ['black', 'indigo', 'teal']
styles = ['--','-.']
scale='log'

for i in range(2):
    e_pred = P[i][Tt==1]
    p_pred = P[i][Tt==0]
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
    axes[0].plot(FPR,TPR, cols[i+1])
    axes[0].vlines(PiC, 0, 0.9, 'k', '--')
    axes[0].hlines(0.9, 0, PiC, 'k', '--')
    axes[0].set_ylabel("$e$-efficiency")
    axes[0].set_xlabel("$\\pi$-contamination")
    axes[0].text(PiC+0.05, 0.4, "$\\varepsilon_\\pi$ = "+ str(np.round(PiC, 3)), fontsize=18)
    axes[0].text(PiC+0.05, 0.2, "AUC = "+ str(np.round(AUC, 2)), fontsize=18)
    axes[0].grid()

    c, b, p = axes[1].hist(p_pred ,label=cnames[0], color=colour[0],bins = 40, alpha = 0.7)
    axes[1].hist(e_pred ,label=cnames[1], color=colour[1], bins = b, alpha = 0.7)
    axes[1].set_yscale(scale)
    axes[1].vlines(thresholds[TPR<0.9][0], 0, max(c), 'k', label="Decision Boundary")
    axes[1].legend()
    axes[1].grid()
    # plt.savefig(plotdir + mname + "_class" + str(i) + ".png")
