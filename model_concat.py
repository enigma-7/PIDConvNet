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
matplotlib.rcParams.update({'font.size': 18})
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt

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

input_main = Input(shape=X.shape[1:], name="tracklet")
x = Conv2D(8, [3,3], activation='relu', padding ='same')(input_main)
x = MaxPool2D([2,2], 2, padding='valid')(x)
x = Conv2D(16, [3,3], activation='relu', padding='same')(x)
x = MaxPool2D([2,2], 2, padding='valid')(x)
flattened = Flatten()(x)
input_aux = Input(shape=I.shape[1:], name="info")
x = concatenate([input_aux, flattened])
x = Dense(256)(x)
x = Dense(64)(x)
output_aux = Dense(1, activation='sigmoid')(x)

model = Model(inputs=[input_main,input_aux], outputs=output_aux)
model.summary()
model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=1e-4), loss='binary_crossentropy',metrics=[METRICS.pion_con])
model.fit([X,I], T, batch_size=100, epochs=2, validation_data=([Xv,Iv],Tv), )

cnames = ["$\\pi$","$e$"]
colour = ['r', 'g']
styles = ['--','-.']
P = model.predict([Xt,It])

plt.figure(figsize=(8,6))
for i in range(2):
    plt.hist(P[Tt==i],label=cnames[i], alpha = 0.5)
plt.legend()
plt.yscale('log')
plt.show()

PLOT.ROC_(P,Tt)
