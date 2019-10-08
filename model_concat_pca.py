import numpy as np
import matplotlib.pyplot as plt
from TOOLS import DATA, MODELS, LOG, METRICS, PLOT
import random, matplotlib, datetime, os
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPool2D, concatenate
import tensorflow as tf
import pandas as pd
from sklearn.decomposition import PCA
from sklearn import preprocessing
from tensorflow.keras.callbacks import TensorBoard, Callback

run_no = '000265378/'
dataname = 'all/'
directory = 'data/output/' + run_no + dataname
raw_data = np.load(directory + '0_tracks.npy')
raw_info = np.load(directory + '0_info_set.npy')
print('Loaded: %s' % directory)

dataset, infoset = DATA.process_1(raw_data, raw_info)
dataset, infoset = DATA.shuffle_(dataset, infoset)

def WBCE(y_true, y_pred, weight = 7.0, from_logits=False):
    y_pred = tf.cast(y_pred, dtype='float32')
    y_true = tf.cast(y_true, y_pred.dtype)
    return K.mean(ML.weighted_binary_crossentropy(y_true, y_pred, weight=weight, from_logits=from_logits), axis=-1)

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
model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=1e-3), loss=WBCE, metrics=[ML.pion_con])
model.fit(x=X, y=T, batch_size = 2**9, epochs=20, validation_data=(Xv,Tv), callbacks=[tensorboard, csvlogger])


columns = ["label", "nsigmae", "nsigmap", "PT", "${dE}/{dx}$", "Momenta [GeV]", "eta", "theta", "phi", "event", "V0trackID",  "track"]
nx = 3
ny = 9
params = infoset[:,nx:ny]
print(columns[nx:ny])

scaler = preprocessing.StandardScaler()
scaled_Pdf =  scaler.fit_transform(params)
scaled_Pdf = pd.DataFrame(scaled_Pdf, columns=columns[nx:ny])
scaled_Pdf.head(7)

pca = PCA()
pca.fit(scaled_Pdf)

X = dataset/1024
T = infoset[:,0]
I = pca.transform(scaled_Pdf)

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
model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=1e-4),
    loss='binary_crossentropy',metrics=[METRICS.pion_con])
model.fit([X,I], T, batch_size=100, epochs=10, validation_split=0.4, )
