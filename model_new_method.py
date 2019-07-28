import numpy as np
import matplotlib.pyplot as plt
from TOOLS import DATA, MODELS, LOG, METRICS, PLOT
import random, matplotlib, datetime, os
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPool2D, concatenate
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard, Callback

run_no = '000265378/'
dataname = 'even/'
directory = 'data/output/' + run_no + dataname
raw_data = np.load(directory + '0_tracks.npy')
raw_info = np.load(directory + '0_info_set.npy')
print('Loaded: %s' % directory)

dataset, infoset = DATA.process_1(raw_data, raw_info)
X, y = DATA.shuffle_(dataset/1024, infoset[:,0])
columns = ["label", "nsigmae", "nsigmap", "PT", "${dE}/{dx}$", "Momenta [GeV]", "eta", "theta", "phi", "event", "V0trackID",  "track"]
infoarray = infoset[:,4:5]
print(columns[4:5])

input_main = Input(shape=X.shape[1:], name="tracklet")
x = Conv2D(8, [3,3], activation='relu', padding ='same')(input_main)
x = MaxPool2D([2,2], 2, padding='valid')(x)
x = Conv2D(16, [3,3], activation='relu', padding='same')(x)
x = MaxPool2D([2,2], 2, padding='valid')(x)
flattened = Flatten()(x)
input_aux = Input(shape=infoarray.shape[1:], name="info")
x = concatenate([input_aux, flattened])
x = Dense(256)(x)
x = Dense(64)(x)
output_aux = Dense(1, activation='sigmoid')(x)
"""
y = Dense(256)(flattened)
y = Dense(64)(y)
output_non = Dense(1, activation='sigmoid', name="output_non")(y)


input_aux = Input(shape=infoarray.shape[1:])
x = Dense(256, activation='relu')(input_aux)
x = Dense(64, activation='relu')(x)
output_aux = Dense(1, activation='sigmoid')(x)
"""

model = Model(inputs=[input_main,input_aux], outputs=output_aux)
model.summary()
model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=1e-4), loss='binary_crossentropy',metrics=['accuracy'])
model.fit([X,infoarray], y, batch_size=100, epochs=10, validation_split=0.4, )
