import numpy as np
import matplotlib.pyplot as plt
from TOOLS import DATA, MODELS, LOG, METRICS, PLOT
import random, matplotlib, datetime, os
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPool2D, concatenate
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard, Callback
from sklearn.decomposition import PCA
from sklearn import preprocessing

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

input_aux = Input(shape=I.shape[1:], name="info")
x = Dense(64)(input_aux)
x = Dense(8)(x)
output_aux = Dense(1, activation='sigmoid')(x)

model = Model(inputs=input_aux, outputs=output_aux)
#model.summary()
model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=1e-4), loss='binary_crossentropy',metrics=[METRICS.pion_con])
model.fit(I, T, batch_size=100, epochs=10, validation_data=(Iv,Tv), )

P = model.predict(It)

PLOT.classification_(P,Tt, scale='linear')
