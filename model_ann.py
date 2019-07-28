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
elec_data, elec_info = DATA.elec_strip_(dataset, infoset)
pion_data, pion_info = DATA.pion_strip_(dataset, infoset)
class_data = [pion_data, elec_data]
class_info = [pion_info, elec_info]
cnames = ["$\\pi$","$e$"]
colour = ['r', 'g']
styles = ['--','-.']

X, T = DATA.shuffle_(dataset/1024, infoset[:,0])
columns = ["label", "nsigmae", "nsigmap", "PT", "${dE}/{dx}$", "Momenta [GeV]", "eta", "theta", "phi", "event", "V0trackID",  "track"]
nx = 5
ny = 4
I = infoset[:,ny:nx+1]
I = (I - I.mean(axis=0))/P.std(axis=0)
# P = P/(P.max(axis=0))
print(columns[ny:nx+1])

plt.figure(figsize=(8,6))
plt.title("Bethe-Bloch")
plt.grid()
for i,c in enumerate(class_data):
    x = class_info[i][:,nx]
    y = class_info[i][:,ny]
    binn = 10
    bins = np.linspace(x.min(), x.max(),binn)
    #inds = np.digitize(x, bins)
    mean = [y[((x>=bins[i]) & (x<=bins[i+1]))].mean() for i in  range(binn-1)]
    binx = (bins[:-1]+bins[1:])/2
    plt.plot(binx, mean,  color = 'k', linestyle = styles[i], label=cnames[i] + "-mean")
    plt.scatter(x, y, alpha=0.06, color = colour[i], label=cnames[i])
    plt.xlabel(columns[nx])
    plt.ylabel(columns[ny])
plt.legend()

input_aux = Input(shape=I.shape[1:], name="info")
x = Dense(256)(input_aux)
#x = Dense(1024)(x)
x = Dense(256)(x)
x = Dense(64)(x)
output_aux = Dense(1, activation='sigmoid')(x)

model = Model(inputs=input_aux, outputs=output_aux)
model.summary()
model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy', METRICS.pion_con])
model.fit(I, T, batch_size=100, epochs=10, validation_split=0.3)
