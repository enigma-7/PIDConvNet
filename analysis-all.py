import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.callbacks import TensorBoard, CSVLogger
import matplotlib, DATA, MODELS, time, datetime,random

run_no = '000265378/'
dataname = 'all/'
directory = 'data/output/' + run_no + dataname
raw_data = np.load(directory + '0_tracks.npy')
raw_info = np.load(directory + '0_info_set.npy')
print("Loaded: %s" % directory)

dataset, infoset = DATA.process_1(raw_data, raw_info)
X, y = DATA.shuffle_(dataset/1024, infoset[:,0])
print("Electron occurence: %.2f" % (100*sum(y)/len(y)))

conv_size1 = 8
conv_size2 = 16
dense_size1 = 256
dense_size2 = 64

stamp = datetime.datetime.now().strftime("%d-%m-%H%M%S")
mname = f"conv-{conv_size1}-{conv_size2}-filters-dense-{dense_size1}-{dense_size2}-nodes-"
tensorboard, csvlogger = MODELS.logger_(run_no, 'test/', mname, stamp)

net1 = MODELS.new
net1.compile(optimizer='adam', loss='binary_crossentropy', metrics=[MODELS.pion_con, MODELS.prec, MODELS.F1])
net1.fit(x=X, y=y, batch_size = 100, epochs=10, validation_split=0.4, callbacks=[tensorboard, csvlogger])
net1.summary()
