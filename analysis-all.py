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

stamp = datetime.datetime.now().strftime("%d-%m-%H%M%S")
fname = run_no + dataname + f"conv-32-64-filters-dense-256-64-nodes-" + stamp

tensorboard = TensorBoard(log_dir='logs-TB/%s'%fname, update_freq=500)
csvlogger = CSVLogger('logs-CSV/%s'%fname)
print(fname)

net1 = MODELS.new
net1.compile(optimizer='adam', loss='binary_crossentropy', metrics=[MODELS.pion_con, MODELS.prec, MODELS.F1])
net1.fit(x=X, y=y, batch_size = 100, epochs=10, validation_split=0.4)#, callbacks=[tensorboard, csvlogger])
net1.summary()
