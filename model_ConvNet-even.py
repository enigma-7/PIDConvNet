import numpy as np
import matplotlib.pyplot as plt
from TOOLS import DATA, MODELS, LOG, METRICS
import random, matplotlib, datetime, os

run_no = '000265378/'
dataname = 'even/'
directory = 'data/output/' + run_no + dataname
raw_data = np.load(directory + '0_tracks.npy')
raw_info = np.load(directory + '0_info_set.npy')
print("Loaded: %s" % directory)

dataset, infoset = DATA.process_1(raw_data, raw_info)
X, y = DATA.shuffle_(dataset/1024, infoset[:,0])            #Tracks and targets
print("Electron occurence: %.2f " % (sum(y)/len(y)))

conv_size1 = 8
conv_size2 = 16
dense_size1 = 256
dense_size2 = 64

stamp = datetime.datetime.now().strftime("%d-%m-%H%M%S")
mname = "conv-%d-%d-filters-dense-%d-%d-nodes-"%(conv_size1,
    conv_size2, dense_size1, dense_size2)
tensorboard, csvlogger = LOG.logger_(run_no, 'test/', mname, stamp)


net1 = MODELS.new
net1.compile(optimizer='adam', loss='binary_crossentropy',
    metrics=['accuracy', METRICS.pion_con, METRICS.F1])

for i in range(1):
    net1.fit(x=X, y=y, batch_size = 100, epochs=10, validation_split=0.4,
        callbacks=[])
    y_pred = net1.predict(X[:2000])
    e_pred = y_pred[y[:2000]==1]
    p_pred = y_pred[y[:2000]==0]
    plt.figure(figsize=(8,6))
    plt.hist(e_pred, alpha=0.5, label = 'positive')
    plt.hist(p_pred, alpha=0.5, label = 'negative')
    plt.legend()
    plt.show()
