import numpy as np
from TOOLS import DATA, MODELS, LOG, METRICS,PLOT
import random, datetime, os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 18})
matplotlib.rcParams['text.usetex'] = True
run_no = '000265378/'
dataname = 'all/'
directory = 'data/output/' + run_no + dataname
raw_data = np.load(directory + '0_tracks.npy')
raw_info = np.load(directory + '0_info_set.npy')
print("Loaded: %s" % directory)

dataset, infoset = DATA.process_1(raw_data, raw_info)
X , infoset = DATA.shuffle_(dataset/1024, infoset)

(X, infoset), (Xv, valid_infoset), (Xt, test_infoset) = DATA.TVT_split_(X, infoset)
T  = infoset[:,0]
Tv = valid_infoset[:,0]
Tt = test_infoset[:,0]
# I  = infoset[:,nx:ny]
# Iv = valid_infoset[:,nx:ny]
# It = test_infoset[:,nx:ny]# I  = infoset[:,nx:ny]

conv_size1 = 8
conv_size2 = 16
dense_size1 = 256
dense_size2 = 64

# stamp = datetime.datetime.now().strftime("%d-%m-%H%M%S")
# mname = "conv-%d-%d-filters-dense-%d-%d-nodes-"%(conv_size1,
#     conv_size2, dense_size1, dense_size2)
# tensorboard, csvlogger = LOG.logger_(run_no, 'test/', mname, stamp)

net1 = MODELS.new
net1.compile(optimizer='adam', loss='binary_crossentropy', metrics=[METRICS.pion_con])
net1.fit(x=X, y=T, batch_size = 100, epochs=1, validation_data=(Xv,Tv),)# callbacks=[tensorboard, csvlogger])

P = net1.predict(Xt)
plt.figure(figsize=(8,6))
for i in range(2):
    plt.hist(P[Tt==i], alpha = 0.5)
plt.legend()
#plt.yscale('log')
plt.show()
PLOT.ROC_(P,Tt)
