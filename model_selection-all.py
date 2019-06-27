import numpy as np
import matplotlib.pyplot as plt
import DATA, MODELS, LOG, METRICS
import random, matplotlib, datetime, os

run_no = '000265378/'
dataname = 'all/'
directory = 'data/output/' + run_no + dataname
raw_data = np.load(directory + '0_tracks.npy')
raw_info = np.load(directory + '0_info_set.npy')
print('Loaded: %s' % directory)

dataset, infoset = DATA.process_1(raw_data, raw_info)
X, y = DATA.shuffle_(dataset/1024, infoset[:,0])
print("Electron occurence: %.2f " % (sum(y)/len(y)))

##      MODEL       ##

conv_sizes1 = [8, 16]
conv_sizes2 = [32, 64]
dense_sizes1 = [512, 1024]
dense_sizes2 = [128, 256]


stamp = datetime.datetime.now().strftime("%d-%m-%H%M%S")
for i in range(1):
    for conv_size1 in conv_sizes1:
        for conv_size2 in conv_sizes2:
            for dense_size1 in dense_sizes1:
                for dense_size2 in dense_sizes2:
                    mname = "conv-%d-%d-filters-dense-%d-%d-nodes-"%(conv_size1,
                        conv_size2, dense_size1, dense_size2)
                    tensorboard, csvlogger = MODELS.logger_(run_no, "test/", mname, stamp)

                    model = MODELS.blank_2_2_(conv_size1, conv_size2, dense_size1, dense_size2)
                    model.compile(optimizer='adam', loss='binary_crossentropy',
                        metrics=[MODELS.pion_con, MODELS.F1])
                    model.fit(X, y, batch_size=1000, epochs=1, validation_split=0.4)

                    model.summary()

"""
conv_layers = [1, 2, 3]
dense_layers = [1, 2]
layer_sizes = [32, 64]

stamp = datetime.datetime.now().strftime("%d-%m-%H%M%S")
for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            mname = f"{conv_layer}-conv-{layer_size}-nodes-{dense_layer}-dense-{layer_size}-nodes-"
            tensorboard, csvlogger = MODELS.logger_(run_no, "test/", mname, stamp)
            model = MODELS.blank_v_v_(conv_layer, dense_layer, layer_size, layer_size)

            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            model.fit(X, y, batch_size=100, epochs=10, validation_split=0.4, callbacks=[tensorboard])
            model.summary()

conv_sizes1 = [16]
conv_sizes2 = [64]
dense_sizes1 = [1024]
dense_sizes2 = [256]
"""
