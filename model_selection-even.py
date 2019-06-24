import numpy as np
import matplotlib.pyplot as plt
import matplotlib, DATA, MODELS, time, datetime, yaml, json, yaml, random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.callbacks import TensorBoard, CSVLogger
from yaml import Loader, Dumper

run_no = '000265378/'
dataname = 'even/'
directory = 'data/output/' + run_no + dataname
raw_data = np.load(directory + '0_tracks.npy')
raw_info = np.load(directory + '0_info_set.npy')
print('Loaded: %s' % directory)

dataset, infoset = DATA.process_1(raw_data, raw_info)
X, y = DATA.shuffle_(dataset/1024, infoset[:,0])
print("Electron occurence: %.2f " % (sum(y)/len(y)))

##      MODEL       ##


"""
conv_layers = [1, 2, 3]
dense_layers = [1, 2]
layer_sizes = [32, 64]

for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            stamp = datetime.datetime.now().strftime("%d-%m-%H%M%S")
            fname = f"{conv_layer}-conv-{layer_size}-nodes-{dense_layer}-dense-{layer_size}-nodes-" + stamp
            tensorboard = TensorBoard(log_dir='logs/%s%s'%(dataname,fname))

            print(fname)
            model = Sequential()

            model.add(Conv2D(layer_size, [3,3], activation='relu', padding ='same'))
            model.add(MaxPool2D([2,2], 2, padding='valid'))

            for l in range(conv_layer-1):
                model.add(Conv2D(layer_size, [3,3], activation='relu', padding ='same'))
                model.add(MaxPool2D([2,2], 2, padding='valid'))

            model.add(Flatten())

            for l in range(dense_layer):
                model.add(Dense(layer_size))

            model.add(Dense(1, activation='sigmoid'))
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            model.fit(X, y, batch_size=100, epochs=10, validation_split=0.4, callbacks=[tensorboard])
            model.summary()


conv_sizes1 = [8, 16, 32]
conv_sizes2 = [32, 64 ,128]
dense_sizes1 = [256, 512, 1024]
dense_sizes2 = [64, 128, 256]


conv_sizes1 = [32]
conv_sizes2 = [32]
dense_sizes1 = [256]
dense_sizes2 = [64]

stamp = datetime.datetime.now().strftime("%d-%m-%H%M%S")
for i in range(1):
    for conv_size1 in conv_sizes1:
        for conv_size2 in conv_sizes2:
            for dense_size1 in dense_sizes1:
                for dense_size2 in dense_sizes2:
                    fname = f"conv-{conv_size1}-{conv_size2}-filters-dense-{dense_size1}-{dense_size2}-nodes-" + stamp
                    #tensorboard = TensorBoard(log_dir='logs/%s%s'%(dataname,fname), update_freq=500)
                    print(fname)

                    model = Sequential([
                        Conv2D(conv_sizes1, [3,3], activation='relu', padding ='same'),
                        MaxPool2D([2,2], 2, padding='valid'),
                        Conv2D(conv_sizes2, [3,3], activation='relu', padding ='same'),
                        MaxPool2D([2,2], 2, padding='valid'),
                        Flatten(),
                        Dense(dense_size1),
                        Dense(dense_size2),
                        Dense(1, activation='sigmoid')])

                    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
                    model.fit(X, y, batch_size=100, epochs=10, validation_split=0.4)

                    model.summary()


print(history.history)
with open('logs/%s%s/'%(dataname,fname) + 'info.yaml', 'w') as outfile:
    yaml.dump(history.history, outfile)
"""
