import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
import tensorflow.keras.backend as K

tf.enable_eager_execution()

old = Sequential([
  Conv2D(64, [2,3], activation='relu', padding ='same'),
  MaxPool2D([2,2], 2, padding='valid'),
  Conv2D(128, [2,3], activation='relu', padding ='same'),
  MaxPool2D([2,2], 2, padding='valid'),
  Flatten(),
  Dense(1024),
  Dense(1, activation='linear', trainable=False),
  Dense(1, activation='sigmoid')
])

new = Sequential([
  Conv2D(8, [3,3], activation='relu', padding ='same'),
  MaxPool2D([2,2], 2, padding='valid'),
  Conv2D(16, [3,3], activation='relu', padding='same'),
  MaxPool2D([2,2], 2, padding='valid'),
  Flatten(),
  Dense(256),
  Dense(64),
  Dense(1, activation='sigmoid')
])

def blank_2_2_(conv_size1, conv_size2, dense_size1, dense_size2, droprate = 0.1):
    #   Two conv layers and two dense layers    #
    model = Sequential([
        Conv2D(conv_size1, [3,3], activation='relu', padding ='same'),
        MaxPool2D([2,2], 2, padding='valid'),
        Conv2D(conv_size2, [3,3], activation='relu', padding ='same'),
        MaxPool2D([2,2], 2, padding='valid'),
        Flatten(),
        Dropout(rate=droprate),
        Dense(dense_size1),
        Dense(dense_size2),
        Dense(1, activation='sigmoid')])
    return model

def blank_v_v_(conv_layer, dense_layer, conv_size, dense_size, droprate = 0.1):
    #   Variable conv layers and variable dense layer   #
    model = Sequential()

    model.add(Conv2D(layer_size, [3,3], activation='relu', padding ='same'))
    model.add(MaxPool2D([2,2], 2, padding='valid'))

    for l in range(conv_layer-1):
        model.add(Conv2D(conv_size, [3,3], activation='relu', padding ='same'))
        model.add(MaxPool2D([2,2], 2, padding='valid'))

    model.add(Flatten())
    model.add(Dropout(rate=droprate))
    for l in range(dense_layer):
        model.add(Dense(dense_size))

    model.add(Dense(1, activation='sigmoid'))
    return model
