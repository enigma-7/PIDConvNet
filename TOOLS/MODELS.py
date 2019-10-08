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
  Conv2D(8, [2,3], activation='relu', padding ='same'),
  MaxPool2D([2,2], 2, padding='valid'),
  Conv2D(16, [2,3], activation='relu', padding='same'),
  MaxPool2D([2,2], 2, padding='valid'),
  Flatten(),
  Dense(128),
  Dense(64),
  Dense(1, activation='sigmoid')
])

def blank_2_2_(conv_size1, conv_size2, dense_size1, dense_size2, droprate = 0.1):
    #   Two conv layers and two dense layers    #
    model = Sequential([
        Conv2D(conv_size1, [2,3], activation='relu', padding ='same'),
        MaxPool2D([2,2], 2, padding='valid'),
        Conv2D(conv_size2, [2,3], activation='relu', padding ='same'),
        MaxPool2D([2,2], 2, padding='valid'),
        Flatten(),
        Dropout(rate=droprate),
        Dense(dense_size1),
        Dense(dense_size2),
        Dense(1, activation='sigmoid')])
    return model

def blank_2_1_(conv_size1, conv_size2, dense_size1, droprate = 0.1):
    model = Sequential([
        Conv2D(conv_size1, [2,3], activation='relu', padding ='same'),
        MaxPool2D([2,2], 2, padding='valid'),
        Conv2D(conv_size2, [2,3], activation='relu', padding ='same'),
        MaxPool2D([2,2], 2, padding='valid'),
        Flatten(),
        Dropout(rate=droprate),
        Dense(dense_size1),
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

class SuperModel(tf.keras.Model):
	def save(self, path):
		self.save_weights(path, save_format='tf')

	def load(self, path):
		self.load_weights(path)

class ComplexConvTrackletPID(SuperModel):
	def __init__(self):
		super(ComplexConvTrackletPID, self).__init__()
		self.conv_model = tf.keras.Sequential([
			# tf.keras.layers.GaussianNoise(0.1),
			tf.keras.layers.Conv2D(filters=8, kernel_size=(3, 4), strides=1, use_bias=True,
								   bias_initializer='normal',
								   activation='relu', padding='same', ),
			tf.keras.layers.MaxPool2D(2),
			tf.keras.layers.Conv2D(filters=16, kernel_size=(2, 2), strides=1, use_bias=True,
								   bias_initializer='normal',
								   activation='relu', padding='same', ),
			tf.keras.layers.MaxPool2D(2),
			tf.keras.layers.Flatten(),
		])

		self.ann_model = tf.keras.Sequential([
			tf.keras.layers.Dense(128, activation=tf.nn.relu, use_bias=True),
            tf.keras.layers.Dropout(rate=0.5),
			tf.keras.layers.Dense(1, activation=tf.nn.sigmoid, use_bias=True),
		])

	def get_conv_units(self):
		return self.conv_model.layers[0].trainable_weights[0]

	def call(self, tracklets):
		return self.ann_model(self.conv_model(tracklets))

class TrackletModelMultiplexer(SuperModel):
	def __init__(self, tracklet_model):
		super(TrackletModelMultiplexer, self).__init__()
		self.tracklet_model = tracklet_model

		self.ann_model = tf.keras.Sequential([
			tf.keras.layers.Dense(1024, activation=tf.nn.relu, use_bias=True, input_shape=(6,)),
			tf.keras.layers.Dropout(rate=0.5),
			tf.keras.layers.Dense(1, activation=tf.nn.sigmoid, use_bias=True),
		])

	def call(self, tracks):
		return self.ann_model(tf.transpose(tf.map_fn(self.tracklet_model, tf.transpose(tracks, (1, 0, 2, 3, 4)) ), (1, 0, 2)) [:,:,0])

class ComplexConvTrackletPID(SuperModel):
	def __init__(self):
		super(ComplexConvTrackletPID, self).__init__()
		self.conv_model = tf.keras.Sequential([
			# tf.keras.layers.GaussianNoise(0.1),
			tf.keras.layers.Conv2D(filters=5, kernel_size=(3, 4), strides=1, use_bias=True,
								   bias_initializer='normal',
								   activation='relu', padding='same', ),
			tf.keras.layers.MaxPool2D(2),
			tf.keras.layers.Conv2D(filters=15, kernel_size=(2, 2), strides=1, use_bias=True,
								   bias_initializer='normal',
								   activation='relu', padding='same', ),
			tf.keras.layers.MaxPool2D(2),
			tf.keras.layers.Flatten(),
		])

		self.ann_model = tf.keras.Sequential([
			tf.keras.layers.Dense(128, activation=tf.nn.relu, use_bias=True),
			tf.keras.layers.Dropout(rate=0.5),
			tf.keras.layers.Dense(1, activation=tf.nn.sigmoid, use_bias=True),
		])
