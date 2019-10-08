import tensorflow as tf
import numpy as np

class SuperModel(tf.keras.Model):
	def save(self, path):
		self.save_weights(path, save_format='tf')

	def load(self, path):
		self.load_weights(path)

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
		return self.ann_model(tf.transpose(tf.map_fn(self.tracklet_model, tf.transpose(tracks, (1, 0, 2, 3, 4))), (1, 0, 2))[:,:,0])

class SimpleSingleTrackletConvPID(SuperModel):
	def __init__(self, num_filters, kernel_size=(17, 24), use_bias=False):
		super(SimpleSingleTrackletConvPID, self).__init__()
		self.conv_model = tf.keras.Sequential([
			tf.keras.layers.Conv2D(filters=num_filters, kernel_size=kernel_size, strides=1, use_bias=use_bias,
								   bias_initializer='normal',
								   activation='relu', padding='valid',),
			tf.keras.layers.Flatten(),
		])

		self.ann_model = tf.keras.Sequential([
			tf.keras.layers.Dense(12, activation=tf.nn.sigmoid, use_bias=True),
			# tf.keras.layers.Dropout(rate=0.6),
			tf.keras.layers.Dense(1, activation=tf.nn.sigmoid, use_bias=True),
		])

	def get_conv_units(self):
		return self.conv_model.layers[0].trainable_weights[0]

	def call(self, tracklets):
		return self.ann_model(self.conv_model(tracklets))

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

	def get_conv_units(self):
		return self.conv_model.layers[0].trainable_weights[0]

	def call(self, tracklets):
		return self.ann_model(self.conv_model(tracklets))

class VeryComplexConvTrackletPID(SuperModel):
	def __init__(self):
		super(VeryComplexConvTrackletPID, self).__init__()
		self.conv_model = tf.keras.Sequential([
			# tf.keras.layers.GaussianNoise(0.1),
			tf.keras.layers.Conv2D(filters=15, kernel_size=(4, 4), strides=1, use_bias=True,
								   bias_initializer='normal',
								   activation='relu', padding='same', ),
			tf.keras.layers.MaxPool2D(2),
			tf.keras.layers.Conv2D(filters=15, kernel_size=(2, 2), strides=1, use_bias=True,
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
			tf.keras.layers.Dense(128, activation=tf.nn.relu, use_bias=True),
			tf.keras.layers.Dropout(rate=0.5),
			tf.keras.layers.Dense(128, activation=tf.nn.relu, use_bias=True),
			tf.keras.layers.Dropout(rate=0.5),
			tf.keras.layers.Dense(1, activation=tf.nn.sigmoid, use_bias=False),
		])

	def get_conv_units(self):
		return self.conv_model.layers[0].trainable_weights[0]

	def call(self, tracklets):
		return self.ann_model(self.conv_model(tracklets))

class SeededComplexConvTrackletPID(SuperModel):
	def __init__(self):
		super(SeededComplexConvTrackletPID, self).__init__()
		self.conv_model = tf.keras.Sequential([
			# tf.keras.layers.GaussianNoise(0.1),
			tf.keras.layers.Conv2D(filters=1, kernel_size=(5, 15), strides=1, use_bias=True,
								   bias_initializer='normal',
								   activation='linear', padding='same',
								   kernel_initializer=tf.constant_initializer(np.expand_dims(np.expand_dims(np.load('py_datatools/deconvolution/inverse_impulse_response.npy'), axis=0), axis=0))),
			tf.keras.layers.Conv2D(filters=15, kernel_size=(2, 2), strides=1, use_bias=True,
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

	def get_conv_units(self):
		return self.conv_model.layers[0].trainable_weights[0]

	def call(self, tracklets):
		return self.ann_model(self.conv_model(tracklets))

class FullTrackletConvPID(SuperModel):
	def __init__(self):
		super(FullTrackletConvPID, self).__init__()
		self.conv_model = tf.keras.Sequential([
			tf.keras.layers.Conv2D(filters=2, kernel_size=(17, 24), strides=[17,24], use_bias=False,
								   bias_initializer='normal',
								   activation='sigmoid', padding='valid', ),
			tf.keras.layers.Flatten(),
		])

	def call(self, tracklets):
		return self.conv_model(tracklets)

class PartialTrackletConvPID(SuperModel):
	def __init__(self):
		super(PartialTrackletConvPID, self).__init__()
		self.conv_model = tf.keras.Sequential([
			tf.keras.layers.Conv2D(filters=10, kernel_size=(17, 12), strides=[1,1], use_bias=False,
								   bias_initializer='normal',
								   activation='relu', padding='valid', ),
			tf.keras.layers.Flatten(),
		])

		self.ann_model = tf.keras.Sequential([
			tf.keras.layers.Dense(128, activation=tf.nn.relu, use_bias=True),
			tf.keras.layers.Dropout(rate=0.4),
			tf.keras.layers.Dense(1, activation=tf.nn.sigmoid, use_bias=True),
		])

	def call(self, tracklets):
		return self.ann_model(self.conv_model(tracklets))

class ComplexConvTrackletMomentumModel(SuperModel):
	def __init__(self):
		super(ComplexConvTrackletMomentumModel, self).__init__()
		self.conv_model = tf.keras.Sequential([
			# tf.keras.layers.GaussianNoise(0.1),
			tf.keras.layers.Conv2D(filters=15, kernel_size=(2, 2), strides=1, use_bias=True,
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
			tf.keras.layers.Dropout(rate=0),
			tf.keras.layers.Dense(128, activation=tf.nn.relu, use_bias=True),
			tf.keras.layers.Dropout(rate=0),
			tf.keras.layers.Dense(1, activation='relu', use_bias=True),
		])

	def get_conv_units(self):
		return self.conv_model.layers[0].trainable_weights[0]

	def call(self, tracklets):
		return self.ann_model(self.conv_model(tracklets))

class TrackletMomentumModelMultiplexer(SuperModel):
	def __init__(self, tracklet_model):
		super(TrackletMomentumModelMultiplexer, self).__init__()
		self.tracklet_model = tracklet_model

		self.ann_model = tf.keras.Sequential([
			tf.keras.layers.Dense(1024, activation=tf.nn.relu, use_bias=True, input_shape=(6,)),
			tf.keras.layers.Dropout(rate=0.2),
			tf.keras.layers.Dense(128, activation=tf.nn.relu, use_bias=True),
			tf.keras.layers.Dropout(rate=0.2),
			tf.keras.layers.Dense(128, activation=tf.nn.relu, use_bias=True),
			tf.keras.layers.Dropout(rate=0.2),
			tf.keras.layers.Dense(1, activation='relu', use_bias=True),
		])

	def call(self, tracks):
		return self.ann_model(tf.transpose(tf.map_fn(self.tracklet_model, tf.transpose(tracks, (1, 0, 2, 3, 4))), (1, 0, 2))[:,:,0])

class ClusterModel(SuperModel):
	def __init__(self, num_clusters):
		super(ClusterModel, self).__init__()

		self.ann_model = tf.keras.Sequential([
			tf.keras.layers.Dense(128, activation=tf.nn.relu, use_bias=True),
			tf.keras.layers.Dropout(rate=0.2),
			tf.keras.layers.Dense(1, activation='sigmoid', use_bias=True),
		])

	def call(self, clusters):
		return self.ann_model(clusters)