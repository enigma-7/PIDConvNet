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


class PIDnet(object):
    def __init__(self, input_shape=(17, 24, 1)):
        self.height, self.width, self.channels= input_shape
#         First Convolutional layer
        self.W_conv1 = self.weight_variable([3, 3, self.channels, 64])             #64 2x3xchannels tensors for convolution
        self.b_conv1 = self.bias_variable([64])
#         Second convolutional layer
        self.W_conv2 = self.weight_variable([2, 3, 64, 128])                                #128 2x3x64 tensors for convolution
        self.b_conv2 = self.bias_variable([128])
#         Fully connected layer
        self.W_fc1 = self.weight_variable([int(self.width/4)*int(self.height/4)*128 , 1024])             #1024 nodes in 1st ann layer
        self.b_fc1 = self.bias_variable([1024])
#         Last fully connected layer
        self.W_fc2 = self.weight_variable([1024, 1])
        self.b_fc2 = self.bias_variable([1])

    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='VALID')

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)    #initialize to random values from normal distribution
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)             #initialize bias to 0.1
        return tf.Variable(initial)

    def __call__(self, dataset):
        self.x = dataset
        self.h_conv1 = tf.nn.relu(self.conv2d(self.x, self.W_conv1) + self.b_conv1)     #ReLu activation
        self.h_pool1 = self.max_pool_2x2(self.h_conv1)                                  #Max_pooling 2x2
        self.h_conv2 = tf.nn.relu(self.conv2d(self.h_pool1, self.W_conv2) + self.b_conv2)
        self.h_pool2 = self.max_pool_2x2(self.h_conv2)
        self.h_pool2_flat = tf.reshape(self.h_pool2, [-1, int(self.width/4)*int(self.height/4)*128])  #unspecified first dimension
        self.h_fc1 = tf.nn.relu(tf.matmul(self.h_pool2_flat, self.W_fc1) + self.b_fc1)
        self.y_conv = tf.matmul(self.h_fc1, self.W_fc2) + self.b_fc2
        self.y_sig = tf.nn.sigmoid(self.y_conv)
        return self.y_sig


def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def pion_con(y_true, y_pred, e_eff = 90, thresh=1e-4):
    e_pred = tf.boolean_mask(y_pred, tf.less(tf.abs(y_true - 1.0), thresh))   #select all positives
    p_pred = tf.boolean_mask(y_pred, tf.less(tf.abs(y_true - 0.0), thresh))   #select all negatives
    argsort = tf.argsort(e_pred)
    cutoff = e_pred[argsort[tf.cast(tf.multiply(tf.cast(tf.shape(argsort)[-1], dtype='float32'), (1 - e_eff / 100)), dtype='int32')]]
    return tf.cast(tf.count_nonzero(p_pred > cutoff) / tf.count_nonzero(tf.equal(y_true, 0)), dtype='float32')

def prec(y_true, y_pred, e_eff = 90, thresh=1e-4):
    e_pred = tf.boolean_mask(y_pred, tf.less(tf.abs(y_true - 1.0), thresh))   #select all positives
    p_pred = tf.boolean_mask(y_pred, tf.less(tf.abs(y_true - 0.0), thresh))   #select all negatives
    argsort = tf.argsort(e_pred)
    cutoff = e_pred[argsort[tf.cast(tf.multiply(tf.cast(tf.shape(argsort)[-1], dtype='float32'), (1 - e_eff / 100)), dtype='int32')]]
    return tf.cast(tf.count_nonzero(e_pred > cutoff) / tf.count_nonzero(y_pred > cutoff), dtype='float32')

def F1(y_true, y_pred, e_eff = 90, thresh=1e-4):
    e_pred = tf.boolean_mask(y_pred, tf.less(tf.abs(y_true - 1.0), thresh))   #select all positives
    p_pred = tf.boolean_mask(y_pred, tf.less(tf.abs(y_true - 0.0), thresh))   #select all negatives
    argsort = tf.argsort(e_pred)
    cutoff = e_pred[argsort[tf.cast(tf.multiply(tf.cast(tf.shape(argsort)[-1], dtype='float32'), (1 - e_eff / 100)), dtype='int32')]]
    TPR = tf.cast(tf.count_nonzero(e_pred > cutoff) / tf.count_nonzero(tf.equal(y_true, 1)), dtype='float32')
    PPV = tf.cast(tf.count_nonzero(e_pred > cutoff) / tf.count_nonzero(y_pred > cutoff), dtype='float32')
    return tf.cast(2*PPV*TPR/(PPV+TPR), dtype='float32')
