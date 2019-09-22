import tensorflow as tf
from tensorflow.keras import backend as K

def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def prec(y_true, y_pred, e_eff = 90, thresh=1e-4):
    e_pred = tf.boolean_mask(y_pred, tf.less(tf.abs(y_true - 1.0), thresh))   #select all positives
    p_pred = tf.boolean_mask(y_pred, tf.less(tf.abs(y_true - 0.0), thresh))   #select all negatives
    argsort = tf.argsort(e_pred)
    cutoff = e_pred[argsort[tf.cast(tf.multiply(tf.cast(tf.shape(argsort)[-1]
        , dtype='float32'), (1 - e_eff / 100)), dtype='int32')]]
    return tf.cast(tf.count_nonzero(e_pred > cutoff) / (tf.count_nonzero(e_pred > cutoff) + tf.count_nonzero(e_pred < cutoff)), dtype='float32')

def pion_con(y_true, y_pred, e_eff = 90, thresh=1e-4):
    e_pred = tf.boolean_mask(y_pred, tf.less(tf.abs(y_true - 1.0), thresh))   #select all positives
    p_pred = tf.boolean_mask(y_pred, tf.less(tf.abs(y_true - 0.0), thresh))   #select all negatives
    argsort = tf.argsort(e_pred)
    cutoff = e_pred[argsort[tf.cast(tf.multiply(tf.cast(tf.shape(argsort)[-1]
        , dtype='float32'), (1 - e_eff / 100)), dtype='int32')]]
    return tf.cast(tf.count_nonzero(p_pred > cutoff) / tf.count_nonzero(tf.equal(y_true, 0)), dtype='float32')

def prec(y_true, y_pred, e_eff = 90, thresh=1e-4):
    e_pred = tf.boolean_mask(y_pred, tf.less(tf.abs(y_true - 1.0), thresh))   #select all positives
    p_pred = tf.boolean_mask(y_pred, tf.less(tf.abs(y_true - 0.0), thresh))   #select all negatives
    argsort = tf.argsort(e_pred)
    cutoff = e_pred[argsort[tf.cast(tf.multiply(tf.cast(tf.shape(argsort)[-1]
        , dtype='float32'), (1 - e_eff / 100)), dtype='int32')]]
    return tf.cast(tf.count_nonzero(e_pred > cutoff) / tf.count_nonzero(y_pred > cutoff), dtype='float32')

def F1(y_true, y_pred, e_eff = 90, thresh=1e-4):
    e_pred = tf.boolean_mask(y_pred, tf.less(tf.abs(y_true - 1.0), thresh))   #select all positives
    p_pred = tf.boolean_mask(y_pred, tf.less(tf.abs(y_true - 0.0), thresh))   #select all negatives
    argsort = tf.argsort(e_pred)
    cutoff = e_pred[argsort[tf.cast(tf.multiply(tf.cast(tf.shape(argsort)[-1]
        , dtype='float32'), (1 - e_eff / 100)), dtype='int32')]]
    TPR = tf.cast(tf.count_nonzero(e_pred > cutoff) / tf.count_nonzero(tf.equal(y_true, 1)), dtype='float32')
    PPV = tf.cast(tf.count_nonzero(e_pred > cutoff) / tf.count_nonzero(y_pred > cutoff), dtype='float32')
    return tf.cast(2*PPV*TPR/(PPV+TPR), dtype='float32')

def WBCE(y_true, y_pred, weight = 1.0, from_logits=False, label_smoothing=0):
    y_pred = tf.cast(y_pred, dtype='float32')
    y_true = tf.cast(y_true, y_pred.dtype)
    return K.mean(weighted_binary_crossentropy(y_true, y_pred, weight=weight, from_logits=from_logits), axis=-1)

def weighted_binary_crossentropy(target, output, weight=1.0, from_logits=False):
    if from_logits:
        output = tf.math.sigmoid(output)
    output = K.clip(output, K.epsilon(), 1.0 - K.epsilon())
    output = -weight * target * K.log(output) - (1.0 - target) * K.log(1.0 - output)
    return output
