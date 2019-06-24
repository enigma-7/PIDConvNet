

def pion_con(y_true, y_pred, thresh=1e-4):
    e_pred = tf.boolean_mask(y_pred, tf.less(tf.abs(y_true - 1.0), thresh))   #select all positives
    p_pred = tf.boolean_mask(y_pred, tf.less(tf.abs(y_true - 0.0), thresh))   #select all negatives
    argsort = tf.argsort(e_pred)
    cutoff = e_pred[argsort[tf.cast(tf.multiply(tf.cast(tf.shape(argsort)[-1], dtype='float32'), (1 - e_eff / 100)), dtype='int32')]]
    return 100.0 * tf.cast(tf.count_nonzero(p_pred > e_eff_90_cutoff) / tf.count_nonzero(tf.equal(y_true, 0)), dtype='float32')

def PionEfficiencyAtElectronEfficiency(e_eff, thresh = 1e-4):
    def PionEfficiencyAtElectronEfficiency(y_true, y_pred):
        e_pred = tf.boolean_mask(y_pred, tf.less(tf.abs(y_true - 1.0), thresh))   #select all true positives
        p_pred = tf.boolean_mask(y_pred, tf.less(tf.abs(y_true - 0.0), thresh))   #select all true negatives
        argsort = tf.argsort(e_pred)                                              #returns indices of sorted array
        e_eff_90_cutoff = e_pred[argsort[tf.cast(tf.multiply(tf.cast(tf.shape(argsort)[-1],
        dtype='float32'), (1 - e_eff / 100)), dtype='int32')]]
        return 100.0 * tf.cast(tf.count_nonzero(p_pred > e_eff_90_cutoff) / tf.count_nonzero(tf.equal(y_true, 0)), dtype='float32')
    return PionEfficiencyAtElectronEfficiency

tracklet = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],
[10, 20, 14, 14, 32, 34, 25, 20, 28, 22, 16, 15, 13, 13, 16, 15, 15, 11, 12, 12, 11, 13, 10, 11 ],
[14, 36, 32, 22, 124, 171, 128, 86, 169, 131, 88, 61, 45, 52, 54, 51, 39, 32, 30, 28, 26, 22, 25, 20 ],
[9, 11, 10, 10, 42, 51, 46, 35, 84, 67, 49, 33, 26, 40, 51, 52, 44, 31, 27, 24, 24, 26, 28, 24 ],
[10, 11, 8, 9, 12, 12, 12, 11, 17, 14, 12, 11, 10, 13, 15, 15, 14, 10, 13, 9, 11, 12, 13, 12 ],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]])
emptyarr = np.array([[],[],[],[],[],[],[],[],[],[]])
zerosarr = np.zeros((17,24))
print('emptyarr:',emptyarr.any(), emptyarr.all())
print('tracklet:',not tracklet.any(), tracklet.all())
print('zerosarr:',not zerosarr.any(), zerosarr.all())

def pion_con(y_true, y_pred, thresholds = np.arange(0,1,0.01)):
    recall = np.zeros(thresholds.shape)
    for i, thresh in enumerate(thresholds):
        TP = np.dot(abs(np.ceil(y_pred - thresh)),y_true)
        FN = np.dot(abs(np.ceil(thresh - y_pred)),y_true)
        recall[i] = TP/(TP+FN+K.epsilon())
    thresh_e90 = thresholds[recall<0.91][0]
    plt.axvline(x=thresh_e90)
    plt.show()
    print('Threshold selected\t: %.2f'%thresh_e90)
    TP = np.dot(abs(np.ceil(y_pred - thresh_e90)),y_true)
    FP = np.dot(abs(np.ceil(y_pred - thresh_e90)),(1-y_true))
    FN = np.dot(abs(np.ceil(thresh_e90 - y_pred)),y_true)
    TN = np.dot(abs(np.ceil(thresh_e90 - y_pred)),(1-y_true))
    tot_P = TP + FN
    tot_N = FP + TN
    TPR = TP/(tot_P+K.epsilon())
    FPR = FP/(tot_N+K.epsilon())
    print('Electron efficiency\t: %.2f'%(TPR))
    print('Pion contamination\t: %.2f'%(FPR))
    print('P (TP,FN)\t\t: %.i (%.i,%.i)'%(tot_P, TP, FN))
    print('N (TN,FP)\t\t: %.i (%.i,%.i)'%(tot_N, TN, FP))

class SuperModel(tf.keras.Model):
	def save(self, path):
		self.save_weights(path, save_format='tf')

	def load(self, path):
		self.load_weights(path)

class TrackletModelMultiplexer(SuperModel):
	def _init_(self, tracklet_model):
		super(TrackletModelMultiplexer, self)._init_()
		self.tracklet_model = tracklet_model

		self.ann_model = tf.keras.Sequential([
			tf.keras.layers.Dense(1024, activation=tf.nn.relu, use_bias=True, input_shape=(6,)),
			tf.keras.layers.Dropout(rate=0.5),
			tf.keras.layers.Dense(1, activation=tf.nn.sigmoid, use_bias=True),
		])

	def call(self, tracks):
		return self.ann_model(tf.transpose(tf.map_fn(self.tracklet_model, tf.transpose(tracks, (1, 0, 2, 3, 4))), (1, 0, 2))[:,:,0])
