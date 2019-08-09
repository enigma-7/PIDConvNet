import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import random

N = 10000
y_true = np.array([random.randint(0,1) for i in range(N)])
positive =  np.array([np.random.normal(0.7,0.3) for i in range(N)]).clip(0,1)
negative =  np.array([np.random.normal(0.2,0.3) for i in range(N)]).clip(0,1)
y_pred = np.array([xv if c else yv for c, xv, yv in zip(y_true==1, positive, negative)])
e_pred = y_pred[y_true.astype(bool)]
p_pred = y_pred[(1-y_true).astype(bool)]
argsort = e_pred.argsort()
cutoff = e_pred[argsort[np.multiply(argsort.shape[-1],(1-90/100)).astype(int)]]
TN = np.sum((p_pred>cutoff).astype(int))
FP = np.sum((p_pred<cutoff).astype(int))
TP = np.sum((e_pred>cutoff).astype(int))
FN = np.sum((e_pred<cutoff).astype(int))
confusion_matrix = [[TP, FN],[FP, TN]]
columns = ("Positive", "Negative")
print('P (TP,FN)\t\t: %.i (%.i,%.i)'%(TP+FN, TP, FN))
print('N (TN,FP)\t\t: %.i (%.i,%.i)'%(FP+TN, TN, FP))
print(confusion_matrix)
# pion_con: x FPR @ 0.9 TPR

fig, axes = plt.subplots(1, 2, figsize=(8,6))

axes[1].hist(positive, alpha=0.5, label = 'positive', color='g')
axes[1].hist(negative, alpha=0.5, label = 'negative', color='r')
axes[1].axvline(x=cutoff, label='Threshold', color='k')
axes[1].legend(loc=9)

axes[0].axis('tight')
axes[0].axis('off')
axes[0].table(cellText=confusion_matrix, colLabels=columns)
plt.show()
