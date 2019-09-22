import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import random
from scipy.interpolate import make_interp_spline, BSpline
import matplotlib
matplotlib.rcParams.update({'font.size': 16})
matplotlib.rcParams['text.usetex'] = True

PosPeak = [0.8, 0.5, 0.3]
NegPeak = [0.2, 0.5, 0.7]
splits = [0.2, 0.5, 0.7]          #Percentage representation of positive class
N = 1000
M = len(PosPeak)
lines = ["-","--",":","-."]
colour = ['lightblue', 'blue', 'purple']
fig, classification = plt.subplots(1, 2, figsize=(16,6))
for j, split in enumerate(splits):
    fig, hists = plt.subplots(1, M, figsize=(16,4))
    for i in range(M):
        PosN = int(N*split)
        NegN = N - PosN
        positive =  np.array([np.random.normal(PosPeak[i],0.25) for j in range(PosN)]).clip(0,1)
        negative =  np.array([np.random.normal(NegPeak[i],0.25) for j in range(NegN)]).clip(0,1)
        b = np.linspace(0,1,40)
        c1, b1 = np.histogram(negative, bins= b)
        c2, b2 = np.histogram(positive, bins = b)
        hists[i].step(b1[1:], c1, color='r', label= "$\\mu_+ =$ " + str(NegPeak[i]))
        hists[i].step(b2[1:], c2, color='g', label= "$\\mu_+ =$ " + str(PosPeak[i]))
        hists[i].legend()
        hists[i].grid()
        # argsort = positive.argsort()
        # cutoff = positive[argsort[np.multiply(argsort.shape[-1],(1-90/100)).astype(int)]]
        # TN = np.sum((negative<cutoff).astype(int))
        # FP = np.sum((negative>=cutoff).astype(int))
        # TP = np.sum((positive>cutoff).astype(int))
        # FN = np.sum((positive<=cutoff).astype(int))
        # confusion_matrix = [[TP, FN],[FP, TN]]
        # columns = ("Positive", "Negative")
        # print('P (TP,FN)\t\t: %.i (%.i,%.i)'%(TP+FN, TP, FN))
        # print('N (TN,FP)\t\t: %.i (%.i,%.i)'%(FP+TN, TN, FP))
        #print(confusion_matrix)
        # pion_con: x FPR @ 0.9 TPR
        #hists[i].axvline(x=cutoff, label='Threshold', color='k')

        thresholds=np.linspace(0,1,10)
        TP = np.array([positive[positive>threshold].sum() for threshold in thresholds])
        FN = np.array([positive[positive<threshold].sum() for threshold in thresholds])
        FP = np.array([negative[negative>threshold].sum() for threshold in thresholds])
        TN = np.array([negative[negative<threshold].sum() for threshold in thresholds])

        TPR = TP/(TP+FN)
        FPR = FP/(FP+TN)
        PPV = TP/(TP+FP)
        for k, val in enumerate(PPV):
            if np.isnan(val):
                PPV[k] = PPV[k-1]
            else:
                continue

        # newFPR = np.linspace(FPR.min(),FPR.max(),300)
        # spl = make_interp_spline(FPR, TPR, k=3)
        # smoothTPR = spl(newFPR)

        classification[0].plot(FPR, TPR, color = colour[j], linestyle = lines[i], label = (str(NegPeak[i]),str(PosPeak[i])))
        classification[1].plot(TPR, PPV, color = colour[j], linestyle = lines[i], label = (str(NegPeak[i]),str(PosPeak[i])))
    hists[0].set_ylabel("Counts")
    hists[1].set_title("POE: %.1f"%split)
    for i in range(2):
        classification[i].grid()
    classification[0].set_ylabel("TPR")
    classification[0].set_xlabel("FPR")
    classification[1].set_ylabel("PPV")
    classification[1].set_xlabel("TPR")
    classification[1].legend(bbox_to_anchor=(1.04, 1), loc='upper left', ncol=1)


#
# PosPeak = [0.5, 0.7, 0.9]
# NegPeak = [0.5, 0.3, 0.1]
# M = len(PosPeak)
# fig, hists = plt.subplots(1, M, figsize=(16,4))
# fig, classification = plt.subplots(1, 2, figsize=(16,4))
# for i in range(M):
#     NegN = 10000
#     PosN = 1000
#     positive =  np.array([np.random.normal(PosPeak[i],0.25) for j in range(PosN)]).clip(0,1)
#     negative =  np.array([np.random.normal(NegPeak[i],0.25) for j in range(NegN)]).clip(0,1)
#     argsort = positive.argsort()
#     cutoff = positive[argsort[np.multiply(argsort.shape[-1],(1-90/100)).astype(int)]]
#     TN = np.sum((negative<cutoff).astype(int))
#     FP = np.sum((negative>=cutoff).astype(int))
#     TP = np.sum((positive>cutoff).astype(int))
#     FN = np.sum((positive<=cutoff).astype(int))
#     confusion_matrix = [[TP, FN],[FP, TN]]
#     # columns = ("Positive", "Negative")
#     # print('P (TP,FN)\t\t: %.i (%.i,%.i)'%(TP+FN, TP, FN))
#     # print('N (TN,FP)\t\t: %.i (%.i,%.i)'%(FP+TN, TN, FP))
#     print(confusion_matrix)
#     # pion_con: x FPR @ 0.9 TPR
#     c, b, p = hists[i].hist(negative, alpha=0.5, label = '-', edgecolor = 'black', color='r', bins= 15)
#     hists[i].hist(positive, alpha=0.5, label = '+', edgecolor = 'black', color='g', bins = b)
#     hists[i].axvline(x=cutoff, label='Threshold', color='k')
#
#     thresholds=np.linspace(0,1,1000)
#     TP = np.array([positive[positive>threshold].sum() for threshold in thresholds])
#     FN = np.array([positive[positive<threshold].sum() for threshold in thresholds])
#     FP = np.array([negative[negative>threshold].sum() for threshold in thresholds])
#     TN = np.array([negative[negative<threshold].sum() for threshold in thresholds])
#
#     TPR = TP/(TP+FN)
#     FPR = FP/(FP+TN)
#     PPV = TP/(TP+FP)
#
#     classification[0].plot(FPR, TPR)
#     classification[0].grid()
#
#     classification[1].plot(TPR, PPV)
#     classification[1].grid()
# hists[M-2].legend(loc=9)
# hists[0].set_ylabel("Counts")
