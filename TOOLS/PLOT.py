import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random, matplotlib
matplotlib.rcParams.update({'font.size': 18})
matplotlib.rcParams['text.usetex'] = True

def surface_(classes, cnames=['$e^-$', '$\\pi$'], colour = ['g','r']):
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    for i,c in enumerate(classes):
        c_track = np.sum(c, axis=0)/len(c)
        X, Y = np.meshgrid(range(24),range(17))
        Z = c_track
        p = ax.plot_surface(X, Y, Z, alpha=0.6)
        p = ax.plot_wireframe(X, Y, Z, label=cnames[i],color = colour[i], alpha = 0.6)
    plt.legend()
    plt.title('Mean tracklet per class')
    #plt.ylabel('Time bin no.')
    #plt.xlabel('Pad no.')
    plt.xticks(np.arange(0, 24, 4))
    plt.yticks(np.arange(0, 17, 3))
    plt.show()

def timespec_(tensor, labels):
    elec = tensor[labels.astype(bool)]
    pion = tensor[(1-labels).astype(bool)]
    classes = [elec,pion]
    cnames = ['$e^-$', '$\\pi$']
    colour = ['g','r']
    fig = plt.figure(figsize=(8,6))
    for i,c in enumerate(classes):
        c_tspec = np.sum(c, axis=(0,1))/c.shape[0]
        plt.plot(range(24), c_tspec ,label=cnames[i])
    plt.legend()
    plt.title('Normalized time spectra')
    plt.ylabel('Time bin no.')
    plt.xlabel('Pad no.')
    plt.grid()
    plt.xticks(np.arange(0, 24, 2))
    plt.show()

def hist_(tensor, labels):
    elec = tensor[labels.astype(bool)]
    pion = tensor[(1-labels).astype(bool)]
    classes = [elec,pion]
    cnames = ['$e^-$', '$\\pi$']
    colour = ['g','r']
    fig, axes = plt.subplots(1, 2, figsize=(12,5))
    classes = [elec,pion]
    for i, cl in enumerate(classes):
        flat_cl = cl.flatten()
        nonzero = flat_cl[flat_cl>0]
        counts, bins, patches = axes[i].hist(flat_cl, edgecolor='black')
        axes[i].set_yscale('log')
        axes[i].set_xticks(bins)
        #axes[i].grid()
        axes[i].set_title("ADC histogram with zeros")

def iter_(data, nplots = 6):
    i = random.randint(1,data.shape[0])
    print("Initial tracklet %i:"%i)
    leny = data.shape[1]
    lenx = data.shape[2]
    X, Y = np.meshgrid(range(lenx),range(leny))
    for j in range(nplots):
        Z = data[i+j]
        fig, ax = plt.subplots(figsize=(8,6))
        p = ax.pcolor(X, Y, Z)
        cb = fig.colorbar(p, ax=ax)
        plt.show()

def single_(dataset, infoset, save=False, plotdir = '/home/jibran/Desktop/neuralnet/plots/'):
    num = random.randint(0,dataset.shape[0]-1)
    j = random.randint(0,dataset.shape[-1]-1)
    tracklet = dataset[num,:,:,j]
    trackids = infoset[num][-3:].astype(int)
    title = "Displaying tracklet %s:"%"/".join([str(i) for i in trackids])
    plt.imshow(tracklet)
    if infoset[num,0] == 1:
        title = "$e$ -- " + title
    else:
        title = "$\\pi$ -- " + title
    plt.title(title)
    plt.xticks(np.arange(0, 24, 4))
    plt.yticks(np.arange(0, 18, 2))
    if save:
        plt.savefig(plotdir)
    print(title, num)

def tileplot_(array, title=None):
    divlist = []
    for x in range(1,array.shape[-1]+1):
        if array.shape[-1] % x == 0:
            divlist.append(x)

    n = divlist[int(len(divlist)/2)]
    m = int(array.shape[-1]/n)
    fig, axes = plt.subplots(n,m, figsize=(16,16))
    k = 0

    for i in range(n):
        for j in range(m):
            axes[i,j].imshow(array[:,:,0,k])
            axes[i,j].axis('off')
            k +=1

    #fig.colorbar(axes[0,0], ax=axes, orientation='horizontal')
    plt.title(title)
    plt.show()

def classification_(predict, targets, filename='str', save = False, thresholds=np.linspace(0,1,1000),
    cnames = ["$\\pi$","$e$"], colour = ['r', 'g'], styles = ['--','-.'], scale='log'):
    e_pred = predict[targets==1]
    p_pred = predict[targets==0]
    argsort = e_pred.argsort()

    TP = np.array([e_pred[e_pred>threshold].sum() for threshold in thresholds])
    FN = np.array([e_pred[e_pred<threshold].sum() for threshold in thresholds])
    FP = np.array([p_pred[p_pred>threshold].sum() for threshold in thresholds])
    TN = np.array([p_pred[p_pred<threshold].sum() for threshold in thresholds])

    TPR = TP/(FN+TP)            #True Positive Rate / Recall
    FPR = FP/(TN+FP)            #False Positive Rate
    PPV = TP/(TP+FP)            #Positive Predictive Value / Precision
    pioncon = FPR[TPR>0.9][-1]          #estimate pion contamination
    decbound = thresholds[TPR<0.905][0]
    AUC = np.sum(np.abs(np.diff(FPR)*TPR[1:]))          #estimate area under curve

    fig, axes = plt.subplots(1, 3, figsize=(15,5))
    axes[0].plot(FPR,TPR)
    axes[0].vlines(pioncon, 0, 0.9, 'k', '--')
    axes[0].hlines(0.9, 0, pioncon, 'k', '--')
    axes[0].set_ylabel("$e$-efficiency")
    axes[0].set_xlabel("$\\pi$-contamination")
    axes[0].text(pioncon+0.05, 0.4, "$\\varepsilon_\\pi$ = "+ str(np.round(pioncon, 3)), fontsize=18)
    axes[0].text(pioncon+0.05, 0.2, "AUC = "+ str(np.round(AUC, 2)), fontsize=18)
    axes[0].grid()

    axes[1].plot(TPR,PPV)
    axes[1].set_xlabel("$e$-efficiency")
    axes[1].set_ylabel("Precision")
    axes[1].grid()

    c1, b1 = np.histogram(p_pred, bins=40)
    c2, b2 = np.histogram(e_pred, bins=b1)

    axes[2].step(b1[:-1], c1, label=cnames[0], color=colour[0])
    axes[2].step(b2[:-1], c2, label=cnames[1], color=colour[1])
    axes[2].set_yscale(scale)
    axes[2].vlines(decbound, 0, max(c), 'k', label="Decision Boundary")
    axes[2].legend()
    axes[2].grid()
    if save:
        plt.savefig(filename)
    plt.show()
