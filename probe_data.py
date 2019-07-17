import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import random, matplotlib
from TOOLS import DATA
matplotlib.rcParams.update({'font.size': 16})
matplotlib.rcParams['text.usetex'] = True

run_no = '000265378/'
dataname = 'all/'
directory = 'data/output/' + run_no + dataname
raw_data = np.load(directory + '0_tracks.npy')
raw_info = np.load(directory + '0_info_set.npy')
print("Loaded: %s" % directory)

plotdir = '/home/jibran/Desktop/neuralnet/plots/'
dataset, infoset = DATA.process_1(raw_data, raw_info)
columns = ["label", "nsigmae", "nsigmap", "PT", "${dE}/{dx}$", "Momenta [GeV]", "eta", "theta", "phi", "event", "V0trackID",  "track"]
infoframe = pd.DataFrame(data=infoset, columns=columns)
infoframe.head()
#infoframe["ADCsum"] = dataset.sum(axis=(1,2,3))
#infoframe["label"] = ["$\\pi$" if l == 0 else "$e^-$" for l in infoset[:,0]]
#g = sns.pairplot(data=infoframe[["label", "dEdX", "P", "ADCsum"]], hue="label", diag_kind='hist', palette={"$\\pi$":'r',"$e^-$":'g'})
#g.map_diag(plt.hist, log=True, edgecolor='black')


elec_data, elec_info = DATA.elec_strip_(dataset, infoset)
pion_data, pion_info = DATA.pion_strip_(dataset, infoset)

class_data = [pion_data, elec_data]
class_info = [pion_info, elec_info]
cnames = ["$\\pi$","$e$"]
colour = ['r', 'g']

fig, axes = plt.subplots(1, 3, figsize=(15,5))
for j,n in enumerate([4,5]):
    for i in range(2):
        x = class_info[i][:,n]
        if i>0:
            axes[j].hist(x, edgecolor='black', color = colour[i],
                bins=bins[bins< np.ceil(x.max())], alpha=0.6, label=cnames[i])
            #axes[0].set_xticks([round(j,2) for j in bins[bins< np.ceil(x.max())]][::1])
        else:
            counts, bins, patches = axes[j].hist(x, edgecolor='black',
                color = colour[i], bins=14, alpha=0.6, label=cnames[i])
        y = x[(x>bins[0]) & (x<bins[1])]
        axes[j].set_xticks([round(j,2) for j in bins][::3])
        axes[j].set_xlabel(columns[n])
        axes[j].set_ylabel("Counts")
        axes[j].set_yscale('log')
        axes[j].legend()


fig, axes = plt.subplots(1, 3, figsize=(15,5))
for i in range(2):
    x = class_info[i][:,5]
    if i>0:
        axes[0].hist(x, edgecolor='black', color = colour[i],
            bins=bins[bins< np.ceil(x.max())], alpha=0.6, label=cnames[i])
        #axes[0].set_xticks([round(j,2) for j in bins[bins< np.ceil(x.max())]][::1])
    else:
        counts, bins, patches = axes[0].hist(x, edgecolor='black',
            color = colour[i], bins=14, alpha=0.6, label=cnames[i])
    y = x[(x>bins[0]) & (x<bins[1])]
    c0, b0, p0 = axes[1].hist(y, edgecolor='black', color = colour[i], alpha=0.6, label=cnames[i])

axes[0].set_xticks([round(j,2) for j in bins][::2])
axes[0].set_xlabel("Momenta [GeV]")
axes[0].set_ylabel("Counts")
axes[0].set_yscale('log')
axes[0].legend()

axes[1].set_xticks([round(i,2) for i in b0][::2])
axes[1].set_xlabel("Momenta [GeV]")
axes[1].set_ylabel("Counts")
#axes[1].set_yscale('log')
axes[1].legend()

elec_momenta = elec_info[:,5]
pion_momenta = pion_info[:,5]
counts_p, bins_p = np.histogram(pion_momenta[pion_momenta<np.ceil(elec_momenta.max())], bins=35)
counts_e, bins_e = np.histogram(elec_momenta, bins=bins_p)
e_occ = 100*counts_e / (counts_e + counts_p)

axes[2].bar(bins_e[:-1], e_occ, width=(bins_e[1:]-bins_e[:-1]).mean()*0.8)
axes[2].set_ylabel("$e^-$ occurence [$\%$]")
axes[2].set_xlabel("Momenta [GeV]")
axes[2].grid()
axes[2].set_xticks([round(i,1) for i in bins_e[bins_e < round(elec_momenta.max(),1)]][::5])

plt.savefig(plotdir + 'momenta_hist.png')
plt.show()

fig, axes = plt.subplots(1, 3, figsize=(17,5))
for i,c in enumerate(class_data):
    Z = np.sum(c, axis=(0,3))/c.shape[0]
    X, Y = np.meshgrid(range(24),np.array(range(17)))
    axes[i].contourf(X, Y, Z)
    axes[i].set_title(cnames[i])
    axes[i].set_xlabel("Time bin no.")
    axes[i].set_ylabel("Pad no.")
    axes[i].grid()
    axes[i].set_xticks(np.arange(0, 24, 4))
    axes[i].set_yticks(np.arange(0, 18, 2))
    axes[2].plot(range(1,25), Z.sum(axis=0)/Z.shape[0], label = cnames[i], color = colour[i])
axes[2].legend()
axes[2].set_xticks(np.arange(1, 24, 2))
axes[2].set_xlabel("Time bin no.")
axes[2].set_ylabel("Mean ADC value per pad")
axes[2].grid()

#fig, axes = plt.subplots(1, 2, figsize=(12,5))
plt.figure(figsize=(8,6))
ADCsum = [x.sum(axis=(1,2,3)) for x in class_data]
plt.boxplot(ADCsum, labels=cnames)
plt.grid()
plt.yscale('log')
plt.ylabel("ADC sum")
plt.title("")
plt.show()
class_data[1][(ADCsum[1]<10.0)].shape

fig, axes = plt.subplots(1, 2, figsize=(12,5))
for i,c in enumerate(class_data):
    Z = np.sum(c, axis=(0,3))/c.shape[0]
    im = axes[i].imshow(Z)
    axes[i].set_title(cnames[i])
    axes[i].set_xlabel("Time bin no.")
    axes[i].set_ylabel("Pad no.")

    axes[i].set_xticks(np.arange(0, 24, 4))
    axes[i].set_yticks(np.arange(0, 18, 2))

    #fig.subplots_adjust(right=0.8)
    #cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
    #fig.colorbar(im)#, cax=cbar_a)

#fig.title('Mean tracklet per class')
