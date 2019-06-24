import numpy as np
import matplotlib.pyplot as plt
import DATA, MODELS, random, matplotlib, PLOT
matplotlib.rcParams.update({'font.size': 16})
matplotlib.rcParams['text.usetex'] = True

run_no = '000265378/'
dataname = 'all/'
directory = 'data/output/' + run_no + dataname
raw_data = np.load(directory + '0_tracks.npy')
raw_info = np.load(directory + '0_info_set.npy')
print("Loaded: %s" % directory)

dataset, infoset = DATA.process_1(raw_data, raw_info)

elec_data, elec_info = DATA.elec_strip_(dataset, infoset)
pion_data, pion_info = DATA.pion_strip_(dataset, infoset)

class_data = [pion_data, elec_data]
class_info = [pion_info, elec_info]
cnames = ["$\\pi$","$e^-$"]
colour = ['r', 'g']

fig, axes = plt.subplots(1, 3, figsize=(17,5))
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
axes[1].set_yscale('log')
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

"""
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
"""
