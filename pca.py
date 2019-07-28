import numpy as np
import matplotlib.pyplot as plt
from TOOLS import DATA, MODELS, LOG, METRICS
import random, matplotlib, datetime, os
from sklearn.decomposition import PCA
from sklearn import preprocessing
import pandas as pd

run_no = '000265378/'
dataname = 'even/'
directory = 'data/output/' + run_no + dataname
raw_data = np.load(directory + '0_tracks.npy')
raw_info = np.load(directory + '0_info_set.npy')
print("Loaded: %s" % directory)

raw_data, raw_info = DATA.shuffle_(raw_data, raw_info)
columns = ["label", "nsigmae", "nsigmap", "PT", "${dE}/{dx}$", "Momenta [GeV]", "eta", "theta", "phi", "event", "V0trackID",  "track"]

cnames = ["$\\pi$","$e$"]
colour = ['r', 'g']
styles = ['--','-.']

T = raw_info[:,0]
nx = 4
ny = 6

columns[nx:ny+1]
params = raw_info[:,nx:ny+1]
parammean = params.sum(axis=0)/params.shape[0]
paramstdv = np.sqrt(np.square((params - parammean)).sum(axis=0)/(params.shape[0]-1))
I = (params - parammean)
Pcovariance = np.matmul(I , I.T)/I.shape[0]

Pdf = pd.DataFrame(params,columns=columns[nx:ny+1])
Pdf.mean()
Pdf.std()
# s = 80
# n = 5
# raw_info[s:s+n,0]
# print(np.around(Pcovariance[s:s+n,s:s+n],2))

scaler = preprocessing.StandardScaler()
scaled_Pdf =  scaler.fit_transform(I)
scaled_Pdf = pd.DataFrame(scaled_Pdf, columns=columns[nx:ny+1])
scaled_Pdf.head(20)

pca = PCA()
pca.fit(scaled_Pdf)
pca_data = pca.transform(scaled_Pdf)

per_var = np.round(pca.explained_variance_ratio_*100,decimals=1)        #percentage variation that each PC accounts for
labels = ["PC" +str(x) for x in range(1,len(per_var)+1)]
plt.figure(figsize=(8,6))
plt.bar(x = range(1,len(per_var)+1), height = per_var, tick_label = labels)
plt.xlabel("Principal Components")
plt.ylabel("Percentage of Explained Variance")
plt.title("Scree Plot")
plt.grid()
plt.show()


from mpl_toolkits.mplot3d import axes3d, Axes3D

T = raw_info[:,0]
pca_df = pd.DataFrame(pca_data, columns=labels)
fig = plt.figure(figsize=(8,6))
ax = Axes3D(fig)
ax.grid()
# ax.xlabel("PC1")
# ax.ylabel("PC2")
# ax.zlabel("PC3")
for i in range(2):
    ax.scatter(pca_df.PC1[T==i], pca_df.PC2[T==i], pca_df.PC3[T==i],color=colour[i], alpha=0.2)

pca_df = pd.DataFrame(pca_data, columns=labels)
plt.figure(figsize=(8,6))
plt.grid()
plt.xlabel("PC1")
plt.ylabel("PC2")
for i in range(2):
    plt.scatter(pca_df.PC1[T==i], pca_df.PC2[T==i],color=colour[i], alpha=0.2)

plt.figure(figsize=(8,6))
plt.grid()
for i in range(2):
    plt.hist(pca_df.PC1[T==i],color=colour[i], edgecolor='black')

plt.figure(figsize=(8,6))
plt.grid()
for i in range(2):
    plt.hist(pca_df.PC2[T==i],color=colour[i], edgecolor='black')
