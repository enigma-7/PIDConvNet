import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import glob
from TOOLS import DATA, MODELS, LOG, METRICS
import random, matplotlib, datetime, os
matplotlib.rcParams.update({'font.size': 16})
matplotlib.rcParams['text.usetex'] = True

run_no = "000265378/"
dataname = "all/"
directory = "data/output/" + run_no + dataname
raw_data = np.load(directory + "0_tracks.npy")
raw_info = np.load(directory + "0_info_set.npy")
print("Loaded: %s" % directory)

dataset, infoset = DATA.process_1(raw_data, raw_info)
dataset, infoset = DATA.shuffle_(dataset/1024, infoset)

(X, infoset), (Xv, valid_infoset), (Xt, test_infoset) = DATA.TVT_split_(dataset, infoset)
T  = infoset[:,0]
Tv = valid_infoset[:,0]
Tt = test_infoset[:,0]

##      MODEL       ##

stamp = datetime.datetime.now().strftime("%d-%m-%H%M%S")

###     Iteration 1     ###
# conv_sizes = [(4, 8), (8, 16), (16, 32), (32, 64), (64, 128)]
#
# for i, conv_size in enumerate(conv_sizes):
#     mname = "conv-%d-%d-dense-1024-"%(conv_size)
#     tensorboard, csvlogger = LOG.logger_(run_no, dataname, stamp, mname)
#     model = MODELS.blank_2_1_(conv_size[0], conv_size[1], 1024)
#     model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=1e-3), loss='binary_crossentropy', metrics=[METRICS.pion_con]) #Change loss function
#     model.fit(x=X, y=T, batch_size = 100, epochs=20, validation_data=(Xv,Tv), callbacks=[tensorboard, csvlogger])
    # model.summary()

directory = "logs-CSV/" + run_no + dataname
glob.glob(directory + "08-08-164626_*")
filenames = [directory + "08-08-164626_" + "conv-%d-%d-dense-1024"%(cs) for cs in conv_sizes]
plotdir = '/home/jibran/Desktop/neuralnet/plots/'

li = []
nm = []
# filenames
# fils = sorted(filenames, key=lambda item: (int(item.partition('conv-')[-1].partition('-')[0])
#     if item.isdigit() else float('inf'), item))
#
# item = '4 sheets'
# int(item.partition(' ')[0])
# filename = 'logs-CSV/000265378/all/08-08-164626_conv-64-128-dense-1024'
# int(filename.partition('conv-')[-1].partition('-')[0])
#
# fils

for i, filename in enumerate(filenames):
    nm.append(filename.split('_')[-1])
    df = pd.read_csv(filename, index_col=None, header=0)
    df['model_name'] = ["%s"%filename.split('_')[-1]]*20
    li.append(df)

frame = pd.concat(li, axis=0, names=nm)
frame = frame.set_index(['model_name', 'epoch'])
frame.head()
frame.sum(level='model_name').sort_values(by='train_time', inplace=True)
array = frame.values.reshape(-1,20,5)
# np.save(directory + "Iter1_08-08-164626_.npy",array)
colours = ['red', 'goldenrod', 'green', 'blue', 'purple']
epochtimes = array[:,:,2].mean(axis=1)
fig, axes = plt.subplots(1, 3, figsize=(17,6))
for j in range(array.shape[-1]):
    axes[0].plot(range(1,21), array[j,:,0], label=nm[j], color=colours[j])
    axes[1].plot(array[j,:,2].cumsum(), array[j,:,0], label=nm[j], color=colours[j])
    axes[2].bar(j+1, epochtimes[j], color = colours[j], width = 0.4)
axes[0].set_ylabel("Loss")
axes[0].grid()
axes[0].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].grid()
axes[1].legend()
axes[1].set_xlabel("Training Time [$s$]")
axes[2].set_xlabel("Model no.")
axes[2].set_ylabel("Average Training Epoch [$s$]")
axes[2].grid()

# plt.savefig(plotdir + 'Iter1.png')

for m in [1,4]:
    fig, axes = plt.subplots(1, 2, figsize=(12,6))
    axes[0].plot(array[m,:,2].cumsum(), array[m,:,0], color=colours[m], label="Training")
    axes[0].plot(array[m,:,2].cumsum(), array[m,:,3], color=colours[m], label="Validation", linestyle='--')
    axes[0].set_ylabel("Loss")
    axes[1].plot(array[m,:,2].cumsum(), array[m,:,1], color=colours[m], label="Training")
    axes[1].plot(array[m,:,2].cumsum(), array[m,:,4], color=colours[m], label="Validation", linestyle='--')
    axes[1].set_ylabel("Pion Contamination")
    for i in range(2):
        axes[i].grid()
        axes[i].set_xlabel("Training Time [$s$]")
    plt.legend()
    # plt.savefig(plotdir + nm[m] + ".png")
