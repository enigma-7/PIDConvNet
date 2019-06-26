import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import DATA, MODELS, random, matplotlib

fileNames = DATA.subdir_('data/input/')
tracks, parameters = DATA.extract_(fileNames)
#dataset, targets, momenta = DATA.process_(tracks, parameters)
directory = 'data/datasets_1/'
raw_dataset = np.load(directory + 'tensor.npy')
raw_targets = np.load(directory + 'labels.npy')
raw_momenta = np.load(directory + 'moment.npy')
print('Data loaded...')

#   Create masks on momentaum   #
mask1 = raw_momenta < 3.0
mask2 = (raw_momenta < 9.4) & (raw_momenta > 3.0)
mask3 = raw_momenta > 9.4
mask2.sum() + mask1.sum() + mask3.sum() - len(raw_momenta)

"""
plt.figure(figsize=(8,6))
plt.hist(momenta, bins=30, edgecolor='black')
plt.yscale('log')
plt.ylabel('Count')
plt.xlabel('Total Momentum - P (GeV)')
"""
#   Create dataset with equal no. of elec and pion

elec, pion = DATA.elec_pion_split_(raw_dataset,raw_targets)
split = 0.5
len_pion = int((1/split-1)*elec[0].shape[0])
dataset = np.concatenate((elec[0], pion[0][:len_pion]))
targets = np.concatenate((elec[1], pion[1][:len_pion]))

#   Create dataset using momentaum mask1     #
"""
dataset = raw_dataset[mask1][:5000]
targets = raw_targets[mask1][:5000]
"""

dataset = dataset.reshape(dataset.shape[0],17,24,1)/1024                       #create NHWC dataset
dataset, targets = DATA.shuffle_(dataset, targets)
dataset, targets = dataset[:5000], targets[:5000]
train, valid = DATA.train_valid_split_(dataset, targets)
"""
epochs = 3
net1 = CNN.old
net1.compile(optimizer='adam', loss=cross_entropy, metrics=[CNN.recall,'accuracy'])

for epoch in range(epochs):
    train_dataset, train_targets = DATA.shuffle_(train[0], train[1])
    valid_dataset, valid_targets = valid[0], valid[1]
    net1.train_on_batch(train_dataset, train_targets, sample_weight=None, class_weight=None)
"""
#   Neural Network  #
net1 = CNN.old
net1.compile(optimizer='adam', loss='binary_crossentropy', metrics=[CNN.pion_con, 'accuracy'])
history = net1.fit(x=dataset, y=targets, batch_size = 100, epochs=10, validation_split=0.2)
net1.summary()
print(history.history)

"""
# Plot training & validation accuracy values
plt.figure(figsize=(8,6))
plt.plot(history.history['recall'])
plt.plot(history.history['val_recall'])
plt.title('Model recall')
plt.ylabel('Recall')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.figure(figsize=(8,6))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

plt.figure(figsize=(8,6))
plt.imshow(net1.trainable_weights[0][:,:,0,0])
net1.trainable_variables
#net1.weights[0][:,:,0,0]
"""

"""
#   Neural Network  #
net1 = MODELS.old
net1.compile(optimizer='adam', loss='binary_crossentropy', metrics=[MODELS.pion_con])
epochs = 2
batch_size = 100
loss = []
pion = []
for i in range(epochs):
    dataset, targets = DATA.shuffle_(dataset, targets)
    train, valid = DATA.train_valid_split_(dataset, targets)
    print('Epoch: %d / %d' %(i+1,epochs))
    for j in range(1, int(dataset.shape[0]/batch_size)):
        dataset_batch, targets_batch = DATA.batch_(dataset, targets, batch_size, j)
        results = net1.train_on_batch(x=dataset_batch, y=targets_batch)
        loss.append(results[0])
        pion.append(results[1])
        print('Loss: %.2f\t pion_con: %.2f'%(results[0],results[1]))
print('Done')

plt.figure(figsize=(8,6))
plt.plot(range(len(loss)),loss)
plt.plot(range(len(pion)),pion)
plt.show()
y_pred = net1.predict(valid[0])
positive = y_pred[valid[1]==1]
negative = y_pred[valid[1]==0]

plt.figure(figsize=(8,6))
plt.hist(positive, alpha=0.5, label = 'positive')
plt.hist(negative, alpha=0.5, label = 'negative')
plt.legend()
plt.show()
"""

"""
def fix_(fileNames):
    for fileName in fileNames:
        print(fileName)
        file = open(fileName,'r')
        bracks = 0
        for line in file:
            line = line.strip()
            for letter in line:
                if letter == '{':
                    bracks+=1
                elif letter =='}':
                    bracks-=1
        if bracks == 1:
            print(fileName)
            file = open(fileName, 'a+')
            file.write('}')
            file.close()
        return "Directory ready for processing"

def extract_(fileNames, paramname = ['pdgCode','P'], save = False):
    paramvals = [[] for i in range(len(paramname))]
    tracks = []
    for fil in fileNames:
        print(fil)
        f = open(fil)
        r = f.read()
        try:
            exec('raw_data = ' + r + '}')
            for dict in raw_data:
                track = []
                for i in range(6):
                    if 'layer ' +str(i) in raw_data[dict].keys():
                        #2D array#
                        tracklet = np.array(raw_data[dict]['layer '+str(i)])
                        if (tracklet.all())or(not tracklet.any()):
                            #if empty or equal to zero matrix, skip#
                            continue
                        else:
                            #3D array#
                            track.append(tracklet)
                if len(track)>0:
                    track = np.array(track)
                    tracks.append(track)
                    for i, p in enumerate(paramname):
                        paramvals[i].append(raw_data[dict][p])
        except Exception as e:
            print(e)
    parameters= {}                      #dictionary
    for i, p in enumerate(paramname):
        paramvals[i] = np.array(paramvals[i])
        parameters[p] = paramvals[i]
    tracks = np.asarray(tracks)         #list of 3D arrays
    if save:
        np.save('data/tracks', tracks)
    return tracks, parameters]

def process_6(tracks, parameters, save=False):
    #   exclude tracks which don't have 6 layers    #
    paramname = parameters.keys()
    dim3  = np.array([tracks[i].shape[0] for i in range(len(tracks))])
    bool = dim3==6
    tracks = tracks[bool]
    for i, par in enumerate(paramname):
        parameters[par] = parameters[par][bool]
    targets = (abs(parameters['pdgCode'])==11).astype(int)
    momenta = parameters['P']
    print('targets created...')
    dataset = []
    for i in range(len(tracks)):
        dataset.append(tracks[i])
    dataset = np.asarray(dataset)
    #dataset = np.swapaxes(np.swapaxes(dataset,1,2),2,3)/1023
    print('dataset created...')
    if save:
        np.save('data/dataset', dataset)
        np.save('data/targets', targets)
    return dataset, targets, momenta

def process_(tracks, parameters, save=True):
    paramname = parameters.keys()
    dim3  = np.array([tracks[i].shape[0] for i in range(len(tracks))])
    trck = [np.sum(dim3==i) for i in range(1,7)]                        #total of tracks having n tracklets
    bool = dim3 > 4                                                     #select tracks with at least 4 tracklets
    for i, par in enumerate(paramname):
        parameters[par] = parameters[par][bool]
    labtemp = (abs(parameters['pdgCode'])==11).astype(int)
    momtemp = parameters['P']
    trktemp = tracks[bool]
    dataset = []
    targets = []
    momenta = []
    for i in range(len(trktemp)):
        t = i
        for j in range(trktemp[i].shape[0]):
            dataset.append(trktemp[i][j])
            targets.append(labtemp[i])
            momenta.append(momtemp[i])
    dataset = np.asarray(dataset)             #3D array NHW
    targets = np.array(targets)               #targets on each tracklet
    momenta = np.array(momenta)               #momentum
    print('%i tracklets found; e- occur at %.2f'%(len(targets),np.sum(targets)/len(targets)))
    if save:
        np.save('data/dataset', dataset)
        np.save('data/targets', targets)
        np.save('data/momenta', momenta)
    return dataset, targets, momenta

fileNames = subdir_('data/input/')
tracks, parameters = extract_(fileNames)
"""
