import numpy as np
import os

def subdir_(directory):
    fileNames = []
    for r, d, f in os.walk(directory):
        for file in f:
            if 'pythonDict.txt' in file:
                fileNames.append(os.path.join(r, file))
    fileNames.sort()
    return fileNames

def process_1(raw_data, raw_info, min_tracklet=1.0, min_adcvalue=10.0, min_momentum=0.0, max_momentum=100.0):
    """
    raw_info[:,0] = label
    raw_info[:,1] = nsigmae
    raw_info[:,2] = nsigmap
    raw_info[:,3] = PT
    raw_info[:,4] = dEdX
    raw_info[:,5] = P
    raw_info[:,6] = eta
    raw_info[:,7] = theta
    raw_info[:,8] = phi
    raw_info[:,9] = event
    raw_info[:,10] = trackid
    raw_info[:,11] = trackval
    raw_info[:,12] = num_tracklets
    raw_info[:,13:19] = present_map
    """
    mask_tracklet = raw_info[:,12] > min_tracklet                          #Discriminate tracks based on no. of tracklets
    mask_adcvalue = raw_data.sum(axis=(1,2,3)) > min_adcvalue              #Sum of ADC per tracklet
    mask_momentum = (raw_info[:,5] > min_momentum) & (raw_info[:,5] < max_momentum) #Select momentum range
    raw_info = raw_info[mask_tracklet & mask_adcvalue & mask_momentum]
    raw_data = raw_data[mask_tracklet & mask_adcvalue & mask_momentum]
    numtracks = raw_info[:,12].astype(int)                                  #Tracklets per track

    infoset = np.zeros((numtracks.sum(), raw_info[:,:12].shape[1]))
    k = 0
    for i in range(len(numtracks)):
        t = i
        for j in range(numtracks[i]):
            infoset[k] = raw_info[i,:12]
            k += 1

    present = raw_info[:,-6:].flatten('C').astype('bool')
    dataset = raw_data.reshape(raw_data.shape[0]*raw_data.shape[1],17,24,1)[present]  #NHWC array
    return dataset, infoset

def shuffle_(dataset, infoset):
    #   Apply random permutation to given dataset.  #
    perm = np.random.permutation(dataset.shape[0])
    dataset = dataset[perm]
    infoset = infoset[perm]
    return dataset, infoset

def elec_strip_(dataset, infoset):
    targets = infoset[:,0].astype('int')
    dataset = dataset[targets==1]
    infoset = infoset[targets==1]
    return dataset, infoset

def pion_strip_(dataset, infoset):
    targets = infoset[:,0].astype('int')
    dataset = dataset[targets==0]
    infoset = infoset[targets==0]
    return dataset, infoset

def batch_(dataset, targets, batch_size, pos):
    batch_dataset = dataset[(pos-1)*batch_size:pos*batch_size]
    batch_targets = targets[(pos-1)*batch_size:pos*batch_size]
    return batch_dataset, batch_targets

def elec_pion_split_(dataset, targets):
    elec_dataset = dataset[targets.astype(bool)]
    pion_dataset = dataset[(1-targets).astype(bool)]
    elec_targets = targets[targets==1]
    pion_targets = targets[targets==0]
    return [elec_dataset, elec_targets], [pion_dataset, pion_targets]

def train_valid_split_(dataset, targets, split=0.2):
    #   Create training and validation sets   #
    N = int((1-split)*dataset.shape[0])
    train_dataset = dataset[:N]
    train_targets = targets[:N]
    valid_dataset = dataset[N:]
    valid_targets = targets[N:]
    return [train_dataset, train_targets], [valid_dataset, valid_targets]
