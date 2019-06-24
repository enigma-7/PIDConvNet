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

def process_1(raw_data, raw_info, min_tracklet=1.0, min_adcvalue=1.0, min_momentum=0.0, max_momentum=100.0):
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
    return tracks, parameters
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
    momenta = np.array(momenta)               #momentaum " " "
    print('%i tracklets found; e- occur at %.2f'%(len(targets),np.sum(targets)/len(targets)))
    if save:
        np.save('data/dataset', dataset)
        np.save('data/targets', targets)
        np.save('data/momenta', momenta)
    return dataset, targets, momenta

fileNames = subdir_('data/input/')
tracks, parameters = extract_(fileNames)
"""
