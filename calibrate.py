import numpy as np
import glob
from TOOLS import DATA, MODELS, LOG, ML, PLOT, DEFAULTS

ocdbdir = DEFAULTS.ocdbdir
datadir = DEFAULTS.datadir
plotdir = DEFAULTS.plotdir

###         Chamber Corrections         ###
calib_params = {}
for fil in glob.glob(DEFAULTS.ocdbdir + 'chamber_info_2016_*.txt'):
	print('Loading calib file:', fil)
	run_no = fil.split('_')[-1].split('.')[0]
	calib_params[run_no] = np.genfromtxt(fil, delimiter=', ')[:,3]

for j in range(1,5):
    dataset_name = 'DS%d'%j
    new_dataset_name = dataset_name + '_chamber'
    print(dataset_name)

    dataset = np.load(datadir + dataset_name + '/0_tracks.npy')
    infoset = np.load(datadir + dataset_name + '/0_info_set.npy')

    for i in range(len(dataset)):
        run_no = infoset[i,9]
        run_gains = calib_params[str(int(run_no))]
        dets = infoset[i,14:20].astype(int)
        track_gains =  np.array([np.tile(run_gains[det],(17,24)) for det in dets])
        dataset[i] *= track_gains

    np.save(datadir + new_dataset_name + '/0_tracks.npy', dataset)
    np.save(datadir + new_dataset_name + '/0_info_set.npy', infoset)

###     Pad Corrections     ###
# calib_params = {}
# print("Now processing pad gain for runs:\n")
# for fil in glob.glob(DEFAULTS.ocdbdir + 'local_gains*.txt'):
#     run_no = fil.split('_')[-1].split('.')[0]
#     print("\t -%s"% run_no)
#     calib_params[run_no] = np.genfromtxt(fil, delimiter=', ')[:,2:].reshape((540, 16, 144))
# print("\nCalibration parameters initialized \n")
#
#
# for j in range(1,5):
#     dataset_name = 'DS%d'%j
#     new_dataset_name = dataset_name + '_pad'
#     print(dataset_name)
#
#     dataset = np.load(datadir + dataset_name + '/0_tracks.npy')
#     infoset = np.load(datadir + dataset_name + '/0_info_set.npy')
#
#     for i in range(len(dataset)):
#     	run_no = infoset[i,9]
#     	run_gains = calib_params[str(int(run_no))]
#     	dets = infoset[i,14:20].astype(int)
#     	rows = infoset[i,20:26].astype(int)
#     	cols = infoset[i,26:32].astype(int)
#     	track_gains = np.expand_dims(run_gains[dets,rows,[cols - 8 + i for i in range(17)]].T, axis=-1)
#     	dataset[i] *= track_gains
#
#     np.save(datadir + new_dataset_name + '/0_tracks.npy', dataset)
#     np.save(datadir + new_dataset_name + '/0_info_set.npy', infoset)

###     Online Corrections     ###
calib_params = {}
print("Now processing pad gain for runs:\n")
for fil in glob.glob(DEFAULTS.ocdbdir + 'online_local_gains_2016_*.txt'):
    run_no = fil.split('_')[-1].split('.')[0]
    print("\t -%s"% run_no)
    calib_params[run_no] = np.genfromtxt(fil, delimiter=', ')[:,2:].reshape((540, 16, 144))
print("\nCalibration parameters initialized \n")


for j in range(1,5):
    dataset_name = 'DS%d'%j
    new_dataset_name = dataset_name + '_online'
    print(dataset_name)

    dataset = np.load(datadir + dataset_name + '/0_tracks.npy')
    infoset = np.load(datadir + dataset_name + '/0_info_set.npy')

    for i in range(len(dataset)):
    	run_no = infoset[i,9]
    	run_gains = calib_params[str(int(run_no))]
    	dets = infoset[i,14:20].astype(int)
    	rows = infoset[i,20:26].astype(int)
    	cols = infoset[i,26:32].astype(int)
    	track_gains = np.expand_dims(run_gains[dets,rows,[cols - 8 + i for i in range(17)]].T, axis=-1)
    	dataset[i] *= track_gains

    np.save(datadir + new_dataset_name + '/0_tracks.npy', dataset)
    np.save(datadir + new_dataset_name + '/0_info_set.npy', infoset)
