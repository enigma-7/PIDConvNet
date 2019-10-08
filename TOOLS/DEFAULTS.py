import glob

datadir = '/home/jibran/Desktop/neuralnet/trdpid/py_datatools/data_processed/'
plotdir = '/home/jibran/Desktop/neuralnet/plots/'
ocdbdir = '/home/jibran/Desktop/neuralnet/trdpid/ocdb/calib_files/'
dictdir = '/home/jibran/Desktop/neuralnet/trdpid/py_datatools/data_raw/'
modldir = 'saved_models/'

DS1 = [265378]
DS2 = [265377, 265381, 265383, 265385, 265388, 265419, 265420, 265425, 265426, 265499] + DS1
DS3 = [265309, 265332, 265334, 265335, 265336, 265339, 265342] + DS2

cnames = ["$\\pi$","$e$"]
colour = ['r', 'g']
letter = ['(a)','(b)','(c)','(d)','(e)','(f)','(g)']
bin_no = 50
styles = ['--','-.']

infocols_track_ = ["label", "nsigmae", "nsigmap", "PT", "{dE}/{dx}", "Momenta [GeV]", "$\\eta$", "$\\theta$", "$\\phi$",
    "run_no", "event", "V0trackID",  "track"]
info_cols_tracklet_ = ["label", "nsigmae", "nsigmap", "PT", "{dE}/{dx}", "Momenta [GeV]", "$\\eta$", "$\\theta$", "$\\phi$",
    "run_no", "event", "V0trackID",  "track",  "Layers", "Detector", "Row", "Column"]
ocdb_cols1 = ["$V_{Anode}$", "$V_{Drift}$", "$G_{c}$", "$v_{D}$", "$E \\times B$"]
ocdb_cols2 = ["G_{p}(%d)"%(i+1) for i in range(17)]

col = info_cols_tracklet_+ ocdb_cols1 + ocdb_cols2

tracklet_V_U1 = ['logs-CSV/DS2/16-09-173700_conv-8-16-dense-128-64']
tracklet_V_U2 = ['logs-CSV/DS3/04-10-092029_Utracklet_C-8-16-D-128-64']
tracklet_V_U3 = ['logs-CSV/DS2/04-10-093904_Utracklet_C-8-16-D-128-64']
tracklet_V_U4 = ['logs-CSV/DS2/04-10-095955_Utracklet_C-8-16-D-128-64']     #shuffled dataset- good results
tracklet_V_U5 = ['logs-CSV/DS3/07-10-102944_tracklet_V_U_wbce7']                  #

tracklet_conv_iter2 = glob.glob('logs-CSV/DS2/02-10-222349_*')
tracklet_conv_iter3 = glob.glob('logs-CSV/DS2/03-10-164208_*')
tracklet_conv_iter4 = glob.glob('logs-CSV/DS2/03-10-192008_*')

track_V_U_dropout_iter1 = glob.glob('logs-CSV/DS2/16-09-183152_*')                      #Discard
track_V_U_dropout_iter2 = glob.glob('logs-CSV/DS5/04-10-124841_track_V_U_dropout_*')    #0.00 dropout
track_V_U_dropout_iter3 = glob.glob('logs-CSV/DS5/04-10-143121_track_V_U_dropout_*')    #Discard
track_V_U_dropout_iter4 = glob.glob('logs-CSV/DS5/05-10-121344_track_V_U_dropout_*')    #Discard
track_V_U_dropout_iter5 = glob.glob('logs-CSV/DS5/05-10-132355_track_V_U_dropout_*')    #0.30 dropout (after flatten)
track_V_U_dropout_iter6 = glob.glob('logs-CSV/DS5/05-10-192656_track_V_U_dropout_*')
