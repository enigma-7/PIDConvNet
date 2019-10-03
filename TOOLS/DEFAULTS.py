import glob

datadir = '/home/jibran/Desktop/neuralnet/trdpid/py_datatools/data_processed/'
plotdir = '/home/jibran/Desktop/neuralnet/plots/'
ocdbdir = '/home/jibran/Desktop/neuralnet/trdpid/ocdb/calib_files/'
dictdir = '/home/jibran/Desktop/neuralnet/trdpid/py_datatools/data_raw/'
modldir = 'saved_models/'

DS1 = [265378]
DS2 = [265377, 265381, 265383, 265385, 265388, 265419, 265420, 265425, 265426, 265499] + DS1
DS3 = [265309, 265332, 265334, 265335, 265336, 265339, 265342] + DS2
DS3
cnames = ["$\\pi$","$e$"]
colour = ['r', 'g']
letter = ['(a)','(b)','(c)','(d)','(e)','(f)','(g)']
bin_no = 50
styles = ['--','-.']

infocols_track_ = ["label", "nsigmae", "nsigmap", "PT", "{dE}/{dx}", "Momenta [GeV]", "$\\eta$", "$\\theta$", "$\\phi$",
    "run_no", "event", "V0trackID",  "track"]
info_cols_tracklet_ = ["label", "nsigmae", "nsigmap", "PT", "{dE}/{dx}", "Momenta [GeV]", "$\\eta$", "$\\theta$", "$\\phi$",
    "run_no", "event", "V0trackID",  "track", "Detector", "Row", "Column"]
ocdb_cols1 = ["$V_{Anode}$", "$V_{Drift}$", "$G_{c}$", "$v_{D}$", "$E \\times B$"]

tracklet_uncalib = ['logs-CSV/DS2/16-09-173700_conv-8-16-dense-128-64']
track_uncalib_dropout_iter = glob.glob('logs-CSV/DS2/16-09-183152_*')
