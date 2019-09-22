import glob

datadir = '/home/jibran/Desktop/neuralnet/trdpid/py_datatools/data_processed/'
plotdir = '/home/jibran/Desktop/neuralnet/plots/'
ocdbdir = '/home/jibran/Desktop/neuralnet/trdpid/ocdb/calib_files/'
dictdir = '/home/jibran/Desktop/neuralnet/trdpid/py_datatools/data_raw/'

cnames = ["$\\pi$","$e$"]
colour = ['r', 'g']
letter = ['(a)','(b)','(c)','(d)','(e)','(f)','(g)']
bin_no = 50
styles = ['--','-.']

info_cols_tracklet_ = ["label", "nsigmae", "nsigmap", "PT", "{dE}/{dx}", "Momenta [GeV]", "$\\eta$", "$\\theta$", "$\\phi$",
    "run_no", "event", "V0trackID",  "track", "Detector", "Row", "Column"]
ocdb_cols = ["Anode Voltage", "Drift Voltage", "Drift Velocity", "ExB"]

tracklet_uncalib = ['logs-CSV/DS2/16-09-173700_conv-8-16-dense-128-64']
track_uncalib_dropout_iter = glob.glob('logs-CSV/DS2/16-09-183152_*')
