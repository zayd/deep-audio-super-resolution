import os
from experiments import experiments
from sklearn.externals import joblib
from dnn import DNN
import numpy as np
# Set GPU here
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

# Set data directory here
OUTPUT_DIR = '../../data/output/'

# Model variables that we are not experimenting with
N_FRAMES = 9 # number of context frames for prediction
BATCH_SIZE = 128
N_EPOCHS = 100

# Used to load the dataset created by prepare_data.py
def create_path(params):
  path = ''
  for key in params.keys():
    path += key + '=' + str(params[key]) + '/'
  return path

# Load up the data specified by the experiment. First need to run prepare_data.py
# to create datasets
for experiment in experiments:
    SAVE_DIR = OUTPUT_DIR + create_path(experiment)
    fe = joblib.load(SAVE_DIR + 'fe')

    if experiment['phase'] in ('imaged', 'cheated'):
      X_train = fe.X_train
      Y_train = fe.Y_train

      X_val = fe.X_val
      Y_val = fe.Y_val

    elif experiment['phase'] == 'regression':
      # The following to train with phase regression
      X_train = np.hstack([fe.X_train, fe.X_train_phase])
      Y_train = np.hstack([fe.Y_train, fe.Y_train_phase])

      X_val = np.hstack([fe.X_val, fe.X_val_phase])
      Y_val = np.hstack([fe.Y_val, fe.Y_val_phase])


    n_train, n_input = X_train.shape
    n_val, _ = X_val.shape

    _, n_output = Y_train.shape

    if experiment['model'] == 'dnn':
      dnn = DNN(name=SAVE_DIR + 'model.snapshot', fe=fe, n_input=n_input*N_FRAMES, n_output=n_output)
      # Create frame generator for training
      data = dnn.frame_generator(X_train, Y_train,
                                 n_frames=N_FRAMES, batch_size=BATCH_SIZE)

      data_val = dnn.frame_generator(X_val, Y_val,
                                   n_frames=N_FRAMES, batch_size=BATCH_SIZE)

      # Create model to train - replace with GMM\linear regression here
      dnn.fit_generator(data, data_val, n_train, n_val, N_EPOCHS)
