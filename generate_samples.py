"""
File to generate sound samples from model
"""
import os
from experiments import experiments
from sklearn.externals import joblib
from feature_extraction import FeatureExtraction
from dnn import DNN
import numpy as np
import librosa
import keras
# Set GPU here
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# Set data directory here
OUTPUT_DIR = '../../data/output/'

# Model variables that we are not experimenting with
N_FRAMES = 9 # number of context frames for prediction
BATCH_SIZE = 128
N_EPOCHS = 100

def create_path(params):
  path = ''
  for key in params.keys():
    path += key + '=' + str(params[key]) + '/'
  return path


MULTISPEAKER_TRAIN = '../../data/vctk/multispeaker/vctk-train-files.txt'
MULTISPEAKER_VAL = '../../data/vctk/multispeaker/vctk-val-files-subset.txt'
MULTISPEAKER_DATA = '../../data/vctk/VCTK-Corpus/wav48/'


for experiment in experiments:
  SAVE_DIR = OUTPUT_DIR + create_path(experiment)
  fe = joblib.load(SAVE_DIR + 'fe')

  #fe = FeatureExtraction(train_files=MULTISPEAKER_TRAIN,
  #                       val_files=MULTISPEAKER_VAL,
  #                       data_dir=MULTISPEAKER_DATA,
  #                       dataset='vctk',
  #                       upsample=experiment['upsample'],
  #                       train_subsample=experiment['subsample'])


  # Put our validation data in the correct input form depending on phase mode
  if experiment['phase'] in ('imaged', 'cheated'):
    X_val = fe.X_val
    Y_val = fe.Y_val

  elif experiment['phase'] == 'regression':
    X_val = np.hstack([fe.X_val, fe.X_val_phase])
    Y_val = np.hstack([fe.Y_val, fe.Y_val_phase])

  n_val, n_input = X_val.shape
  _, n_output = Y_val.shape

  # Use model to predict the high frequency magnitude and phase
  if experiment['model'] == 'dnn':
    from dnn import psnr_metric # Keras quirk
    dnn = DNN(fe=fe, name=SAVE_DIR + 'model.snapshot', n_input=n_input*N_FRAMES, n_output=n_output)
    dnn.model = keras.models.load_model(SAVE_DIR + 'model.snapshot', custom_objects={'psnr_metric': psnr_metric})

    idx = 0
    # Need to load up original waveforms to reconstruct them
    for waveform, rate in fe.ld._load_data(fe.ld.val_files, 1.0):
      X = fe.stft(waveform, fe.high_window_size, fe.high_window_shift)
      X_log_magnitude, X_phase = fe.decompose_stft(X)
      X_low, X_high, X_low_phase, X_high_phase =\
              fe.extract_low_high(X_log_magnitude, X_phase)

      X_low_whitened = fe.whiten_low.transform(X_low)

      if experiment['phase'] == 'cheated':
        X_context = fe.frame_creator(X_low_whitened, n_frames=N_FRAMES)
        Yhat_val = dnn.model.predict(X_context)
        Yhat_val = fe.whiten_high.inverse_transform(Yhat_val)

        n_samples = len(waveform)

        # Phase cheated
        Xhat_log_magnitude, X_phase = fe.reconstruct_low_high(X_low, Yhat_val, X_low_phase, X_high_phase)
        Xhat = fe.compose_stft(Xhat_log_magnitude, X_phase)
        xhat = fe.istft(Xhat, n_samples, fe.high_window_shift)

        print 'Saving', str(idx) + '.wav'
        librosa.output.write_wav(SAVE_DIR + str(idx) + '.wav', xhat, 16000.)

      elif experiment['phase'] == 'regression':
        X_low_phase_whitened = fe.whiten_low_phase.transform(X_low_phase)
        X_low_both = np.hstack([X_low_whitened, X_low_phase_whitened])
        X_context = fe.frame_creator(X_low_both, n_frames=N_FRAMES)
        Yhat_val = dnn.model.predict(X_context)
        Yhat_val, Yhat_val_phase = np.split(Yhat_val, 2, axis=1)

        Yhat_val = fe.whiten_high.inverse_transform(Yhat_val)
        Yhat_val_phase = fe.whiten_high_phase.inverse_transform(Yhat_val_phase)

        n_samples = len(waveform)
        # Phase predicted
        Xhat_log_magnitude, Xhat_phase = fe.reconstruct_low_high(X_low, Yhat_val, X_low_phase, Yhat_val_phase)
        Xhat = fe.compose_stft(Xhat_log_magnitude, X_phase)
        xhat = fe.istft(Xhat, n_samples, fe.high_window_shift)

        print 'Saving new with regression phase', str(idx) + '.wav'
        librosa.output.write_wav(SAVE_DIR + str(idx) + '.wav', xhat, 16000.)

      idx += 1

  else:
    raise Exception('phase mode not supported')
