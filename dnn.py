import os
import time
import pickle

import numpy as np
import numpy.random as random
import keras
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.callbacks import Callback, ProgbarLogger, ModelCheckpoint, EarlyStopping

from feature_extraction import FeatureExtraction
from sklearn.externals import joblib

class DNN(object):
  """ Implements Li and Lee 2015 DNN High Bandwidth Estimator """

  def __init__(self, name,
                     n_input=1161,  # n_frames (9) * narrow_band_window (256/2 + 1)
                     n_output=128,  # high_band_window (512/4 + 1)
                     n_hidden=2048,
                     n_layers=3,
                     fe=None):

    self.n_input = n_input
    self.n_output = n_output
    self.n_hidden = n_hidden
    self.n_layers = n_layers

    self.fe = fe
    self.name = name

    self.model = Sequential()
    self.model.add(Dense(self.n_hidden, input_shape=(self.n_input,)))
    self.model.add(Activation('relu'))

    for idx in range(self.n_layers):
      self.model.add(Dense(n_hidden))
      self.model.add(Activation('relu'))

    self.model.add(Dense(self.n_output))
    self.model.compile(optimizer='adam', loss='mse')#, metrics=[])

    self.progbar_logger = ProgbarLogger()
    self.psnr_metrics = PsnrMetrics(self)
    self.model_checkpoint = ModelCheckpoint(self.name, monitor='val_loss',
                                            verbose=1,
                                            save_best_only=True, mode='min')

    self.early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.0,
                                        patience=8, mode='min')

  def fit_generator(self, data, data_val, n_train, n_val, n_epochs):
    self.model.fit_generator(data, validation_data=data_val,
                             samples_per_epoch=n_train,
                             nb_epoch=n_epochs, verbose=2,
                             nb_val_samples=n_val,
                             callbacks=[self.psnr_metrics,
                                        self.model_checkpoint,
                                        self.early_stopping])

  def predict(self, X, fe, whiten_input=False):
    """ Passes concatenated context window through DNN """

    if whiten_input:
      X = fe.whiten_low.inverse_transform(X)

    Y = self.model.predict(X)
    return fe.whiten_high.inverse_transform(Y)

  def compute_val_psnr(self, fe):
    """ Compute psnr of model on validation """
    idx = 0
    psnrs = []
    fe.val_waveforms = fe.ld._load_data(fe.ld.val_files, subsample=1.0)

    for waveform, rate in fe.val_waveforms:
      X = fe.stft(waveform, fe.high_window_size, fe.high_window_shift)

      X_log_magnitude, X_phase = fe.decompose_stft(X)
      X_low, X_high = fe.extract_low_high(X_log_magnitude)
      X_low = fe.whiten_low.inverse_transform(X_low)

      X_context = fe.frame_creator(X_low, 9)
      Yhat_val = self.model.predict(X_context)
      Yhat_val = fe.whiten_high.inverse_transform(Yhat_val)

      n_samples = len(waveform)
      # X_high -> Yhat_val
      Xhat_log_magnitude = fe.reconstruct_low_high(X_low, Yhat_val)
      Xhat = fe.compose_stft(Xhat_log_magnitude, X_phase)
      xhat = fe.istft(Xhat, n_samples, fe.high_window_shift)

      l2_loss = np.mean((waveform - xhat)**2)
      psnrs.append(20. * np.log(np.max(waveform) / np.sqrt(l2_loss) + 1e-8) / np.log(10.))

    print "psnr on validation set with cheated phase {0}".format(np.mean(psnrs))
    return {'mean_psnr': np.mean(psnrs)}

  def frame_generator(self, X, Y, n_frames, batch_size):
    """ Generator that takes X and Y and generates batches to train on """

    def neighbor_indices(indices, n_behind, n_forward):
      """ Assumes odd number of context"""

      neighbor_indices = []
      for idx in indices:
        neighbor_indices += range(idx-n_behind, idx+n_forward+1)

      # We zero pad the input matrix
      padded_neighbor_indices = np.array(neighbor_indices) + n_behind

      return padded_neighbor_indices

    n_examples, n_features = X.shape

    assert(n_frames % 2 == 1) # assume n_frames is odd
    n_behind = (n_frames-1)/2
    X_padded = np.pad(X, ((n_behind, n_behind), (0, 0)), mode='constant')

    # Continue indefinitely
    while True:
      minibatch_idx = random.randint(0, high=n_examples, size=batch_size)

      neighbor_minibatch_idx = neighbor_indices(minibatch_idx,
                                                n_behind, n_behind)

      X_minibatch = X_padded[neighbor_minibatch_idx].reshape(batch_size,
                                                             n_frames*n_features)
      Y_minibatch = Y[minibatch_idx]

      yield X_minibatch, Y_minibatch

class PsnrMetrics(Callback):
    def __init__(self, dnn):
      super(PsnrMetrics, self).__init__()
      self.metrics = []
      self.dnn = dnn

    def on_epoch_end(self, epoch, logs={}):
      #metric = self.dnn.compute_val_psnr(dnn.fe)
      #metric['epoch'] = epoch
      #print metric
      #self.metrics.append(metric)
      #joblib.dump(self.dnn.fe.whiten_high, self.dnn.name + '-high.pkl')
      #joblib.dump(self.dnn.fe.whiten_low, self.dnn.name + '-low.pkl')
      #self.model.save(self.dnn.name)
      return

def psnr_metric(y_true, y_pred):
  l2_loss = K.mean((y_true - y_pred)**2)
  psnr = 20. * K.log(K.max(y_true) / K.sqrt(l2_loss) + 1e-8) / K.log(10.)
  return psnr

if __name__ == "__main__":
  fe = FeatureExtraction(train_subsample=0.25, val_subsample=1.0)

  batch_size = 128
  n_epochs = 100
  n_frames = 9

  data = fe.frame_generator(fe.X_train, fe.Y_train, n_frames=9, batch_size=128)
  n_train, _ = fe.X_train.shape

  data_val = fe.frame_generator(fe.X_val, fe.Y_val, n_frames=9, batch_size=128)
  n_val, _ = fe.X_val.shape

  dnn = DNN(fe=fe)
  dnn.fit_generator(data, data_val, n_train, n_val, n_epochs)
  import IPython; IPython.embed()
