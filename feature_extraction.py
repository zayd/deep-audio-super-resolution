import scipy as sp
import scipy.signal as sps
import numpy as np
import numpy.random as random

from sklearn.decomposition import PCA

import sys
from load_data import LoadData

class FeatureExtraction(object):
  """ Implements Li and Lee 2015 DNN Feature Extraction """

  def __init__(self, high_window_size=512,
                     high_window_shift=256,
                     low_window_size=256,
                     low_window_shift=128,
                     sampling_rate=16000.,
                     train_subsample=1.0,
                     val_subsample=1.0,
                     train_files='../../data/vctk/multispeaker/vctk-train-files.txt',
                     val_files='../../data/vctk/multispeaker/vctk-val-files.txt',
                     data_dir='../../data/vctk/VCTK-Corpus/wav48/',
                     dataset='vctk',
                     upsample=2
                     ):

    self.high_window_size= high_window_size
    self.high_window_shift = high_window_shift
    self.low_window_size = low_window_size
    self.low_window_shift = low_window_shift

    self.train_subsample = train_subsample
    self.val_subsample = val_subsample
    self.dataset = dataset
    self.upsample = upsample

    self.ld = LoadData(sampling_rate=sampling_rate,
                       train_files=train_files,
                       val_files=val_files,
                       data_dir=data_dir,
                       train_subsample=self.train_subsample)


    if self.dataset == 'vctk':
      train_waveforms = self.ld._load_data(self.ld.train_files, self.train_subsample)
      val_waveforms = self.ld._load_data(self.ld.val_files, self.val_subsample)
    elif self.dataset == 'music':
      train_waveforms = np.load(train_files)
      if self.train_subsample < 1.0:
        n_examples, _ = train_waveforms.shape
        subsample = np.random.choice(np.arange(n_examples),
                                     size=int(n_examples*self.train_subsample))
        train_waveforms = train_waveforms[subsample]

      val_waveforms = np.load(val_files)
      if self.val_subsample < 1.0:
        n_examples, _ = val_waveforms.shape
        subsample = np.random.choice(np.arange(n_examples),
                                     size=int(n_examples*self.val_subsample))
        val_waveforms = val_waveforms[subsample]


    # Manually setting these numbers based on self.upsample
    if self.upsample == 2:
      self.low_band_size = self.low_window_size/2 + 1
      self.high_band_size = self.high_window_size/4
    elif self.upsample == 4:
      self.low_band_size = self.low_window_size/4 + 1
      self.high_band_size = self.high_window_size/4 + 64
    elif self.upsample == 6:
      self.low_band_size = int(np.ceil(self.low_window_size/6 + 1))
      self.high_band_size = self.high_window_size/4 + (64 + 21)
    elif self.upsample == 8:
      self.low_band_size = self.low_window_size/8 + 1
      self.high_band_size = self.high_window_size/4 + (64 + 32)

    self.whiten_low = PCA(n_components=self.low_band_size, whiten=True)
    self.whiten_high = PCA(n_components=self.high_band_size, whiten=True)
    self.whiten_low_phase = PCA(n_components=self.low_band_size, whiten=True)
    self.whiten_high_phase = PCA(n_components=self.high_band_size, whiten=True)

    self.create_training_set(train_waveforms, val_waveforms)

  def frame_creator(self, X, n_frames):
    """Creates context frames from  X"""
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

    # Get all
    idx = np.arange(n_examples)

    neighbor_idx = neighbor_indices(idx, n_behind, n_behind)
    X_context = X_padded[neighbor_idx].reshape(n_examples,
                                               n_frames*n_features)
    return X_context

  def create_training_set(self, train_waveforms, val_waveforms):
    """
    Create training and validation set
    and compute mean and correllation matrix
    """
    print "Extracting features..."

    X_train, Y_train, X_train_phase, Y_train_phase =\
      self.pipeline(train_waveforms)
    self.X_train = np.vstack(X_train)
    self.Y_train = np.vstack(Y_train)
    self.X_train_phase = np.vstack(X_train_phase)
    self.Y_train_phase = np.vstack(Y_train_phase)

    X_val, Y_val, X_val_phase, Y_val_phase = self.pipeline(val_waveforms)
    self.X_val = np.vstack(X_val)
    self.Y_val = np.vstack(Y_val)
    self.X_val_phase = np.vstack(X_val_phase)
    self.Y_val_phase = np.vstack(Y_val_phase)

    print "Computing mean and covariance. Whitening training data..."
    self.X_train = self.whiten_low.fit_transform(self.X_train)
    self.Y_train = self.whiten_high.fit_transform(self.Y_train)
    self.X_train_phase = self.whiten_low_phase.fit_transform(self.X_train_phase)
    self.Y_train_phase = self.whiten_high_phase.fit_transform(self.Y_train_phase)

    print "Whitening validation data..."
    self.X_val = self.whiten_low.transform(self.X_val)
    self.Y_val = self.whiten_high.transform(self.Y_val)
    self.X_val_phase = self.whiten_low_phase.transform(self.X_val_phase)
    self.Y_val_phase = self.whiten_high_phase.transform(self.Y_val_phase)

  def pipeline(self, waveforms):
    """ Takes generator of waveforms and returns generator of
        low-band and high-band features """

    if self.dataset == 'vctk':
      X_lows, X_highs, X_lows_phase, X_highs_phase = [], [], [], []

      for waveform, rate in waveforms:
        # First high band features
        X = self.stft(waveform, self.high_window_size, self.high_window_shift)
        X_log_magnitude, X_phase = self.decompose_stft(X)
        X_low, X_high, X_low_phase, X_high_phase =\
          self.extract_low_high(X_log_magnitude, X_phase)

        # Then extract the low band features from the downsampled signal
        # Assume our filter is perfect - use the extracted signal
        #waveform_ds = sps.decimate(waveform, self.upsample, zero_phase=True)
        #X = self.stft(waveform_ds, self.low_window_size, self.low_window_shift)
        #X_log_magnitude, X_phase = self.decompose_stft(X)
        #X_low = self.extract_low_high(X_log_magnitude, split=False)

        X_lows.append(X_low)
        X_highs.append(X_high)
        X_lows_phase.append(X_low_phase)
        X_highs_phase.append(X_high_phase)

      return X_lows, X_highs, X_lows_phase, X_highs_phase

    elif self.dataset == 'music':
      X_lows, X_highs, X_lows_phase, X_highs_phase = [], [], [], []

      for waveform in waveforms:
        # First high band features
        X = self.stft(waveform, self.high_window_size, self.high_window_shift)
        X_log_magnitude, X_phase = self.decompose_stft(X)
        X_low, X_high, X_low_phase, X_high_phase =\
          self.extract_low_high(X_log_magnitude, X_phase)

        X_lows.append(X_low)
        X_highs.append(X_high)
        X_lows_phase.append(X_low_phase)
        X_highs_phase.append(X_high_phase)

      return X_lows, X_highs, X_lows_phase, X_highs_phase

  def stft(self, x, window_size, window_shift):
    """ STFT with non-symmetric Hamming window """
    w = sps.hamming(window_size, sym=False)
    X = sp.array([sp.fft(w*x[i:i+window_size])
                  for i in range(0, len(x)-window_size, window_shift)])
    return X

  def istft(self, X, n_samples, window_shift):
    """ iSTFT with symmetric Hamming window """
    n_windows, window_size = X.shape
    #x_len = window_size + (n_windows-1)*window_shift

    x = sp.zeros(n_samples)

    for n,i in enumerate(range(0, len(x)-window_size, window_shift)):
        x[i:i+window_size] += sp.real(sp.ifft(X[n]))
    return x

  def decompose_stft(self, X):
    """ Takes windowed STFT and compute ln mag and phase """
    # Replace zeros with fudge
    X[X == 0] = 1e-8
    X_log_magnitude = 2*np.log(np.absolute(X))
    X_phase = np.angle(X, deg=False)

    return X_log_magnitude, X_phase

  def extract_low_high(self, X_log_magnitude, X_phase, split=True):
    """ Extract high and low bands from X_log_magnitude """

    def split(X, n):
      """ Takes as input array X and returns a split column at X[:,n] """
      return X[:, :n], X[:, n:]

    windows, N = X_log_magnitude.shape

    # Conjugate symmetric only take non-redundant points
    X_log_magnitude = X_log_magnitude[:, :(N/2)+1]
    # Conjugate symmetric only take non-redundant points
    X_phase = X_phase[:, :(N/2)+1]

    # If we want to split into high and low components
    # I break out the cases manually because it's easier to follow than eqn
    if split:
      if self.upsample == 2:
        X_low, X_high = split(X_log_magnitude, (N/4)+1)
        X_low_phase, X_high_phase = split(X_phase, (N/4)+1)
      elif self.upsample == 4:
        X_low, X_high = split(X_log_magnitude, (N/8)+1)
        X_low_phase, X_high_phase = split(X_phase, (N/8)+1)
      elif self.upsample == 6:
        X_low, X_high = split(X_log_magnitude, int(np.ceil((N/12)+1)))
        X_low_phase, X_high_phase = split(X_phase, int(np.ceil((N/12)+1)))
      elif self.upsample == 8:
        X_low, X_high = split(X_log_magnitude, (N/16)+1)
        X_low_phase, X_high_phase = split(X_phase, (N/16)+1)

      return X_low, X_high, X_low_phase, X_high_phase
    else:
      return X_log_magnitude

  def reconstruct_low_high(self, X_low, X_high, X_low_phase=None, X_high_phase=None):
    """ Reconstruct from X_low, Y_high and assume conjugate symmetry """

    # bug in preprocessing
    if X_high.shape[1] == 129:
      # Slice off first index
      X_high = X_high[:, 1:]

    #windows, N = X_log_magnitude.shape
    X_log_magnitude =  np.hstack([X_low, X_high])

    # Conjugate symmetric only take non-redundant points
    # Slice last two indices and flip
    flipped = X_log_magnitude[:, 1:-1][:, ::-1]
    X_log_magnitude =  np.hstack([X_log_magnitude, flipped])

    if X_low_phase is not None and X_high_phase is not None:
      X_phase = np.hstack([X_low_phase, X_high_phase])
      # Multipl by -1 to take complex conjugate
      flipped_phase = -1*X_phase[:, 1:-1][:, ::-1]
      X_phase = np.hstack([X_phase, flipped_phase])
      return X_log_magnitude, X_phase
    else:
      return X_log_magnitude


  def compose_stft(self, X_log_magnitude, X_phase):
    """ Do reverse operation of decompose_stft """
    return np.exp(0.5*X_log_magnitude + 1j*X_phase)

if __name__ == "__main__":
  fe = FeatureExtraction()

  import IPython; IPython.embed();
