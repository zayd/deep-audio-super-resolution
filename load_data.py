"""
Implements wav data file loading to extract LFCC features
"""

import os
import scipy.io.wavfile
import librosa

import numpy as np

class LoadData(object):

  def __init__(self, data_dir='../../data/vctk/VCTK-Corpus/wav48/p225/',
               train_files='../../data/vctk/speaker1/speaker1-train-files.txt',
               val_files='../../data/vctk/speaker1/speaker1-val-files.txt',
               sampling_rate=8000.,
               train_subsample=1.0 # Train on this fraction of training set
              ):

    self.data_dir = data_dir
    self.train_files = train_files
    self.val_files = val_files
    self.train_subsample = train_subsample

    self.sampling_rate = sampling_rate # must match sampling rate in SpeechFeatures
    self.patch_size = 0.02 # patch size in seconds
    self.num_samples = self.patch_size*self.sampling_rate

  def generate_data(self):
    print "Generating data..."

    X_train = [self.to_patches(waveform)
         for waveform, rate in self._load_data(self.train_files,
                                               subsample=self.train_subsample)]
    X_train = np.vstack(X_train)

    X_val = [self.to_patches(waveform)
         for waveform, rate in self._load_data(self.val_files)]
    X_val = np.vstack(X_val)

    return X_train, X_val

  def _load_data(self, input_files, subsample=1.0):
    """
    Internal helper function to load files
    """

    file_list = []
    file_extensions = {'.wav'}
    with open(input_files) as f:
      for line in f:
        if subsample < 1.0 and np.random.uniform() > subsample:
          continue

        filename = line.strip()
        ext = os.path.splitext(filename)[1]
        if ext in file_extensions:
          file_list.append(os.path.join(self.data_dir, filename))

    print "In one set {0} files".format(len(file_list))

    waveforms = []
    for file_path in file_list:
      waveform, rate = librosa.load(file_path, sr=self.sampling_rate)
      waveforms.append((waveform, rate))

    return waveforms

  def to_patches(self, waveform):
    """ Break up waveform to patches """
    num_patches = len(waveform) // self.num_samples

    # chop off end so equal patches
    remainder = int(len(waveform) % self.num_samples)

    if remainder != 0:
      waveform = waveform[:-remainder]

    patches = np.split(waveform, num_patches)

    return np.vstack(patches)

if __name__ == "__main__":

  ld = LoadData()

  import IPython; IPython.embed();
