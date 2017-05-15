from experiments import experiments
from sklearn.externals import joblib

from feature_extraction import FeatureExtraction
import os


SPEAKER1_TRAIN = '../../data/vctk/speaker1/speaker1-train-files.txt'
SPEAKER1_VAL = '../../data/vctk/speaker1/speaker1-val-files.txt'
SPEAKER1_DATA = '../../data/vctk/VCTK-Corpus/wav48/p225/'

MULTISPEAKER_TRAIN = '../../data/vctk/multispeaker/vctk-train-files.txt'
MULTISPEAKER_VAL = '../../data/vctk/multispeaker/vctk-val-files-subset.txt'
MULTISPEAKER_DATA = '../../data/vctk/VCTK-Corpus/wav48/'

MUSIC_TRAIN = '../../data/music/music_train.npy'
MUSIC_VAL = '../../data/music/music_valid.npy'
MUSIC_DATA = ''

OUTPUT_DIR = '../../data/output/'

def create_path(params):
  path = ''
  for key in params.keys():
    path += key + '=' + str(params[key]) + '/'
  return path

# Loop over the experiments create necessary datasets and save to paths
for experiment in experiments:
    if experiment['dataset'] == 'speaker1':
      fe = FeatureExtraction(train_files=SPEAKER1_TRAIN,
                             val_files=SPEAKER1_VAL,
                             data_dir=SPEAKER1_DATA,
                             dataset='vctk',
                             upsample=experiment['upsample'])

      SAVE_DIR = OUTPUT_DIR + create_path(experiment)

    elif experiment['dataset'] == 'multispeaker':
      fe = FeatureExtraction(train_files=MULTISPEAKER_TRAIN,
                             val_files=MULTISPEAKER_VAL,
                             data_dir=MULTISPEAKER_DATA,
                             dataset='vctk',
                             upsample=experiment['upsample'],
                             train_subsample=experiment['subsample'])

      SAVE_DIR = OUTPUT_DIR + create_path(experiment)

    elif experiment['dataset'] == 'music':
      fe = FeatureExtraction(train_files=MUSIC_TRAIN,
                             val_files=MUSIC_VAL,
                             data_dir=MUSIC_DATA,
                             dataset='music',
                             upsample=experiment['upsample'])

      SAVE_DIR = OUTPUT_DIR + create_path(experiment)

    print "Saving output to:", SAVE_DIR
    os.makedirs(SAVE_DIR)
    joblib.dump(fe, SAVE_DIR + 'fe')
