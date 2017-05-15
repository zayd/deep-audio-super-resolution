# deep-audio-super-resolution

## Framework to train audio super-resolution neural nets on spectrogram features

## Running an experiment

Append a experiment parameters to the `experiments` list in `models/dnn/experiments.py`. e.g.

```
experiments += [OrderedDict([('dataset', 'speaker1'), ('upsample', 2),('model', 'dnn'), ('phase', 'regression')])]
```

### Options:
`dataset`: `speaker1`, `multispeaker`, `music`

`upsample`: 2, 4, 6, 8

`model`: `dnn`

`phase`: `cheated`, `regression`

Then prepare the dataset

```
cd models/dnn/
python prepare_datasets.py
```
This will create the dataset in a path definted by `OUTPUT_DIR` and the parameters of the experiment

Then run the experiment

```
cd models/dnn/
python run_experiments.py
```

This will save a snapshot of the model in the same path as the data dir with the filename `model.snapshot`

## Generating Samples
```
cd models/dnn/
python generate_samples.py
```
This will create the samples in a path definted by `OUTPUT_DIR` and the parameters of the experiment
