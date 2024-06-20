# Malassezia native microscopy classifier

These are the software for a binary classifier for native microscopy (tape strip) specimens for the presence of *Malassezia spp.* yeast.

## Training
A conda environment for running this software is provided in `environment.yaml`:
```
$ conda env create
$ conda activate malassezia-ai
```

Then, appropriate data in the format `{test,train}/{negative,positive}/{fn}.png` has to be provided in a folder (default `./data.split/`).

The networks can be trained using:
```
$ python train-{network}.py [-a] [-w] [-p DATA_PATH] runid
```
where `runid` is a string identifying this training run.

The training script creates checkpoints for every increase in validation accuracy.

## Inference

Inference can be run using `validate.py`:
```
$ python validate.py MODEL.keras DATA_PATH [-o CSV_OUT]
```

The output csv (default: `out.csv`) contains the columns `id,real,predicted,raw`.

Metrics on `out.csv` can be generated using `metrics.py`:
```
$ python metrics.py
```

## Parameters used for the model in the paper
For training:
```
$ python train-vit.py -a -w -p data.split
```

The network (layout and weights) can be downloaded from https://sebastian.sitaru.eu/projects/malassezia-classifier/vitb32_paper_model.keras.

Base network (backbone): `vit_keras.vit.vit_b32`

Classifier head: `Dense(256, activation="relu")` layer, followed by `Dense(1, activation="sigmoid")` 

Hyperparameters
- Data augmentation: `RandomCrop()` from 500x500 to 384x384px, `RandomBrightness(factor=0.3), RandomContrast(factor=0.3), RandomFlip()`
- Epochs: 200
- Batch size: 16
- Optimizer: Adam with default parameters

Dataset
- Training: 884 images (430 negative, 457 positive)
- Test: 222 images (109 negative, 115 positive)

Performance
```
      class     metric      true     lower     upper  sample_size
0  negative  precision  0.943925  0.896552  0.982456          108
1  negative     recall  0.935185  0.885714  0.978261          108
2  negative   f1_score  0.939535  0.905263  0.970000          108
3  positive  precision  0.939130  0.894737  0.981132          114
4  positive     recall  0.947368  0.900901  0.983333          114
5  positive   f1_score  0.943231  0.910798  0.972332          114
```