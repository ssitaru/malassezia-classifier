#!/usr/bin/python 

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras.applications import Xception
from tensorflow.keras import layers
import argparse
from pathlib import Path
import pandas as pd
from vit_keras import vit

BATCH_SIZE=16

parser = argparse.ArgumentParser(description='Validate binary model, generating a .csv')
parser.add_argument("model", help="tf keras .model file")
parser.add_argument("datapath", help="path to test dataset (with two subdirs corresponding to the labels)")
parser.add_argument("-o", "--output", help="Path to output", default='out.csv')
parser.add_argument("-c", "--checkpoint", help="Use another checkpoint for mopdel weights")
args = parser.parse_args()

labels = [l.stem for l in Path(args.datapath).glob('*')]

print('loading model...')
mdl = tf.keras.models.load_model(args.model, safe_mode=False)
mdl.summary()
config = mdl.get_config()
input_size = config["layers"][0]["config"]["batch_input_shape"]

if args.checkpoint:
    mdl.load_weights(args.checkpoint)

test_ds_filepaths = tf.data.Dataset.list_files(str(args.datapath)+'/*/*', shuffle=False)
print('dataset info')
print('n =', tf.data.experimental.cardinality(test_ds_filepaths).numpy())

def get_label(file_path):
  # Convert the path to a list of path components
  parts = tf.strings.split(file_path, os.path.sep)
  # The second to last is the class-directory
  one_hot = parts[-2] == labels
  # Integer encode the label
  return tf.argmax(one_hot)

def decode_img(img):
  img = tf.io.read_file(img)
  img = tf.image.decode_jpeg(img, channels=3)
  # Resize the image to the desired size
  return tf.image.resize(img, [input_size[1], input_size[2]])

def process_path(file_path):
  label = get_label(file_path)
  # Load the raw data from the file as a string
  img = decode_img(file_path)
  return img, label

test_ds = test_ds_filepaths.map(process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)

output = pd.DataFrame(columns=['id', 'real', 'predicted', 'raw'])
raw_output = mdl.predict(test_ds.cache().batch(BATCH_SIZE))

output.loc[:,'id'] = [Path(str(e.numpy())).stem for e in test_ds_filepaths]
output.loc[:,'real'] = [labels[l.numpy()] for d,l in test_ds]
output.loc[:,'raw'] = [e[0] for e in list(raw_output)]
output.loc[output.raw < 0.5,'predicted'] = labels[0]
output.loc[output.raw >= 0.5,'predicted'] = labels[1]

output.to_csv(args.output, index=False)