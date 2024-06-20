#!/usr/bin/python 

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
from vit_keras import vit
import argparse

parser = argparse.ArgumentParser(description='Convert keras model to tfjs format')
parser.add_argument("model", help="tf keras .model file")
parser.add_argument("-o", "--output", help="Path to output", required=True)
args = parser.parse_args()

print('loading model...')
mdl = tf.keras.models.load_model(args.model, safe_mode=False)
mdl.summary()

mdl.save(args.output)