#!/usr/bin/python

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import argparse
from tensorflow.keras.applications import Xception
from tensorflow.keras import layers
from glob import glob
import datetime

IMG_LOAD_SIZE=500
IMG_SIZE=299
BATCH_SIZE=16

parser = argparse.ArgumentParser(description='Train XCeption')
parser.add_argument("runid", help="run ID")
parser.add_argument("-p", "--data-path", help="Path to data e.g. images/{test,train}", required=True)
parser.add_argument("-w", "--init-weights", action='store_true', default=False, help="Init weights from ImageNet")
parser.add_argument("-a", "--data-augmentation", action='store_true', default=False, help="Use data augmentation")
parser.add_argument("-e", "--epochs", default=80, help="Epochs")
parser.add_argument("-l", "--learning-rate", default=2e-5, help="Learning rate")
args = parser.parse_args()
dp = args.data_path

# Convert Images to tensors suitable for the model
TRAINDATA = keras.utils.image_dataset_from_directory('./'+dp+'/train/', batch_size=BATCH_SIZE, image_size=(IMG_LOAD_SIZE, IMG_LOAD_SIZE), shuffle=True, label_mode='binary')
TESTDATA = keras.utils.image_dataset_from_directory('./'+dp+'/test/', batch_size=BATCH_SIZE, image_size=(IMG_LOAD_SIZE, IMG_LOAD_SIZE), shuffle=True, label_mode='binary')
nTrain = len(list(glob('./'+dp+'/train/*/*.jpg')))
nTest = len(list(glob('./'+dp+'/test/*/*.jpg')))

# Setting up model
input = keras.Input(shape=(IMG_LOAD_SIZE, IMG_LOAD_SIZE, 3))
x = tf.cast(input, tf.float32)
x = tf.keras.layers.RandomCrop(IMG_SIZE, IMG_SIZE)(x)
if args.data_augmentation:
      l_dataAugmentation = tf.keras.Sequential([
           tf.keras.layers.RandomBrightness(factor=0.4),
           tf.keras.layers.RandomFlip(),
           ])
      x = l_dataAugmentation(x)
x = tf.keras.applications.xception.preprocess_input(x)
if args.init_weights:  
    core = Xception(include_top=False, weights='imagenet')
else:
    core = Xception(include_top=False, weights=None)
x = core(x)
l_flatten = layers.Flatten()
x = l_flatten(x)
#x = layers.Dropout(0.2)(x)
l_dense1 = layers.Dense(256,  activation='relu')
x = l_dense1(x)
l_dense2 = layers.Dense(1, activation="sigmoid")
x = l_dense2(x)
model = tf.keras.Model(inputs=input, outputs=x)

model.compile(optimizer=keras.optimizers.Adam(),
    loss=keras.losses.BinaryCrossentropy(),
    metrics=["accuracy"])

model.summary()

print('HYPERPARAMETERS')
print('===============')
print('image load size:', IMG_LOAD_SIZE)
print('image size:', IMG_SIZE)
print('batch size:', BATCH_SIZE)
print('data augmentation:', args.data_augmentation)
print('use ImageNet weights:', args.init_weights)
print('steps per epoch:', int(nTrain/BATCH_SIZE))
print('epochs:', args.epochs)
print('validation steps:', int(nTest/BATCH_SIZE))
print('===============')

now = datetime.datetime.now()
filepath= './xception_{runid}_{iso}'.format(runid=args.runid,iso=now.strftime("%Y%m%d_%H%M%S"))
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_accuracy', mode='max', save_best_only=True)
callbacks_list = [checkpoint]

hist = model.fit(TRAINDATA.repeat(),
                         steps_per_epoch = int(nTrain/BATCH_SIZE),
                         epochs = int(args.epochs),
                         validation_data = TESTDATA.repeat(),
                         validation_steps = int(nTest/BATCH_SIZE),
                         callbacks=callbacks_list)

print(hist.history)

eval = model.evaluate(TESTDATA)
print(eval)

model.save(filepath+'.keras')
