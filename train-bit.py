import tensorflow
from tensorflow import keras
import numpy as np
from keras.applications.xception import Xception
from keras.preprocessing.image import ImageDataGenerator
import argparse
import tensorflow_hub as hub

bit_model_url = "https://tfhub.dev/google/bit/m-r50x1/1"
bit_module = hub.KerasLayer(bit_model_url)


parser = argparse.ArgumentParser(description='Train BiT')
parser.add_argument("runid", help="run ID")
parser.add_argument("-p", "--data-path", help="Path to data e.g. images/{test,train}", required=True)
parser.add_argument("-w", "--no-init-weights", action='store_true', default=False, help="No init weights from ImageNet")
parser.add_argument("-a", "--data-augmentation", action='store_true', default=False, help="Use data augmentation")
parser.add_argument("-s", "--steps-per-epoch", default=80, help="Steps per epoch")
parser.add_argument("-e", "--epochs", default=80, help="Epochs")
parser.add_argument("-l", "--learning-rate", default=2e-5, help="Learning rate")
args = parser.parse_args()
dp = args.data_path

# Convert Images to tensors suitable for the model
if args.data_augmentation:
      TRDATA = ImageDataGenerator(
            zoom_range=0.15,
            horizontal_flip=True,
            vertical_flip=True)
else:
      TRDATA = ImageDataGenerator()
TRAINDATA = TRDATA.flow_from_directory(directory='./'+dp+'/train/',target_size=(224,224))
TSDATA = ImageDataGenerator()
TESTDATA = TSDATA.flow_from_directory(directory='./'+dp+'/test/', target_size=(224,224))

#Setting up model
class MyBiTModel(keras.Model):
    def __init__(self, num_classes, module, **kwargs):
        super().__init__(**kwargs)

        self.num_classes = num_classes
        self.head = keras.layers.Dense(num_classes, kernel_initializer="zeros")
        self.bit_model = module

    def call(self, images):
        bit_embedding = self.bit_model(images)
        return self.head(bit_embedding)


model = MyBiTModel(num_classes=2, module=bit_module)
# Compiling model for usage
model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=args.learning_rate),
      loss=keras.losses.BinaryCrossentropy(),
      metrics=["accuracy"])
model.build((None, 224, 224, 3))
model.summary()

train_callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=2, restore_best_weights=True
    )
]


hist = model.fit(TRAINDATA,
                         steps_per_epoch = int(args.steps_per_epoch),
                         epochs = int(args.epochs),
                         validation_data = TESTDATA,
                         validation_steps = 30,
                         validation_freq=1,
                         callbacks=train_callbacks)

print(hist.history)

eval = model.evaluate(TESTDATA)
print(eval)

model.save('./BiT_{runid}.h5'.format(runid=args.runid,))