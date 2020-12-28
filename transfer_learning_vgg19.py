# NOTE: comment this if train using GPU
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from preprocess import load_batch, load_batch_bootstrap, load_validation
import tensorflow as tf

import keras
from keras.applications.vgg19 import VGG19
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.losses import CategoricalCrossentropy
from keras.metrics import CategoricalAccuracy
from keras.optimizers import Adam
from tensorflow.python.keras.optimizer_v2.learning_rate_schedule import ExponentialDecay

from matplotlib import pyplot as plt

# NOTE: uncomment this if train using GPU
# physical_devices = tf.config.list_physical_devices('GPU') 
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Hyperparameters:
epochs = 10
n_batches = 5
batch_size = 32
preprocess_batch_path = 'Preprocess_batch'
trained_model_path = 'Trained_model' # Please set this to a different value to create different folder for different model
random_state = None # Please set this to a number if train using bootstrap

# Load model without classifier layers
base_model = VGG19(
    weights = 'imagenet',
    input_shape = (224, 224, 3),
    include_top = False)

base_model.trainable = False

inputs = keras.Input(shape = (224, 224, 3), dtype=tf.float32)
x = base_model(inputs)
x = GlobalAveragePooling2D()(x)
outputs = Dense(7)(x)
model = Model(inputs, outputs)

lr_schedule = ExponentialDecay(
    initial_learning_rate=1e-3,
    decay_steps=1000,
    decay_rate=0.9)
optimizer = Adam(learning_rate=lr_schedule)

model.compile(optimizer=optimizer,
            loss=CategoricalCrossentropy(from_logits=True),
            metrics=[CategoricalAccuracy()])

history = model.fit(load_batch_bootstrap(preprocess_batch_path, n_batches, batch_size, random_state),
                    epochs=epochs,
                    steps_per_epoch=197*n_batches, 
                    validation_data=load_validation(preprocess_batch_path, batch_size), validation_steps=109)

model.save("Trained_model")

print(model.summary())
plt.figure(1)

# summarize history for accuracy

plt.subplot(211)
plt.plot(history.history['categorical_accuracy'])
plt.plot(history.history['val_categorical_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='lower right')

# summarize history for loss

plt.subplot(212)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='upper right')

plt.tight_layout()

plt.show()