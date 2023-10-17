import tempfile
import os
import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras import layers
mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()


train_images = train_images.astype('float32')/ 255
test_images = test_images.astype('float32')/ 255

train_images =np.expand_dims(train_images,-1)
test_images =np.expand_dims(test_images,-1)

train_labels = keras.utils.to_categorical(train_labels,10)
test_labels = keras.utils.to_categorical(test_labels,10)

batch_size = 200
epochs = 10
validation_split = 0.2

model = keras.Sequential([
    layers.Input(shape=(28,28,1)),
    layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(10, activation="softmax"),
])

model.compile(loss="categorical_crossentropy",
                       optimizer="adam", metrics=["accuracy"])

model.fit(
  train_images,
  train_labels,
  batch_size=batch_size,
  epochs=epochs,
  validation_split=validation_split,
)

_, baseline_model_accuracy = model.evaluate(
    test_images, test_labels, verbose=0)

print('Baseline test accuracy:', baseline_model_accuracy)
model.save("base_model.h5")

import tensorflow_model_optimization as tfmot

prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

num_images = train_images.shape[0] * (1 - validation_split)
end_step = np.ceil(num_images / batch_size).astype(np.int32) * epochs

pruning_params = {
      'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.50,
                                                               final_sparsity=0.80,
                                                               begin_step=0,
                                                               end_step=end_step)
}

model_for_pruning = prune_low_magnitude(model, **pruning_params)


model_for_pruning.compile(optimizer='adam',
                          loss="categorical_crossentropy",
                          metrics=['accuracy'])

model_for_pruning.summary()

logdir = tempfile.mkdtemp()

callbacks = [
  tfmot.sparsity.keras.UpdatePruningStep(),
  tfmot.sparsity.keras.PruningSummaries(log_dir=logdir),
]

model_for_pruning.fit(train_images, train_labels,
                  batch_size=batch_size, epochs=epochs, validation_split=validation_split,
                  callbacks=callbacks)

_, model_for_pruning_accuracy = model_for_pruning.evaluate(
   test_images, test_labels, verbose=0)

model_for_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)

model_for_export.summary()

model_for_export.save("pruned_model.h5")

converter = tf.lite.TFLiteConverter.from_keras_model(model_for_export)

pruned_tflite_model = converter.convert()

with open('pruned_model.tflite', 'wb') as f:
  f.write(pruned_tflite_model)

def representative_data_gen():
  for input_value in tf.data.Dataset.from_tensor_slices(train_images).batch(1).take(100):
    yield [input_value]

converter = tf.lite.TFLiteConverter.from_keras_model(model_for_export)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
# Ensure that if any ops can't be quantized, the converter throws an error
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# Set the input and output tensors to uint8 (APIs added in r2.3)
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

quantized_and_pruned_tflite_model = converter.convert()

with open('quantized_pruned_model.tflite', 'wb') as f:
  f.write(quantized_and_pruned_tflite_model)


print('Baseline test accuracy:', baseline_model_accuracy) 
print('Pruned test accuracy:', model_for_pruning_accuracy)
print('Baseline model (h5) size:', os.path.getsize('base_model.h5'),'Bytes')
print('Pruned model (h5) size:', os.path.getsize('pruned_model.h5'),'Bytes')
print('Pruned model (tflite) size:', os.path.getsize('pruned_model.tflite'),'Bytes')
print('Quantized and Pruned model (tflite) size:', os.path.getsize('quantized_pruned_model.tflite'),'Bytes')






