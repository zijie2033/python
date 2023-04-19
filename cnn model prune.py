from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Flatten,Dense
from keras.datasets import mnist 
from keras.utils import np_utils 
import tensorflow as tf 
import tensorflow_model_optimization as tfmot
import numpy as np
(train_feature,train_label),(test_feature,test_label)=mnist.load_data()
train_images = train_feature/255
test_images = test_feature/255
test_labels = test_label
train_feature_vector = train_feature.reshape(len(train_feature),28,28,1).astype('float32')
test_feature_vector = test_feature.reshape(len(test_feature),28,28,1).astype('float32')
train_feature_normalize = train_feature_vector/255
test_feature_normalize = test_feature_vector/255
train_label_onehot = np_utils.to_categorical(train_label)
test_label_onehot = np_utils.to_categorical(test_label)
model = Sequential()
model.add(Conv2D(filters=10,kernel_size=(3,3),padding='same',
                    input_shape=(28,28,1),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(filters=20,kernel_size=(3,3),padding='same',
                    activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(units=256,activation='relu'))
model.add(Dense(units=10,activation='softmax'))

model.compile(loss='categorical_crossentropy',
                optimizer='adam',metrics=['accuracy'])
model.fit(x=train_feature_normalize,y=train_label_onehot,
            validation_split=0.2,epochs=10,batch_size=100,verbose=1)
print(model.weights[1])
scores = model.evaluate(test_feature_normalize,test_label_onehot)
print('\n準確率=',scores[1])
model.save('Mnist_cnn_model.h5')

prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
batch_size = 100
epochs = 10
validation_split = 0.1
num_images = train_images.shape[0] * (1- validation_split)
end_step = np.ceil(num_images / batch_size).astype(np.int32) * epochs 
pruning_params = {
      'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.50,
                                                              final_sparsity=0.80,
                                                              begin_step=0,
                                                              end_step=end_step)
}

model_for_pruning = prune_low_magnitude(model, **pruning_params)
model_for_pruning.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model_for_pruning.summary()
model_for_pruning.weights[1]
callbacks = [
  tfmot.sparsity.keras.UpdatePruningStep()
]
model_for_pruning.fit(train_images, train_label,
                  batch_size=batch_size, epochs=epochs, validation_split=validation_split,
                  callbacks=callbacks)
model_for_pruning = tfmot.sparsity.keras.strip_pruning(model_for_pruning)
prune_scores = model_for_pruning.evaluate(test_images,test_labels,verbose=0)
model_for_pruning.save('Mnist_cnn_model_pruned.h5')

print('\n準確率=',scores[1])
print('\n剪枝後準確率=',prune_scores[1])

