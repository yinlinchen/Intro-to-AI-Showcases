#Code reference: https://www.tensorflow.org/tutorials/images/classification 

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import matplotlib.pyplot as plt


batch_size = 32
img_height = 180
img_width = 180

  
train_data = tf.keras.utils.image_dataset_from_directory(directory='dogSkinDisease/train',
                                                    label_mode='categorical',
                                                    seed=123,
                                                    image_size=(img_height, img_width),
                                                    batch_size=batch_size)

val_data = tf.keras.utils.image_dataset_from_directory(directory='dogSkinDisease/validation',
                                                            seed=123,
                                                            label_mode='categorical',
                                                            image_size=(img_height, img_width),
                                                            batch_size=batch_size)


num_classes = 4

model = Sequential([
  layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes, activation='softmax')
])



model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])


epochs = 30
history = model.fit(
  train_data,
  validation_data=val_data,
  epochs=epochs
)

model.save("SkinDisease.h5")

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


