import os
import zipfile
import sys
import tensorflow as tf
import matplotlib.pyplot as plt
import pathlib
import tensorflow_hub as hub
from tqdm import tqdm
import numpy as np
import tensorflow_model_optimization as tfmot
from keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dropout, BatchNormalization, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Sequential
import cv2


if tf.__version__.split('.')[0] != '2':
    raise Exception((f"The script is developed and tested for tensorflow 2. "
                     f"Current version: {tf.__version__}"))

if sys.version_info.major < 3:
    raise Exception((f"The script is developed and tested for Python 3. "
                     f"Current version: {sys.version_info.major}"))


train_dir = 'archive/train'
validation_dir = 'archive/test'

def to_grayscale(image):
    if len(image.shape) == 3 and image.shape[2] == 3:  # Check if the image has 3 channels
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image  # If the image is already grayscale, return it as is

# Use ImageDataGenerator to load images from the directories
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    preprocessing_function=to_grayscale)

train_generator = train_datagen.flow_from_directory(
        'archive/train',
        target_size=(96,96),
        batch_size=32,
        color_mode='grayscale',  
        class_mode='categorical')


validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    preprocessing_function=to_grayscale)



early_stop = EarlyStopping(monitor='val_loss', patience=5)

validation_generator = validation_datagen.flow_from_directory(
        'archive/test',
        target_size=(96,96),
        batch_size=32,
        color_mode='grayscale',  # This is important. It tells keras to load images in grayscale
        class_mode='categorical')




model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(96, 96, 1)), 
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(256, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(6, activation='softmax')
])

# Modelo 10MB, <6414628 bytes
# loss: 0.7376 - accuracy: 0.7272 - val_loss: 1.0521 - val_accuracy: 0.6264 

# Modelo 5MB,
# loss: 0.8846 - accuracy: 0.6685 - val_loss: 1.0887 - val_accuracy: 0.6005

# Modelo 6MB,
# loss: 0.5839 - accuracy: 0.7798 - val_loss: 1.0983 - val_accuracy: 0.6535

# model = tf.keras.models.Sequential([
#     Conv2D(32, (3,3), activation='relu', input_shape=(100, 100, 3)),
#     Conv2D(64, (3,3), activation='relu'),
#     Conv2D(32, (3,3), activation='relu'),
#     Conv2D(16, (3,3), activation='relu'),
#     Conv2D(6, (3,3), activation='relu'),
# ])

print(model.summary())

#quantize_model = tfmot.quantization.keras.quantize_model

# `quantize_model` requiere un modelo que aún no ha sido compilado
#q_aware_model = quantize_model(model)

model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.001),
              metrics=['accuracy'])

# q_aware_model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.0001),
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])

train_steps = train_generator.samples // train_generator.batch_size
validation_steps = validation_generator.samples // validation_generator.batch_size

lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)

history = model.fit(train_generator,
    validation_data=validation_generator,
    steps_per_epoch=train_steps,
    epochs=85,
    validation_steps=validation_steps,
    verbose=1,
    callbacks=[lr_scheduler])

model.save("ball_classification_model.keras")

export_dir = "saved_model/model1"
model.export(export_dir)

""" SE HACE POST-TRAINING QUANTIZATION """

# Convertir a tflite
# converter = tf.lite.TFLiteConverter.from_saved_model(export_dir) # Busca en el directorio el modelo que se va a convertir a tflite
# converter.optimizations = [tf.lite.Optimize.DEFAULT] # Cuantizar

# def representative_data_gen(): # Cuantizar con data representativa
#     for i in range(100):
#         # Obtener un lote de datos
#         input_value, _ = next(validation_generator)
#         yield [input_value]


# converter.representative_dataset = representative_data_gen
# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

# tflite_model = converter.convert() # Finalmente se convierte
# tflite_model_file = pathlib.Path("saved_model/model_original.tflite") # Donde se guarda
# tflite_model_file.write_bytes(tflite_model) # Ns q wea

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.xlim([0,60])
plt.ylim([0.0,1.0])
plt.show()
#q_aware_model.fit(train_generator,batch_size=32, epochs=25, validation_split=0.1, verbose=1,validation_data=validation_generator)


# q_aware_model.save("q_aware_ball_classification_model.keras")
# export_dir = "saved_model1/model1"
# q_aware_model.export(export_dir)

# loaded_model = tf.lite.TFLiteConverter.from_saved_model(export_dir)

# tflite_model_file = pathlib.Path("saved_model1/model_original.tflite") # Donde se guarda
# tflite_model_file.write_bytes(loaded_model) # Se escribe el archivo

# tflite_model_file = 'saved_model/model_original.tflite'
# interpreter = tf.lite.Interpreter(model_path=tflite_model_file)
# interpreter.allocate_tensors()

# loaded_model = tf.keras.models.load_model("q_aware_ball_classification_model.keras")
# loaded_model.summary()
# converter = tf.lite.TFLiteConverter.from_keras_model(q_aware_model)
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# quantized_tflite_model = converter.convert()

# with open('q_aware_ball_classification_model.tflite', 'wb') as f:
#     f.write(quantized_tflite_model)





