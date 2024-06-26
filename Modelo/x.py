import os
import zipfile
import sys
import tensorflow as tf
import matplotlib.pyplot as plt
import pathlib
import tensorflow_hub as hub
from tqdm import tqdm
import numpy as np
import cv2


def to_grayscale(image):
    if len(image.shape) == 3 and image.shape[2] == 3:  # Check if the image has 3 channels
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image  # If the image is already grayscale, return it as is

validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    preprocessing_function=to_grayscale)

validation_generator = validation_datagen.flow_from_directory(
        'archive/test',
        target_size=(96,96),
        batch_size=32,
        color_mode='grayscale',  # This is important. It tells keras to load images in grayscale
        class_mode='categorical')

def representative_dataset():
  num_samples = min(500, len(validation_generator) - 1)
  for i in range(num_samples):
    input_data, _ = validation_generator[i]
    yield [input_data.astype('float32')]

converter = tf.lite.TFLiteConverter.from_saved_model("saved_model/model1")
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
converter.representative_dataset = representative_dataset
tflite_model_quant = converter.convert()

tflite_model_file_quant = pathlib.Path('model_quant_ball_clasification6.tflite')
tflite_model_file_quant.write_bytes(tflite_model_quant)

interpreter = tf.lite.Interpreter(model_content=tflite_model_quant)
interpreter.allocate_tensors()

# Obtenemos dimensiones del interpreter
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print(input_details)
print(output_details)




