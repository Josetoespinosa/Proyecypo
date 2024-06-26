import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img


def different_quant(interpreter):
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    test_images_folder = "archive/test" 

    correct_predictions = 0
    total_images = 0

    class_mapping = {'Basketball': 0, 'Soccer': 1, 'Rugby': 2, 'Table_tennis': 3, 'Tennis': 4, 'Volleyball': 5}
    for class_folder in os.listdir(test_images_folder):
        class_folder_path = os.path.join(test_images_folder, class_folder)
        if not os.path.isdir(class_folder_path):
            continue

        true_label = class_mapping.get(class_folder)
        if true_label is None:
            continue 

        for image_filename in os.listdir(class_folder_path):
            image_path = os.path.join(class_folder_path, image_filename)
            image = load_img(image_path, target_size=(96,96), color_mode='grayscale')  
            image_array = img_to_array(image) / 255.0  
            image_array = np.expand_dims(image_array, axis=0)  

            interpreter.set_tensor(input_details[0]['index'], image_array)

            interpreter.invoke()

            output = interpreter.get_tensor(output_details[0]['index'])

            predicted_label = np.argmax(output)

            if predicted_label == true_label:
                correct_predictions += 1
                
            total_images += 1
    
    print("Total images: ", total_images)
    print("Correct predictions: ", correct_predictions)

    accuracy = correct_predictions / total_images * 100
    print("Accuracy: {:.2f}%".format(accuracy))
    
    


print("Using Post Training Quantization: ")
different_quant(tf.lite.Interpreter(model_path="model_quant_ball_clasification6.tflite"))
print()
print("Using Quantization Aware Training:")
different_quant(tf.lite.Interpreter(model_path="Good_Models/q_aware_ball_classification_model.tflite"))

    