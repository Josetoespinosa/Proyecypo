import tensorflow as tf
import numpy as np

class_mapping = {'Basketball': 0, 'Soccer': 1, 'Rugby': 2, 'Table_tennis': 3, 'Tennis': 4, 'Volleyball': 5}
inverse_class_mapping = {v: k for k, v in class_mapping.items()}

model = tf.keras.models.load_model('ball_classification_model.keras', compile=False)

image = tf.keras.preprocessing.image.load_img("archive/predict/x/image5.jpg", target_size=(100,100))  
image_array = tf.keras.preprocessing.image.img_to_array(image) / 255.0  
image_array = np.expand_dims(image_array, axis=0)

x = model.predict(image_array, verbose=2)
predicted_label = np.argmax(x)
#print(x)

predicted_label_name = inverse_class_mapping[predicted_label]

print(predicted_label_name)