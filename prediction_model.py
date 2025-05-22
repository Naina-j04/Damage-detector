import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import json

# Config
MODEL_PATH = r"C:\Users\naina\Resume projects\s24_dataset\models\simple_damage_classifier.keras"
CLASS_INDEX_PATH = r"C:\Users\naina\Resume projects\s24_dataset\models\class_indices.json"
IMG_SIZE = (150, 150)

# Load model and class index
model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASS_INDEX_PATH, "r") as f:
    class_indices = json.load(f)

# Map 0 or 1 back to class name
index_to_class = {v: k for k, v in class_indices.items()}

def predict_image(img_path):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]
    predicted_label = 1 if prediction >= 0.5 else 0
    class_name = index_to_class[predicted_label]

    # Confidence calculation
    confidence = prediction if predicted_label == 1 else 1 - prediction

    # ✅ Add this confidence threshold check
    if confidence < 0.65:
        return "uncertain", confidence

    return class_name, confidence
