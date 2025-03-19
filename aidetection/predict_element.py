import tensorflow as tf
import numpy as np
import cv2
import os

# Load trained AI model
model = tf.keras.models.load_model("form_detector.h5")

# Define categories
element_types = ["text_field", "email_field", "dropdown", "checkbox", "radio_button", "button"]

# Load images from dataset to test
test_images = [
    "dataset/text_field/text_field_1.png",
    "dataset/email_field/email_field_2.png",
    "dataset/dropdown/dropdown_3.png",
    "dataset/checkbox/checkbox_4.png",
    "dataset/radio_button/radio_button_5.png",
    "dataset/button/button_6.png"
]

# Predict for each test image
for img_path in test_images:
    if not os.path.exists(img_path):
        print(f"⚠️ Skipping missing file: {img_path}")
        continue
    
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img, (64, 64))
    img_array = np.array(img_resized).reshape(-1, 64, 64, 1) / 255.0

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)

    print(f"🔍 Image: {img_path} → Predicted: {element_types[predicted_class]}")
