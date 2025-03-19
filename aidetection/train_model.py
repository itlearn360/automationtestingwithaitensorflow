import tensorflow as tf
import numpy as np
import cv2
import os
import random

# Define categories for form elements
categories = ["text_field", "email_field", "dropdown", "checkbox", "radio_button", "button"]
IMG_SIZE = 64  # Image size for model training
data_dir = "dataset/"

# Prepare dataset
training_data = []
labels = []

for category in categories:
    path = os.path.join(data_dir, category)
    if not os.path.exists(path):
        print(f"⚠️ Skipping missing folder: {path}")
        continue

    class_num = categories.index(category)

    images = os.listdir(path)
    random.shuffle(images)  # Shuffle dataset for better learning

    for img_name in images:
        img_path = os.path.join(path, img_name)

        # Ensure file is an image
        if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            print(f"⚠️ Skipping non-image file: {img_path}")
            continue

        img_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        # Skip empty images
        if img_array is None or img_array.size == 0:
            print(f"❌ Skipping corrupted image: {img_path}")
            continue

        resized_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
        
        # Data Augmentation
        flipped = cv2.flip(resized_array, 1)  # Horizontal flip
        blurred = cv2.GaussianBlur(resized_array, (3, 3), 0)  # Blur
        rotated = cv2.rotate(resized_array, cv2.ROTATE_90_CLOCKWISE)  # Rotate 90 degrees
        
        training_data.append(resized_array)
        labels.append(class_num)
        
        training_data.append(flipped)
        labels.append(class_num)
        
        training_data.append(blurred)
        labels.append(class_num)

        training_data.append(rotated)
        labels.append(class_num)

print(f"✅ Successfully loaded {len(training_data)} valid images for training.")

# Convert to NumPy arrays
if len(training_data) == 0:
    raise ValueError("No valid training data found! Please check dataset.")

X = np.array(training_data).reshape(-1, IMG_SIZE, IMG_SIZE, 1) / 255.0
y = np.array(labels)

# Define AI Model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(len(categories), activation='softmax')
])

# Compile and Train Model
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Train model with validation split to prevent overfitting
if len(X) > 0:
    model.fit(X, y, epochs=25, batch_size=16, validation_split=0.2)
    model.save("form_detector.h5")
    print("🎯 AI Model Trained & Saved!")
else:
    print("❌ No valid images found. Please capture new images and retry.")
