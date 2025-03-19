import cv2
import numpy as np
import os
import time
from selenium import webdriver

# **STEP 1: Start Selenium WebDriver & Capture Screenshot**
driver = webdriver.Chrome()
driver.get("https://training.qaonlinetraining.com/testPage.php")
time.sleep(3)

# **Capture Screenshot**
screenshot_path = "dataset/form_screenshot.png"
driver.save_screenshot(screenshot_path)
print(f"📸 Screenshot saved: {screenshot_path}")

# **STEP 2: Load Screenshot for AI Processing**
img = cv2.imread(screenshot_path, cv2.IMREAD_GRAYSCALE)

# **Apply Image Preprocessing**
thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

# **Find Contours (Form Elements)**
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# **Create Directory for Cropped Elements**
cropped_dir = "dataset/cropped"
os.makedirs(cropped_dir, exist_ok=True)

# **Crop Form Elements Dynamically**
cropped_elements = []
for i, contour in enumerate(contours):
    x, y, w, h = cv2.boundingRect(contour)
    
    if 10 < w < 400 and 10 < h < 300:  # **Detect Small & Large Form Elements**
        cropped = img[y:y+h, x:x+w]
        cropped_path = os.path.join(cropped_dir, f"element_{i}.png")
        cv2.imwrite(cropped_path, cropped)
        cropped_elements.append(cropped_path)
        print(f"✅ Cropped Element Saved: {cropped_path}")

driver.quit()

import tensorflow as tf

# **STEP 1: Load Trained AI Model**
model = tf.keras.models.load_model("form_detector.h5")

# **Define Categories**
element_types = ["text_field", "email_field", "dropdown", "checkbox", "radio_button", "button"]

# **STEP 2: Process Each Cropped Image for AI Prediction**
detected_elements = []
for cropped_path in cropped_elements:
    img = cv2.imread(cropped_path, cv2.IMREAD_GRAYSCALE)
    
    # **Resize Image for AI Model**
    img_resized = cv2.resize(img, (64, 64))
    img_array = np.array(img_resized).reshape(-1, 64, 64, 1) / 255.0

    # **AI Prediction**
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    detected_element = element_types[predicted_class]

    detected_elements.append(detected_element)
    print(f"🔍 AI detected element: {detected_element}")

from selenium.webdriver.common.by import By
import time

# **STEP 1: Start WebDriver**
driver = webdriver.Chrome()
driver.get("https://training.qaonlinetraining.com/testPage.php")
time.sleep(3)

# **Find All Input Elements Dynamically (Using AI Detection)**
all_inputs = driver.find_elements(By.TAG_NAME, "input")  # Gets all input elements

# **Track Used Elements to Prevent Duplicates**
used_elements = set()

# **STEP 2: Match AI Predictions with Real HTML Elements**
for detected_element in detected_elements:
    if detected_element in used_elements:
        continue  # **Skip duplicate input**
    
    for input_element in all_inputs:
        input_type = input_element.get_attribute("type")  # Get input type

        # **Text Fields**
        if detected_element == "text_field" and input_type == "text":
            input_element.send_keys("John Doe")

        # **Email Fields**
        elif detected_element == "email_field" and input_type == "email":
            input_element.send_keys("johndoe@example.com")

        # **Checkboxes**
        elif detected_element == "checkbox" and input_type == "checkbox":
            if not input_element.is_selected():
                driver.execute_script("arguments[0].click();", input_element)

        # **Radio Buttons**
        elif detected_element == "radio_button" and input_type == "radio":
            driver.execute_script("arguments[0].click();", input_element)

    # **Mark element as used**
    used_elements.add(detected_element)

# **STEP 3: Submit Form Using Correct Input Type**
submit_buttons = driver.find_elements(By.TAG_NAME, "input")  # Get all inputs
submit_clicked = False  # Track submit button click

for button in submit_buttons:
    if button.get_attribute("type") == "submit" and not submit_clicked:
        button.click()
        print("✅ Form Submitted!")
        submit_clicked = True  # Prevent duplicate submissions

# **STEP 4: Close Browser**
time.sleep(3)
driver.quit()
print("✅ AI-based form submission completed for testPage.php!")
