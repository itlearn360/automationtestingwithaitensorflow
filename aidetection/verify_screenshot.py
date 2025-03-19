import cv2
import numpy as np
import os
import time
# Load Screenshot
screenshot_path = "dataset/form_screenshot.png"
img = cv2.imread(screenshot_path, cv2.IMREAD_GRAYSCALE)

# Apply Adaptive Thresholding (Highlights Small Elements like Checkboxes)
thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

# Apply Morphological Operations to Connect Edges
kernel = np.ones((3,3), np.uint8)
morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

# Find Contours (Detect Form Elements)
contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Create Directory for Cropped Images
cropped_dir = "dataset/cropped"
os.makedirs(cropped_dir, exist_ok=True)

# Iterate Through Contours & Crop Elements (Including Checkboxes)
cropped_elements = []
for i, contour in enumerate(contours):
    x, y, w, h = cv2.boundingRect(contour)
    
    # **Detect checkboxes & small elements (Allow very small boxes)**
    if 5 < w < 300 and 5 < h < 300:  
        cropped = img[y:y+h, x:x+w]
        cropped_path = os.path.join(cropped_dir, f"element_{i}.png")
        cv2.imwrite(cropped_path, cropped)
        cropped_elements.append(cropped_path)
        print(f"✅ Cropped Element Saved: {cropped_path}")

# Show Cropped Form Elements (For Debugging)
for cropped_path in cropped_elements:
    cropped_img = cv2.imread(cropped_path, cv2.IMREAD_GRAYSCALE)
    cv2.imshow(cropped_path, cropped_img)

cv2.waitKey(0)
cv2.destroyAllWindows()

import tensorflow as tf

# Load AI Model
model = tf.keras.models.load_model("form_detector.h5")

# Define Form Element Types
element_types = ["text_field", "email_field", "dropdown", "checkbox", "radio_button", "button"]

# Process Each Cropped Image for AI Prediction
detected_elements = []
for cropped_path in cropped_elements:
    img = cv2.imread(cropped_path, cv2.IMREAD_GRAYSCALE)
    
    # Resize for AI Model
    img_resized = cv2.resize(img, (64, 64))
    img_array = np.array(img_resized).reshape(-1, 64, 64, 1) / 255.0

    # AI Prediction
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    detected_element = element_types[predicted_class]

    detected_elements.append(detected_element)
    print(f"🔍 AI detected element: {detected_element}")

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import Select
import time

# Start Selenium WebDriver
driver = webdriver.Chrome()
driver.get("http://localhost/form_test.html")  # Update with your local form URL
time.sleep(3)

# Define Form Elements in Selenium
elements = {
    "text_field": driver.find_element(By.ID, "name"),
    "email_field": driver.find_element(By.ID, "email"),
    "dropdown": driver.find_element(By.ID, "course"),
    "checkbox": driver.find_element(By.NAME, "agree"),  # Use NAME instead of ID
    "radio_button": driver.find_element(By.ID, "intermediate"),
    "button": driver.find_element(By.ID, "submitBtn")
}

# **Prevent Multiple Entries**
used_elements = set()

for detected_element in detected_elements:
    if detected_element in used_elements:
        continue  # **Skip duplicate input**

    if detected_element == "text_field":
        elements["text_field"].send_keys("John Doe")

    elif detected_element == "email_field":
        elements["email_field"].send_keys("johndoe@example.com")

    elif detected_element == "dropdown":
        select = Select(elements["dropdown"])
        select.select_by_index(1)  # Select first option

    elif detected_element == "checkbox":
        checkboxes = driver.find_elements(By.NAME, "agree")  # Get all checkboxes
        for checkbox in checkboxes:
            if not checkbox.is_selected():
                driver.execute_script("arguments[0].click();", checkbox)  # Click each checkbox
                print("✅ Agreement Checkbox Selected!")

    elif detected_element == "radio_button":
        driver.execute_script("arguments[0].click();", elements["radio_button"])
        print("✅ Radio button clicked!")

    elif detected_element == "button":
        elements["button"].click()
        print("✅ Form Submitted!")

    # **Mark element as used**
    used_elements.add(detected_element)

# **STEP 6: CLOSE BROWSER AFTER COMPLETION**
time.sleep(5)
driver.quit()
print("✅ AI-based form submission completed dynamically!")
