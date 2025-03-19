from selenium import webdriver
from selenium.webdriver.common.by import By
import time
import os

# Define dataset directory
dataset_dir = "dataset"

# Define categories (subfolders)
categories = ["text_field", "email_field", "dropdown", "checkbox", "radio_button", "button"]

# Ensure each category folder exists
for category in categories:
    os.makedirs(os.path.join(dataset_dir, category), exist_ok=True)

# Start Selenium WebDriver
driver = webdriver.Chrome()
driver.get("http://localhost/form_test.html")  # Update with your local form URL

# Wait for page to load
time.sleep(3)

# Define form elements with locators
elements = {
    "text_field": driver.find_element(By.ID, "name"),
    "email_field": driver.find_element(By.ID, "email"),
    "dropdown": driver.find_element(By.ID, "course"),
    "checkbox": driver.find_element(By.ID, "skill1"),
    "radio_button": driver.find_element(By.ID, "intermediate"),
    "button": driver.find_element(By.ID, "submitBtn")
}

# Capture multiple screenshots for each element with sequential numbering
for i in range(50):  # Capture 50 images per element
    for element_name, element in elements.items():
        try:
            element_path = os.path.join(dataset_dir, element_name, f"{element_name}_{i}.png")
            element.screenshot(element_path)
            print(f"📸 Saved: {element_path}")
            time.sleep(0.5)  # Small delay to avoid duplicate images
        except Exception as e:
            print(f"❌ Failed to capture {element_name}: {e}")

# Close browser
driver.quit()
print("✅ Images captured successfully without timestamps!")
