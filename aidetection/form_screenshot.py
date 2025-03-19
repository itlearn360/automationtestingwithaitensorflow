from selenium import webdriver
import time
import os

# Ensure dataset directory exists
dataset_dir = "dataset"
os.makedirs(dataset_dir, exist_ok=True)

# Start Selenium WebDriver
driver = webdriver.Chrome()
driver.get("http://localhost/form_test.html")  # Update with your local form URL

# Wait for page to load
time.sleep(3)

# Capture Full Page Screenshot
screenshot_path = os.path.join(dataset_dir, "form_screenshot.png")
driver.save_screenshot(screenshot_path)
print(f"📸 Screenshot saved: {screenshot_path}")

driver.quit()
