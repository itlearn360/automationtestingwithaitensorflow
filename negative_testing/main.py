from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select
from selenium.common.exceptions import NoSuchElementException
import time
import tensorflow as tf
import numpy as np

# Load or define an AI model for anomaly detection
def load_ai_model(model_path):
    try:
        return tf.keras.models.load_model(model_path)
    except Exception as e:
        print(f"Error loading AI model: {e}")
        return None

# Use AI to validate input data
def validate_data_with_ai(ai_model, data):
    # Content-based validation
    name_valid = 1 if data.get("name") else 0
    email_valid = 1 if data.get("email") and "@" in data["email"] and "." in data["email"] else 0
    website_valid = 1 if data.get("website") and (data["website"].startswith("http://") or data["website"].startswith("https://")) else 0
    comment_valid = 1 if data.get("comment") else 0
    gender_valid = 1 if data.get("gender") in ["female", "male", "Other"] else 0
    country_valid = 1 if data.get("country") in ["USA", "Canada", "UK"] else 0  # Example valid countries

    # Prepare the input for the AI model
    input_data = np.array([[name_valid, email_valid, website_valid, comment_valid, gender_valid, country_valid]])
    prediction = ai_model.predict(input_data)
    print(f"Debug: Input Data: {input_data}, Prediction: {prediction[0][0]}")
    return "error" if prediction[0][0] < 0.5 else "success"

def safe_find_element(driver, by, value, description):
    try:
        element = driver.find_element(by, value)
        return element
    except NoSuchElementException:
        print(f"Locator not found for: {description} ({value})")
        return None

# Function to automate testing with AI integration
def automate_test_case_with_ai(data, expected_result, driver_path, ai_model):
    driver = webdriver.Chrome()
    driver.get("https://training.qaonlinetraining.com/testPage.php")
    actual_result = ""

    try:
        # Validate input data using AI model
        ai_validation_result = validate_data_with_ai(ai_model, data)
        print(f"AI Validation Result: {ai_validation_result}")
        
        if ai_validation_result == "error":
            print("AI flagged this input as invalid. Skipping execution.")
            return

        # (Original Selenium automation process...)
        # Fill 'Name' field
        if data.get("name") is not None:
            name_field = safe_find_element(driver, By.NAME, "name", "Name field")
            if name_field:
                name_field.clear()
                name_field.send_keys(data["name"])

        # Fill 'E-mail' field
        if data.get("email") is not None:
            email_field = safe_find_element(driver, By.NAME, "email", "Email field")
            if email_field:
                email_field.clear()
                email_field.send_keys(data["email"])

        # Fill 'Website' field
        if data.get("website"):
            website_field = safe_find_element(driver, By.NAME, "website", "Website field")
            if website_field:
                website_field.clear()
                website_field.send_keys(data["website"])

        # Fill 'Comment' field
        if data.get("comment"):
            comment_field = safe_find_element(driver, By.NAME, "comment", "Comment field")
            if comment_field:
                comment_field.clear()
                comment_field.send_keys(data["comment"])

        # Select 'Gender'
        if data.get("gender") in ["female", "male", "Other"]:
            gender_radio = safe_find_element(driver, By.XPATH, f"//input[@name='gender'][@value='{data['gender']}']", "Gender radio button")
            if gender_radio:
                gender_radio.click()
            else:
                print(f"Gender option '{data['gender']}' is not available. Skipping gender selection.")


        # Select 'Country'
        if data.get("country"):
            country_select_element = safe_find_element(driver, By.NAME, "country", "Country dropdown")
            if country_select_element:
                select = Select(country_select_element)
                try:
                    select.select_by_visible_text(data["country"])
                except Exception:
                    print(f"Invalid country: '{data['country']}' is not available in the dropdown.")
            else:
                print("Country dropdown is missing. Skipping country selection.")


        # Submit the form
        submit_button = safe_find_element(driver, By.NAME, "submit", "Submit button")
        if submit_button:
            submit_button.click()
            time.sleep(2)  # Wait for the page to load or update
            if "is required" in driver.page_source.lower() or "invalid" in driver.page_source.lower():
                actual_result = "error"
            else:
                actual_result = "success"

    except Exception as e:
        actual_result = "error"
        print(f"Exception occurred during test execution: {e}")
    finally:
        driver.quit()

    # Compare actual and expected results
    if actual_result == expected_result:
        print(f"Test Passed: Expected '{expected_result}', got '{actual_result}'.")
    else:
        print(f"Test Failed: Expected '{expected_result}', got '{actual_result}'.")

# Main script for test cases
if __name__ == "__main__":
    # Load the AI model (update the path)
    AI_MODEL_PATH = "form_validation_model.h5"
    ai_model = load_ai_model(AI_MODEL_PATH)

    # Define test cases
    test_cases = [
        {
            "data": {
                "name": "John Doe",
                "email": "john.doe@example.com",
                "website": "https://johndoe.com",
                "comment": "Looking forward to the training!",
                "gender": "male",
                "country": "USA",
            },
            "expected_result": "success",  # Positive test case
        },
        {
            "data": {
                "name": "",
                "email": "invalid-email@",  # Invalid email
                "website": "http://invalid-website",
                "comment": "Testing invalid inputs.",
                "gender": "InvalidGender",  # Invalid gender
                "country": "InvalidCountry",  # Invalid country
            },
            "expected_result": "error",  # Negative test case
        },
        {
            "data": {
                "name": "John",
                "email": "no-domain-email",  # Invalid email
                "website": None,
                "comment": "",
                "gender": None,  # No gender selected
                "country": None,  # No country selected
            },
            "expected_result": "error",  # Negative test case
        },
        {
            "data": {
                "name": "Alice",
                "email": "alice@example.com",
                "website": "https://alice.com",
                "comment": "This is a valid input.",
                "gender": "female",
                "country": "Canada",
            },
            "expected_result": "success",  # Positive test case
        },
        {
            "data": {
                "name": "",
                "email": "",
                "website": "",
                "comment": "",
                "gender": "",
                "country": "",
            },
            "expected_result": "error",  # Negative test case
        },
    ]


    # Path to ChromeDriver (update this to the correct path)
    CHROME_DRIVER_PATH = "/path/to/chromedriver"

    # Execute test cases
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nExecuting Test Case {i} with AI...")
        automate_test_case_with_ai(test_case["data"], test_case["expected_result"], CHROME_DRIVER_PATH, ai_model)
