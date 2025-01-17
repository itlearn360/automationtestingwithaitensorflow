from selenium import webdriver
from selenium.webdriver.common.by import By
import tensorflow as tf
import pandas as pd
import joblib
import logging
import time
logging.basicConfig(filename='logs/main.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

MODEL_PATH = 'models/element_model.h5'
FEATURE_COLUMNS_PATH = 'models/feature_columns.txt'
SCALER_PATH = 'models/scaler.pkl'

def load_model_and_scaler(model_path, feature_columns_path, scaler_path):
    model = tf.keras.models.load_model(model_path)
    with open(feature_columns_path, 'r') as f:
        feature_columns = f.read().splitlines()
    scaler = joblib.load(scaler_path)
    return model, scaler, feature_columns

def preprocess_input(features, scaler, feature_columns):
    input_df = pd.DataFrame([features])
    input_encoded = pd.get_dummies(input_df, columns=['id', 'class', 'text', 'tag', 'type', 'value'])
    input_encoded = input_encoded.reindex(columns=feature_columns, fill_value=0)
    input_scaled = scaler.transform(input_encoded)
    return input_scaled

def predict_and_act(driver, model, scaler, feature_columns):
    """
    Predict element type and locator type, validate predictions, and perform an action using Selenium.
    """
    # Example input features
    element_features = {
        'id': 'email-field',
        'class': 'email-class',
        'text': 'Email',
        'tag': 'input',
        'value': '',
        'type': '',
        'name': 'name'
    }

    # Preprocess input features
    input_scaled = preprocess_input(element_features, scaler, feature_columns)

    # Predict element type and locator type
    predictions = model.predict(input_scaled)
    element_type_probs = predictions[0][0]  # Extract probabilities for element type
    locator_type_probs = predictions[1][0]  # Extract probabilities for locator type

    # Define maps
    label_map = {0: 'button', 1: 'input', 2: 'link', 3: 'select', 4: 'password'}
    locator_map = {0: 'id', 1: 'class', 2: 'text', 3: 'value', 4: 'type', 5: 'name'}

    # Extract predictions separately
    predicted_type = label_map[int(element_type_probs.argmax())]
    predicted_locator_order = [
        locator_map[int(i)]
        for i in locator_type_probs.argsort()[::-1]
        if element_features.get(locator_map[int(i)], '')  # Only include non-blank locators
    ]

    logging.info(f"Predicted Element Type: {predicted_type}")
    logging.info(f"Predicted Locator Priority: {predicted_locator_order}")

    # Validate predicted type based on HTML tag
    if predicted_type != 'input' and element_features['tag'] == 'input':
        logging.warning(f"Predicted type '{predicted_type}' corrected to 'input' based on HTML tag.")
        predicted_type = 'input'

    # Locate and interact with the element
    element = None
    for locator in predicted_locator_order:
        try:
            if locator == 'id':
                element = driver.find_element(By.ID, element_features['id'])
            elif locator == 'class':
                element = driver.find_element(By.CLASS_NAME, element_features['class'])
            elif locator == 'text':
                element = driver.find_element(By.XPATH, f"//*[text()='{element_features['text']}']")
            elif locator == 'value':
                element = driver.find_element(By.XPATH, f"//*[@value='{element_features['value']}']")
            elif locator == 'type':
                element = driver.find_element(By.XPATH, f"//*[@type='{element_features['type']}']")
            elif locator == 'name':
                element = driver.find_element(By.NAME, element_features['name'])

            if element:
                logging.info(f"Element located using {locator}.")
                break
        except Exception as e:
            logging.warning(f"Failed to locate element using {locator}: {e}")

    if not element:
        raise ValueError("Unable to locate element with any available locators.")

    # Perform action based on predicted type
    if predicted_type == 'input':
        element.send_keys("Entered Value")
        logging.info("Text entered in input field.")
    elif predicted_type == 'button':
        element.click()
        logging.info("Button clicked.")


if __name__ == "__main__":
    model, scaler, feature_columns = load_model_and_scaler(MODEL_PATH, FEATURE_COLUMNS_PATH, SCALER_PATH)
    driver = webdriver.Chrome()
    driver.get("https://training.qaonlinetraining.com/testPage.php")
    time.sleep(4)
    predict_and_act(driver, model, scaler, feature_columns)
    time.sleep(4)  # Pause for 2 seconds
    print("Execution resumes after 2 seconds.")
    driver.quit()
