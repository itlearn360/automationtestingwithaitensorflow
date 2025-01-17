# AI-Powered Automation Testing Framework

## **Overview**
This framework leverages AI and machine learning to enhance automation testing. By integrating TensorFlow, Pandas, NumPy, and Selenium, the system dynamically identifies web elements, predicts the most stable locators, and performs actions efficiently. The framework is designed to be robust, adaptable, and capable of handling dynamic web applications.

---

## **Features**
- **AI-Based Locator Prediction**: Utilizes a trained model to predict the best locator (e.g., `id`, `class`, `name`) for identifying elements.
- **Self-Healing Scripts**: Adapts to changes in the web UI by dynamically selecting alternative locators.
- **Comprehensive Preprocessing**: Ensures data integrity and compatibility with machine learning models.
- **End-to-End Automation**: Combines AI predictions with Selenium for seamless web element interactions.

---

## **Prerequisites**
1. Python 3.7+
2. Required Python Libraries:
   - TensorFlow
   - Pandas
   - NumPy
   - Selenium
   - scikit-learn
   - joblib
3. Google Chrome and Chromedriver (for Selenium)

---

## **Setup Instructions**

### **1. Clone the Repository**
```bash
git clone <repository_url>
cd <repository_folder>
```

### **2. Install Dependencies**
```bash
pip install -r requirements.txt
```

### **3. Prepare the Dataset**
Ensure the dataset (`element_data.csv`) contains the following columns:
- `id`, `class`, `text`, `tag`, `value`, `type`, `name`, `label`, `locator`

Example row:
```csv
id,class,text,tag,value,type,name,label,locator
name-field,input-class,Name,input,Name,text,name,input,name
```
Place the dataset in the `data/` directory.

### **4. Preprocess the Dataset**
Run the preprocessing script to clean and encode the dataset:
```bash
python train_model.py
```

### **5. Train the Model**
Train the AI model using the preprocessed dataset:
```bash
python train_model.py
```
The trained model will be saved in the `models/` directory.

### **6. Run the Selenium AI Test**
Execute the test script to perform AI-powered automation testing:
```bash
python selenium_ai_test.py
```

---

## **Folder Structure**
```
├── data
│   ├── element_data.csv         # Dataset for training
│   ├── processed                # Processed data folder
├── models
│   ├── element_model.h5         # Trained TensorFlow model
│   ├── scaler.pkl               # Saved scaler for preprocessing
│   ├── feature_columns.txt      # List of feature columns
├── logs
│   ├── model_training.log       # Logs for training
├── selenium_ai_test.py          # Selenium AI testing script
├── train_model.py               # Model training script
├── requirements.txt             # Python dependencies
```

---

## **How It Works**
1. **Dataset Preparation**:
   - Web element attributes are used as features.
   - The `label` column specifies the element type (e.g., `button`, `input`).
   - The `locator` column specifies the preferred locator type.

2. **Preprocessing**:
   - Missing values are filled with defaults.
   - Categorical columns are one-hot encoded.
   - Hybrid encoding ensures all locator types are represented.

3. **Model Training**:
   - A TensorFlow model predicts both element type and locator type.
   - Separate output layers are used for multi-class classification.

4. **Selenium Integration**:
   - The model's predictions guide Selenium to locate and interact with elements.
   - If a predicted locator fails, alternative locators are attempted.

---

## **Example Usage**
### **Element Features for Prediction**
```python
{
    'id': 'email-field',
    'class': 'email-class',
    'text': '',
    'tag': 'input',
    'value': '',
    'type': '',
    'name': 'email'
}
```

### **AI Prediction and Automation**
```python
# Load the model and scaler
model = tf.keras.models.load_model('./models/element_model.h5')
scaler = joblib.load('./models/scaler.pkl')

# Preprocess the input
input_scaled = preprocess_input(element_features, scaler, feature_columns)

# Predict
predictions = model.predict(input_scaled)
locator_map = ['id', 'class', 'text', 'value', 'type', 'name']
predicted_locator = locator_map[np.argmax(predictions[1][0])]

# Locate element and perform action
if predicted_locator == 'name':
    element = driver.find_element(By.NAME, 'email')
    element.send_keys("test@example.com")
```

---

## **Logs and Debugging**
Logs are stored in the `logs/` folder to track training progress and Selenium interactions. Example:
```plaintext
2025-01-17 14:00:00 - INFO - Model training completed successfully.
2025-01-17 14:05:00 - INFO - Predicted locator: name
2025-01-17 14:05:02 - INFO - Element located and action performed.
```

---

## **Future Enhancements**
1. Support for additional locators such as XPath.
2. Integrating reinforcement learning for self-healing scripts.
3. Expanding dataset for multilingual web applications.

---