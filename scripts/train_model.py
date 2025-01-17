import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Paths
DATA_PATH = 'data/element_data.csv'
PROCESSED_DATA_PATH = 'data/processed/processed_data.csv'
MODEL_SAVE_PATH = 'models/element_model.h5'
SCALER_SAVE_PATH = 'models/scaler.pkl'
FEATURE_COLUMNS_PATH = 'models/feature_columns.txt'

# Locator types for hybrid encoding
LOCATOR_TYPES = ['id', 'class', 'text', 'value', 'type', 'name']

# Logging setup
import logging
logging.basicConfig(
    filename='logs/model_training.log',
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)

def hybrid_encode_locators(data, locator_types):
    """
    Ensures all locator types are present and assigns a default value for missing locators.
    """
    for locator in locator_types:
        if f'locator_{locator}' not in data.columns:
            data[f'locator_{locator}'] = 0
    return data

def preprocess_data(input_path, output_path, feature_columns_path):
    """
    Preprocesses the raw dataset by encoding locator types and saving feature columns.
    """
    try:
        # Load raw data
        data = pd.read_csv(input_path)

        # Fill missing values with empty strings
        data.fillna('', inplace=True)

        # One-hot encode relevant columns
        data = pd.get_dummies(data, columns=['label', 'locator', 'id', 'class', 'text', 'tag', 'value', 'type', 'name'])

        # Apply hybrid encoding for locators
        data = hybrid_encode_locators(data, LOCATOR_TYPES)

        # Convert any boolean columns to integers
        data = data.applymap(lambda x: int(x) if isinstance(x, bool) else x)

        # Save feature columns for consistency during prediction
        os.makedirs(os.path.dirname(feature_columns_path), exist_ok=True)
        with open(feature_columns_path, 'w') as f:
            f.write('\n'.join(data.columns.tolist()))

        # Save the processed dataset
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        data.to_csv(output_path, index=False)
        logging.info(f"Processed data saved to {output_path}")
    except Exception as e:
        logging.error(f"Error in data preprocessing: {e}")
        raise

def train_model(data_path, model_path, scaler_path, feature_columns_path):
    """
    Trains a model to predict element type and locator type.
    """
    try:
        # Load processed data
        data = pd.read_csv(data_path)

        # Separate features and targets
        X = data.drop(columns=[col for col in data.columns if col.startswith('label_') or col.startswith('locator_')])
        y_label = data[[col for col in data.columns if col.startswith('label_')]]  # Element type
        y_locator = data[[col for col in data.columns if col.startswith('locator_')]]  # Locator type

        # Save feature columns
        os.makedirs(os.path.dirname(feature_columns_path), exist_ok=True)
        with open(feature_columns_path, 'w') as f:
            f.write('\n'.join(X.columns.tolist()))

        # Train-test split
        X_train, X_test, y_label_train, y_label_test = train_test_split(X, y_label, test_size=0.2, random_state=42)
        _, _, y_locator_train, y_locator_test = train_test_split(X, y_locator, test_size=0.2, random_state=42)

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train).astype('float32')  # Explicitly convert to float32
        X_test_scaled = scaler.transform(X_test).astype('float32')  # Explicitly convert to float32

        # Save the scaler
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        joblib.dump(scaler, scaler_path)

        # Define the model
        inputs = tf.keras.layers.Input(shape=(X_train_scaled.shape[1],))
        hidden = tf.keras.layers.Dense(128, activation='relu')(inputs)
        hidden = tf.keras.layers.Dense(64, activation='relu')(hidden)

        # Output layers
        output_label = tf.keras.layers.Dense(y_label_train.shape[1], activation='softmax', name='label')(hidden)
        output_locator = tf.keras.layers.Dense(y_locator_train.shape[1], activation='softmax', name='locator')(hidden)

        # Model definition
        model = tf.keras.Model(inputs=inputs, outputs=[output_label, output_locator])
        model.compile(
            optimizer='adam',
            loss={'label': 'categorical_crossentropy', 'locator': 'categorical_crossentropy'},
            metrics={'label': 'accuracy', 'locator': 'accuracy'}
        )

        # Train the model
        model.fit(
            X_train_scaled,
            {'label': y_label_train, 'locator': y_locator_train},
            validation_data=(X_test_scaled, {'label': y_label_test, 'locator': y_locator_test}),
            epochs=10,
            batch_size=32,
            verbose=2
        )

        # Save the model
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        model.save(model_path)
        logging.info(f"Model saved to {model_path}")

    except Exception as e:
        logging.error(f"Error in training model: {e}")
        raise

if __name__ == "__main__":
    preprocess_data(DATA_PATH, PROCESSED_DATA_PATH, FEATURE_COLUMNS_PATH)
    train_model(PROCESSED_DATA_PATH, MODEL_SAVE_PATH, SCALER_SAVE_PATH, FEATURE_COLUMNS_PATH)