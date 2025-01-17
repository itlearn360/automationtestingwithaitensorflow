import pandas as pd
import os
import logging

# Configure logging
logging.basicConfig(filename='logs/data_preparation.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

RAW_DATA_PATH = 'data/element_data.csv'
PROCESSED_DATA_PATH = 'data/processed/processed_data.csv'
FEATURE_COLUMNS_PATH = 'models/feature_columns.txt'

def preprocess_data(input_path, output_path, feature_columns_path):
    try:
        # Load raw data
        data = pd.read_csv(input_path)

        # Fill missing values
        data.fillna('', inplace=True)

        # One-hot encode the `label` and `locator` columns (targets)
        data = pd.get_dummies(data, columns=['label', 'locator', 'id', 'class', 'text', 'tag', 'value', 'type', 'name'])
        print(data.columns)
        # Save feature columns for consistency in prediction
        os.makedirs(os.path.dirname(feature_columns_path), exist_ok=True)
        with open(feature_columns_path, 'w') as f:
            f.write('\n'.join(data.columns.tolist()))

        # Save processed data
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        data.to_csv(output_path, index=False)
        logging.info(f"Processed data saved to {output_path}")
    except Exception as e:
        logging.error(f"Error in data preprocessing: {e}")
        raise

if __name__ == "__main__":
    preprocess_data(RAW_DATA_PATH, PROCESSED_DATA_PATH, FEATURE_COLUMNS_PATH)
