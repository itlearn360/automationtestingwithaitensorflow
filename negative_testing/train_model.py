import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load the dataset
dataset = pd.read_csv('dataset.csv')

# Encode the target variable
label_encoder = LabelEncoder()
dataset['Result'] = label_encoder.fit_transform(dataset['Result'])  # 0 for "error", 1 for "success"

# Prepare features and target
X = dataset[['Name Valid', 'Email Valid', 'Website Valid', 'Comment Valid', 'Gender Valid', 'Country Valid']]
y = dataset['Result']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model
model = Sequential([
    Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=8, validation_data=(X_test, y_test))

# Save the model
model.save('form_validation_model.h5')

print("Model training completed and saved as 'form_validation_model.h5'.")
