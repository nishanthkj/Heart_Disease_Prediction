import pandas as pd

# Load the dataset
dataset_path = 'data/dataset.csv'
data = pd.read_csv(dataset_path)

# Display the first few rows of the dataset to understand its structure
data.head()
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
import h5py
import numpy as np

# Split the dataset into features and target variable
X = data.drop(columns='target')
y = data['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the Random Forest model
rf = RandomForestClassifier()

# Fit the model on the training data
rf.fit(X_train, y_train)

# Save the model to an HDF5 file
model_file_path = 'models/heart_disease_prediction_model.h5'
with h5py.File(model_file_path, 'w') as h5f:
    model_data = pickle.dumps(rf)
    h5f.create_dataset('model', data=np.void(model_data))

model_file_path
