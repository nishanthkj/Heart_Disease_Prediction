
# Heart Disease Prediction System

## Overview

This project is a web-based application that predicts the likelihood of heart disease based on user input. It uses a pre-trained Random Forest model to make predictions. The application is built using Flask, a lightweight web framework for Python.

## Project Structure

```
Heart_Disease_Prediction/
│
├── models/
│   └── random_forest_model.h5
|   └── heart_disease_prediction_model.h5
│
├── templates/
│   └── index.html
│
├── app.py
└── model.py
```

## Files Description

- **models/random_forest_model.h5**: The pre-trained Random Forest model saved in HDF5 format.
- **templates/index.html**: The HTML template for the web form.
- **app.py**: The main Flask application file.
- **model.py**: Script used to train and save the Random Forest model.

## Setup Instructions

### Prerequisites

- Python 3.12
- Flask
- scikit-learn
- h5py
- numpy
- pandas

### Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/nishanthkj/Heart_Disease_Prediction
    cd Heart_Disease_Prediction
    ```

2. **Create a virtual environment**:
    ```bash
    # On Windows use
    python -m venv venv
    venv\Scripts\activate
    ```

3. **Install the required packages**:
    ```bash
    pip install flask scikit-learn h5py numpy pandas
    ```

4. **Ensure the model file exists**:
    Make sure `heart_disease_prediction_model.h5` is present in the `models` directory.

## Usage

1. **Run the Flask application**:
    ```bash
    python app.py
    ```

2. **Access the application**:
    Open your web browser and go to `http://127.0.0.1:5000/`.

3. **Enter the required details**:
    Fill in the form with the necessary details and click on the "Predict" button to get the prediction.

## Code Explanation

### `app.py`

This is the main Flask application file.

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import h5py
import pickle
import numpy as np
from flask import Flask, request, render_template

app = Flask(__name__)

# Load the model once when the application starts
model_file_path = 'models/random_forest_model.h5'
with h5py.File(model_file_path, 'r') as h5f:
    model_data = h5f['model'][()]
rf = pickle.loads(model_data)

@app.route('/', methods=['GET', 'POST'])
def predict():
    result = None
    if request.method == 'POST':
        try:
            # Retrieve values from form
            features = [
                int(request.form['age']),
                int(request.form['sex']),
                int(request.form['cp']),
                int(request.form['trestbps']),
                int(request.form['chol']),
                int(request.form['fbs']),
                int(request.form['restecg']),
                float(request.form['thalach']),
                int(request.form['exang']),
                float(request.form['oldpeak']),
                int(request.form['slope']),
                int(request.form['ca']),
                int(request.form['thal'])
            ]

            # Make prediction
            prediction = rf.predict([features])

            # Interpret result
            if prediction[0] == 0:
                result = "No Heart Disease"
            else:
                result = "Possibility of Heart Disease"
        except ValueError:
            result = "Please enter valid values in all fields"

    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
```

### `templates/index.html`

This is the HTML template for the web form.

```html
<!DOCTYPE html>
<html>
<head>
    <title>Heart Disease Prediction System</title>
</head>
<body>
    <h2>Heart Disease Prediction System</h2>
    <form method="post">
        <label>Enter Your Age:</label>
        <input type="number" name="age" required><br>
        <label>Male Or Female [1/0]:</label>
        <input type="number" name="sex" required><br>
        <label>Enter Value of CP:</label>
        <input type="number" name="cp" required><br>
        <label>Enter Value of trestbps:</label>
        <input type="number" name="trestbps" required><br>
        <label>Enter Value of chol:</label>
        <input type="number" name="chol" required><br>
        <label>Enter Value of fbs:</label>
        <input type="number" name="fbs" required><br>
        <label>Enter Value of restecg:</label>
        <input type="number" name="restecg" required><br>
        <label>Enter Value of thalach:</label>
        <input type="number" name="thalach" step="any" required><br>
        <label>Enter Value of exang:</label>
        <input type="number" name="exang" required><br>
        <label>Enter Value of oldpeak:</label>
        <input type="number" name="oldpeak" step="any" required><br>
        <label>Enter Value of slope:</label>
        <input type="number" name="slope" required><br>
        <label>Enter Value of ca:</label>
        <input type="number" name="ca" required><br>
        <label>Enter Value of thal:</label>
        <input type="number" name="thal" required><br>
        <input type="submit" value="Predict">
    </form>
    {% if result is not none %}
        <h3>Prediction: {{ result }}</h3>
    {% endif %}
</body>
</html>
```

## Model Training (`model.py`)

This script is used to train the Random Forest model and save it to an HDF5 file.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import h5py
import pickle
import numpy as np

# Load dataset
data = pd.read_csv('path_to_your_dataset.csv')

# Split the dataset into features and target variable
X = data.drop(columns='target')
y = data['target']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the Random Forest model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Evaluate the model
y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save the model to an HDF5 file
model_file_path = 'models/random_forest_model.h5'
with h5py.File(model_file_path, 'w') as h5f:
    model_data = pickle.dumps(rf)
    h5f.create_dataset('model', data=np.void(model_data))
```
## Screenshots
![image](https://github.com/nishanthkj/Heart_Disease_Prediction/assets/138886231/64d81d3e-5c73-4cce-bea4-67baa38b12fa)

## Conclusion

This documentation provides a comprehensive guide to setting up and running the Heart Disease Prediction System. By following the steps outlined, you should be able to deploy the application and make predictions based on user input. If you encounter any issues, ensure that all dependencies are installed and that the model file is correctly placed in the `models` directory.
