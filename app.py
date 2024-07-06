import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import h5py
import pickle
import numpy as np
from flask import Flask, request, render_template

app = Flask(__name__)

# Load the model once when the application starts
with h5py.File('models/heart_disease_prediction_model.h5', 'r') as h5f:
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
