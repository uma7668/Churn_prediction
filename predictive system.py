# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
import pickle

# Load the trained model
model_path = 'C:/Users/srini/Desktop/New folder/MachineLearning/trained_model.sav'
loaded_model = pickle.load(open(model_path, 'rb'))

# Define raw input data (as dictionary)
input_data = {
    'gender': 'Female',
    'SeniorCitizen': 0,
    'Partner': 'Yes',
    'Dependents': 'No',
    'tenure': 1,
    'PhoneService': 'No',
    'MultipleLines': 'No phone service',
    'InternetService': 'DSL',
    'OnlineSecurity': 'No',
    'OnlineBackup': 'Yes',
    'DeviceProtection': 'No',
    'TechSupport': 'No',
    'StreamingTV': 'No',
    'StreamingMovies': 'No',
    'Contract': 'Month-to-month',
    'PaperlessBilling': 'Yes',
    'PaymentMethod': 'Electronic check',
    'MonthlyCharges': 29.85,
    'TotalCharges': 29.85
}

# Convert to DataFrame
input_data_df = pd.DataFrame([input_data])

encoder_path = 'C:/Users/srini/Desktop/New folder/MachineLearning/encoders.pkl'
loaded_encoders = pickle.load(open(encoder_path, 'rb'))


for column, encoder in loaded_encoders.items():
  input_data_df[column] = encoder.transform(input_data_df[column])

# Make prediction
prediction = loaded_model.predict(input_data_df)
prediction_proba = loaded_model.predict_proba(input_data_df)

print(f"Prediction: {'Churn' if prediction[0] == 1 else 'No Churn'}")
print(f"Prediction Probability: {prediction_proba}")
