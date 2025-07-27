
import numpy as np
import pandas as pd
import pickle
import streamlit as st

# Load the model and encoders

loaded_model = pickle.load(open('trained_model.sav', 'rb'))
loaded_encoders = pickle.load(open('encoders.pkl', 'rb'))

# Prediction function
def churn_prediction(input_data):
    input_data_df = pd.DataFrame([input_data])

    # Apply encoders
    for column, encoder in loaded_encoders.items():
        try:
            input_data_df[column] = encoder.transform(input_data_df[[column]])
        except Exception as e:
            st.error(f"Encoding error in column '{column}': {e}")
            return

    # Predict
    prediction = loaded_model.predict(input_data_df)
    prediction_proba = loaded_model.predict_proba(input_data_df)

    return prediction[0], prediction_proba[0][1]  # Return class and probability

# Streamlit UI
def main():
    st.title('📊 Customer Churn Prediction')

    # Form inputs
    gender = st.selectbox("Gender", ['Female', 'Male'])
    SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
    Partner = st.selectbox("Partner", ['Yes', 'No'])
    Dependents = st.selectbox("Dependents", ['Yes', 'No'])
    tenure = st.number_input("Tenure (in months)", min_value=0)
    PhoneService = st.selectbox("Phone Service", ['Yes', 'No'])
    MultipleLines = st.selectbox("Multiple Lines", ['Yes', 'No', 'No phone service'])
    InternetService = st.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No'])
    OnlineSecurity = st.selectbox("Online Security", ['Yes', 'No'])
    OnlineBackup = st.selectbox("Online Backup", ['Yes', 'No'])
    DeviceProtection = st.selectbox("Device Protection", ['Yes', 'No'])
    TechSupport = st.selectbox("Tech Support", ['Yes', 'No'])
    StreamingTV = st.selectbox("Streaming TV", ['Yes', 'No'])
    StreamingMovies = st.selectbox("Streaming Movies", ['Yes', 'No'])
    Contract = st.selectbox("Contract", ['Month-to-month', 'One year', 'Two year'])
    PaperlessBilling = st.selectbox("Paperless Billing", ['Yes', 'No'])
    PaymentMethod = st.selectbox("Payment Method", ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
    MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0)
    TotalCharges = st.number_input("Total Charges", min_value=0.0)

    # When button clicked
    if st.button("Predict"):
        input_data = {
            'gender': gender,
            'SeniorCitizen': SeniorCitizen,
            'Partner': Partner,
            'Dependents': Dependents,
            'tenure': tenure,
            'PhoneService': PhoneService,
            'MultipleLines': MultipleLines,
            'InternetService': InternetService,
            'OnlineSecurity': OnlineSecurity,
            'OnlineBackup': OnlineBackup,
            'DeviceProtection': DeviceProtection,
            'TechSupport': TechSupport,
            'StreamingTV': StreamingTV,
            'StreamingMovies': StreamingMovies,
            'Contract': Contract,
            'PaperlessBilling': PaperlessBilling,
            'PaymentMethod': PaymentMethod,
            'MonthlyCharges': MonthlyCharges,
            'TotalCharges': TotalCharges
        }

        result, proba = churn_prediction(input_data)
        st.success(f"Prediction: {'Churn' if result == 1 else 'No Churn'}")
        st.info(f"Churn Probability: {proba:.2%}")

# Run the app
if __name__ == '__main__':
    main()



