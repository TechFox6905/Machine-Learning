import streamlit as st
import numpy as np
import joblib
import tensorflow as tf
import pandas as pd

# Load model and preprocessor
model = tf.keras.models.load_model("heart_disease_model.h5")
preprocessor = joblib.load("preprocessor.pkl")

# Get expected feature names from the preprocessor
expected_features = preprocessor.feature_names_in_

# App title
st.title("🫀 Heart Disease Prediction App")
st.write("Enter patient details below to predict the risk of heart disease.")

# User input fields
col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=50)
    height = st.number_input("Height (cm)", min_value=100, max_value=250, value=170)
    weight = st.number_input("Weight (kg)", min_value=30, max_value=200, value=70)
    ap_hi = st.number_input("Systolic BP (Higher)", min_value=80, max_value=200, value=120)
with col2:
    ap_lo = st.number_input("Diastolic BP (Lower)", min_value=40, max_value=150, value=80)
    cholesterol = st.selectbox("Cholesterol", [1, 2, 3], format_func=lambda x: ["Normal", "Above Normal", "High"][x - 1])
    gluc = st.selectbox("Glucose", [1, 2, 3], format_func=lambda x: ["Normal", "Above Normal", "High"][x - 1])
    smoke = st.selectbox("Smoking", [0, 1], format_func=lambda x: "Yes" if x else "No")
    alco = st.selectbox("Alcohol Intake", [0, 1], format_func=lambda x: "Yes" if x else "No")
    active = st.selectbox("Physical Activity", [0, 1], format_func=lambda x: "Active" if x else "Inactive")

# Compute BMI
bmi = weight / (height / 100) ** 2

# Prepare input data as DataFrame with expected feature names
input_data = pd.DataFrame([[age, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active, bmi]],
                          columns=["age", "height", "weight", "ap_hi", "ap_lo", "cholesterol", "gluc", "smoke", "alco", "active", "bmi"])

# Ensure input data matches the trained model's feature set
for missing_feature in set(expected_features) - set(input_data.columns):
    input_data[missing_feature] = 0  # Add missing columns with default values

# Reorder columns to match training order
input_data = input_data[expected_features]

# Transform input data
input_data = preprocessor.transform(input_data)

# Predict and display results
if st.button("Predict"):
    prediction = model.predict(input_data)[0][0]
    
    if prediction > 0.5:
        st.error("⚠️ High risk of heart disease! Consult a doctor.")
        st.warning(f"⚕️ Prediction Score: {prediction:.2f} (Higher means more risk)")
    else:
        st.success("✅ Low risk of heart disease. Keep maintaining a healthy lifestyle!")
        st.info(f"⚕️ Prediction Score: {prediction:.2f} (Lower means less risk)")
