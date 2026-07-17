import streamlit as st
import numpy as np
import pickle
from sklearn import svm                      
from sklearn.preprocessing import StandardScaler

# Load the saved model and scaler
model = pickle.load(open('Models/diabetes_model.sav', 'rb'))
scaler = pickle.load(open('Models/scaler.sav', 'rb'))

# Set up page configuration
st.set_page_config(page_title="Diabetes Prediction Dashboard", layout="centered")

st.title("🩺 PIMA Diabetes Prediction Dashboard")
st.write("Enter the patient's medical details below to predict diabetes vulnerability.")

# Create input fields for user data based on dataset columns
col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1, step=1)
    glucose = st.number_input("Glucose Level (mg/dL)", min_value=0, max_value=300, value=120)
    blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=200, value=70)
    skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=20)

with col2:
    insulin = st.number_input("Insulin Level (mu U/ml)", min_value=0, max_value=900, value=80)
    bmi = st.number_input("BMI (Body Mass Index)", min_value=0.0, max_value=70.0, value=32.0, format="%.1f")
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, format="%.3f")
    age = st.number_input("Age (years)", min_value=1, max_value=120, value=30, step=1)

# Code for Prediction
diagnosis = ""

if st.button("Predict Test Result", type="primary"):
    # Arrange features in the exact same order as model training
    input_data = np.asarray([pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age])
    
    # Reshape input for a single instance prediction
    input_data_reshaped = input_data.reshape(1, -1)
    
    # Standardize using the loaded scaler
    std_data = scaler.transform(input_data_reshaped)
    
    # Predict
    prediction = model.predict(std_data)
    
    if prediction[0] == 1:
        st.error("🚨 **Prediction:** The patient is likely to be diabetic.")
    else:
        st.success("✅ **Prediction:** The patient is likely non-diabetic.")
