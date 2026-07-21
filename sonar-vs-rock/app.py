import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Set page configuration
st.set_page_config(page_title="Sonar Rock vs. Mine Predictor", layout="wide")

# Get the directory where app.py is located
current_dir = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(current_dir, 'model.pkl')
data_path = os.path.join(current_dir, 'sonar data.csv')

# Load model, dataset, and calculate accuracy dynamically
@st.cache_resource
def load_data_model_and_metrics():
    model = joblib.load(model_path)
    sonar_data = pd.read_csv(data_path, header=None)
    
    # Recalculate train/test split accuracy metrics (removed axis=1)
    X = sonar_data.drop(columns=60)
    Y = sonar_data[60]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, stratify=Y, random_state=1)
    
    train_accuracy = accuracy_score(model.predict(X_train), Y_train)
    test_accuracy = accuracy_score(model.predict(X_test), Y_test)
    
    return model, sonar_data, train_accuracy, test_accuracy

model, sonar_data, train_accuracy, test_accuracy = load_data_model_and_metrics()

st.title("Sonar Navigation: Rock vs. Mine Dashboard")
st.write("An interactive web application built with Streamlit to analyze sonar return frequencies and evaluate machine learning predictions.")

# Sidebar navigation
page = st.sidebar.selectbox("Choose a View", ["Dataset Overview", "Make a Prediction"])

if page == "Dataset Overview":
    st.subheader("Dataset Summary & Model Performance")
    
    # Display metrics cards for accuracy data
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Total Dataset Records", value=sonar_data.shape[0])
    with col2:
        st.metric(label="Training Data Accuracy", value=f"{train_accuracy * 100:.2f}%")
    with col3:
        st.metric(label="Test Data Accuracy", value=f"{test_accuracy * 100:.2f}%")
        
    st.markdown("---")
    st.subheader("Class Distribution")
    class_counts = sonar_data[60].value_counts().reindex(['M', 'R'])
    st.bar_chart(class_counts)

elif page == "Make a Prediction":
    st.subheader("Test Sonar Instance Prediction")
    st.write("Click the button below to test a random instance from the dataset through your Logistic Regression model.")
    
    # Show test accuracy on this page as well for reference
    st.info(f"Model Current Test Accuracy: **{test_accuracy * 100:.2f}%**")
    
    if st.button("Run Prediction on Random Sample", type="primary"):
        # Pick a random row from the dataset each time
        random_index = random.randint(0, len(sonar_data) - 1)
        
        sample_input = sonar_data.drop(columns=60).iloc[random_index].values.reshape(1, -1)
        pred = model.predict(sample_input)[0]
        actual = sonar_data.iloc[random_index][60]
        
        st.write(f"**Tested Row Index from Dataset:** {random_index}")
        st.write(f"**Actual Label in Dataset:** {'Mine (M)' if actual == 'M' else 'Rock (R)'}")
        
        if pred == 'M':
            st.error("Prediction Result: Mine (M)")
        else:
            st.success("Prediction Result: Rock (R)")
