import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Invoice Intelligence Dashboard", page_icon="📦", layout="centered"
)

st.title("📦 Invoice Intelligence: Freight Predictor")
st.write(
    "Upload or input your invoice financial details to estimate the freight cost."
)

# Load the trained model
@st.cache_resource
def load_model():
    return joblib.load("best_invoice_model.pkl")


try:
    model = load_model()
except Exception as e:
    st.error(
        f"Model file not found. Please make sure 'best_invoice_model.pkl' is in the repository. Error: {e}"
    )

# User input widget
dollars = st.number_input(
    "Enter Invoice Dollars ($):", min_value=0.0, value=1500.0, step=50.0
)

if st.button("Calculate Prediction"):
    # Make prediction using the loaded model
    input_data = pd.DataFrame({"Dollars": [dollars]})
    prediction = model.predict(input_data)

    st.success(f"Predicted Freight Cost: **${prediction[0]:.2f}**")
