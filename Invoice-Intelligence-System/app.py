import os
import joblib
import streamlit as st

st.set_page_config(
    page_title="Invoice Intelligence Dashboard", page_icon="📦", layout="centered"
)

# Get the absolute path of the directory where app.py is located
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "best_invoice_model.pkl")

st.title("📦 Invoice Intelligence: Freight Predictor")
st.write(
    "Upload or input your invoice financial details to estimate the freight cost."
)


@st.cache_resource
def load_model():
    return joblib.load(model_path)


try:
    model = load_model()
except Exception as e:
    st.error(
        f"Could not load model from {model_path}. Please check file placement. Error: {e}"
    )

# User input widget
dollars = st.number_input(
    "Enter Invoice Dollars ($):", min_value=0.0, value=1500.0, step=50.0
)

if st.button("Calculate Prediction"):
    import pandas as pd

    input_data = pd.DataFrame({"Dollars": [dollars]})
    prediction = model.predict(input_data)
    st.success(f"Predicted Freight Cost: **${prediction[0]:.2f}**")
