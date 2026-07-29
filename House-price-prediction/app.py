from pathlib import Path
import pickle
import streamlit as st
from sklearn.datasets import fetch_california_housing
import pandas as pd
import numpy as np

# Page Configuration
st.set_page_config(
    page_title="California House Price Prediction",
    page_icon="🏡",
    layout="wide",
)

# Robust path resolution so Streamlit Cloud can find the model file regardless of working directory
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "xgboost_house_model.pkl"

# Load the trained model safely
@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()

# App Header
st.title("🏡 California House Price Prediction Dashboard")
st.markdown("""
This web application uses an **XGBoost Regressor** machine learning model trained on the California Housing Dataset. 
Use the sidebar inputs to adjust housing features and predict the median house value.
""")

# Sidebar Input Controls for Features
st.sidebar.header("Input Features")

def user_input_features():
    med_inc = st.sidebar.slider("Median Income (10k USD)", 0.5, 15.0, 3.5)
    house_age = st.sidebar.slider("House Age (Years)", 1.0, 52.0, 28.0)
    ave_rooms = st.sidebar.slider("Average Rooms per Household", 1.0, 10.0, 5.0)
    ave_bedrms = st.sidebar.slider("Average Bedrooms per Household", 0.5, 2.0, 1.0)
    population = st.sidebar.slider("Block Group Population", 10.0, 10000.0, 1400.0)
    ave_occup = st.sidebar.slider("Average House Occupancy", 1.0, 10.0, 3.0)
    latitude = st.sidebar.slider("Latitude", 32.5, 42.0, 35.6)
    longitude = st.sidebar.slider("Longitude", -124.3, -114.3, -119.5)
    
    # Passing data as a numpy array or standard dataframe columns matching the training order
    data = np.array([[med_inc, house_age, ave_rooms, ave_bedrms, population, ave_occup, latitude, longitude]])
    features = pd.DataFrame(data, columns=['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude'])
    return features

input_df = user_input_features()

# Main Display Panel
st.subheader("User Specified Parameters")
st.write(input_df)

# Prediction execution
prediction = model.predict(input_df)

st.subheader("Prediction Result")
# Target variable is expressed in hundreds of thousands of dollars ($100,000)
predicted_price_usd = prediction[0] * 100000

st.metric(label="Estimated Median House Value", value=f"${predicted_price_usd:,.2f}")

# Dataset Overview Section
if st.checkbox("Show Raw Dataset Summary"):
    housing = fetch_california_housing()
    df = pd.DataFrame(housing.data, columns=housing.feature_names)
    df['Price'] = housing.target
    st.write(df.head())
    st.markdown("### Statistical Summary")
    st.write(df.describe())
