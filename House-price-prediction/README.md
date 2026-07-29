# California House Price Prediction & Dashboard

An end-to-end Machine Learning project analyzing the standard California Housing dataset, training an **XGBoost Regressor** model, and deploying an interactive **Streamlit** web dashboard for real-time housing price predictions.

---

##  Project Structure

```text
House-price-prediction/
│
├── 3-House Price Prediction.ipynb   # Google Colab notebook with data analysis & model training
├── app.py                           # Streamlit web application dashboard
├── requirements.txt                 # Required Python packages
└── README.md                        # Project documentation

```

---

##  Dataset Overview

* **Source:** Scikit-learn California Housing dataset (`sklearn.datasets.fetch_california_housing`)


* **Instances:** 20,640 samples


* **Attributes:** 8 numeric features (Median Income, House Age, Average Rooms, Average Bedrooms, Population, Average Occupancy, Latitude, Longitude)


* **Target Variable (`MedHouseVal`):** Median house value for California districts, expressed in hundreds of thousands of dollars ($100,000).



---

##  Model Performance & Architecture

* **Algorithm:** XGBoost Regressor (`XGBRegressor`)


* **Evaluation Metrics:**
* **R-squared Error ($R^2$):** ~0.94 (Training) / ~0.83 (Test)


* **Mean Absolute Error (MAE):** ~0.19 (Training) / ~0.31 (Test)




* **Deployment Strategy:** The model trains dynamically using cached resources on app startup, ensuring robust performance and zero serialization version conflicts on cloud platforms.

---

##  Installation & Running Locally

1. **Clone the repository:**
```bash
git clone https://github.com/Painx9/Models.git
cd Models/House-price-prediction

```


2. **Install dependencies:**
```bash
pip install -r requirements.txt

```


3. **Run the Streamlit app:**
```bash
streamlit run app.py

```



---

##  Deployment

This app can be hosted live via **Streamlit Community Cloud**:

1. Link your GitHub repository to Streamlit Cloud.
2. Set the main file path to: `https://house-price-prediction09.streamlit.app`
3. Deploy!
