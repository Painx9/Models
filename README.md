# 🚀 Machine Learning & Data Science Portfolio Hub

Welcome to my centralized Machine Learning portfolio! This repository serves as a production-ready showcase of end-to-end data science projects. Each project spans data preprocessing, feature engineering, statistical model training, evaluation, and interactive web dashboard deployment.

---

## 🧭 Repository Blueprint

To keep the repository highly scalable for 10-20+ individual projects, every project is isolated within its own dedicated subdirectory containing its model artifacts, requirements, notebook, and web app.

```text
📁 ML-Portfolio-Hub/ (Root)
├── README.md                           # Main portfolio guide (This file)
│
├── 📁 diabetes-prediction-ml/          # 🩺 PIMA Diabetes Classification
│   ├── app.py                          # Streamlit UI dashboard
│   ├── requirements.txt                # Dependencies (scikit-learn, etc.)
│   ├── diabetes_model.sav              # Serialized trained SVM Model
│   ├── scaler.sav                      # Serialized StandardScaler
│   └── 2-Diabetic Predictions.ipynb    # Jupyter Notebook training file #Credits:** This project was built as part of an educational learning path guided by the [Siddhardhan](https://www.youtube.com/watch?v=xUE7SjVx9bQ&list=PLfFghEzKVmjvuSA67LszN1dZ-Dd_pkus6&index=2).
|                                       I extended the project by implementing standard file packaging and converting the static notebook into a live Streamlit dashboard.
│
├── 📁 02-Loan-Eligibility-Prediction/  # 💰 Financial Risk Assessment

```
🛠️ General Technical Stack
Across these workflows, I heavily leverage the following core data science ecosystem:
Data Handling & Exploration: Python, Pandas, NumPy
Machine Learning Infrastructure: Scikit-Learn (Pipelines, StandardScalers, GridSearch Cross-Validation)
Dashboard Deployment: Streamlit Cloud Framework
Model Management: Serialization via Python ```pickle```


# 🏃 Launching Apps Locally
Every project folder is fully standalone. To run any individual dashboard on your local machine:

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Painx9/Models.git](https://github.com/Painx9/Models.git)
   cd Models
   ```

2. **Navigate into your project folder of choice:**
  ```bash
   # Jump into your specific project folder (e.g., diabetes-prediction-ml)
   cd diabetes-prediction-ml
   ```

3. **Install the project isolated dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch the Streamlit instance:**
   ```bash
   streamlit run app.py
   ```
## 📜 License & Credits
* **Credits:** Core machine learning models inspired by educational tutorials from the data science community on YouTube.
* **License:** This project is licensed under the MIT License.
