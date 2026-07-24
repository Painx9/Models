import joblib
import pandas as pd
from data_preprocessing import load_vendor_invoice_data, prepare_feature, split_data
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression


def run_pipeline():
    # 1. Load and prepare data from your database[cite: 1]
    db_path = r"Machine Learning Project/data/inventory.db"
    raw_df = load_vendor_invoice_data(db_path)
    clean_df = prepare_feature(raw_df)

    # ==========================================
    # PART A: REGRESSION TASK (Predicting Freight Cost)
    # ==========================================
    print("\n--- Training Regression Models ---")
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = split_data(
        clean_df, target_column="Freight"
    )

    reg_models = {
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(max_depth=6, random_state=42),
    }

    best_reg_name = None
    best_reg_model = None
    best_reg_score = -float("inf")  # Higher R2 is better

    for name, model in reg_models.items():
        model.fit(X_train_reg, y_train_reg)
        preds = model.predict(X_test_reg)
        score = r2_score(y_test_reg, preds)  # Using R2 Score
        print(f"{name} R2 Score: {score:.4f}")

        if score > best_reg_score:
            best_reg_score = score
            best_reg_name = name
            best_reg_model = model

    print(
        f"\n🏆 Best Regression Model: {best_reg_name} with R2 Score: {best_reg_score:.4f}"
    )

    # ==========================================
    # PART B: CLASSIFICATION TASK (Predicting High/Low Freight Tier)
    # ==========================================
    print("\n--- Training Classification Models ---")
    # Create a classification target: 1 if Freight is above median, else 0
    class_df = clean_df.copy()
    median_freight = class_df["Freight"].median()
    class_df["Freight_Tier"] = (class_df["Freight"] > median_freight).astype(int)

    # Split data for classification using Dollars as feature
    X_cls = class_df[["Dollars"]]
    y_cls = class_df["Freight_Tier"]
    from sklearn.model_selection import train_test_split

    X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(
        X_cls, y_cls, test_size=0.2, random_state=42
    )

    cls_models = {
        "Logistic Regression": LogisticRegression(),
        "Random Forest Classifier": RandomForestClassifier(
            max_depth=6, random_state=42
        ),
    }

    best_cls_name = None
    best_cls_model = None
    best_cls_score = 0.0  # Higher F1-Score is better

    for name, model in cls_models.items():
        model.fit(X_train_cls, y_train_cls)
        preds = model.predict(X_test_cls)
        score = f1_score(y_test_cls, preds)  # Using F1-Score
        print(f"{name} F1-Score: {score:.4f}")

        if score > best_cls_score:
            best_cls_score = score
            best_cls_name = name
            best_cls_model = model

    print(
        f"\n🏆 Best Classification Model: {best_cls_name} with F1-Score: {best_cls_score:.4f}"
    )

    # ==========================================
    # PART C: SELECT OVERALL BEST PARADIGM
    # ==========================================
    print("\n==========================================")
    if best_reg_score > 0.5:  # Arbitrary threshold to compare domains
        print(
            f"🎯 FINAL DECISION: Regression approach wins! Deploying **{best_reg_name}** for precise cost forecasting."
        )
        joblib.dump(best_reg_model, "best_invoice_model.pkl")
    else:
        print(
            f"🎯 FINAL DECISION: Classification approach wins! Deploying **{best_cls_name}** for tier classification."
        )
        joblib.dump(best_cls_model, "best_invoice_model.pkl")
    print("Model successfully saved to disk as 'best_invoice_model.pkl'!")


if __name__ == "__main__":
    run_pipeline()
