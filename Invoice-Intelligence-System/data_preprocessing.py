import pandas as pd
import sqlite3
from sklearn.model_selection import train_test_split


def load_vendor_invoice_data(db_path: str) -> pd.DataFrame:
    """Load vendor invoice data directly from the SQLite database."""
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM vendor_invoice", conn)
    conn.close()
    print(f"Successfully loaded vendor_invoice data. Shape: {df.shape}")
    return df


def prepare_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Clean data, handle missing values, and select features matching your model workflow."""
    processed_df = df.copy()

    # Drop duplicates if any
    processed_df = processed_df.drop_duplicates()

    # Handle missing values for numerical and categorical features
    for col in processed_df.select_dtypes(include=["number"]).columns:
        processed_df[col] = processed_df[col].fillna(processed_df[col].median())

    for col in processed_df.select_dtypes(include=["object"]).columns:
        processed_df[col] = processed_df[col].fillna("Unknown")

    print(f"Data preparation completed. Processed shape: {processed_df.shape}")
    return processed_df


def split_data(
    df: pd.DataFrame, target_column: str = "Freight", test_size: float = 0.2, random_state: int = 42
):
    """Split the dataframe into training and testing feature/target sets based on your notebook setup."""
    # Following your notebook's feature selection (using Dollars to predict Freight)
    X = df[["Dollars"]]
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    print("Data split successfully:")
    print(f" - Training features shape: {X_train.shape}")
    print(f" - Testing features shape: {X_test.shape}")

    return X_train, X_test, y_train, y_test
