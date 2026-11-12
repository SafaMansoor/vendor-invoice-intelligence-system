import sqlite3
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class DataPreprocessor:

  def __init__(self):
    pass

  def load_data(self, db_path: str):
    """Load vendor invoice data from SQLite database."""
    print(f"Loading data from {db_path}...")
    with sqlite3.connect(db_path) as conn:
      df = pd.read_sql_query("SELECT * FROM vendor_invoice", conn)
    print(f"✅ Data Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

  def clean_freight_data(self, df: pd.DataFrame):
    """Validate and clean records required for freight modelling."""
    df = df.copy()

    required = ["InvoiceDate", "Dollars", "Freight"]
    missing_columns = [col for col in required if col not in df.columns]

    if missing_columns:
      raise ValueError(f"Missing required columns: {missing_columns}")

    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")

    for col in ["Dollars", "Freight"]:
      df[col] = pd.to_numeric(df[col], errors="coerce")

    invalid_rows = (
        df["InvoiceDate"].isna()
        | ~np.isfinite(df["Dollars"])
        | ~np.isfinite(df["Freight"])
    )

    print("Invalid records removed:", invalid_rows.sum())
    df = df.loc[~invalid_rows].copy()

    # Filter non-negative values
    df = df[(df["Dollars"] >= 0) & (df["Freight"] >= 0)]

    return df

  def prepare_features(self, df: pd.DataFrame):
    """Select features and target variable.

    X: Dollars y: Freight
    """
    X = df[["Dollars"]]
    y = df["Freight"]
    return X, y

  def split_data(self, df, cutoff="2024-11-01"):
    """Perform chronological train-test split."""
    df = self.clean_freight_data(df)

    cutoff = pd.Timestamp(cutoff)

    train_mask = df["InvoiceDate"] < cutoff
    test_mask = df["InvoiceDate"] >= cutoff

    X, y = self.prepare_features(df)

    X_train = X.loc[train_mask]
    y_train = y.loc[train_mask]

    X_test = X.loc[test_mask]
    y_test = y.loc[test_mask]

    if X_train.empty or X_test.empty:
      raise ValueError("Train or test set is empty. Check the cutoff date.")

    print("Training records:", len(X_train))
    print("Testing records:", len(X_test))
    print("Cutoff date:", cutoff.date())

    return X_train, X_test, y_train, y_test

  def prepare_data(self, db_path: str, cutoff="2024-11-01"):
    """Complete end-to-end preprocessing and splitting pipeline."""
    print("\n" + "=" * 60)
    print("STARTING INVOICE DATA PREPROCESSING PIPELINE")
    print("=" * 60)


    df = self.load_data(db_path)

    
    X_train, X_test, y_train, y_test = self.split_data(df, cutoff=cutoff)

    print("\n" + "=" * 60)
    print("✅ PREPROCESSING PIPELINE COMPLETE")
    print("=" * 60)

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
  from pathlib import Path
  
  preprocessor = DataPreprocessor()

  project_root = Path(__file__).resolve().parent.parent
  db_path = project_root / "data" / "inventory.db"
  if not db_path.exists():
    db_path = Path("data/inventory.db")

  X_train, X_test, y_train, y_test = preprocessor.prepare_data(str(db_path))

  print(f"\n📊 Verification Shapes:")
  print(f"   X_train: {X_train.shape}, y_train: {y_train.shape}")
  print(f"   X_test: {X_test.shape}, y_test: {y_test.shape}")