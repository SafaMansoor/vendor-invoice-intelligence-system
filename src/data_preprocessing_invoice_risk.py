import sqlite3
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from scipy.stats import ttest_ind
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class InvoiceDataPreprocessor:
    """
    A comprehensive data preprocessing, anomaly detection, and scaling pipeline 
    for the Invoice Intelligence System.
    """
    def __init__(self, csv_path: str = None, db_path: str = None):
        self.csv_path = csv_path
        self.db_path = db_path
        self.df = None
        self.flagged_df = None
        self.normal_df = None
        self.significant_features = []
        self.non_significant_features = []
        self.scaler = StandardScaler()

    def load_data(self) -> pd.DataFrame:
        """Loads processed invoice data from a CSV file."""
        self.df = pd.read_csv(self.csv_path)
        print(f"Data loaded successfully from CSV. Shape: {self.df.shape}")
        return self.df

    def load_data_from_sqlite(self, query: str) -> pd.DataFrame:
        """Loads raw data from a SQLite inventory database using a custom SQL query."""
        conn = sqlite3.connect(self.db_path)
        self.df = pd.read_sql_query(query, conn)
        conn.close()
        print(f"Data loaded successfully from SQLite. Shape: {self.df.shape}")
        return self.df

    def detect_anomalies(self, features: list = None, contamination: float = 0.1, random_state: int = 42) -> pd.DataFrame:
        """
        Applies an unsupervised Isolation Forest model to flag risky/abnormal invoices.
        Maps output into a binary 'flag_invoice' column (1 = anomaly/risk, 0 = normal).
        """
        if features is None:
            features = [
                'invoice_quantity', 'invoice_dollars', 'Freight', 
                'days_po_to_invoice', 'total_brands', 'avg_receiving_delay',
            ]
        
        iso = IsolationForest(contamination=contamination, random_state=random_state)
        preds = iso.fit_predict(self.df[features])
        
        # Convert -1 (anomaly) to 1, and 1 (normal) to 0
        self.df["flag_invoice"] = [1 if x == -1 else 0 for x in preds]
        
        # Split into flagged and normal subsets for downstream analysis
        self.flagged_df = self.df[self.df['flag_invoice'] == 1]
        self.normal_df = self.df[self.df['flag_invoice'] == 0]
        
        print("Anomaly Detection Complete:")
        print(self.df["flag_invoice"].value_counts())
        return self.df

    def perform_statistical_tests(self, metrics: list = None) -> tuple:
        """
        Performs Independent Two-Sample T-Tests (Welch's t-test) to evaluate 
        which features show statistically significant differences between flagged and normal invoices.
        """
        if metrics is None:
            metrics = [
                'invoice_quantity', 'invoice_dollars', 'Freight', 'days_po_to_invoice',
                'days_to_pay', 'total_brands', 'total_item_quantity', 
                'total_item_dollars', 'avg_receiving_delay'
            ]

        self.significant_features = []
        self.non_significant_features = []
        results = []

        for metric in metrics:
            flagged_mean = self.flagged_df[metric].mean()
            normal_mean = self.normal_df[metric].mean()

            t_stat, p_value = ttest_ind(
                self.flagged_df[metric].dropna(),
                self.normal_df[metric].dropna(),
                equal_var=False
            )

            if p_value < 0.05:
                self.significant_features.append(metric)
                results.append({
                    "metric": metric,
                    "flagged_mean": round(float(flagged_mean), 2),
                    "normal_mean": round(float(normal_mean), 2),
                    "p_value": round(float(p_value), 3)
                })
            else:
                self.non_significant_features.append(metric)

        print(f"Significant Features (p < 0.05): {self.significant_features}")
        print(f"Non-Significant Features (p >= 0.05): {self.non_significant_features}")
        return self.significant_features, self.non_significant_features, results

    def get_modeling_matrices(self) -> tuple:
        """
        Extracts final feature matrix (X) and target vector (y) 
        using statistically significant features for downstream machine learning tasks.
        """
        if not self.significant_features:
            raise ValueError("Run perform_statistical_tests() first to identify features.")
        
        X = self.df[self.significant_features]
        y = self.df['flag_invoice']
        return X, y

    def split_and_scale_data(self, test_size: float = 0.2, random_state: int = 42) -> tuple:
        """
        Splits data into train/test subsets and applies StandardScaler 
        (fitted strictly on X_train to prevent data leakage).
        """
        X, y = self.get_modeling_matrices()
        
        # 1. Stratified train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state, 
            stratify=y
        )
        
        # 2. Standardize features (Fit on train, transform both train and test)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Convert arrays back to DataFrames to retain feature column names
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=self.significant_features, index=X_train.index)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=self.significant_features, index=X_test.index)
        
        print("Data split and scaled successfully!")
        print(f"Training shape: {X_train_scaled.shape}, Testing shape: {X_test_scaled.shape}")
        return X_train_scaled, X_test_scaled, y_train, y_test

    def save_processed_data(self, output_path: str):
        """Saves the processed dataframe containing risk flags to a CSV file."""
        if self.df is not None:
            self.df.to_csv(output_path, index=False)
            print(f"Saved processed data successfully to {output_path}!")
        else:
            raise ValueError("No dataframe available to save. Load and process data first.")

    def save_scaler(self, output_path: str):
        """Saves the fitted StandardScaler to disk for inference."""
        import joblib
        joblib.dump(self.scaler, output_path)
        print(f"Saved scaler successfully to {output_path}!")



if __name__ == "__main__":
    from pathlib import Path
    
    # Locate project root and files robustly
    project_root = Path(__file__).resolve().parent.parent
    csv_file = project_root / "notebook" / "processed_invoices.csv"
    if not csv_file.exists():
        csv_file = Path("processed_invoices.csv")
    
    # Initialize processor with dataset path
    preprocessor = InvoiceDataPreprocessor(csv_path=str(csv_file))
    
    # 1. Load data
    preprocessor.load_data()
    
    # 2. Run Isolation Forest anomaly detection
    preprocessor.detect_anomalies()
    
    # 3. Perform T-test statistical validation
    preprocessor.perform_statistical_tests()
    
    # 4. Split and scale data for machine learning models
    X_train, X_test, y_train, y_test = preprocessor.split_and_scale_data(test_size=0.2, random_state=42)
    
    # 5. Export results and scaler
    out_csv = project_root / "notebook" / "processed_invoices.csv"
    preprocessor.save_processed_data(str(out_csv))
    
    scaler_path = project_root / "models" / "risk_scaler.pkl"
    preprocessor.save_scaler(str(scaler_path))