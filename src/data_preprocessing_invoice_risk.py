import sqlite3
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import ttest_ind
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class InvoiceDataPreprocessor:
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

    @staticmethod
    def _create_invoice_risk_label(row):
        if abs(row["invoice_dollars"] - row["total_item_dollars"]) > 5:
            return 1
        if row["avg_receiving_delay"] > 10:
            return 1
        return 0

    def apply_labels(self) -> pd.DataFrame:
        """Applies risk labels and splits dataframe into flagged and normal subsets."""
        if self.df is None:
            raise ValueError("Load data first before applying labels.")
        
        self.df["flag_invoice"] = self.df.apply(self._create_invoice_risk_label, axis=1)
        self.flagged_df = self.df[self.df["flag_invoice"] == 1]
        self.normal_df = self.df[self.df["flag_invoice"] == 0]
        print(f"Labels applied. Flagged: {len(self.flagged_df)}, Normal: {len(self.normal_df)}")
        return self.df

    def perform_statistical_tests(self, metrics: list = None) -> tuple:
        """Performs Welch's t-test to evaluate statistically significant features."""
        if self.flagged_df is None or self.normal_df is None:
            raise ValueError("Run apply_labels() before performing statistical tests.")

        if metrics is None:
            metrics = [
                'invoice_quantity', 'invoice_dollars', 'Freight',
                'days_po_to_invoice', 'days_to_pay', 'total_brands',
                'total_item_quantity', 'total_item_dollars', 'avg_receiving_delay'
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
        if not self.significant_features:
            raise ValueError("Run perform_statistical_tests() first to identify features.")
        X = self.df[self.significant_features]
        y = self.df['flag_invoice']
        return X, y

    def split_and_scale_data(self, test_size: float = 0.2, random_state: int = 42) -> tuple:
        """Splits data into train/test subsets and applies StandardScaler."""
        X, y = self.get_modeling_matrices()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state,
        )

        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=self.significant_features,
            index=X_train.index
        )
        X_test_scaled = pd.DataFrame(
            self.scaler.transform(X_test),
            columns=self.significant_features,
            index=X_test.index
        )

        print("Data split and scaled successfully!")
        print(f"Training shape: {X_train_scaled.shape}, Testing shape: {X_test_scaled.shape}")
        return X_train_scaled, X_test_scaled, y_train, y_test

    def save_processed_data(self, output_path: str):
        if self.df is not None:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            self.df.to_csv(output_path, index=False)
            print(f"Saved processed data successfully to {output_path}!")
        else:
            raise ValueError("No dataframe available to save.")

    def save_scaler(self, output_path: str):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.scaler, output_path)
        print(f"Saved scaler successfully to {output_path}!")
