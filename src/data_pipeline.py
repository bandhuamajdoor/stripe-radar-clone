import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
import os

class TransactionGenerator:
    """
    Generates synthetic transaction data that mimics the Kaggle Credit Card Fraud dataset.
    Features:
        - Time: Seconds elapsed since the start.
        - V1 to V28: PCA features.
        - Amount: Transaction amount.
        - Class: 0 (Normal) or 1 (Fraudulent).
    """
    def __init__(self, random_state=42):
        self.random_state = random_state
        np.random.seed(self.random_state)
        
    def generate_batch(self, n_samples=1000, fraud_ratio=0.0017, start_time=0):
        """
        Generates a batch of transactions.
        """
        n_fraud = int(np.ceil(n_samples * fraud_ratio))
        n_normal = n_samples - n_fraud
        
        # 1. Generate normal transactions
        # V1-V28 for normal transactions centered around 0 with low variance
        v_normal = np.random.normal(loc=0.0, scale=1.0, size=(n_normal, 28))
        
        # Amounts for normal transactions (log-normal, typically smaller)
        amount_normal = np.random.lognormal(mean=3.0, sigma=0.8, size=n_normal)
        
        # 2. Generate fraudulent transactions
        # V1-V28 for fraud are outliers (larger variance, shifted means)
        v_fraud = np.random.normal(loc=1.5, scale=2.5, size=(n_fraud, 28))
        # Add correlation in some specific features commonly indicating fraud
        v_fraud[:, 0:5] += 2.0  # Shift V1 to V5 further away
        
        # Amounts for fraud transactions (typically larger on average)
        amount_fraud = np.random.lognormal(mean=5.0, sigma=1.2, size=n_fraud)
        
        # 3. Combine
        classes = np.array([0] * n_normal + [1] * n_fraud)
        features_v = np.vstack((v_normal, v_fraud))
        amounts = np.concatenate((amount_normal, amount_fraud))
        
        # Time progression (e.g., average 10 seconds between transactions)
        times = start_time + np.cumsum(np.random.exponential(scale=10.0, size=n_samples))
        
        # 4. Create DataFrame
        columns = [f"V{i}" for i in range(1, 29)]
        df = pd.DataFrame(features_v, columns=columns)
        df["Time"] = times
        df["Amount"] = amounts
        df["Class"] = classes
        
        # Shuffle
        df = df.sample(frac=1.0, random_state=self.random_state).reset_index(drop=True)
        return df

class DataPreprocessor:
    """
    Handles preprocessing for transaction data.
    Scales 'Amount' and 'Time' using RobustScaler (less sensitive to outliers).
    """
    def __init__(self):
        self.amount_scaler = RobustScaler()
        self.time_scaler = RobustScaler()
        self.is_fitted = False
        
    def fit(self, df):
        """
        Fits scalers on Amount and Time features.
        """
        self.amount_scaler.fit(df[["Amount"]])
        self.time_scaler.fit(df[["Time"]])
        self.is_fitted = True
        return self
        
    def transform(self, df):
        """
        Transforms Amount and Time in a copy of the dataframe.
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor has not been fitted yet.")
            
        df_transformed = df.copy()
        df_transformed["scaled_Amount"] = self.amount_scaler.transform(df[["Amount"]])
        df_transformed["scaled_Time"] = self.time_scaler.transform(df[["Time"]])
        
        # Reorder columns to match standard training structure:
        # scaled_Time, scaled_Amount, V1...V28, and Class if present
        cols_order = ["scaled_Time", "scaled_Amount"] + [f"V{i}" for i in range(1, 29)]
        
        X = df_transformed[cols_order]
        
        if "Class" in df_transformed.columns:
            y = df_transformed["Class"]
            return X, y
        return X

    def fit_transform(self, df):
        """
        Fits scalers and transforms the dataframe.
        """
        return self.fit(df).transform(df)

def prepare_and_save_data(data_dir="data", n_samples=50000, fraud_ratio=0.0017, random_state=42):
    """
    Generates train and test datasets and saves them.
    """
    os.makedirs(data_dir, exist_ok=True)
    
    generator = TransactionGenerator(random_state=random_state)
    df = generator.generate_batch(n_samples=n_samples, fraud_ratio=fraud_ratio)
    
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["Class"], random_state=random_state)
    
    train_path = os.path.join(data_dir, "train_transactions.csv")
    test_path = os.path.join(data_dir, "test_transactions.csv")
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"Generated {len(train_df)} training and {len(test_df)} testing samples.")
    print(f"Train frauds: {train_df['Class'].sum()}, Test frauds: {test_df['Class'].sum()}")
    return train_path, test_path
