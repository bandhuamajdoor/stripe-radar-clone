import os
import pandas as pd
import numpy as np
import pytest
import sys
sys.path.append('src')

from data_pipeline import TransactionGenerator, DataPreprocessor

def test_transaction_generator_shape():
    generator = TransactionGenerator(random_state=42)
    n_samples = 500
    fraud_ratio = 0.02
    
    df = generator.generate_batch(n_samples=n_samples, fraud_ratio=fraud_ratio)
    
    # Assert type and shape
    assert isinstance(df, pd.DataFrame)
    assert len(df) == n_samples
    assert df.shape[1] == 31  # V1-V28 + Time + Amount + Class
    
    # Assert essential columns exist
    assert "Time" in df.columns
    assert "Amount" in df.columns
    assert "Class" in df.columns
    for i in range(1, 29):
        assert f"V{i}" in df.columns

def test_transaction_generator_fraud_ratio():
    generator = TransactionGenerator(random_state=42)
    n_samples = 1000
    fraud_ratio = 0.05
    
    df = generator.generate_batch(n_samples=n_samples, fraud_ratio=fraud_ratio)
    
    # Assert number of frauds matches expected ceiling
    n_expected_fraud = int(np.ceil(n_samples * fraud_ratio))
    assert df["Class"].sum() == n_expected_fraud

def test_data_preprocessor_scaling():
    generator = TransactionGenerator(random_state=42)
    df = generator.generate_batch(n_samples=200, fraud_ratio=0.01)
    
    preprocessor = DataPreprocessor()
    
    # Assert fit and transform
    with pytest.raises(ValueError):
        preprocessor.transform(df)  # Throws error if not fitted
        
    preprocessor.fit(df)
    X, y = preprocessor.transform(df)
    
    assert X.shape == (200, 30)  # scaled_Time, scaled_Amount, V1-V28
    assert "scaled_Time" in X.columns
    assert "scaled_Amount" in X.columns
    assert "Time" not in X.columns
    assert "Amount" not in X.columns
    assert len(y) == 200

def test_generated_files_exist():
    # Verify that prepare_and_save_data output files exist
    assert os.path.exists("data/train_transactions.csv"), "Train dataset CSV not found"
    assert os.path.exists("data/test_transactions.csv"), "Test dataset CSV not found"
    
    # Load and check shape
    train_df = pd.read_csv("data/train_transactions.csv")
    assert len(train_df) > 0
    assert train_df.shape[1] == 31
