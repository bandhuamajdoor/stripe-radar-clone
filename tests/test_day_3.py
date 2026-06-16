import os
import joblib
import numpy as np
import pytest
import sys
sys.path.append('src')

from metrics import calculate_metrics
from model import train_baseline_rf

def test_metrics_calculation():
    # Mock labels and predictions
    # 5 actual positives, 5 actual negatives
    y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    # 4 true positives, 1 false positive, 1 false negative, 4 true negatives
    y_pred = np.array([0, 0, 0, 0, 1, 0, 1, 1, 1, 1])
    
    metrics = calculate_metrics(y_true, y_pred)
    
    # Precision = TP / (TP + FP) = 4 / (4 + 1) = 0.8
    assert abs(metrics["precision"] - 0.8) < 1e-5
    
    # Recall = TP / (TP + FN) = 4 / (4 + 1) = 0.8
    assert abs(metrics["recall"] - 0.8) < 1e-5
    
    # F1 = 2 * (P * R) / (P + R) = 2 * (0.64) / 1.6 = 0.8
    assert abs(metrics["f1_score"] - 0.8) < 1e-5
    
    # F2 = (1 + 2^2) * (P * R) / (2^2 * P + R) = 5 * (0.64) / (4 * 0.8 + 0.8) = 3.2 / 4.0 = 0.8
    assert abs(metrics["f2_score"] - 0.8) < 1e-5

def test_saved_model_artifacts():
    # Assert model artifacts exist
    assert os.path.exists("models/baseline_rf.joblib"), "baseline_rf.joblib model not found"
    assert os.path.exists("models/preprocessor.joblib"), "preprocessor.joblib scaler not found"
    
    # Load model and preprocessor
    rf_model = joblib.load("models/baseline_rf.joblib")
    preprocessor = joblib.load("models/preprocessor.joblib")
    
    # Verify loaded types
    from sklearn.ensemble import RandomForestClassifier
    from data_pipeline import DataPreprocessor
    assert isinstance(rf_model, RandomForestClassifier)
    assert isinstance(preprocessor, DataPreprocessor)

def test_model_predict_format():
    rf_model = joblib.load("models/baseline_rf.joblib")
    
    # Create a dummy transaction row (Time, Amount, V1-V28)
    dummy_input = np.random.normal(0, 1, size=(5, 30))  # 5 samples, 30 features
    
    y_pred = rf_model.predict(dummy_input)
    y_prob = rf_model.predict_proba(dummy_input)[:, 1]
    
    assert y_pred.shape == (5,)
    assert y_prob.shape == (5,)
    assert np.all((y_pred == 0) | (y_pred == 1))
    assert np.all((y_prob >= 0.0) & (y_prob <= 1.0))
