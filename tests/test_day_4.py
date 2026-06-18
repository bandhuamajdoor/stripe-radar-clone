import os
import joblib
import numpy as np
import pytest
import sys
sys.path.append('src')

from sklearn.ensemble import IsolationForest

def test_isolation_forest_artifacts():
    assert os.path.exists("models/isolation_forest.joblib"), "isolation_forest.joblib model not found"
    
    iforest = joblib.load("models/isolation_forest.joblib")
    assert isinstance(iforest, IsolationForest)

def test_isolation_forest_prediction():
    iforest = joblib.load("models/isolation_forest.joblib")
    
    # 5 dummy transactions with 30 preprocessed features
    dummy_input = np.random.normal(0, 1, size=(5, 30))
    
    # Predict inliers (1) vs outliers (-1)
    raw_pred = iforest.predict(dummy_input)
    assert raw_pred.shape == (5,)
    assert np.all((raw_pred == 1) | (raw_pred == -1))
    
    # Check mapping
    y_pred = np.where(raw_pred == -1, 1, 0)
    assert np.all((y_pred == 0) | (y_pred == 1))
    
    # Risk scores (positive)
    raw_scores = iforest.score_samples(dummy_input)
    y_prob = -raw_scores
    assert y_prob.shape == (5,)
    assert np.all(y_prob > 0.0)  # -score_samples should be positive
