import os
import joblib
import numpy as np
import pytest
import sys
sys.path.append('src')

from sklearn.neural_network import MLPRegressor

def test_autoencoder_artifacts():
    assert os.path.exists("models/autoencoder.joblib"), "autoencoder.joblib model not found"
    assert os.path.exists("models/autoencoder_threshold.joblib"), "autoencoder_threshold.joblib threshold not found"
    
    autoencoder = joblib.load("models/autoencoder.joblib")
    threshold = joblib.load("models/autoencoder_threshold.joblib")
    
    assert isinstance(autoencoder, MLPRegressor)
    assert isinstance(threshold, float) or isinstance(threshold, np.float64)

def test_autoencoder_reconstruction():
    autoencoder = joblib.load("models/autoencoder.joblib")
    threshold = joblib.load("models/autoencoder_threshold.joblib")
    
    # 5 dummy transactions with 30 preprocessed features
    dummy_input = np.random.normal(0, 1, size=(5, 30))
    
    # Reconstruct input
    reconstructed = autoencoder.predict(dummy_input)
    assert reconstructed.shape == (5, 30)
    
    # Calculate Mean Squared Error (MSE) per sample
    mse = np.mean(np.square(dummy_input - reconstructed), axis=1)
    assert mse.shape == (5,)
    
    # Predict based on threshold
    y_pred = np.where(mse > threshold, 1, 0)
    assert y_pred.shape == (5,)
    assert np.all((y_pred == 0) | (y_pred == 1))
