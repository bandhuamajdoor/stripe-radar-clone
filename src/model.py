import os
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from data_pipeline import DataPreprocessor
from metrics import calculate_metrics, print_metrics_summary

def train_baseline_rf(data_dir="data", models_dir="models", random_state=42):
    """
    Fits preprocessing pipeline and trains a baseline Random Forest model.
    Saves the preprocessor and model artifacts.
    """
    os.makedirs(models_dir, exist_ok=True)
    
    # 1. Load Data
    train_path = os.path.join(data_dir, "train_transactions.csv")
    test_path = os.path.join(data_dir, "test_transactions.csv")
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError("Train or test transaction data files are missing.")
        
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # 2. Preprocess
    print("Fitting preprocessor scaler...")
    preprocessor = DataPreprocessor()
    X_train, y_train = preprocessor.fit_transform(train_df)
    X_test, y_test = preprocessor.transform(test_df)
    
    # Save preprocessor scaler
    preprocessor_path = os.path.join(models_dir, "preprocessor.joblib")
    joblib.dump(preprocessor, preprocessor_path)
    print(f"Saved preprocessor to {preprocessor_path}")
    
    # 3. Train Baseline Random Forest
    print("Training baseline Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    
    # Save baseline model
    model_path = os.path.join(models_dir, "baseline_rf.joblib")
    joblib.dump(rf_model, model_path)
    print(f"Saved baseline Random Forest model to {model_path}")
    
    # 4. Evaluate
    y_pred = rf_model.predict(X_test)
    y_prob = rf_model.predict_proba(X_test)[:, 1]
    
    metrics = calculate_metrics(y_test, y_pred, y_prob)
    print_metrics_summary(metrics, model_name="Baseline Random Forest")
    return rf_model, preprocessor, metrics

def train_isolation_forest(data_dir="data", models_dir="models", contamination=0.01, random_state=42):
    """
    Trains an Isolation Forest model for anomaly detection.
    Saves the model artifact.
    """
    os.makedirs(models_dir, exist_ok=True)
    
    # Load Data
    train_path = os.path.join(data_dir, "train_transactions.csv")
    test_path = os.path.join(data_dir, "test_transactions.csv")
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError("Train or test transaction data files are missing.")
        
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Load preprocessor
    preprocessor_path = os.path.join(models_dir, "preprocessor.joblib")
    if os.path.exists(preprocessor_path):
        preprocessor = joblib.load(preprocessor_path)
        X_train, y_train = preprocessor.transform(train_df)
        X_test, y_test = preprocessor.transform(test_df)
    else:
        preprocessor = DataPreprocessor()
        X_train, y_train = preprocessor.fit_transform(train_df)
        X_test, y_test = preprocessor.transform(test_df)
        joblib.dump(preprocessor, preprocessor_path)
        
    print(f"Training Isolation Forest (contamination={contamination})...")
    iforest = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1
    )
    iforest.fit(X_train)
    
    # Save model
    model_path = os.path.join(models_dir, "isolation_forest.joblib")
    joblib.dump(iforest, model_path)
    print(f"Saved Isolation Forest model to {model_path}")
    
    # Evaluate: -1 is outlier (fraud), 1 is inlier (normal)
    y_pred_raw = iforest.predict(X_test)
    y_pred = np.where(y_pred_raw == -1, 1, 0)
    
    # Anomaly score (higher score = more anomalous)
    raw_scores = iforest.score_samples(X_test)
    y_prob = -raw_scores
    
    metrics = calculate_metrics(y_test, y_pred, y_prob)
    print_metrics_summary(metrics, model_name="Isolation Forest")
    return iforest, metrics

if __name__ == "__main__":
    train_baseline_rf()
    train_isolation_forest()
