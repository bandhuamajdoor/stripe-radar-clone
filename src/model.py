import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
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
    # probabilities for fraud class (1)
    y_prob = rf_model.predict_proba(X_test)[:, 1]
    
    metrics = calculate_metrics(y_test, y_pred, y_prob)
    print_metrics_summary(metrics, model_name="Baseline Random Forest")
    return rf_model, preprocessor, metrics

if __name__ == "__main__":
    train_baseline_rf()
