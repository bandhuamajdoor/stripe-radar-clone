import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, fbeta_score, precision_recall_curve, auc

def calculate_metrics(y_true, y_pred, y_prob=None):
    """
    Calculates key evaluation metrics for highly imbalanced fraud detection.
    Metrics:
        - Precision: Out of all flagged transactions, how many were actually fraud?
        - Recall: Out of all actual fraud, how many did we catch?
        - F1-Score: Harmonic mean of precision and recall.
        - F2-Score: Weighting recall higher than precision (crucial for catching fraud).
        - PR-AUC: Area under the Precision-Recall curve (better than ROC-AUC for imbalanced data).
    """
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    f2 = fbeta_score(y_true, y_pred, beta=2, zero_division=0)
    
    metrics = {
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "f2_score": float(f2),
        "pr_auc": 0.0
    }
    
    if y_prob is not None:
        precision_pts, recall_pts, _ = precision_recall_curve(y_true, y_prob)
        metrics["pr_auc"] = float(auc(recall_pts, precision_pts))
        
    return metrics

def print_metrics_summary(metrics, model_name="Model"):
    """
    Helper to print a clean summary of metrics.
    """
    print(f"=== {model_name} Performance Summary ===")
    print(f"Precision:  {metrics['precision']:.4f}")
    print(f"Recall:     {metrics['recall']:.4f}")
    print(f"F1-Score:   {metrics['f1_score']:.4f}")
    print(f"F2-Score:   {metrics['f2_score']:.4f} (recall-optimized)")
    print(f"PR-AUC:     {metrics['pr_auc']:.4f}")
    print("=======================================")
