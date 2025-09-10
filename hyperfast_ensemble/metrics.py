from __future__ import annotations
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from typing import Dict, Any, Optional, Tuple

def multiclass_metrics(y_true, y_pred) -> Dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
    }

def binary_metrics(y_true, y_pred, y_prob=None) -> Dict[str, float]:
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, average="binary"),
        "precision": precision_score(y_true, y_pred, average="binary", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="binary", zero_division=0),
    }
    # Handle AUC robustly
    try:
        if y_prob is not None:
            out["auc"] = roc_auc_score(y_true, y_prob)
    except Exception:
        pass
    return out

def per_class_confusion(y_true, y_pred) -> np.ndarray:
    return confusion_matrix(y_true, y_pred)
