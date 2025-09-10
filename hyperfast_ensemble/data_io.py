from __future__ import annotations
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from typing import Tuple, Dict, Any, Optional

def load_tcga_csv(path: str, label_col: str = "label", drop_cols: Optional[list] = None) -> Tuple[pd.DataFrame, pd.Series]:
    """Load TCGA-like tabular data where rows=samples, columns=features, and one column is the label.
    - `label_col`: column name for labels (str).
    - `drop_cols`: list of columns to drop before modeling (ids, patient ids, etc.).
    Returns (X_df, y_series).
    """
    df = pd.read_csv(path)
    if drop_cols:
        df = df.drop(columns=drop_cols, errors="ignore")
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in: {path}")
    y = df[label_col]
    X = df.drop(columns=[label_col])
    # Coerce to numeric (non-numeric to NaN, to be imputed later)
    X = X.apply(pd.to_numeric, errors="coerce")
    return X, y

def label_encode(y: pd.Series) -> Tuple[np.ndarray, LabelEncoder, Dict[int, str]]:
    le = LabelEncoder()
    y_enc = le.fit_transform(y.astype(str))
    inv_map = {i: lab for i, lab in enumerate(le.classes_)}
    return y_enc, le, inv_map

def stratified_split(X: pd.DataFrame, y_enc: np.ndarray, test_size: float = 0.2, seed: int = 42):
    return train_test_split(X, y_enc, test_size=test_size, random_state=seed, stratify=y_enc)

def one_vs_rest_labels(y_str: pd.Series, positive_class: str) -> np.ndarray:
    return (y_str.astype(str) == str(positive_class)).astype(int).values
