from __future__ import annotations
import lightgbm as lgb
import numpy as np
from sklearn.utils.class_weight import compute_sample_weight

def get_lgbm_classifier(objective: str = "multiclass", **kwargs):
    params = dict(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=-1,
        num_leaves=64,
        subsample=0.9,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective=objective,
        n_jobs=-1,
    )
    params.update(kwargs)
    return lgb.LGBMClassifier(**params)

def fit_lgbm(model, X_train, y_train, class_weight: str | None = None):
    if class_weight == "balanced":
        sw = compute_sample_weight(class_weight="balanced", y=y_train)
        model.fit(X_train, y_train, sample_weight=sw)
    else:
        model.fit(X_train, y_train)
    return model
