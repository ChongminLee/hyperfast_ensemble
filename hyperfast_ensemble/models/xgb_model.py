from __future__ import annotations
import xgboost as xgb
from sklearn.utils.class_weight import compute_sample_weight

def get_xgb_classifier(objective: str = "multi:softprob", **kwargs):
    params = dict(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective=objective,
        n_jobs=-1,
        tree_method="hist",
    )
    params.update(kwargs)
    return xgb.XGBClassifier(**params)

def fit_xgb(model, X_train, y_train, class_weight: str | None = None):
    if class_weight == "balanced":
        sw = compute_sample_weight(class_weight="balanced", y=y_train)
        model.fit(X_train, y_train, sample_weight=sw, eval_metric="mlogloss")
    else:
        model.fit(X_train, y_train, eval_metric="mlogloss")
    return model
