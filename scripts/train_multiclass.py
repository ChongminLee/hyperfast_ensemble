from __future__ import annotations
import argparse, yaml, os, json
import numpy as np
from pathlib import Path
from hyperfast_ensemble.data_io import load_tcga_csv, label_encode, stratified_split
from hyperfast_ensemble.preprocess import PCATransformer
from hyperfast_ensemble.models.hyperfast_wrapper import HyperfastWrapper
from hyperfast_ensemble.models.xgb_model import get_xgb_classifier, fit_xgb
from hyperfast_ensemble.models.lgbm_model import get_lgbm_classifier, fit_lgbm
from hyperfast_ensemble.ensemble import majority_vote
from hyperfast_ensemble.metrics import multiclass_metrics, per_class_confusion
from hyperfast_ensemble.utils import set_seed

def main():
    ap = argparse.ArgumentParser(description="Multiclass TCGA classification (Hyperfast + XGB + LGBM).")
    ap.add_argument("--data-path", required=True, help="CSV with features and a 'label' column.")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--drop-cols", nargs="*", default=None)
    ap.add_argument("--n-pcs", type=int, default=500)
    ap.add_argument("--out-dir", default="./outputs/multiclass")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    set_seed(args.seed)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    # Load config if provided
    cfg = {}
    if args.config and os.path.exists(args.config):
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)

    X_df, y = load_tcga_csv(args.data_path, label_col=args.label_col, drop_cols=args.drop_cols)
    y_enc, le, inv_map = label_encode(y)

    X_tr, X_te, y_tr, y_te = stratified_split(X_df, y_enc, test_size=0.2, seed=args.seed)

    # PCA
    n_pcs = args.n_pcs
    pca = PCATransformer(n_components=n_pcs, whiten=bool(cfg.get("pca", {}).get("whiten", False)))
    X_tr_pca = pca.fit_transform(X_tr)
    X_te_pca = pca.transform(X_te)

    # Hyperfast (with fallback)
    hf = HyperfastWrapper(use_fallback=False).fit(X_tr_pca, y_tr)
    y_hf = hf.predict(X_te_pca)

    # XGBoost
    xgb_obj = "multi:softprob"
    xgb = get_xgb_classifier(objective=xgb_obj, **cfg.get("xgboost", {}))
    xgb = fit_xgb(xgb, X_tr_pca, y_tr, class_weight="balanced")
    y_xgb = xgb.predict(X_te_pca)

    # LightGBM
    lgbm_obj = "multiclass"
    lgbm = get_lgbm_classifier(objective=lgbm_obj, **cfg.get("lightgbm", {}))
    lgbm = fit_lgbm(lgbm, X_tr_pca, y_tr, class_weight="balanced")
    y_lgbm = lgbm.predict(X_te_pca)

    # Majority vote
    y_ens = majority_vote([y_hf, y_xgb, y_lgbm])

    # Metrics
    m_hf = multiclass_metrics(y_te, y_hf)
    m_xg = multiclass_metrics(y_te, y_xgb)
    m_lb = multiclass_metrics(y_te, y_lgbm)
    m_en = multiclass_metrics(y_te, y_ens)

    results = {
        "n_pcs": n_pcs,
        "hyperfast": m_hf,
        "xgboost": m_xg,
        "lightgbm": m_lb,
        "ensemble_hard_vote": m_en,
        "confusion_matrix_ensemble": per_class_confusion(y_te, y_ens).tolist(),
        "label_map": inv_map,
    }
    with open(os.path.join(args.out_dir, f"results_multiclass_{n_pcs}.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
