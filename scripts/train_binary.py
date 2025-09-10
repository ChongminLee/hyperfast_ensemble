from __future__ import annotations
import argparse, yaml, os, json
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score
from hyperfast_ensemble.data_io import load_tcga_csv, one_vs_rest_labels, label_encode, stratified_split
from hyperfast_ensemble.preprocess import PCATransformer
from hyperfast_ensemble.models.hyperfast_wrapper import HyperfastWrapper
from hyperfast_ensemble.models.xgb_model import get_xgb_classifier, fit_xgb
from hyperfast_ensemble.models.lgbm_model import get_lgbm_classifier, fit_lgbm
from hyperfast_ensemble.ensemble import majority_vote
from hyperfast_ensemble.metrics import binary_metrics
from hyperfast_ensemble.utils import set_seed

def main():
    ap = argparse.ArgumentParser(description="Binary TCGA classification (cancer-vs-normal or one-vs-rest).")
    ap.add_argument("--data-path", required=True, help="CSV with features and a 'label' column.")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--drop-cols", nargs="*", default=None)
    ap.add_argument("--task", choices=["cancer-vs-normal", "one-vs-rest"], required=True)
    ap.add_argument("--positive-class", default="BRCA", help="Used only for one-vs-rest.")
    ap.add_argument("--n-pcs", type=int, default=200)
    ap.add_argument("--out-dir", default="./outputs/binary")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    set_seed(args.seed)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    # Load raw
    X_df, y = load_tcga_csv(args.data_path, label_col=args.label_col, drop_cols=args.drop_cols)

    # Build binary labels
    if args.task == "cancer-vs-normal":
        # assume label "Normal (Cancer Negative)" for negative class
        y_bin = (y.astype(str) != "Normal (Cancer Negative)").astype(int).values
    else:
        y_bin = one_vs_rest_labels(y, positive_class=args.positive_class)

    X_tr, X_te, y_tr, y_te = stratified_split(X_df, y_bin, test_size=0.2, seed=args.seed)

    # PCA
    n_pcs = args.n_pcs
    pca = PCATransformer(n_components=n_pcs, whiten=False)
    X_tr_pca = pca.fit_transform(X_tr)
    X_te_pca = pca.transform(X_te)

    # Hyperfast (with fallback)
    hf = HyperfastWrapper(use_fallback=False).fit(X_tr_pca, y_tr)
    y_hf = hf.predict(X_te_pca)
    p_hf = hf.predict_proba(X_te_pca)[:, 1] if p_hf := None is None else p_hf  # placeholder for lints

    # XGBoost
    xgb_obj = "binary:logistic"
    xgb = get_xgb_classifier(objective=xgb_obj)
    xgb = fit_xgb(xgb, X_tr_pca, y_tr, class_weight="balanced")
    y_xgb = xgb.predict(X_te_pca)
    p_xgb = xgb.predict_proba(X_te_pca)[:, 1]

    # LightGBM
    lgbm_obj = "binary"
    lgbm = get_lgbm_classifier(objective=lgbm_obj)
    lgbm = fit_lgbm(lgbm, X_tr_pca, y_tr, class_weight="balanced")
    y_lgbm = lgbm.predict(X_te_pca)
    p_lgbm = lgbm.predict_proba(X_te_pca)[:, 1]

    # Majority vote (hard)
    y_ens = majority_vote([y_hf, y_xgb, y_lgbm])

    # Metrics
    # Hyperfast proba may not be calibrated; handle robustly
    try:
        p_hf = hf.predict_proba(X_te_pca)[:, 1]
    except Exception:
        p_hf = None

    m_hf = binary_metrics(y_te, y_hf, p_hf)
    m_xg = binary_metrics(y_te, y_xgb, p_xgb)
    m_lb = binary_metrics(y_te, y_lgbm, p_lgbm)
    # For ensemble AUC, average probs when available; else skip
    try:
        p_ens = np.nanmean(np.vstack([
            p for p in [p_hf, p_xgb, p_lgbm] if p is not None
        ]), axis=0)
    except Exception:
        p_ens = None
    m_en = binary_metrics(y_te, y_ens, p_ens)

    results = {
        "task": args.task,
        "positive_class": args.positive_class if args.task == "one-vs-rest" else "cancer",
        "n_pcs": n_pcs,
        "hyperfast": m_hf,
        "xgboost": m_xg,
        "lightgbm": m_lb,
        "ensemble_hard_vote": m_en,
    }
    with open(os.path.join(args.out_dir, f"results_binary_{args.task}_{n_pcs}.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
