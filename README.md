# Hyperfast + Boosted Trees Ensemble for Biomarker-Based Cancer Classification

This repository reproduces the pipeline from the manuscript:
> *Provably_Robust_Pre_Trained_Ensembles_for_Biomarker_Based_Cancer_Classification*.

## Overview
We classify TCGA cancer types from biomarker features (liquid-biopsy-like tabular features) using:
- **Hyperfast** (pre-trained hypernetwork, prototype-style last layer),
- **LightGBM** and **XGBoost** (high-performance gradient-boosted trees),
- A **hard-vote (majority)** ensemble *(Hyperfast + LightGBM + XGBoost)*.

We also provide a mathematically grounded analysis (prototype margins and hard-vote bounds) in the paper appendix.

## Data
- Primary data: **TCGA**. We use processed features.
- Preprocessed TCGA and training code (anonymous) are hosted: **[link in the paper]**.
- For convenience, you can point this code to your local copies via CLI flags.

## Pipeline
1. **Load and validate data** (tabular features `X`, labels `y`).
2. **Preprocess**: impute, standardize, and reduce via **PCA** (variable #PCs).
3. **Train models**: Hyperfast (or a prototype fallback), LightGBM, XGBoost.
4. **Evaluate** with accuracy, balanced accuracy, AUC (binary), F1, per-class metrics; compute bootstrap CIs.
5. **Ensemble** via **majority vote** and re-evaluate.
6. Optional **visualizations**: t-SNE, performance vs. number of PCs.

## Reproducibility
- Script entry points under `scripts/` reproduce **binary** (cancer vs. non-cancer; one-vs-rest) and **multiclass** results.
- All random seeds are controlled.
- Model hyperparameters are in `configs/default.yaml` (edit as needed).

## Quick Start
```bash
# (Recommended) create a virtual environment, then:
pip install -r requirements.txt

# Multiclass example (500 PCs)
python -m scripts.train_multiclass --data-path /path/to/TCGA_processed.csv --n-pcs 500 --out-dir ./outputs/MC_500

# Binary cancer vs. non-cancer
python -m scripts.train_binary --data-path /path/to/TCGA_processed.csv --task cancer-vs-normal --n-pcs 200 --out-dir ./outputs/BIN_200

# One-vs-rest (e.g., BRCA vs. non-BRCA)
python -m scripts.train_binary --data-path /path/to/TCGA_processed.csv --task one-vs-rest --positive-class BRCA --n-pcs 500 --out-dir ./outputs/BRCA_OVR_500
```

## Folder Structure
```
hyperfast_ensemble_repo/
├─ hyperfast_ensemble/
│  ├─ data_io.py           # loading, label encoding, splitting
│  ├─ preprocess.py        # impute, scale, PCA, t-SNE
│  ├─ metrics.py           # metrics + bootstrap CIs
│  ├─ ensemble.py          # majority voting utilities
│  ├─ utils.py             # seeding, logging
│  ├─ plots.py             # (optional) t-SNE & performance plots
│  └─ models/
│     ├─ hyperfast_wrapper.py  # wrapper around HyperFastClassifier (with prototype fallback)
│     ├─ xgb_model.py          # XGBoost helpers
│     └─ lgbm_model.py         # LightGBM helpers
├─ scripts/
│  ├─ train_multiclass.py   # full multiclass pipeline
│  ├─ train_binary.py       # cancer-vs-normal and one-vs-rest
│  └─ reproduce_all.py      # convenience orchestrator
├─ configs/
│  └─ default.yaml          # default parameters
├─ requirements.txt
├─ CITATION.cff
└─ README.md
```

## Notes
- Download the hyperfast package (hyperfast.ckpt) from the original paper by Bonet.
- If `hyperfast` is unavailable on your system, the wrapper **falls back** to a prototype (nearest-centroid-like) classifier in PCA space so that the ensemble still runs end-to-end.
- For transparency and speed, we keep hyperparameter grids modest by default; expand as needed for thorough sweeps.

