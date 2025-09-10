from __future__ import annotations
import argparse, os, subprocess, sys, json
from pathlib import Path

def main():
    ap = argparse.ArgumentParser(description="Reproduce key experiments: multiclass (500 PCs), binary tasks.")
    ap.add_argument("--data-path", required=True)
    ap.add_argument("--out-root", default="./outputs/repro")
    args = ap.parse_args()

    Path(args.out_root).mkdir(parents=True, exist_ok=True)

    # Multiclass at 500 PCs
    subprocess.run([sys.executable, "-m", "scripts.train_multiclass",
                    "--data-path", args.data_path,
                    "--n-pcs", "500",
                    "--out-dir", str(Path(args.out_root)/"multiclass_500")], check=True)

    # Cancer vs. Normal at 200 PCs
    subprocess.run([sys.executable, "-m", "scripts.train_binary",
                    "--data-path", args.data_path,
                    "--task", "cancer-vs-normal",
                    "--n-pcs", "200",
                    "--out-dir", str(Path(args.out_root)/"bin_cancer_vs_normal_200")], check=True)

    # BRCA one-vs-rest at 500 PCs
    subprocess.run([sys.executable, "-m", "scripts.train_binary",
                    "--data-path", args.data_path,
                    "--task", "one-vs-rest",
                    "--positive-class", "BRCA",
                    "--n-pcs", "500",
                    "--out-dir", str(Path(args.out_root)/"bin_brca_ovr_500")], check=True)

    print("Reproduction complete. See:", args.out_root)

if __name__ == "__main__":
    main()
