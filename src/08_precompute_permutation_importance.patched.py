"""Precompute permutation importance for final trained models (patched).
Output folder aligned to notebook number via downstream notebook_09.

Key change:
- Loads the locked winning classifier, including Elastic Net when selected.
"""
import argparse
import warnings
from pathlib import Path

import joblib
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import cohen_kappa_score, make_scorer

warnings.filterwarnings("ignore")
MODELS = [
    ("M1a_overall_survival", "binary"),
    ("M1b_cancer_specific_survival", "binary"),
    ("M3_pam50_subtype", "multi"),
    ("M4_histologic_grade", "ordinal"),
]


def model_suffix(name):
    if name == "RF":
        return "rf"
    if isinstance(name, str) and ("elastic" in name.lower() or name == "EN (L1+L2)" or name.lower() == "en"):
        return "en"
    return "lr"


def resolve_model_path(models_dir, key):
    best_name_path = models_dir / f"{key}_best_name.joblib"
    if best_name_path.exists():
        best_name = joblib.load(best_name_path)
        candidate = models_dir / f"{key}_{model_suffix(best_name)}.joblib"
        if candidate.exists():
            return candidate
    for suffix in ["rf", "en", "lr"]:
        candidate = models_dir / f"{key}_{suffix}.joblib"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No saved model found for {key}")


def main():
    p = argparse.ArgumentParser(description="Precompute permutation importance")
    p.add_argument("--splits-dir", type=Path, default=Path("outputs") / "splits")
    p.add_argument("--models-dir", type=Path, default=Path("outputs") / "models")
    p.add_argument("--output-dir", type=Path, default=Path("outputs") / "permutation_importance")
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    kappa_scorer = make_scorer(cohen_kappa_score, weights="quadratic")
    for key, task in MODELS:
        feat_path = args.models_dir / f"{key}_features.joblib"
        if not feat_path.exists():
            print(f"Skipping {key}: features not found")
            continue
        feats = joblib.load(feat_path)
        model_path = resolve_model_path(args.models_dir, key)
        clf = joblib.load(model_path)
        X = pd.read_csv(args.splits_dir / key / "X_test.csv")[feats]
        y_raw = pd.read_csv(args.splits_dir / key / "y_test.csv")
        y = y_raw.iloc[:, 0]
        scoring = "roc_auc" if task == "binary" else ("f1_macro" if task == "multi" else kappa_scorer)
        res = permutation_importance(clf, X, y, n_repeats=10, random_state=42, scoring=scoring)
        df = pd.DataFrame({"feature": X.columns, "importance_mean": res.importances_mean, "importance_std": res.importances_std}).sort_values("importance_mean", ascending=False)
        out = args.output_dir / f"{key}_perm_imp.csv"
        df.to_csv(out, index=False)
        print(f"Saved: {out}")


if __name__ == "__main__":
    main()
