"""Notebook 09 — Model Interpretation (patched)
Outputs to outputs/notebook_09/

Key change:
- Linear-coefficient panel resolves the locked linear winner (LR or EN) when available.
"""
import argparse
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
plt.rcParams.update({"figure.dpi": 150, "figure.facecolor": "white", "axes.facecolor": "#f8f8f8", "axes.spines.top": False, "axes.spines.right": False})
MODEL_DEFS = [("M1a_overall_survival", "M1a overall survival"), ("M1b_cancer_specific_survival", "M1b cancer-specific survival"), ("M3_pam50_subtype", "M3 PAM50 subtype"), ("M4_histologic_grade", "M4 histologic grade")]


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


def load_perm(perm_dir, key):
    return pd.read_csv(perm_dir / f"{key}_perm_imp.csv")


def feat_color(f):
    return "#2ecc71" if f.startswith("gene_programme") else ("#9b59b6" if f.startswith("ohe_") else "#3498db")


def model_suffix(name):
    if name == "RF":
        return "rf"
    if isinstance(name, str) and ("elastic" in name.lower() or name == "EN (L1+L2)" or name.lower() == "en"):
        return "en"
    return "lr"


def resolve_linear_model(models_dir, key):
    best_name_path = models_dir / f"{key}_best_name.joblib"
    if best_name_path.exists():
        best_name = joblib.load(best_name_path)
        if model_suffix(best_name) in {"lr", "en"}:
            candidate = models_dir / f"{key}_{model_suffix(best_name)}.joblib"
            if candidate.exists():
                return candidate, str(best_name)
    for suffix, label in [("en", "EN (L1+L2)"), ("lr", "LR (L2)")]:
        candidate = models_dir / f"{key}_{suffix}.joblib"
        if candidate.exists():
            return candidate, label
    return None, None


def fig_global(perm_dir, models_dir, out):
    fig, axes = plt.subplots(3, 2, figsize=(16, 16))
    axes = axes.flatten()
    for ax, (key, label) in zip(axes[:4], MODEL_DEFS):
        pi = load_perm(perm_dir, key).head(20)
        ax.barh(pi["feature"][::-1], pi["importance_mean"][::-1], xerr=pi["importance_std"][::-1], color=[feat_color(f) for f in pi["feature"][::-1]], alpha=0.85)
        ax.set_title(label)
        ax.axvline(0, color="black", lw=0.8)
    for ax, key, title in [(axes[4], "M2a_overall_survival_cox", "M2a overall survival — Cox coefficients"), (axes[5], "M2b_cancer_specific_cox", "M2b cancer-specific — Cox coefficients")]:
        cph = joblib.load(models_dir / f"{key}_cox.joblib")
        top = cph.summary.assign(abs_coef=lambda d: d["coef"].abs()).sort_values("abs_coef", ascending=False).head(20)
        ax.barh(top.index[::-1], top["coef"][::-1], color=["#e74c3c" if v > 0 else "#3498db" for v in top["coef"][::-1]], alpha=0.85)
        ax.set_title(title)
        ax.axvline(0, color="black", lw=0.8)
    fig.tight_layout()
    save(fig, out / "35_global_importance.png")


def fig_linear(models_dir, out):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    for ax, (key, label) in zip(axes, MODEL_DEFS):
        model_path, model_label = resolve_linear_model(models_dir, key)
        if model_path is None:
            ax.axis("off")
            ax.text(0.02, 0.8, f"{label}: no saved linear winner or fallback linear model", fontsize=10)
            continue
        clf = joblib.load(model_path)
        feats = joblib.load(models_dir / f"{key}_features.joblib")
        coef = np.abs(clf.coef_).mean(axis=0) if getattr(clf.coef_, "ndim", 1) == 2 and clf.coef_.shape[0] > 1 else np.abs(clf.coef_[0])
        ser = pd.Series(coef, index=feats).sort_values(ascending=False).head(20)
        ax.barh(ser.index[::-1], ser.values[::-1], color=[feat_color(f) for f in ser.index[::-1]], alpha=0.85)
        ax.set_title(f"{label} — {model_label}")
    fig.tight_layout()
    save(fig, out / "36_linear_coefficients.png")


def main():
    p = argparse.ArgumentParser(description="Notebook 09 — Model interpretation")
    p.add_argument("--perm-dir", type=Path, default=Path("outputs") / "permutation_importance")
    p.add_argument("--models-dir", type=Path, default=Path("outputs") / "models")
    p.add_argument("--output-dir", type=Path, default=Path("outputs") / "notebook_09")
    args = p.parse_args()
    fig_global(args.perm_dir, args.models_dir, args.output_dir)
    fig_linear(args.models_dir, args.output_dir)


if __name__ == "__main__":
    main()
