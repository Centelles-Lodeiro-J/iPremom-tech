"""
Notebook 03b — Feature Selection Support Plots
Adds:
  fs01_selected_feature_overlap.png
"""
import argparse, warnings
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")
plt.rcParams.update({
    "figure.dpi": 150, "figure.facecolor": "white",
    "axes.facecolor": "#f8f8f8", "axes.spines.top": False,
    "axes.spines.right": False,
})

MODELS = [
    "M1a_overall_survival","M1b_cancer_specific_survival",
    "M2a_overall_survival_cox","M2b_cancer_specific_cox",
    "M3_pam50_subtype","M4_histologic_grade",
]

def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")

def main():
    p=argparse.ArgumentParser(description="Feature selection support plots")
    p.add_argument("--splits-dir", type=Path, default=Path("outputs")/"splits")
    p.add_argument("--output-dir", type=Path, default=Path("outputs")/"notebook_03b")
    args=p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    feat_sets = {}
    for m in MODELS:
        path = args.splits_dir / m / "feature_selection" / "selected_feature_list.csv"
        if path.exists():
            feat_sets[m] = pd.read_csv(path)["feature"].astype(str).tolist()
    all_feats = sorted(set(f for vals in feat_sets.values() for f in vals))
    if not all_feats:
        raise FileNotFoundError("No selected_feature_list.csv files found. Run Notebook 03 first.")
    counts = {f: sum(f in vals for vals in feat_sets.values()) for f in all_feats}
    top_feats = sorted(all_feats, key=lambda f: (-counts[f], f))[:60]
    mat = pd.DataFrame(0, index=top_feats, columns=list(feat_sets))
    for m, feats in feat_sets.items():
        mat.loc[mat.index.isin(feats), m] = 1
    fig, ax = plt.subplots(figsize=(10, max(8, 0.22*len(top_feats))))
    sns.heatmap(mat, cmap=sns.color_palette(["#ecf0f1","#2c3e50"]), cbar=False,
                linewidths=0.5, linecolor="#d0d7de", ax=ax)
    ax.set_title("Selected-feature overlap across tasks\nTop 60 most shared selected features")
    ax.set_xlabel("Model")
    ax.set_ylabel("Feature")
    ax.set_xticklabels([c.replace("_"," ") for c in mat.columns], rotation=30, ha="right")
    ax.set_yticklabels([i.replace("_"," ")[:40] for i in mat.index], fontsize=7)
    fig.tight_layout()
    save(fig, args.output_dir / "fs01_selected_feature_overlap.png")
    print("\nDone.")
if __name__=="__main__":
    main()
