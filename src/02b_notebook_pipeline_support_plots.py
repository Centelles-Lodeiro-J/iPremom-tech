"""
Notebook 02b — Additional Pipeline Support Plots
Adds:
  15b_pipeline_flowchart.png
  15c_before_after_cleaning_summary.png
  15d_feature_family_composition.png
"""
import argparse, warnings
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
plt.rcParams.update({
    "figure.dpi": 150, "figure.facecolor": "white",
    "axes.facecolor": "#f8f8f8", "axes.spines.top": False,
    "axes.spines.right": False, "axes.titlesize": 11,
    "axes.titleweight": "bold", "axes.labelsize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
})
HEAD_BG = "#2c3e50"; BORDER = "#c8d0db"

def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")

def fig_pipeline_flowchart(out):
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis("off")
    boxes = [
        (0.05, 0.78, 0.18, 0.12, "Raw CSV\n1,987 × 523", "#e8f4fd"),
        (0.28, 0.78, 0.18, 0.12, "Stage 2\nDeterministic fixes", "#fff8e1"),
        (0.51, 0.78, 0.18, 0.12, "Stage 3\nShared stratified split", "#edf7ed"),
        (0.74, 0.78, 0.18, 0.12, "Metadata\ntrain/test indices", "#f3e5f5"),
        (0.05, 0.52, 0.18, 0.12, "Stage 4\nTrain-only imputation", "#fff8e1"),
        (0.28, 0.52, 0.18, 0.12, "Stage 5\nFeature engineering", "#e8f4fd"),
        (0.51, 0.52, 0.18, 0.12, "Stage 6\nTrain-only NMF", "#edf7ed"),
        (0.74, 0.52, 0.18, 0.12, "NMF outputs\ncomponents + model", "#f3e5f5"),
        (0.16, 0.26, 0.18, 0.12, "Stage 7\nPer-model splits", "#fff8e1"),
        (0.40, 0.26, 0.18, 0.12, "Stage 8\nConfig + imputation log", "#f3e5f5"),
        (0.64, 0.26, 0.18, 0.12, "Downstream notebooks\n03–10", "#e8f4fd"),
    ]
    for x,y,w,h,txt,color in boxes:
        ax.add_patch(patches.FancyBboxPatch((x,y),w,h, boxstyle="round,pad=0.01,rounding_size=0.02",
                                            facecolor=color, edgecolor=BORDER, lw=1.2, transform=ax.transAxes))
        ax.text(x+w/2, y+h/2, txt, transform=ax.transAxes, ha="center", va="center",
                fontsize=10, color=HEAD_BG, fontweight="bold")
    arrows = [
        ((0.23,0.84),(0.28,0.84)), ((0.46,0.84),(0.51,0.84)), ((0.69,0.84),(0.74,0.84)),
        ((0.14,0.78),(0.14,0.64)), ((0.37,0.78),(0.37,0.64)), ((0.60,0.78),(0.60,0.64)),
        ((0.69,0.58),(0.74,0.58)), ((0.23,0.58),(0.28,0.58)), ((0.46,0.58),(0.51,0.58)),
        ((0.60,0.52),(0.73,0.38)), ((0.23,0.52),(0.25,0.38)), ((0.58,0.32),(0.64,0.32)),
    ]
    for (x1,y1),(x2,y2) in arrows:
        ax.annotate("", xy=(x2,y2), xytext=(x1,y1), xycoords=ax.transAxes,
                    arrowprops=dict(arrowstyle="->", lw=1.8, color="#34495e"))
    ax.text(0.5, 0.96, "End-to-end pipeline flowchart", transform=ax.transAxes,
            ha="center", va="top", fontsize=14, fontweight="bold", color=HEAD_BG)
    ax.text(0.5, 0.06,
            "Key defense point: split first, then fit all repair/transformation steps on training rows only.\nEvery downstream notebook depends on these saved artifacts.",
            transform=ax.transAxes, ha="center", fontsize=9, color="#555")
    save(fig, out / "15b_pipeline_flowchart.png")

def fig_before_after_cleaning(raw, out):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    er_before = raw["er_status_measured_by_ihc"].fillna("Missing").astype(str).str.strip().value_counts()
    corrected = raw["er_status_measured_by_ihc"].fillna("Missing").astype(str).str.strip().replace({"Posyte":"Positive","Positve":"Positive"})
    er_after = corrected.value_counts()
    top_before = er_before.head(6)
    axes[0,0].bar(top_before.index.astype(str), top_before.values, color="#e74c3c", alpha=0.8)
    axes[0,0].set_title("ER IHC labels — before correction")
    axes[0,0].tick_params(axis="x", rotation=25)
    top_after = er_after.head(6)
    axes[0,1].bar(top_after.index.astype(str), top_after.values, color="#2ecc71", alpha=0.8)
    axes[0,1].set_title("ER IHC labels — after correction")
    axes[0,1].tick_params(axis="x", rotation=25)
    age = raw["age_at_diagnosis"]
    age_clean = age.where(age.between(18,100))
    axes[1,0].hist(age.dropna(), bins=30, color="#3498db", alpha=0.8, edgecolor="white")
    axes[1,0].axvline(18, ls="--", lw=1, color="#e74c3c")
    axes[1,0].axvline(100, ls="--", lw=1, color="#e74c3c")
    axes[1,0].set_title("Age at diagnosis — raw")
    axes[1,1].hist(age_clean.dropna(), bins=30, color="#2ecc71", alpha=0.8, edgecolor="white")
    axes[1,1].set_title("Age at diagnosis — after plausible-range filter")
    fig.suptitle("Before/after cleaning summary", fontsize=13, fontweight="bold")
    fig.tight_layout()
    save(fig, out / "15c_before_after_cleaning_summary.png")

def classify_feature(col):
    if col.startswith("gene_programme_"):
        return "Gene programmes"
    if col.startswith("ohe_"):
        return "One-hot categorical"
    if col.endswith("_was_missing") or col.endswith("_missing_flag"):
        return "Missingness flags"
    if col.endswith("_bin"):
        return "Binary clinical"
    if col.endswith("_ord"):
        return "Ordinal clinical"
    return "Continuous clinical"

def fig_feature_family_composition(splits_dir, out):
    models = [
        "M1a_overall_survival","M1b_cancer_specific_survival",
        "M2a_overall_survival_cox","M2b_cancer_specific_cox",
        "M3_pam50_subtype","M4_histologic_grade",
    ]
    rows=[]
    families = ["Continuous clinical","Binary clinical","Ordinal clinical","One-hot categorical","Missingness flags","Gene programmes"]
    for m in models:
        x_path = splits_dir / m / "X_train.csv"
        if not x_path.exists():
            continue
        cols = pd.read_csv(x_path, nrows=1).columns.tolist()
        counts = {fam:0 for fam in families}
        for c in cols:
            fam = classify_feature(c)
            counts[fam] = counts.get(fam,0)+1
        for fam in families:
            rows.append({"model":m, "family":fam, "count":counts.get(fam,0)})
    df = pd.DataFrame(rows)
    pivot = df.pivot(index="model", columns="family", values="count").fillna(0)[families]
    fig, ax = plt.subplots(figsize=(12,7))
    left = np.zeros(len(pivot))
    colors = ["#3498db","#2ecc71","#9b59b6","#f39c12","#95a5a6","#e74c3c"]
    for fam, color in zip(families, colors):
        vals = pivot[fam].values
        ax.barh(range(len(pivot)), vals, left=left, label=fam, color=color, alpha=0.85)
        left += vals
    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels([m.replace("_"," ") for m in pivot.index])
    ax.set_xlabel("Number of features")
    ax.set_title("Feature-family composition by final model")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    save(fig, out / "15d_feature_family_composition.png")

def main():
    p=argparse.ArgumentParser(description="Additional pipeline support plots")
    p.add_argument("--input", type=Path, default=Path("data")/"FCS_ml_test_input_data_rna_mutation.csv")
    p.add_argument("--pipeline-outputs", type=Path, default=Path("outputs"))
    p.add_argument("--output-dir", type=Path, default=Path("outputs")/"notebook_02b")
    args=p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw=pd.read_csv(args.input)
    print(f"Generating figures → {args.output_dir}/\n")
    fig_pipeline_flowchart(args.output_dir)
    fig_before_after_cleaning(raw, args.output_dir)
    fig_feature_family_composition(args.pipeline_outputs/"splits", args.output_dir)
    print("\nDone.")
if __name__=="__main__":
    main()
