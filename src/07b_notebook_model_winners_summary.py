"""
Notebook 07b — Model Winners Summary
Adds:
  47_model_winners_summary.png
"""
import argparse, warnings
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

warnings.filterwarnings("ignore")
plt.rcParams.update({"figure.dpi": 150, "figure.facecolor": "white", "axes.facecolor": "#f8f8f8",
                     "axes.spines.top": False, "axes.spines.right": False})

METRICS = {
    "M1a_overall_survival":"AUC-ROC",
    "M1b_cancer_specific_survival":"AUC-ROC",
    "M2a_overall_survival_cox":"C-index",
    "M2b_cancer_specific_cox":"C-index",
    "M3_pam50_subtype":"Macro-F1",
    "M4_histologic_grade":"QW-Kappa",
}

def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")

def choose_best_row(path, model_key):
    df = pd.read_csv(path)
    score_col = next((c for c in ["score","auc_roc","f1_macro","qw_kappa","c_index"] if c in df.columns), None)
    if score_col is None:
        num_cols = df.select_dtypes(include="number").columns.tolist()
        if not num_cols:
            raise ValueError(f"No numeric score column found in {path}")
        score_col = num_cols[-1]
    row = df.sort_values(score_col, ascending=False).iloc[0]
    return {"model":model_key, "algorithm":str(row.get("algorithm","CoxPH")), "score":float(row[score_col]), "score_col":score_col}

def main():
    p=argparse.ArgumentParser(description="Model winners summary")
    p.add_argument("--cv-dir", type=Path, default=Path("outputs")/"cv_results")
    p.add_argument("--output-dir", type=Path, default=Path("outputs")/"notebook_07b")
    args=p.parse_args()
    rows=[]
    for m in METRICS:
        path = args.cv_dir / m / "cv_results.csv"
        if path.exists():
            rows.append(choose_best_row(path, m))
    if not rows:
        raise FileNotFoundError("No cv_results.csv files found.")
    df = pd.DataFrame(rows)
    df["label"] = df["model"].str.replace("_"," ", regex=False)
    fig, ax = plt.subplots(figsize=(11,6))
    colors = []
    for alg in df["algorithm"].str.lower():
        if "cox" in alg:
            colors.append("#9b59b6")
        elif "rf" in alg or "forest" in alg:
            colors.append("#3498db")
        elif "elastic" in alg or alg=="en":
            colors.append("#e67e22")
        else:
            colors.append("#2ecc71")
    y = range(len(df))
    ax.barh(y, df["score"], color=colors, alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(df["label"])
    ax.invert_yaxis()
    ax.set_xlabel("Best validation score")
    ax.set_title("Final outcomes at a glance — winning algorithm by task")
    for i, r in df.reset_index(drop=True).iterrows():
        metric = METRICS.get(r["model"], r["score_col"])
        ax.text(r["score"] + 0.01, i, f"{r['algorithm']} | {metric}={r['score']:.3f}", va="center", fontsize=8)
    fig.text(0.5, 0.02, "Color coding: CoxPH=purple, RF=blue, Elastic Net=orange, LR=green", ha="center", fontsize=8, color="#555")
    fig.tight_layout(rect=[0,0.04,1,1])
    save(fig, args.output_dir / "47_model_winners_summary.png")
    print("\nDone.")
if __name__=="__main__":
    main()
