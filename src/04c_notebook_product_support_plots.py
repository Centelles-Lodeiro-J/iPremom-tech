"""
Notebook 04c — Product Support Plots
Adds:
  28c_m1b_calibration.png
  28d_m2b_km_risk_groups.png
"""
import argparse, warnings
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
plt.rcParams.update({
    "figure.dpi": 150, "figure.facecolor": "white",
    "axes.facecolor": "#f8f8f8", "axes.spines.top": False,
    "axes.spines.right": False, "axes.titlesize": 11,
    "axes.titleweight": "bold"
})

def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")

def _choose_best_model(cv_path):
    df = pd.read_csv(cv_path)
    score_col = next((c for c in ["score","auc_roc","f1_macro","qw_kappa"] if c in df.columns), None)
    if score_col is None:
        raise ValueError(f"No known score column found in {cv_path}")
    best = df.sort_values(score_col, ascending=False).iloc[0]
    alg = str(best["algorithm"]).lower()
    if "elastic" in alg or alg == "en":
        suffix = "en"
    elif "rf" in alg or "forest" in alg:
        suffix = "rf"
    else:
        suffix = "lr"
    return suffix, best["algorithm"], float(best[score_col])

def fig_m1b_calibration(splits_dir, models_dir, cv_dir, out):
    key = "M1b_cancer_specific_survival"
    suffix, alg_name, best_score = _choose_best_model(cv_dir / key / "cv_results.csv")
    model = joblib.load(models_dir / f"{key}_{suffix}.joblib")
    feats = joblib.load(models_dir / f"{key}_features.joblib")
    X_te = pd.read_csv(splits_dir / key / "X_test.csv")[feats]
    y_te = pd.read_csv(splits_dir / key / "y_test.csv").iloc[:,0].astype(int)
    if not hasattr(model, "predict_proba"):
        raise TypeError(f"Best model {alg_name} for {key} does not expose predict_proba().")
    p = model.predict_proba(X_te)[:,1]
    tmp = pd.DataFrame({"p":p, "y":y_te})
    tmp["bin"] = pd.qcut(tmp["p"], q=min(10, tmp["p"].nunique()), duplicates="drop")
    cal = tmp.groupby("bin", observed=False).agg(pred=("p","mean"), obs=("y","mean"), n=("y","size")).reset_index(drop=True)
    fig, axes = plt.subplots(1,2, figsize=(12,5))
    axes[0].plot([0,1],[0,1], ls="--", color="#7f8c8d")
    axes[0].plot(cal["pred"], cal["obs"], marker="o", lw=2, color="#2c3e50")
    axes[0].set_xlabel("Predicted probability")
    axes[0].set_ylabel("Observed event rate")
    axes[0].set_title(f"M1b calibration on test set\nBest model: {alg_name}  score={best_score:.3f}")
    for _, r in cal.iterrows():
        axes[0].text(r["pred"], r["obs"]+0.02, f"n={int(r['n'])}", fontsize=7, ha="center")
    axes[1].hist(p, bins=20, color="#3498db", alpha=0.85, edgecolor="white")
    axes[1].set_xlabel("Predicted cancer-specific risk")
    axes[1].set_ylabel("Patients")
    axes[1].set_title("Distribution of M1b predicted probabilities")
    fig.suptitle("M1b product-support plot: calibration + risk distribution", fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, out / "28c_m1b_calibration.png")

def km_curve(times, events):
    df = pd.DataFrame({"t": times, "e": events}).sort_values("t")
    S = 1.0
    n = len(df)
    curve = [(0.0, 1.0)]
    for t in sorted(df["t"].unique()):
        d = int(((df["t"] == t) & (df["e"] == 1)).sum())
        if d > 0 and n > 0:
            S *= 1 - d / n
        n -= int((df["t"] == t).sum())
        curve.append((float(t), S))
    return list(zip(*curve))

def fig_m2b_km_risk_groups(splits_dir, models_dir, out):
    key = "M2b_cancer_specific_cox"
    cph = joblib.load(models_dir / f"{key}_cox.joblib")
    feats = joblib.load(models_dir / f"{key}_features.joblib")
    X_te = pd.read_csv(splits_dir / key / "X_test.csv")[feats]
    y_te = pd.read_csv(splits_dir / key / "y_test.csv")
    risk = pd.Series(np.asarray(cph.predict_partial_hazard(X_te)).reshape(-1), index=X_te.index)
    groups = pd.qcut(risk.rank(method="first"), q=3, labels=["Low risk","Medium risk","High risk"])
    colors = {"Low risk":"#2ecc71","Medium risk":"#f39c12","High risk":"#e74c3c"}
    fig, axes = plt.subplots(1,2, figsize=(13,5))
    for g in ["Low risk","Medium risk","High risk"]:
        mask = groups == g
        t, s = km_curve(y_te.loc[mask,"time"].values, y_te.loc[mask,"event"].values)
        axes[0].step(t, s, where="post", lw=2, color=colors[g], label=f"{g} (n={int(mask.sum())})")
    axes[0].set_xlabel("Time (months)")
    axes[0].set_ylabel("Survival probability")
    axes[0].set_ylim(0,1.05)
    axes[0].set_title("M2b KM curves by predicted risk group")
    axes[0].legend(fontsize=8)
    grp = pd.DataFrame({"risk":risk, "group":groups, "event":y_te["event"], "time":y_te["time"]})
    summ = grp.groupby("group", observed=False).agg(
        median_risk=("risk","median"), event_rate=("event","mean"), median_time=("time","median"), n=("event","size")
    ).reindex(["Low risk","Medium risk","High risk"])
    x = np.arange(len(summ))
    axes[1].bar(x, summ["event_rate"], color=[colors[g] for g in summ.index], alpha=0.85)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(summ.index, rotation=20)
    axes[1].set_ylim(0,1)
    axes[1].set_ylabel("Cancer-specific event rate")
    axes[1].set_title("Observed event rate by predicted M2b risk group")
    for i, (_, r) in enumerate(summ.iterrows()):
        axes[1].text(i, r["event_rate"]+0.03, f"n={int(r['n'])}\nmed.time={r['median_time']:.0f}", ha="center", fontsize=8)
    fig.suptitle("M2b product-support plot: survival stratification on test set", fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, out / "28d_m2b_km_risk_groups.png")

def main():
    p=argparse.ArgumentParser(description="Product support plots for M1b and M2b")
    p.add_argument("--splits-dir", type=Path, default=Path("outputs")/"splits")
    p.add_argument("--models-dir", type=Path, default=Path("outputs")/"models")
    p.add_argument("--cv-dir", type=Path, default=Path("outputs")/"cv_results")
    p.add_argument("--output-dir", type=Path, default=Path("outputs")/"notebook_04c")
    args=p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Generating figures → {args.output_dir}/\n")
    fig_m1b_calibration(args.splits_dir, args.models_dir, args.cv_dir, args.output_dir)
    fig_m2b_km_risk_groups(args.splits_dir, args.models_dir, args.output_dir)
    print("\nDone.")
if __name__=="__main__":
    main()
