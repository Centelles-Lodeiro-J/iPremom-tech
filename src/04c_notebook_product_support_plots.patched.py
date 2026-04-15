"""Notebook 04c — Product Support Plots (advanced)
Adds:
  28c_m1b_calibration.png
  28d_m2b_km_risk_groups.png
  28e_m1b_threshold_tradeoffs.png

Advanced additions beyond the enhanced version:
- Formal threshold reporting for M1b on the locked test set.
- Calibration intercept/slope summaries saved as CSV.
- KM step data for M2b risk-group plots saved as CSV.
"""
import argparse
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

warnings.filterwarnings("ignore")
plt.rcParams.update({
    "figure.dpi": 150,
    "figure.facecolor": "white",
    "axes.facecolor": "#f8f8f8",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
})


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


def model_suffix(name):
    if name == "RF":
        return "rf"
    if isinstance(name, str) and ("elastic" in name.lower() or name == "EN (L1+L2)" or name.lower() == "en"):
        return "en"
    return "lr"


def resolve_best_model(key, models_dir, cv_dir):
    best_name_path = models_dir / f"{key}_best_name.joblib"
    if best_name_path.exists():
        best_name = joblib.load(best_name_path)
    else:
        final_path = cv_dir / key / "final_test_results.csv"
        if not final_path.exists():
            raise FileNotFoundError(f"No locked-winner metadata found for {key}.")
        best_name = str(pd.read_csv(final_path).iloc[0]["algorithm"])
    suffix = model_suffix(best_name)
    final_path = cv_dir / key / "final_test_results.csv"
    final_row = pd.read_csv(final_path).iloc[0].to_dict() if final_path.exists() else {}
    return suffix, str(best_name), final_row


def compute_expected_calibration_error(probs, y_true, n_bins=10):
    probs = np.asarray(probs, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    mask = np.isfinite(probs) & np.isfinite(y_true)
    probs = probs[mask]
    y_true = y_true[mask]
    if probs.size == 0:
        return np.nan
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        left, right = bins[i], bins[i + 1]
        sel = (probs >= left) & (probs <= right) if i == n_bins - 1 else (probs >= left) & (probs < right)
        if not np.any(sel):
            continue
        ece += abs(y_true[sel].mean() - probs[sel].mean()) * (sel.sum() / probs.size)
    return float(ece)


def compute_binary_metrics_summary(probs, y_true):
    y_true = pd.Series(y_true).astype(int)
    probs = pd.Series(probs, dtype=float)
    out = {
        "roc_auc": np.nan,
        "pr_auc": np.nan,
        "brier_score": np.nan,
        "ece": np.nan,
        "prevalence": float(y_true.mean()) if len(y_true) else np.nan,
        "n": int(len(y_true)),
    }
    if len(y_true) == 0:
        return out
    if y_true.nunique() >= 2:
        out["roc_auc"] = float(roc_auc_score(y_true, probs))
        out["pr_auc"] = float(average_precision_score(y_true, probs))
    out["brier_score"] = float(brier_score_loss(y_true, probs))
    out["ece"] = compute_expected_calibration_error(probs, y_true)
    return out


def compute_calibration_summary(probs, y_true, max_bins=10):
    probs = pd.Series(probs, dtype=float)
    y_true = pd.Series(y_true, dtype=float)
    tmp = pd.DataFrame({"p": probs, "y": y_true}).dropna()
    if tmp.empty:
        raise ValueError("Calibration summary cannot be computed on empty inputs.")
    n_bins = min(max_bins, max(1, int(tmp["p"].nunique())))
    tmp["bin"] = pd.qcut(tmp["p"], q=n_bins, duplicates="drop")
    cal = tmp.groupby("bin", observed=False).agg(pred=("p", "mean"), obs=("y", "mean"), n=("y", "size")).reset_index(drop=True)
    return cal


def compute_calibration_regression(probs, y_true):
    probs = pd.Series(probs, dtype=float)
    y_true = pd.Series(y_true).astype(int)
    eps = 1e-6
    probs = probs.clip(eps, 1 - eps)
    x = np.log(probs / (1 - probs)).to_numpy().reshape(-1, 1)
    y = y_true.to_numpy()
    if len(np.unique(y)) < 2:
        return {"calibration_intercept": np.nan, "calibration_slope": np.nan}
    clf = LogisticRegression(C=1e6, solver="lbfgs", max_iter=5000)
    clf.fit(x, y)
    return {
        "calibration_intercept": float(clf.intercept_[0]),
        "calibration_slope": float(clf.coef_.ravel()[0]),
    }


def compute_threshold_metrics(probs, y_true, thresholds=(0.10, 0.20, 0.30, 0.40, 0.50)):
    probs = pd.Series(probs, dtype=float)
    y_true = pd.Series(y_true).astype(int)
    rows = []
    for thr in thresholds:
        pred = (probs >= thr).astype(int)
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        tn = int(((pred == 0) & (y_true == 0)).sum())
        fn = int(((pred == 0) & (y_true == 1)).sum())
        sens = tp / (tp + fn) if (tp + fn) else np.nan
        spec = tn / (tn + fp) if (tn + fp) else np.nan
        ppv = tp / (tp + fp) if (tp + fp) else np.nan
        npv = tn / (tn + fn) if (tn + fn) else np.nan
        rows.append({
            "threshold": float(thr),
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "sensitivity": sens,
            "specificity": spec,
            "ppv": ppv,
            "npv": npv,
            "predicted_positive_rate": float(pred.mean()),
            "accuracy": float((pred == y_true).mean()),
        })
    return pd.DataFrame(rows)


def fig_m1b_calibration(splits_dir, models_dir, cv_dir, out):
    key = "M1b_cancer_specific_survival"
    suffix, alg_name, final_row = resolve_best_model(key, models_dir, cv_dir)
    model = joblib.load(models_dir / f"{key}_{suffix}.joblib")
    feats = joblib.load(models_dir / f"{key}_features.joblib")
    X_te = pd.read_csv(splits_dir / key / "X_test.csv")[feats]
    y_te = pd.read_csv(splits_dir / key / "y_test.csv").iloc[:, 0].astype(int)
    if not hasattr(model, "predict_proba"):
        raise TypeError(f"Best model {alg_name} for {key} does not expose predict_proba().")
    p = model.predict_proba(X_te)[:, 1]
    cal = compute_calibration_summary(p, y_te)
    cal.to_csv(out / "28c_m1b_calibration_summary.csv", index=False)
    metrics = compute_binary_metrics_summary(p, y_te)
    pd.DataFrame([metrics]).to_csv(out / "28c_m1b_binary_metrics_summary.csv", index=False)
    calib_reg = compute_calibration_regression(p, y_te)
    pd.DataFrame([calib_reg]).to_csv(out / "28c_m1b_calibration_regression.csv", index=False)
    thr_df = compute_threshold_metrics(p, y_te)
    thr_df.to_csv(out / "28c_m1b_threshold_metrics.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot([0, 1], [0, 1], ls="--", color="#7f8c8d")
    axes[0].plot(cal["pred"], cal["obs"], marker="o", lw=2, color="#2c3e50")
    axes[0].set_xlabel("Predicted probability")
    axes[0].set_ylabel("Observed event rate")
    test_score = final_row.get("test_score", np.nan)
    pr_auc = final_row.get("pr_auc", metrics["pr_auc"])
    brier = final_row.get("brier_score", metrics["brier_score"])
    ece = final_row.get("ece", metrics["ece"])
    axes[0].set_title(
        "M1b calibration on locked test set\n"
        f"Winner: {alg_name} | AUC={test_score:.3f} | PR-AUC={pr_auc:.3f}\n"
        f"Brier={brier:.3f} | ECE={ece:.3f}"
    )
    for _, r in cal.iterrows():
        axes[0].text(r["pred"], r["obs"] + 0.02, f"n={int(r['n'])}", fontsize=7, ha="center")
    axes[1].hist(p, bins=20, color="#3498db", alpha=0.85, edgecolor="white")
    axes[1].set_xlabel("Predicted cancer-specific risk")
    axes[1].set_ylabel("Patients")
    axes[1].set_title("Distribution of M1b predicted probabilities")
    fig.suptitle("M1b product-support plot: calibration + risk distribution", fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, out / "28c_m1b_calibration.png")

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.plot(thr_df["threshold"], thr_df["sensitivity"], marker="o", lw=2, label="Sensitivity")
    ax2.plot(thr_df["threshold"], thr_df["specificity"], marker="o", lw=2, label="Specificity")
    ax2.plot(thr_df["threshold"], thr_df["ppv"], marker="o", lw=2, label="PPV")
    ax2.plot(thr_df["threshold"], thr_df["npv"], marker="o", lw=2, label="NPV")
    ax2.set_xlabel("Decision threshold")
    ax2.set_ylabel("Metric value")
    ax2.set_ylim(0, 1.05)
    ax2.set_title(
        "M1b threshold trade-offs on locked test set\n"
        f"Calibration intercept={calib_reg['calibration_intercept']:.3f}, slope={calib_reg['calibration_slope']:.3f}"
    )
    ax2.legend(fontsize=8, ncol=2)
    fig2.tight_layout()
    save(fig2, out / "28e_m1b_threshold_tradeoffs.png")


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


def compute_risk_group_summary(risk, times, events, q=3):
    risk = pd.Series(risk, dtype=float)
    times = pd.Series(times, dtype=float)
    events = pd.Series(events, dtype=float)
    groups = pd.qcut(risk.rank(method="first"), q=q, labels=["Low risk", "Medium risk", "High risk"])
    grp = pd.DataFrame({"risk": risk, "group": groups, "event": events, "time": times})
    summ = grp.groupby("group", observed=False).agg(
        median_risk=("risk", "median"),
        event_rate=("event", "mean"),
        median_time=("time", "median"),
        n=("event", "size"),
    ).reindex(["Low risk", "Medium risk", "High risk"])
    return summ.reset_index(), groups


def fig_m2b_km_risk_groups(splits_dir, models_dir, out):
    key = "M2b_cancer_specific_cox"
    cph = joblib.load(models_dir / f"{key}_cox.joblib")
    feats = joblib.load(models_dir / f"{key}_features.joblib")
    X_te = pd.read_csv(splits_dir / key / "X_test.csv")[feats]
    y_te = pd.read_csv(splits_dir / key / "y_test.csv")
    risk = pd.Series(np.asarray(cph.predict_partial_hazard(X_te)).reshape(-1), index=X_te.index)
    summary_df, groups = compute_risk_group_summary(risk, y_te["time"], y_te["event"])
    summary_df.to_csv(out / "28d_m2b_risk_group_summary.csv", index=False)
    pd.DataFrame({"row": np.arange(len(groups)), "group": pd.Series(groups, dtype=str)}).to_csv(out / "28d_m2b_risk_group_assignments.csv", index=False)

    colors = {"Low risk": "#2ecc71", "Medium risk": "#f39c12", "High risk": "#e74c3c"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    km_rows = []
    for g in ["Low risk", "Medium risk", "High risk"]:
        mask = groups == g
        t, s = km_curve(y_te.loc[mask, "time"].values, y_te.loc[mask, "event"].values)
        axes[0].step(t, s, where="post", lw=2, color=colors[g], label=f"{g} (n={int(mask.sum())})")
        km_rows.extend([{"group": g, "time": float(tt), "survival": float(ss)} for tt, ss in zip(t, s)])
    pd.DataFrame(km_rows).to_csv(out / "28d_m2b_km_curve_data.csv", index=False)
    axes[0].set_xlabel("Time (months)")
    axes[0].set_ylabel("Survival probability")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("M2b KM curves by predicted risk group")
    axes[0].legend(fontsize=8)

    x = np.arange(len(summary_df))
    axes[1].bar(x, summary_df["event_rate"], color=[colors[g] for g in summary_df["group"]], alpha=0.85)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(summary_df["group"], rotation=20)
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Cancer-specific event rate")
    axes[1].set_title("Observed event rate by predicted M2b risk group")
    for i, r in summary_df.iterrows():
        axes[1].text(i, r["event_rate"] + 0.03, f"n={int(r['n'])}\nmed.time={r['median_time']:.0f}", ha="center", fontsize=8)
    fig.suptitle("M2b product-support plot: survival stratification on locked test set", fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, out / "28d_m2b_km_risk_groups.png")


def main():
    p = argparse.ArgumentParser(description="Product support plots for M1b and M2b")
    p.add_argument("--splits-dir", type=Path, default=Path("outputs") / "splits")
    p.add_argument("--models-dir", type=Path, default=Path("outputs") / "models")
    p.add_argument("--cv-dir", type=Path, default=Path("outputs") / "cv_results")
    p.add_argument("--output-dir", type=Path, default=Path("outputs") / "notebook_04c")
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Generating figures → {args.output_dir}/\\n")
    fig_m1b_calibration(args.splits_dir, args.models_dir, args.cv_dir, args.output_dir)
    fig_m2b_km_risk_groups(args.splits_dir, args.models_dir, args.output_dir)
    print("\\nDone.")


if __name__ == "__main__":
    main()
