"""Notebook 10 — Clinical Discussion
Outputs to outputs/notebook_10/

Advanced additions beyond the enhanced version:
- Reads threshold-reporting artefacts from notebook_04c when available.
- Reads M2b competing-risks sensitivity summaries from notebook_11 when available.
- Saves CSV-backed numeric summary tables alongside figures.
"""
import argparse
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import cohen_kappa_score, f1_score, roc_auc_score

warnings.filterwarnings("ignore")
plt.rcParams.update({"figure.dpi": 150, "figure.facecolor": "white", "axes.facecolor": "#f8f8f8", "axes.spines.top": False, "axes.spines.right": False})
HEAD_BG = "#2c3e50"
ROW_A = "#f0f4f8"
ROW_B = "white"


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


def fallback_compute_classification_score(key, label, task, splits_dir, models_dir):
    feats = joblib.load(models_dir / f"{key}_features.joblib")
    best_name = joblib.load(models_dir / f"{key}_best_name.joblib") if (models_dir / f"{key}_best_name.joblib").exists() else "LR (L2)"
    suffix = model_suffix(best_name)
    clf = joblib.load(models_dir / f"{key}_{suffix}.joblib")
    X = pd.read_csv(splits_dir / key / "X_test.csv")[feats]
    y = pd.read_csv(splits_dir / key / "y_test.csv").iloc[:, 0]
    yp = clf.predict(X)
    sc = roc_auc_score(y, clf.predict_proba(X)[:, 1]) if task == "binary" else (f1_score(y, yp, average="macro", zero_division=0) if task == "multi" else cohen_kappa_score(y, yp, weights="quadratic"))
    row = {
        "Model": label,
        "Metric": "AUC-ROC" if task == "binary" else ("Macro-F1" if task == "multi" else "QW-Kappa"),
        "Score": float(sc),
        "Can do": "Risk stratification" if task == "binary" else ("Subtype prediction" if task == "multi" else "Grade support"),
        "Cannot do": "External validation still needed",
    }
    return row


def load_scores(splits_dir, models_dir, cv_dir):
    items = []
    defs = [
        ("M1a_overall_survival", "M1a overall survival", "binary"),
        ("M1b_cancer_specific_survival", "M1b cancer-specific survival", "binary"),
        ("M3_pam50_subtype", "M3 PAM50 subtype", "multi"),
        ("M4_histologic_grade", "M4 histologic grade", "ordinal"),
    ]
    for key, label, task in defs:
        final_path = cv_dir / key / "final_test_results.csv"
        if final_path.exists():
            row = pd.read_csv(final_path).iloc[0]
            out = {
                "Model": label,
                "Metric": str(row.get("metric", "score")),
                "Score": float(row.get("test_score", row.get("score"))),
                "Can do": "Risk stratification" if task == "binary" else ("Subtype prediction" if task == "multi" else "Grade support"),
                "Cannot do": "External validation still needed",
            }
            for extra in ["algorithm", "pr_auc", "brier_score", "ece", "ci_low", "ci_high", "prevalence"]:
                if extra in row.index:
                    out[extra] = row.get(extra)
            items.append(out)
        else:
            items.append(fallback_compute_classification_score(key, label, task, splits_dir, models_dir))

    for m2key, m2label in [("M2a_overall_survival_cox", "M2a overall survival Cox"), ("M2b_cancer_specific_cox", "M2b cancer-specific Cox")]:
        final_path = cv_dir / m2key / "final_test_results.csv"
        if final_path.exists():
            row = pd.read_csv(final_path).iloc[0]
            cidx = float(row.get("test_score", row.get("score")))
        else:
            cph = joblib.load(models_dir / f"{m2key}_cox.joblib")
            feats = joblib.load(models_dir / f"{m2key}_features.joblib")
            Xte = pd.read_csv(splits_dir / m2key / "X_test.csv")[feats]
            Xte = Xte.apply(pd.to_numeric, errors="coerce").replace([float("inf"), float("-inf")], pd.NA)
            Xte = Xte.fillna(Xte.median(numeric_only=True)).fillna(0.0)
            yte = pd.read_csv(splits_dir / m2key / "y_test.csv")
            df = Xte.copy(); df["time"] = yte["time"].values; df["event"] = yte["event"].values
            cidx = float(cph.score(df, scoring_method="concordance_index"))
        items.append({
            "Model": m2label,
            "Metric": "C-index",
            "Score": cidx,
            "Can do": "Ranks patients by survival risk with censoring handled",
            "Cannot do": "Provide exact individual survival times",
        })
    return pd.DataFrame(items)


def fig_capability(scores, out):
    scores.to_csv(out / "42_capability_assessment.csv", index=False)
    fig, ax = plt.subplots(figsize=(16, 4 + 0.7 * len(scores)))
    ax.axis("off")
    ax.text(0.5, 0.99, "Capability assessment — final models", ha="center", va="top", transform=ax.transAxes, fontsize=12, fontweight="bold")
    y = 0.9
    for i, (_, r) in enumerate(scores.iterrows()):
        bg = ROW_A if i % 2 == 0 else ROW_B
        ax.add_patch(plt.Rectangle((0.0, y - 0.08), 0.98, 0.08, facecolor=bg, transform=ax.transAxes))
        ax.text(0.01, y - 0.04, f"{r['Model']} | {r['Metric']}={r['Score']:.3f}", transform=ax.transAxes, fontsize=10, fontweight="bold")
        ax.text(0.36, y - 0.04, f"Can do: {r['Can do']}", transform=ax.transAxes, fontsize=9)
        ax.text(0.68, y - 0.04, f"Cannot do: {r['Cannot do']}", transform=ax.transAxes, fontsize=9)
        y -= 0.09
    fig.tight_layout()
    save(fig, out / "42_capability_assessment.png")


def fig_binary_metrics(scores, out):
    bin_df = scores[scores["Model"].isin(["M1a overall survival", "M1b cancer-specific survival"])].copy()
    req = {"pr_auc", "brier_score", "ece"}
    if bin_df.empty or not req.issubset(bin_df.columns):
        return
    bin_df.to_csv(out / "44_binary_metrics_detail.csv", index=False)
    fig, ax = plt.subplots(figsize=(14, 3.5 + 0.8 * len(bin_df)))
    ax.axis("off")
    ax.text(0.5, 0.98, "Binary-task diagnostic metrics", ha="center", va="top", transform=ax.transAxes, fontsize=12, fontweight="bold")
    headers = ["Model", "AUC-ROC", "PR-AUC", "Brier", "ECE", "Algorithm"]
    col_x = [0.00, 0.30, 0.46, 0.60, 0.72, 0.82]
    y = 0.84
    for h, x in zip(headers, col_x):
        ax.text(x, y, h, transform=ax.transAxes, fontsize=10, fontweight="bold")
    y -= 0.08
    for i, (_, r) in enumerate(bin_df.iterrows()):
        bg = ROW_A if i % 2 == 0 else ROW_B
        ax.add_patch(plt.Rectangle((0.0, y - 0.04), 0.98, 0.065, facecolor=bg, transform=ax.transAxes))
        vals = [
            r["Model"],
            f"{r['Score']:.3f}",
            f"{float(r.get('pr_auc', float('nan'))):.3f}",
            f"{float(r.get('brier_score', float('nan'))):.3f}",
            f"{float(r.get('ece', float('nan'))):.3f}",
            str(r.get("algorithm", "")),
        ]
        for txt, x in zip(vals, col_x):
            ax.text(x, y, txt, transform=ax.transAxes, fontsize=9)
        y -= 0.08
    ax.text(0.01, 0.05, "Lower Brier and lower ECE indicate better-calibrated probabilities.", transform=ax.transAxes, fontsize=8, color="#555")
    fig.tight_layout()
    save(fig, out / "44_binary_metrics_detail.png")


def fig_thresholds(threshold_csv, out):
    if not threshold_csv.exists():
        return
    thr_df = pd.read_csv(threshold_csv)
    fig, ax = plt.subplots(figsize=(10, 5))
    for metric in ["sensitivity", "specificity", "ppv", "npv"]:
        ax.plot(thr_df["threshold"], thr_df[metric], marker="o", lw=2, label=metric.upper())
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Metric value")
    ax.set_ylim(0, 1.05)
    ax.set_title("M1b threshold report on locked test set")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    save(fig, out / "45_m1b_threshold_report.png")


def fig_competing_risks(summary_csv, out):
    if not summary_csv.exists():
        return
    df = pd.read_csv(summary_csv)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x = range(len(df))
    labels = df["group"].tolist()
    axes[0].bar(x, df["cif_cancer_60m"], color=["#2ecc71", "#f39c12", "#e74c3c"], alpha=0.85)
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels(labels)
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("60-month CIF")
    axes[0].set_title("Cancer-specific cumulative incidence")
    axes[1].bar(x, df["cif_other_cause_60m"], color=["#2ecc71", "#f39c12", "#e74c3c"], alpha=0.85)
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(labels)
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("60-month CIF")
    axes[1].set_title("Other-cause cumulative incidence")
    fig.suptitle("M2b competing-risks sensitivity summary\nAalen–Johansen sensitivity, separate from the main cause-specific Cox model", fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, out / "46_m2b_competing_risk_summary.png")


def fig_findings(scores, out):
    fig, ax = plt.subplots(figsize=(14, 7.0))
    ax.axis("off")
    ax.text(0.5, 0.98, "Key findings and limitations", ha="center", va="top", transform=ax.transAxes, fontsize=12, fontweight="bold")
    lines = [
        "1. M1b restores cancer-specific risk modelling rather than only all-cause mortality.",
        "2. M2 uses lifelines Cox modelling and C-index with locked test evaluation.",
        "3. M2b now has an explicit competing-risks sensitivity analysis reported separately.",
        "4. M3 and M4 remain classification tasks and are reported in Macro-F1 / QW-Kappa.",
        "5. Classification winners are chosen by training-only CV, not by test-set comparison.",
        "6. Binary tasks now retain PR-AUC, Brier score, ECE, and threshold tables alongside AUC-ROC.",
        "7. Cohort transport sensitivity and repeated outer validation are reported descriptively from training data.",
        "Limitations: no external validation yet; competing-risks sensitivity is non-parametric rather than Fine-Gray.",
    ]
    y = 0.88
    for line in lines:
        ax.text(0.03, y, line, transform=ax.transAxes, fontsize=10)
        y -= 0.095
    fig.tight_layout()
    save(fig, out / "43_key_findings_and_limitations.png")


def main():
    p = argparse.ArgumentParser(description="Notebook 10 — Clinical discussion")
    p.add_argument("--splits-dir", type=Path, default=Path("outputs") / "splits")
    p.add_argument("--models-dir", type=Path, default=Path("outputs") / "models")
    p.add_argument("--cv-dir", type=Path, default=Path("outputs") / "cv_results")
    p.add_argument("--support-dir", type=Path, default=Path("outputs") / "notebook_04c")
    p.add_argument("--survival-sensitivity-dir", type=Path, default=Path("outputs") / "notebook_11")
    p.add_argument("--output-dir", type=Path, default=Path("outputs") / "notebook_10")
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scores = load_scores(args.splits_dir, args.models_dir, args.cv_dir)
    fig_capability(scores, args.output_dir)
    fig_findings(scores, args.output_dir)
    fig_binary_metrics(scores, args.output_dir)
    fig_thresholds(args.support_dir / "28c_m1b_threshold_metrics.csv", args.output_dir)
    fig_competing_risks(args.survival_sensitivity_dir / "54_m2b_competing_risk_summary.csv", args.output_dir)


if __name__ == "__main__":
    main()
