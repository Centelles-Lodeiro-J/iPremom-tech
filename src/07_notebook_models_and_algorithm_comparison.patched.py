"""Notebook 07 — Models and Algorithm Comparison (advanced)
Outputs to outputs/notebook_07/

Advanced additions beyond the enhanced version:
- Saves repeated outer-validation sensitivity summaries for classification tasks.
- Saves CSV-backed numeric summaries for feature-representation, model summary,
  cohort transport, and repeated outer validation.
- Keeps locked final test evaluation unchanged.
"""
import argparse
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    cohen_kappa_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
plt.rcParams.update({
    "figure.dpi": 150,
    "figure.facecolor": "white",
    "axes.facecolor": "#f8f8f8",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

ALGO_COLORS = {"LR (L2)": "#3498db", "EN (L1+L2)": "#9b59b6", "RF": "#2ecc71"}
SET_COLORS = {"A": "#95a5a6", "B": "#3498db", "C": "#e67e22", "D": "#9b59b6"}
N_TOP_GENES = 15
CLASS_MODELS = [
    ("M1a_overall_survival", "M1a overall survival", "binary", "AUC-ROC"),
    ("M1b_cancer_specific_survival", "M1b cancer-specific survival", "binary", "AUC-ROC"),
    ("M3_pam50_subtype", "M3 PAM50 subtype", "multi", "Macro-F1"),
    ("M4_histologic_grade", "M4 histologic grade", "ordinal", "QW-Kappa"),
]
M2_KEYS = [
    ("M2a_overall_survival_cox", "M2a overall survival Cox"),
    ("M2b_cancer_specific_cox", "M2b cancer-specific Cox"),
]


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


def load_split(splits_dir, key):
    d = splits_dir / key
    return (
        pd.read_csv(d / "X_train.csv"),
        pd.read_csv(d / "X_test.csv"),
        pd.read_csv(d / "y_train.csv"),
        pd.read_csv(d / "y_test.csv"),
    )


def make_lr():
    return LogisticRegression(
        C=1.0,
        class_weight="balanced",
        penalty="l2",
        solver="lbfgs",
        max_iter=2000,
        random_state=42,
    )


def make_en():
    return LogisticRegression(
        C=1.0,
        class_weight="balanced",
        penalty="elasticnet",
        solver="saga",
        l1_ratio=0.5,
        max_iter=3000,
        random_state=42,
    )


def make_rf():
    return RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )


def model_suffix(name):
    if name == "RF":
        return "rf"
    if name == "EN (L1+L2)":
        return "en"
    return "lr"


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
        conf = probs[sel].mean()
        acc = y_true[sel].mean()
        ece += np.abs(acc - conf) * (sel.sum() / probs.size)
    return float(ece)


def compute_binary_metric_bundle(y_true, probs):
    y_true = pd.Series(y_true).astype(int)
    probs = pd.Series(probs, dtype=float)
    metrics = {
        "roc_auc": np.nan,
        "pr_auc": np.nan,
        "brier_score": np.nan,
        "ece": np.nan,
        "prevalence": float(y_true.mean()) if len(y_true) else np.nan,
    }
    if len(y_true) == 0:
        return metrics
    if y_true.nunique() >= 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, probs))
        metrics["pr_auc"] = float(average_precision_score(y_true, probs))
    metrics["brier_score"] = float(brier_score_loss(y_true, probs))
    metrics["ece"] = compute_expected_calibration_error(probs, y_true)
    return metrics


def compute_metric(clf, X, y, task):
    yp = clf.predict(X)
    if task == "binary":
        return roc_auc_score(y, clf.predict_proba(X)[:, 1])
    if task == "multi":
        return f1_score(y, yp, average="macro", zero_division=0)
    return cohen_kappa_score(y, yp, weights="quadratic")


def bootstrap_ci(clf, X, y, task, n=100, seed=42):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        Xb = X.iloc[idx]
        yb = y.iloc[idx]
        try:
            vals.append(compute_metric(clf, Xb, yb, task))
        except Exception:
            pass
    return np.percentile(vals, [2.5, 97.5]) if vals else (np.nan, np.nan)


def choose_cv_splits(y, max_splits=5):
    counts = pd.Series(y).value_counts()
    if counts.empty:
        raise ValueError("Cannot choose CV folds from an empty target.")
    min_count = int(counts.min())
    if min_count < 2:
        raise ValueError("At least two samples are required in every class for StratifiedKFold.")
    return min(max_splits, min_count)


def cross_validate_classifier(X, y, clf, task):
    n_splits = choose_cv_splits(y)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    rows = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
        est = clone(clf)
        Xtr_f, Xva_f = X.iloc[tr_idx], X.iloc[va_idx]
        ytr_f, yva_f = y.iloc[tr_idx], y.iloc[va_idx]
        est.fit(Xtr_f, ytr_f)
        score = float(compute_metric(est, Xva_f, yva_f, task))
        rows.append({"fold": fold, "score": score})
    return pd.DataFrame(rows)


def repeated_outer_validation(X, y, clf, task, metric_label, n_repeats=10, test_size=0.2):
    y = pd.Series(y).reset_index(drop=True)
    X = X.reset_index(drop=True)
    splitter = StratifiedShuffleSplit(n_splits=n_repeats, test_size=test_size, random_state=42)
    rows = []
    for rep, (tr_idx, va_idx) in enumerate(splitter.split(X, y), start=1):
        est = clone(clf)
        Xtr_f, Xva_f = X.iloc[tr_idx], X.iloc[va_idx]
        ytr_f, yva_f = y.iloc[tr_idx], y.iloc[va_idx]
        est.fit(Xtr_f, ytr_f)
        row = {
            "repeat": rep,
            "metric": metric_label,
            "valid_score": float(compute_metric(est, Xva_f, yva_f, task)),
            "n_train": int(len(tr_idx)),
            "n_valid": int(len(va_idx)),
        }
        if task == "binary" and hasattr(est, "predict_proba"):
            probs = est.predict_proba(Xva_f)[:, 1]
            extras = compute_binary_metric_bundle(yva_f, probs)
            row.update({
                "valid_pr_auc": extras["pr_auc"],
                "valid_brier_score": extras["brier_score"],
                "valid_ece": extras["ece"],
                "valid_prevalence": extras["prevalence"],
            })
        rows.append(row)
    return pd.DataFrame(rows)


def build_feature_sets(X_tr, X_te, y_tr_series, feats):
    clin = [f for f in feats if not f.startswith("gene_programme")]
    gp = [f for f in feats if f.startswith("gene_programme")]
    gene_cols = [c for c in X_tr.columns if c.startswith("g_")]
    notes = []
    X_tr_r = X_tr[clin].reset_index(drop=True)
    X_te_r = X_te[clin].reset_index(drop=True)
    gp_tr = X_tr[gp].reset_index(drop=True)
    gp_te = X_te[gp].reset_index(drop=True)
    if gene_cols:
        mi = mutual_info_classif(X_tr[gene_cols].values, y_tr_series.values, random_state=42)
        top_genes = pd.Series(mi, index=gene_cols).nlargest(min(N_TOP_GENES, len(gene_cols))).index.tolist()
        scaler = StandardScaler()
        gtr = pd.DataFrame(scaler.fit_transform(X_tr[top_genes]), columns=top_genes)
        gte = pd.DataFrame(scaler.transform(X_te[top_genes]), columns=top_genes)
    else:
        gtr = pd.DataFrame(index=X_tr.index)
        gte = pd.DataFrame(index=X_te.index)
        notes.append("Raw gene columns not present in split files: C==A and D==B.")
    sets_tr = {
        "A": X_tr_r,
        "B": pd.concat([X_tr_r, gp_tr], axis=1),
        "C": pd.concat([X_tr_r, gtr.reset_index(drop=True)], axis=1),
        "D": pd.concat([X_tr_r, gp_tr, gtr.reset_index(drop=True)], axis=1),
    }
    sets_te = {
        "A": X_te_r,
        "B": pd.concat([X_te_r, gp_te], axis=1),
        "C": pd.concat([X_te_r, gte.reset_index(drop=True)], axis=1),
        "D": pd.concat([X_te_r, gp_te, gte.reset_index(drop=True)], axis=1),
    }
    return sets_tr, sets_te, notes


def infer_cohort_labels(X):
    if "cohort" in X.columns:
        return pd.to_numeric(X["cohort"], errors="coerce")
    ohe_cols = [c for c in X.columns if c.startswith("ohe_cohort_")]
    if not ohe_cols:
        return None
    mat = X[ohe_cols].fillna(0).to_numpy()
    idx = mat.argmax(axis=1)
    max_vals = mat[np.arange(len(mat)), idx]
    labels = []
    for i, col_idx in enumerate(idx):
        if max_vals[i] <= 0:
            labels.append(np.nan)
        else:
            suffix = ohe_cols[col_idx].replace("ohe_cohort_", "")
            labels.append(pd.to_numeric(suffix, errors="coerce"))
    return pd.Series(labels, index=X.index)


def can_score_subset(y, task):
    y = pd.Series(y)
    if len(y) < 2:
        return False
    return y.nunique() >= 2


def leave_one_cohort_out_transport(X, y, cohort_labels, clf, task, metric_label):
    if cohort_labels is None:
        return pd.DataFrame()
    cohort_labels = pd.Series(cohort_labels).reset_index(drop=True)
    X = X.reset_index(drop=True)
    y = pd.Series(y).reset_index(drop=True)
    rows = []
    for cohort in sorted(pd.Series(cohort_labels.dropna().unique()).tolist()):
        va_mask = cohort_labels == cohort
        tr_mask = ~va_mask
        Xtr_c, Xva_c = X.loc[tr_mask], X.loc[va_mask]
        ytr_c, yva_c = y.loc[tr_mask], y.loc[va_mask]
        row = {
            "cohort": cohort,
            "n_train": int(tr_mask.sum()),
            "n_valid": int(va_mask.sum()),
            "metric": metric_label,
            "valid_score": np.nan,
        }
        if task == "binary":
            row["valid_prevalence"] = float(yva_c.mean()) if len(yva_c) else np.nan
            row["valid_pr_auc"] = np.nan
            row["valid_brier_score"] = np.nan
            row["valid_ece"] = np.nan
        if len(Xtr_c) == 0 or len(Xva_c) == 0 or not can_score_subset(ytr_c, task):
            rows.append(row)
            continue
        est = clone(clf)
        try:
            est.fit(Xtr_c, ytr_c)
            if can_score_subset(yva_c, task):
                row["valid_score"] = float(compute_metric(est, Xva_c, yva_c, task))
            if task == "binary" and hasattr(est, "predict_proba"):
                probs = est.predict_proba(Xva_c)[:, 1]
                extras = compute_binary_metric_bundle(yva_c, probs)
                row["valid_pr_auc"] = extras["pr_auc"]
                row["valid_brier_score"] = extras["brier_score"]
                row["valid_ece"] = extras["ece"]
        except Exception:
            pass
        rows.append(row)
    return pd.DataFrame(rows)


def fig_feature_representation(splits_dir, models_dir, out):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    note_lines = []
    rows_out = []
    for ax, (key, label, task, metric) in zip(axes, CLASS_MODELS):
        Xtr, Xte, ytr_raw, _ = load_split(splits_dir, key)
        ytr = ytr_raw.iloc[:, 0]
        feats = joblib.load(models_dir / f"{key}_features.joblib") if (models_dir / f"{key}_features.joblib").exists() else Xtr.columns.tolist()
        sets_tr, _, notes = build_feature_sets(Xtr, Xte, ytr, feats)
        note_lines.extend([f"{label}: {n}" for n in notes])
        scores = []
        labels = []
        for s in ["A", "B", "C", "D"]:
            fold_df = cross_validate_classifier(sets_tr[s], ytr.reset_index(drop=True), make_lr(), task)
            mean_score = float(fold_df["score"].mean())
            scores.append(mean_score)
            labels.append(f"{s}\n({sets_tr[s].shape[1]}f)")
            rows_out.append({"task_key": key, "Model": label, "feature_set": s, "n_features": int(sets_tr[s].shape[1]), "cv_score_mean": mean_score, "metric": metric})
        ax.bar(labels, scores, color=[SET_COLORS[s] for s in ["A", "B", "C", "D"]], alpha=0.85)
        ax.set_title(label)
        ax.set_ylabel(f"Training-CV {metric}")
    if note_lines:
        fig.text(0.02, 0.01, " | ".join(dict.fromkeys(note_lines)), fontsize=8)
    fig.suptitle("Feature representation comparison (training-CV only; descriptive)", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    pd.DataFrame(rows_out).to_csv(out / "29_feature_representation_scores.csv", index=False)
    save(fig, out / "29_feature_representation.png")


def fig_cohort_transport(transport_df, out):
    if transport_df.empty:
        return
    transport_df.to_csv(out / "31_cohort_transport_sensitivity.csv", index=False)
    keep = transport_df.dropna(subset=["valid_score"]).copy()
    if keep.empty:
        return
    models = keep["Model"].unique().tolist()
    fig, axes = plt.subplots(len(models), 1, figsize=(12, 3.4 * len(models)), squeeze=False)
    for ax, model in zip(axes.flatten(), models):
        sub = keep.loc[keep["Model"] == model].sort_values("cohort")
        x = np.arange(len(sub))
        ax.bar(x, sub["valid_score"], color="#3498db", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels([str(int(c)) if pd.notna(c) else "NA" for c in sub["cohort"]])
        ax.set_ylabel(sub["metric"].iloc[0])
        ax.set_xlabel("Held-out cohort")
        ax.set_title(f"{model} — leave-one-cohort-out sensitivity")
        for i, r in sub.reset_index(drop=True).iterrows():
            ax.text(i, r["valid_score"] + 0.01, f"n={int(r['n_valid'])}", ha="center", fontsize=8)
    fig.suptitle("Cohort transport sensitivity (training data only; descriptive)", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    save(fig, out / "31_cohort_transport_sensitivity.png")


def fig_repeated_outer_validation(repeat_df, out):
    if repeat_df.empty:
        return
    repeat_df.to_csv(out / "32_repeated_outer_validation.csv", index=False)
    keep = repeat_df.dropna(subset=["valid_score"]).copy()
    if keep.empty:
        return
    models = keep["Model"].unique().tolist()
    fig, axes = plt.subplots(len(models), 1, figsize=(12, 3.4 * len(models)), squeeze=False)
    for ax, model in zip(axes.flatten(), models):
        sub = keep.loc[keep["Model"] == model].copy()
        ax.plot(sub["repeat"], sub["valid_score"], marker="o", lw=1.5)
        mean_sc = sub["valid_score"].mean()
        sd_sc = sub["valid_score"].std(ddof=1) if len(sub) > 1 else 0.0
        ax.axhline(mean_sc, color="#e74c3c", ls="--", lw=1)
        ax.set_xlabel("Repeat")
        ax.set_ylabel(sub["metric"].iloc[0])
        ax.set_title(f"{model} — repeated outer validation\nmean={mean_sc:.3f} ± {sd_sc:.3f}")
    fig.suptitle("Repeated outer validation on training data (descriptive sensitivity)", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    save(fig, out / "32_repeated_outer_validation.png")


def write_final_test_results(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(path, index=False)


def fit_and_save_models(splits_dir, models_dir, cv_dir):
    models_dir.mkdir(parents=True, exist_ok=True)
    cv_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    transport_rows = []
    repeated_rows = []

    # Classification models: select by training-only CV, then lock, refit, test once.
    for key, label, task, metric in CLASS_MODELS:
        Xtr, Xte, ytr_raw, yte_raw = load_split(splits_dir, key)
        ytr = ytr_raw.iloc[:, 0]
        yte = yte_raw.iloc[:, 0]
        feat_path = splits_dir / key / "feature_selection" / "selected_feature_list.csv"
        feats = pd.read_csv(feat_path)["feature"].tolist() if feat_path.exists() else Xtr.columns.tolist()
        feats = [f for f in feats if f in Xtr.columns]
        Xtr_sel = Xtr[feats].reset_index(drop=True)
        Xte_sel = Xte[feats].reset_index(drop=True)
        ytr = ytr.reset_index(drop=True)
        yte = yte.reset_index(drop=True)

        algos = {"LR (L2)": make_lr(), "EN (L1+L2)": make_en(), "RF": make_rf()}
        cv_rows = []
        best_name = None
        best_cv = -np.inf
        for name, clf in algos.items():
            fold_df = cross_validate_classifier(Xtr_sel, ytr, clf, task)
            cv_mean = float(fold_df["score"].mean())
            cv_sd = float(fold_df["score"].std(ddof=1)) if len(fold_df) > 1 else 0.0
            cv_rows.append({
                "algorithm": name,
                "cv_score_mean": cv_mean,
                "cv_score_sd": cv_sd,
                "n_folds": int(len(fold_df)),
                "selection_basis": "training_cv",
            })
            if cv_mean > best_cv:
                best_cv = cv_mean
                best_name = name

        best_clf = clone(algos[best_name])
        best_clf.fit(Xtr_sel, ytr)
        suffix = model_suffix(best_name)
        joblib.dump(best_clf, models_dir / f"{key}_{suffix}.joblib")
        joblib.dump(feats, models_dir / f"{key}_features.joblib")
        joblib.dump(best_name, models_dir / f"{key}_best_name.joblib")

        test_score = float(compute_metric(best_clf, Xte_sel, yte, task))
        ci_low, ci_high = bootstrap_ci(best_clf, Xte_sel, yte, task)

        task_dir = cv_dir / key
        task_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(cv_rows).sort_values("cv_score_mean", ascending=False).to_csv(task_dir / "cv_results.csv", index=False)

        final_row = {
            "algorithm": best_name,
            "metric": metric,
            "test_score": test_score,
            "ci_low": float(ci_low),
            "ci_high": float(ci_high),
            "n_test": int(len(yte)),
            "selection_basis": "winner locked by training_cv before test evaluation",
        }
        if task == "binary":
            probs = best_clf.predict_proba(Xte_sel)[:, 1]
            bundle = compute_binary_metric_bundle(yte, probs)
            final_row.update({
                "pr_auc": bundle["pr_auc"],
                "brier_score": bundle["brier_score"],
                "ece": bundle["ece"],
                "prevalence": bundle["prevalence"],
            })
        write_final_test_results(task_dir / "final_test_results.csv", final_row)
        summary.append({"Model": label, "Metric": metric, "Best algorithm": best_name, "Test score": test_score})

        cohort_labels = infer_cohort_labels(Xtr)
        transport_df = leave_one_cohort_out_transport(Xtr_sel, ytr, cohort_labels, algos[best_name], task, metric)
        if not transport_df.empty:
            transport_df.insert(0, "Model", label)
            transport_df.insert(1, "task_key", key)
            transport_df.to_csv(task_dir / "cohort_transport_cv.csv", index=False)
            transport_rows.append(transport_df)

        repeat_df = repeated_outer_validation(Xtr_sel, ytr, algos[best_name], task, metric, n_repeats=10, test_size=0.2)
        if not repeat_df.empty:
            repeat_df.insert(0, "Model", label)
            repeat_df.insert(1, "task_key", key)
            repeat_df.insert(2, "algorithm", best_name)
            repeat_df.to_csv(task_dir / "repeated_outer_validation.csv", index=False)
            repeated_rows.append(repeat_df)

    from lifelines import CoxPHFitter

    def prep(X):
        X = X.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        med = X.median(numeric_only=True)
        X = X.fillna(med).fillna(0.0)
        keep = X.var(numeric_only=True)
        keep = keep[keep > 1e-12].index.tolist()
        X = X[keep]
        return X.reset_index(drop=True), keep, None

    for m2key, m2label in M2_KEYS:
        Xtr, Xte, ytr, yte = load_split(splits_dir, m2key)
        feat_path = splits_dir / m2key / "feature_selection" / "selected_feature_list.csv"
        feats = pd.read_csv(feat_path)["feature"].tolist() if feat_path.exists() else Xtr.columns.tolist()
        feats = [f for f in feats if f in Xtr.columns]
        clin = [f for f in feats if not f.startswith("gene_programme")]
        full = list(feats)
        cv_rows = []
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        strat = ytr["event"].astype(int)
        best_label = None
        best_score = -np.inf
        best_feats = None
        for feat_label, feat_list in [("A: Clinical", clin), ("B: Clin+NMF", full)]:
            fold_scores = []
            for tr_idx, va_idx in skf.split(Xtr, strat):
                Xtr_f, Xva_f = Xtr.iloc[tr_idx][feat_list], Xtr.iloc[va_idx][feat_list]
                ytr_f, yva_f = ytr.iloc[tr_idx], ytr.iloc[va_idx]
                Xtr_p, keep, sc = prep(Xtr_f)
                Xva_p = Xva_f.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
                Xva_p = Xva_p.fillna(Xtr_f.median(numeric_only=True)).fillna(0.0)
                Xva_p = Xva_p[keep].reset_index(drop=True)
                dtr = Xtr_p.copy(); dtr["time"] = ytr_f["time"].values; dtr["event"] = ytr_f["event"].values
                dva = Xva_p.copy(); dva["time"] = yva_f["time"].values; dva["event"] = yva_f["event"].values
                cph = CoxPHFitter(penalizer=1.0, l1_ratio=0.0)
                cph.fit(dtr, duration_col="time", event_col="event", show_progress=False)
                fold_scores.append(float(cph.score(dva, scoring_method="concordance_index")))
            mean_sc = float(np.mean(fold_scores))
            sd_sc = float(np.std(fold_scores, ddof=1)) if len(fold_scores) > 1 else 0.0
            cv_rows.append({"feature_set": feat_label, "cv_score_mean": mean_sc, "cv_score_sd": sd_sc, "n_folds": len(fold_scores), "selection_basis": "training_cv"})
            if mean_sc > best_score:
                best_score = mean_sc
                best_label = feat_label
                best_feats = feat_list
        task_dir = cv_dir / m2key
        task_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(cv_rows).to_csv(task_dir / "cv_results.csv", index=False)

        Xtr_p, keep, sc = prep(Xtr[best_feats])
        Xte_p = Xte[best_feats].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        Xte_p = Xte_p.fillna(Xtr[best_feats].median(numeric_only=True)).fillna(0.0)
        Xte_p = Xte_p[keep].reset_index(drop=True)
        dtr = Xtr_p.copy(); dtr["time"] = ytr["time"].values; dtr["event"] = ytr["event"].values
        dte = Xte_p.copy(); dte["time"] = yte["time"].values; dte["event"] = yte["event"].values
        cph = CoxPHFitter(penalizer=1.0, l1_ratio=0.0)
        cph.fit(dtr, duration_col="time", event_col="event", show_progress=False)
        cidx = float(cph.score(dte, scoring_method="concordance_index"))
        joblib.dump(cph, models_dir / f"{m2key}_cox.joblib")
        joblib.dump(keep, models_dir / f"{m2key}_features.joblib")
        joblib.dump(sc, models_dir / f"{m2key}_scaler.joblib")
        joblib.dump(best_label, models_dir / f"{m2key}_best_name.joblib")
        write_final_test_results(task_dir / "final_test_results.csv", {
            "algorithm": "CoxPH",
            "metric": "C-index",
            "test_score": cidx,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "n_test": int(len(yte)),
            "selection_basis": "feature set locked by training_cv before test evaluation",
        })
        summary.append({"Model": m2label, "Metric": "C-index", "Best algorithm": "CoxPH", "Test score": cidx})

    transport_df = pd.concat(transport_rows, ignore_index=True) if transport_rows else pd.DataFrame()
    repeated_df = pd.concat(repeated_rows, ignore_index=True) if repeated_rows else pd.DataFrame()
    return pd.DataFrame(summary), transport_df, repeated_df


def fig_summary(summary_df, out):
    summary_df.to_csv(out / "30_model_summary.csv", index=False)
    fig, ax = plt.subplots(figsize=(12, 4 + 0.6 * len(summary_df)))
    ax.axis("off")
    ax.text(0.5, 0.98, "Final model summary", ha="center", va="top", transform=ax.transAxes, fontsize=12, fontweight="bold")
    y = 0.88
    for _, r in summary_df.iterrows():
        ax.text(0.02, y, f"{r['Model']}: {r['Best algorithm']} — {r['Metric']}={r['Test score']:.3f}", fontsize=10)
        y -= 0.09
    fig.text(0.5, 0.02, "Classification winners were selected by training-only CV; test scores are final locked estimates.", ha="center", fontsize=8, color="#555")
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    save(fig, out / "30_model_summary.png")


def main():
    p = argparse.ArgumentParser(description="Notebook 07 — Models and algorithm comparison")
    p.add_argument("--splits-dir", type=Path, default=Path("outputs") / "splits")
    p.add_argument("--models-dir", type=Path, default=Path("outputs") / "models")
    p.add_argument("--cv-dir", type=Path, default=Path("outputs") / "cv_results")
    p.add_argument("--output-dir", type=Path, default=Path("outputs") / "notebook_07")
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig_feature_representation(args.splits_dir, args.models_dir, args.output_dir)
    summary_df, transport_df, repeated_df = fit_and_save_models(args.splits_dir, args.models_dir, args.cv_dir)
    fig_summary(summary_df, args.output_dir)
    fig_cohort_transport(transport_df, args.output_dir)
    fig_repeated_outer_validation(repeated_df, args.output_dir)


if __name__ == "__main__":
    main()
