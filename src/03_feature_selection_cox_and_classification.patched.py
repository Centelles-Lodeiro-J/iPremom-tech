"""Feature Selection — patched version
METABRIC Breast Cancer Dataset

Main changes:
- Binary tasks now use L1-penalized logistic regression instead of LassoCV.
- Ordinal grade keeps a regression-style proxy, but it is labelled explicitly as a heuristic surrogate.
- All selection remains fitted on training data only.
"""
import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LassoCV, LogisticRegression, LogisticRegressionCV
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
plt.rcParams.update({"figure.dpi": 150, "figure.facecolor": "white", "axes.facecolor": "#f8f8f8", "axes.spines.top": False, "axes.spines.right": False})
PAL = {"rf": "#3498db", "logit_l1": "#e74c3c", "ordinal_proxy": "#e74c3c", "mi": "#2ecc71", "cox": "#9b59b6", "consensus": "#2c3e50"}

MODELS = {
    "M1a_overall_survival": {"task": "binary"},
    "M1b_cancer_specific_survival": {"task": "binary"},
    "M2a_overall_survival_cox": {"task": "cox"},
    "M2b_cancer_specific_cox": {"task": "cox"},
    "M3_pam50_subtype": {"task": "multi"},
    "M4_histologic_grade": {"task": "ordinal"},
}


def parse_args():
    p = argparse.ArgumentParser(description="Feature selection — patched version")
    p.add_argument("--splits-dir", type=Path, default=Path("outputs") / "splits")
    p.add_argument("--output-dir", type=Path, default=Path("outputs"))
    p.add_argument("--min-votes", type=int, default=2)
    p.add_argument("--top-n", type=int, default=30)
    return p.parse_args()


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {path}")


def load_split(splits_dir, model_key):
    d = splits_dir / model_key
    X_tr = pd.read_csv(d / "X_train.csv")
    X_te = pd.read_csv(d / "X_test.csv")
    y_tr_raw = pd.read_csv(d / "y_train.csv")
    y_te_raw = pd.read_csv(d / "y_test.csv")
    return X_tr, X_te, y_tr_raw, y_te_raw


def run_rf(X, y, top_n):
    rf = RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1, class_weight="balanced")
    rf.fit(X, y)
    imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    return imp, set(imp.head(top_n).index)


def run_logit_l1_binary(X, y, top_n):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    clf = LogisticRegressionCV(
        Cs=[0.01, 0.1, 1.0, 10.0],
        cv=5,
        penalty="l1",
        solver="saga",
        scoring="roc_auc",
        class_weight="balanced",
        max_iter=5000,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(Xs, y)
    coef = pd.Series(np.abs(clf.coef_).ravel(), index=X.columns).sort_values(ascending=False)
    selected = set(coef[coef > 0].index)
    if not selected:
        selected = set(coef.head(top_n).index)
    return coef, selected, float(np.median(clf.C_))


def run_ordinal_proxy(X, y):
    lasso = LassoCV(cv=5, random_state=42, max_iter=10000, n_jobs=-1)
    lasso.fit(X, y)
    coef = pd.Series(np.abs(lasso.coef_), index=X.columns).sort_values(ascending=False)
    return coef, set(coef[coef > 0].index), round(float(lasso.alpha_), 6)


def run_lr_multi(X, y, top_n):
    clf = LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        random_state=42,
    )
    clf.fit(X, y)
    coef = pd.Series(np.abs(clf.coef_).mean(axis=0), index=X.columns).sort_values(ascending=False)
    return coef, set(coef.head(top_n).index)


def run_mi(X, y, top_n):
    scores = mutual_info_classif(X, y, random_state=42, n_neighbors=5)
    mi = pd.Series(scores, index=X.columns).sort_values(ascending=False)
    return mi, set(mi.head(top_n).index)


def prepare_cox_X(X):
    Xc = X.copy()
    Xc = Xc.apply(pd.to_numeric, errors="coerce")
    Xc = Xc.replace([np.inf, -np.inf], np.nan)
    med = Xc.median(axis=0)
    med = med.fillna(0.0)
    Xc = Xc.fillna(med)
    nunique = Xc.nunique(dropna=False)
    keep = nunique[nunique > 1].index.tolist()
    Xc = Xc[keep]
    var = Xc.var(axis=0)
    keep = var[var > 1e-12].index.tolist()
    Xc = Xc[keep]
    if Xc.shape[1] == 0:
        raise ValueError("No usable features remain for Cox feature selection after cleaning.")
    scaler = StandardScaler()
    Xs = pd.DataFrame(scaler.fit_transform(Xc), columns=Xc.columns, index=Xc.index)
    return Xs


def run_cox_univariate(X, y_df, top_n):
    from lifelines import CoxPHFitter
    rows = []
    for col in X.columns:
        df = pd.DataFrame({"time": y_df["time"].values, "event": y_df["event"].values, col: X[col].values})
        try:
            cph = CoxPHFitter(penalizer=0.1)
            cph.fit(df, duration_col="time", event_col="event", show_progress=False)
            s = cph.summary.loc[col]
            rows.append({"feature": col, "score": abs(float(s["coef"])), "p": float(s["p"])})
        except Exception:
            rows.append({"feature": col, "score": 0.0, "p": 1.0})
    out = pd.DataFrame(rows).sort_values(["p", "score"], ascending=[True, False]).reset_index(drop=True)
    return out.set_index("feature")["score"], set(out.head(top_n)["feature"])


def fit_penalized_cox(df):
    from lifelines import CoxPHFitter
    last_err = None
    for pen in [0.1, 0.3, 1.0, 3.0, 10.0]:
        try:
            cph = CoxPHFitter(penalizer=pen, l1_ratio=0.0)
            cph.fit(df, duration_col="time", event_col="event", show_progress=False)
            return cph, pen
        except Exception as e:
            last_err = e
    raise last_err


def run_cox_penalized(X, y_df, top_n):
    df = X.copy(); df["time"] = y_df["time"].values; df["event"] = y_df["event"].values
    cph, pen = fit_penalized_cox(df)
    coef = cph.summary["coef"].abs().sort_values(ascending=False)
    return coef, set(coef.head(top_n).index), pen


def run_cox_dropcol(X, y_df, top_n):
    df = X.copy(); df["time"] = y_df["time"].values; df["event"] = y_df["event"].values
    full, pen = fit_penalized_cox(df)
    base = full.concordance_index_
    rows = []
    for col in X.columns:
        d = df.drop(columns=[col])
        try:
            cph, _ = fit_penalized_cox(d)
            score = base - cph.concordance_index_
        except Exception:
            score = 0.0
        rows.append({"feature": col, "score": score})
    out = pd.DataFrame(rows).sort_values("score", ascending=False)
    return out.set_index("feature")["score"], set(out.head(top_n)["feature"]), pen


def build_consensus(sets, all_cols, min_votes):
    rows = []
    method_names = list(sets.keys())
    for f in all_cols:
        row = {"feature": f}
        votes = 0
        for m, s in sets.items():
            row[m] = int(f in s)
            votes += int(f in s)
        row["votes"] = votes
        rows.append(row)
    df = pd.DataFrame(rows).sort_values(["votes"] + method_names, ascending=False)
    selected = set(df[df["votes"] >= min_votes]["feature"])
    return df, selected


def plot_model(score_series, title, out):
    fig, axes = plt.subplots(1, len(score_series), figsize=(6 * len(score_series), 8))
    if len(score_series) == 1:
        axes = [axes]
    for ax, (label, ser) in zip(axes, score_series.items()):
        top = ser.head(25)
        ax.barh(range(len(top))[::-1], top.values[::-1], color=PAL.get(label, "#3498db"), alpha=0.8)
        ax.set_yticks(range(len(top))[::-1])
        ax.set_yticklabels([f.replace("_", " ")[:30] for f in top.index[::-1]], fontsize=7)
        ax.set_title(label)
    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, out)


def run_model(model_key, label, splits_dir, top_n, min_votes):
    X_tr, X_te, y_tr_raw, _ = load_split(splits_dir, model_key)
    outdir = splits_dir / model_key / "feature_selection"
    outdir.mkdir(parents=True, exist_ok=True)
    task = MODELS[model_key]["task"]
    if task == "binary":
        y = y_tr_raw.iloc[:, 0].astype(int)
        rf_imp, rf_sel = run_rf(X_tr, y, top_n)
        logit_coef, logit_sel, best_c = run_logit_l1_binary(X_tr, y, top_n)
        mi_scores, mi_sel = run_mi(X_tr, y, top_n)
        rf_imp.to_csv(outdir / "rf_importances.csv", header=["importance"])
        logit_coef.to_csv(outdir / "l1_logistic_coefficients.csv", header=["coef_abs"])
        with open(outdir / "l1_logistic_info.txt", "w", encoding="utf-8") as f:
            f.write(f"Best C (median across folds): {best_c}\n")
        mi_scores.to_csv(outdir / "mutual_information.csv", header=["mi"])
        consensus_df, selected = build_consensus({"rf": rf_sel, "logit_l1": logit_sel, "mi": mi_sel}, X_tr.columns, min_votes)
        plot_model({"rf": rf_imp, "logit_l1": logit_coef, "mi": mi_scores}, label, outdir / "plot.png")
    elif task == "multi":
        y = y_tr_raw.iloc[:, 0]
        rf_imp, rf_sel = run_rf(X_tr, y, top_n)
        lr_coef, lr_sel = run_lr_multi(X_tr, y, top_n)
        mi_scores, mi_sel = run_mi(X_tr, y, top_n)
        rf_imp.to_csv(outdir / "rf_importances.csv", header=["importance"])
        lr_coef.to_csv(outdir / "multinomial_lr_coefficients.csv", header=["coef_abs"])
        mi_scores.to_csv(outdir / "mutual_information.csv", header=["mi"])
        consensus_df, selected = build_consensus({"rf": rf_sel, "lr_multi": lr_sel, "mi": mi_sel}, X_tr.columns, min_votes)
        plot_model({"rf": rf_imp, "lr_multi": lr_coef, "mi": mi_scores}, label, outdir / "plot.png")
    elif task == "ordinal":
        y = y_tr_raw.iloc[:, 0]
        rf_imp, rf_sel = run_rf(X_tr, y, top_n)
        proxy_coef, proxy_sel, alpha = run_ordinal_proxy(X_tr, y)
        mi_scores, mi_sel = run_mi(X_tr, y, top_n)
        rf_imp.to_csv(outdir / "rf_importances.csv", header=["importance"])
        proxy_coef.to_csv(outdir / "ordinal_proxy_coefficients.csv", header=["coef_abs"])
        with open(outdir / "ordinal_proxy_info.txt", "w", encoding="utf-8") as f:
            f.write("Heuristic surrogate used for ordinal grade feature ranking.\n")
            f.write(f"Lasso alpha: {alpha}\n")
        mi_scores.to_csv(outdir / "mutual_information.csv", header=["mi"])
        consensus_df, selected = build_consensus({"rf": rf_sel, "ordinal_proxy": proxy_sel, "mi": mi_sel}, X_tr.columns, min_votes)
        plot_model({"rf": rf_imp, "ordinal_proxy": proxy_coef, "mi": mi_scores}, label, outdir / "plot.png")
    else:
        y = y_tr_raw[["time", "event"]].copy()
        Xc = prepare_cox_X(X_tr)
        kept_cols = Xc.columns.tolist()
        uni, uni_sel = run_cox_univariate(Xc, y, top_n)
        pen, pen_sel, pen_used = run_cox_penalized(Xc, y, top_n)
        drop, drop_sel, drop_pen = run_cox_dropcol(Xc, y, top_n)
        uni.to_csv(outdir / "cox_univariate.csv", header=["score"])
        pen.to_csv(outdir / "cox_penalized_coefficients.csv", header=["coef_abs"])
        drop.to_csv(outdir / "cox_dropcolumn_cindex.csv", header=["delta_cindex"])
        pd.DataFrame({"kept_feature": kept_cols}).to_csv(outdir / "cox_kept_features_after_cleaning.csv", index=False)
        with open(outdir / "cox_fit_info.txt", "w", encoding="utf-8") as f:
            f.write(f"Penalizer used (multivariable): {pen_used}\n")
            f.write(f"Penalizer used (drop-column base): {drop_pen}\n")
            f.write(f"Input features after Cox cleaning: {len(kept_cols)}\n")
        consensus_df, selected = build_consensus({"cox_uni": uni_sel, "cox_pen": pen_sel, "cox_drop": drop_sel}, kept_cols, min_votes)
        plot_model({"cox_uni": uni, "cox_pen": pen, "cox_drop": drop}, label + " (lifelines Cox only)", outdir / "plot.png")
    consensus_df.to_csv(outdir / "selected_features.csv", index=False)
    pd.DataFrame({"feature": sorted(selected)}).to_csv(outdir / "selected_feature_list.csv", index=False)
    return {"n_selected": len(selected), "selected": selected}


def main():
    args = parse_args()
    if not args.splits_dir.exists():
        raise FileNotFoundError(f"Splits not found: {args.splits_dir}. Run pipeline.py first.")
    print("=" * 80)
    print("Feature selection — patched")
    print("=" * 80)
    for model_key in MODELS:
        print(f"\n[{model_key}]")
        res = run_model(model_key, model_key.replace("_", " "), args.splits_dir, args.top_n, args.min_votes)
        print(f"Selected {res['n_selected']} features")
    print("\nDone.")


if __name__ == "__main__":
    main()
