"""Notebook 11 — Survival Sensitivity Analyses (advanced)
Outputs to outputs/notebook_11/

Implements an explicit competing-risks sensitivity analysis for M2b.
This does NOT replace the main cause-specific Cox model. It adds a
non-parametric Aalen–Johansen cumulative-incidence sensitivity analysis on the
locked test set, stratified by M2b predicted risk group.
"""
import argparse
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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


def apply_deterministic_fixes(raw):
    df = raw.copy()
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()
    mask = df["cancer_type"] == "Breast Sarcoma"
    df.loc[mask, "cancer_type"] = "Breast Cancer"
    mask = df["er_status_measured_by_ihc"].isin(["Posyte", "Positve"])
    df.loc[mask, "er_status_measured_by_ihc"] = "Positive"
    mask = df["cancer_type_detailed"] == "Breast"
    df.loc[mask, "cancer_type_detailed"] = np.nan
    mask = df["geo_location_id"] == 0
    df.loc[mask, "geo_location_id"] = np.nan
    mask = ~df["age_at_diagnosis"].between(18, 100)
    df.loc[mask, "age_at_diagnosis"] = np.nan
    return df


def reconstruct_split_frame(input_path):
    raw = pd.read_csv(input_path)
    if raw["patient_id"].duplicated().any():
        raw = raw.sort_values("patient_id").drop_duplicates(subset=["patient_id"], keep="first").reset_index(drop=True)
    df = apply_deterministic_fixes(raw)
    missing_target_mask = df["overall_survival"].isna() | df["overall_survival_months"].isna()
    if int(missing_target_mask.sum()) > 0:
        df = df.loc[~missing_target_mask].reset_index(drop=True)
    return df


def status_code_from_label(s):
    if s == "Died of Disease":
        return 1
    if s == "Died of Other Causes":
        return 2
    return 0


def make_risk_groups(risk, q=3):
    risk = pd.Series(risk, dtype=float)
    return pd.qcut(risk.rank(method="first"), q=q, labels=["Low risk", "Medium risk", "High risk"])


def cumulative_incidence_at_times(times, status, event_of_interest, horizons):
    from lifelines import AalenJohansenFitter

    times = pd.Series(times, dtype=float).reset_index(drop=True)
    status = pd.Series(status).astype(int).reset_index(drop=True)
    ajf = AalenJohansenFitter()
    ajf.fit(times, status, event_of_interest=event_of_interest)
    cif = ajf.cumulative_density_.copy()
    cif.columns = ["cif"]
    out = []
    for h in horizons:
        sub = cif.loc[cif.index <= h]
        val = float(sub.iloc[-1, 0]) if not sub.empty else 0.0
        out.append({"horizon_months": float(h), "cif": val})
    return pd.DataFrame(out), cif.reset_index().rename(columns={"event_at": "time"}) if "event_at" in cif.reset_index().columns else cif.reset_index().rename(columns={cif.reset_index().columns[0]: "time"})


def competing_risk_group_summary(risk, times, status, horizons=(36, 60, 120)):
    groups = make_risk_groups(risk)
    risk = pd.Series(risk, dtype=float).reset_index(drop=True)
    times = pd.Series(times, dtype=float).reset_index(drop=True)
    status = pd.Series(status).astype(int).reset_index(drop=True)
    rows = []
    curve_rows = []
    for g in ["Low risk", "Medium risk", "High risk"]:
        mask = groups == g
        t = times.loc[mask]
        s = status.loc[mask]
        row = {
            "group": g,
            "n": int(mask.sum()),
            "median_risk": float(risk.loc[mask].median()),
            "cancer_deaths": int((s == 1).sum()),
            "other_cause_deaths": int((s == 2).sum()),
            "alive_or_censored": int((s == 0).sum()),
        }
        for event_code, label in [(1, "cancer"), (2, "other_cause")]:
            horizon_df, curve_df = cumulative_incidence_at_times(t, s, event_code, horizons)
            for _, r in horizon_df.iterrows():
                row[f"cif_{label}_{int(r['horizon_months'])}m"] = float(r["cif"])
            curve_df["group"] = g
            curve_df["event_type"] = label
            curve_rows.append(curve_df)
        rows.append(row)
    return pd.DataFrame(rows), pd.concat(curve_rows, ignore_index=True), pd.DataFrame({"group": pd.Series(groups, dtype=str)})


def fig_competing_risks(curves_df, summary_df, out):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = {"Low risk": "#2ecc71", "Medium risk": "#f39c12", "High risk": "#e74c3c"}
    for ax, event_type, title in [
        (axes[0], "cancer", "Cancer-specific cumulative incidence"),
        (axes[1], "other_cause", "Competing other-cause cumulative incidence"),
    ]:
        sub = curves_df.loc[curves_df["event_type"] == event_type].copy()
        for g in ["Low risk", "Medium risk", "High risk"]:
            ss = sub.loc[sub["group"] == g].sort_values("time")
            ax.step(ss["time"], ss["cif"], where="post", lw=2, color=colors[g], label=g)
        ax.set_xlabel("Time (months)")
        ax.set_ylabel("Cumulative incidence")
        ax.set_ylim(0, 1.0)
        ax.set_title(title)
        ax.legend(fontsize=8)
    fig.suptitle("M2b competing-risks sensitivity on locked test set\nAalen–Johansen cumulative-incidence curves by predicted M2b risk group", fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, out / "54_m2b_competing_risks_sensitivity.png")


def main():
    p = argparse.ArgumentParser(description="M2b competing-risks sensitivity analysis")
    p.add_argument("--input", type=Path, default=Path("data") / "FCS_ml_test_input_data_rna_mutation.csv")
    p.add_argument("--splits-dir", type=Path, default=Path("outputs") / "splits")
    p.add_argument("--models-dir", type=Path, default=Path("outputs") / "models")
    p.add_argument("--metadata-dir", type=Path, default=Path("outputs") / "metadata")
    p.add_argument("--output-dir", type=Path, default=Path("outputs") / "notebook_11")
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = reconstruct_split_frame(args.input)
    test_idx = pd.read_csv(args.metadata_dir / "shared_test_indices.csv")["row_index"].astype(int).tolist()
    df_test = df.iloc[test_idx].reset_index(drop=True)

    key = "M2b_cancer_specific_cox"
    cph = joblib.load(args.models_dir / f"{key}_cox.joblib")
    feats = joblib.load(args.models_dir / f"{key}_features.joblib")
    X_te = pd.read_csv(args.splits_dir / key / "X_test.csv")[feats]

    if len(df_test) != len(X_te):
        raise ValueError(
            f"Shared test-set reconstruction length mismatch for M2b: raw-derived={len(df_test)} vs split={len(X_te)}."
        )

    risk = np.asarray(cph.predict_partial_hazard(X_te)).reshape(-1)
    status = df_test["death_from_cancer"].map(status_code_from_label)
    times = df_test["overall_survival_months"].astype(float)

    summary_df, curves_df, group_df = competing_risk_group_summary(risk, times, status)
    summary_df.to_csv(args.output_dir / "54_m2b_competing_risk_summary.csv", index=False)
    curves_df.to_csv(args.output_dir / "54_m2b_competing_risk_curves.csv", index=False)
    group_df.to_csv(args.output_dir / "54_m2b_competing_risk_groups.csv", index=False)

    note_df = pd.DataFrame([
        {
            "analysis_type": "Aalen-Johansen competing-risks sensitivity",
            "main_model": "M2b cause-specific Cox remains unchanged",
            "interpretation": "This is a sensitivity analysis, not a Fine-Gray regression replacement.",
        }
    ])
    note_df.to_csv(args.output_dir / "54_m2b_competing_risk_notes.csv", index=False)
    fig_competing_risks(curves_df, summary_df, args.output_dir)


if __name__ == "__main__":
    main()
