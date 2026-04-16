"""
Clean Pipeline — METABRIC Breast Cancer
========================================
Single script that runs the complete preprocessing, encoding,
splitting, NMF, and model preparation pipeline correctly.

Design principles
-----------------
1.  ONE shared stratified 80/20 split created first — all models use it
2.  ALL imputation fitted on training rows only, applied to all rows
3.  NMF fitted on training rows only, applied to all rows
4.  NPI excluded from all models (structural redundancy + M4 leakage)
5.  ohe_3gene_* excluded from M1/M2 (r=0.80 with her2_status_bin)
6.  PAM50 rare classes (NC/Other/Unknown) masked from M3 after split
7.  Target _was_missing flags excluded from their own model
8.  All metadata saved for reproducibility

Pipeline stages
---------------
  Stage 1   Load and validate raw data
  Stage 2   Deterministic fixes (typos, recoding, whitespace)
  Stage 3   Shared stratified split (stratify on overall_survival)
  Stage 4   Imputation (fitted on train, applied to all)
  Stage 5   Feature engineering (encoding, log transforms, winsorize)
  Stage 6   NMF gene clustering (fitted on train, applied to all)
  Stage 7   Per-model splits (feature sets, scaling, save)
  Stage 8   Save metadata

Models
------
  M1a overall_survival          binary classification
  M1b cancer_specific_survival  binary classification
  M2a  survival time (Cox)      Cox
  M2b  cancer specific survival time (Cox) Cox
  M3  pam50_subtype             multiclass (6 classes)
  M4  neoplasm_histologic_grade ordinal (1/2/3)

M5 (chemotherapy) dropped — confounded by indication.

Outputs
-------
  outputs/splits/<model>/X_train.csv, X_test.csv, y_train.csv, y_test.csv
  outputs/splits/<model>/scaler.joblib
  outputs/splits/<model>/feature_selection/  (populated by feature_selection.py)
  outputs/nmf/nmf_model.joblib
  outputs/nmf/nmf_minmax_scaler.joblib
  outputs/METABRIC_nmf_components.csv
  outputs/pam50_label_mapping.csv
  outputs/metadata/shared_train_indices.csv
  outputs/metadata/shared_test_indices.csv
  outputs/metadata/kept_genes.txt
  outputs/metadata/dropped_genes.txt
  outputs/metadata/imputation_values.json
  outputs/metadata/pipeline_config.json

Usage
-----
  python src/pipeline.py
  python src/pipeline.py --input data/your_file.csv --test-size 0.2
"""

import argparse
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import (
    LabelEncoder, MinMaxScaler, StandardScaler,
)

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
CFG = {
    # Split
    "test_size":         0.2,
    "random_state":      42,
    # NMF — tol=1e-4 converges in ~461 iterations on this dataset
    "nmf_k":             15,
    "nmf_max_iter":      5000,
    "nmf_tol":           1e-4,
    "nmf_seed":          42,
    # Gene filtering (NZV threshold — training variance)
    "nzv_threshold":     0.5,
    # PAM50 rare classes dropped from M3
    "pam50_rare":        ["NC", "Other", "Unknown"],
    # Age valid range
    "age_min":           18,
    "age_max":           100,
    # Columns to standardise (gene programmes added dynamically)
    "scale_cols": [
        "age_at_diagnosis_imputed",
        "lymph_nodes_examined_positive_log",
        "mutation_count_log",
        "tumor_size_log",
        "overall_survival_months",
        "tumor_stage_ord",
        "cellularity_ord",
    ],
    # Binary encoding maps
    "binary_maps": {
        "er_status":                {"Positive": 1, "Negative": 0},
        "her2_status":              {"Positive": 1, "Negative": 0},
        "pr_status":                {"Positive": 1, "Negative": 0},
        "inferred_menopausal_state":{"Post": 1, "Pre": 0},
        "type_of_breast_surgery":   {"MASTECTOMY": 1, "BREAST CONSERVING": 0},
        "primary_tumor_laterality": {"Right": 1, "Left": 0},
    },
    # Ordinal encoding maps
    "ordinal_maps": {
        "cellularity": {"Low": 1, "Moderate": 2, "High": 3},
    },
    # Rare category threshold for one-hot encoding (group below this into Other)
    "rare_threshold": 0.01,
}

# ── Column taxonomy ────────────────────────────────────────────────────────────
CLINICAL = [
    "patient_id","age_at_diagnosis","geo_location_id","ethnicity",
    "type_of_breast_surgery","cancer_type","cancer_type_detailed",
    "cellularity","chemotherapy","pam50_+_claudin-low_subtype","cohort",
    "er_status_measured_by_ihc","er_status","neoplasm_histologic_grade",
    "her2_status_measured_by_snp6","her2_status","tumor_other_histologic_subtype",
    "hormone_therapy","inferred_menopausal_state","integrative_cluster",
    "primary_tumor_laterality","lymph_nodes_examined_positive","mutation_count",
    "nottingham_prognostic_index","oncotree_code","overall_survival_months",
    "overall_survival","pr_status","radio_therapy","3-gene_classifier_subtype",
    "tumor_size","tumor_stage","death_from_cancer",
]



# ── Expression-derived biology signatures (current-data upgrades) ─────────────

BIO_SIGNATURES = {
    "pathway_proliferation_score": {
        "family": "pathway",
        "positive": ["AURKA", "CCNB1", "CDC25A", "CDK1", "CDK2", "CDK4", "CDK6"],
    },
    "pathway_her2_score": {
        "family": "pathway",
        "positive": ["ERBB2", "ERBB3", "ERBB4", "EGFR", "CDH1", "EPCAM"],
    },
    "pathway_luminal_score": {
        "family": "pathway",
        "positive": ["GATA3", "MAPT", "RAB25", "CDH1", "EPCAM"],
    },
    "pathway_immune_ifn_score": {
        "family": "pathway",
        "positive": ["STAT1", "STAT2", "JAK1", "JAK2", "HLA-G", "CSF1R"],
    },
    "pathway_emt_stroma_score": {
        "family": "pathway",
        "positive": ["FN1", "MMP2", "MMP9", "MMP11", "MMP14", "COL6A3", "COL12A1", "COL22A1"],
    },
    "pathway_dna_repair_score": {
        "family": "pathway",
        "positive": ["BRCA1", "BRCA2", "FANCA", "FANCD2", "RAD50", "RAD51", "RAD51C", "RAD51D"],
    },
    "immune_innate_score": {
        "family": "microenvironment",
        "positive": ["CSF1", "CSF1R", "STAT1", "STAT2", "JAK1", "JAK2", "HLA-G"],
    },
    "stromal_matrix_score": {
        "family": "microenvironment",
        "positive": ["COL6A3", "COL12A1", "COL22A1", "FN1", "MMP2", "MMP11", "MMP14"],
    },
    "epithelial_differentiation_score": {
        "family": "microenvironment",
        "positive": ["EPCAM", "CDH1", "ERBB3", "ERBB4", "RAB25"],
    },
    "cell_cycle_checkpoint_score": {
        "family": "pathway",
        "positive": ["CDC25A", "CDK1", "CDK2", "CDK4", "CDK6", "CDKN1A", "CDKN1B", "CDKN2A", "CDKN2B", "CDKN2C"],
    },
}


# ── All target-related columns — excluded from all model feature sets ──────────
ALL_TARGETS = [
    "overall_survival",
    "overall_survival_months",
    "death_from_cancer",
    "tumor_stage_ord",
    "neoplasm_histologic_grade_ord",
    "chemotherapy",
    "hormone_therapy",
    "radio_therapy",
    "pam50_target",
    # NPI excluded globally — structural redundancy + direct leakage into M4
    "nottingham_prognostic_index",
]

# ── ohe_3gene columns — r=0.80 with her2_status_bin ───────────────────────────
OHE_3GENE = [
    "ohe_3gene_erplus_her2neg_high_prolif",
    "ohe_3gene_erplus_her2neg_low_prolif",
    "ohe_3gene_erneg_her2neg",
    "ohe_3gene_her2plus",
    "ohe_3gene_unknown",
]

# ── Per-model leakage and target was_missing exclusions ───────────────────────
MODEL_CFG = {
    "M1a_overall_survival": {
        "target":        "overall_survival",
        "task":          "binary",
        "extra_exclude": [
            "overall_survival_months",   # post-outcome
            "death_from_cancer",         # different target
            *OHE_3GENE,
        ],
        "target_was_missing": [],
    },
    "M1b_cancer_specific_survival": {
        "target":        "death_from_cancer",
        "task":          "binary",
        "extra_exclude": [
            "overall_survival_months",   # post-outcome timing retained only for M2
            "overall_survival",          # all-cause target
            *OHE_3GENE,
        ],
        "target_was_missing": [],
    },
    "M2a_overall_survival_cox": {
        "target":        ["overall_survival_months", "overall_survival"],
        "task":          "cox",
        "extra_exclude": [
            "death_from_cancer",
            *OHE_3GENE,
        ],
        "target_was_missing": [],
        "cox_event_col": "overall_survival",
        "note": "Lifelines Cox target: time=overall_survival_months, event=overall_survival.",
    },
    "M2b_cancer_specific_cox": {
        "target":        ["overall_survival_months", "death_from_cancer"],
        "task":          "cox",
        "extra_exclude": [
            "overall_survival",
            *OHE_3GENE,
        ],
        "target_was_missing": [],
        "cox_event_col": "death_from_cancer",
        "note": "Lifelines Cox target: time=overall_survival_months, event=death_from_cancer (cancer-specific).",
    },
    "M3_pam50_subtype": {
        "target":        "pam50_target",
        "task":          "multiclass",
        "extra_exclude": [
            # PAM50 OHE IS the target encoded
            "ohe_pam50_basal","ohe_pam50_her2","ohe_pam50_luma",
            "ohe_pam50_lumb","ohe_pam50_normal","ohe_pam50_other",
            "ohe_pam50_unknown","ohe_pam50_claudinneglow",
            # integrative_cluster uses PAM50 as input
            # (all ohe_integrative_cluster_* added dynamically)
        ],
        "target_was_missing": [],
        "rare_mask": True,               # mask rare PAM50 rows after split
    },
    "M4_histologic_grade": {
        "target":        "neoplasm_histologic_grade_ord",
        "task":          "ordinal",
        "extra_exclude": [
            "neoplasm_histologic_grade_ord",   # IS the target
        ],
        "target_was_missing": ["neoplasm_histologic_grade_was_missing"],
    },
}


def parse_args():
    p = argparse.ArgumentParser(
        description="Clean pipeline — METABRIC breast cancer")
    p.add_argument("--input", type=Path,
                   default=Path("data") /
                   "FCS_ml_test_input_data_rna_mutation.csv")
    p.add_argument("--output-dir",    type=Path, default=Path("outputs"))
    p.add_argument("--test-size",     type=float, default=CFG["test_size"])
    p.add_argument("--random-state",  type=int,   default=CFG["random_state"])
    return p.parse_args()


# ── Helpers ────────────────────────────────────────────────────────────────────
def log1p_safe(series):
    """log1p after clipping negatives to 0."""
    return np.log1p(series.clip(lower=0))


def group_rare(series, threshold):
    """Replace categories below threshold prevalence with 'Other'."""
    freq = series.value_counts(normalize=True)
    rare = freq[freq < threshold].index
    return series.replace(rare, "Other")


def ohe_col(df, col, prefix):
    """One-hot encode column, return new columns dict."""
    vals = df[col].dropna().unique()
    return {f"{prefix}_{str(v).lower().replace(' ','_').replace('/','_').replace('+','plus').replace('-','neg').replace('(','').replace(')','').replace('.','')}" : (df[col] == v).astype(int)
            for v in vals}


def compute_expression_signature_scores(gene_df, signature_defs, min_genes=2):
    """Compute simple mean Z-score signatures from available expression genes."""
    upper_to_actual = {str(col).upper(): col for col in gene_df.columns}
    score_frames = {}
    manifest_rows = []
    for name, spec in signature_defs.items():
        pos_cols = [upper_to_actual[g] for g in spec.get("positive", []) if g in upper_to_actual]
        neg_cols = [upper_to_actual[g] for g in spec.get("negative", []) if g in upper_to_actual]
        used_cols = pos_cols + neg_cols
        if len(used_cols) < min_genes:
            continue

        pos = gene_df[pos_cols].mean(axis=1) if pos_cols else 0.0
        neg = gene_df[neg_cols].mean(axis=1) if neg_cols else 0.0
        score_frames[name] = pos - neg
        manifest_rows.append({
            "signature": name,
            "family": spec.get("family", "unspecified"),
            "n_genes_used": len(used_cols),
            "positive_genes_used": ",".join(pos_cols),
            "negative_genes_used": ",".join(neg_cols),
        })

    scores = pd.DataFrame(score_frames, index=gene_df.index)
    manifest = pd.DataFrame(manifest_rows)
    return scores, manifest


def get_feature_cols(master, model_key):
    """Return feature column list for a model."""
    drop = set(ALL_TARGETS)
    drop.update(MODEL_CFG[model_key].get("extra_exclude", []))
    drop.update(MODEL_CFG[model_key].get("target_was_missing", []))
    if model_key == "M3_pam50_subtype":
        drop.update(c for c in master.columns if c.startswith("ohe_integrative_cluster"))
    return [c for c in master.columns if c not in drop]


def scale_and_save(X_tr, X_te, scale_cols, model_dir):
    """Fit scaler on train, transform both, save scaler."""
    actual = [c for c in scale_cols if c in X_tr.columns]
    scaler = StandardScaler()
    X_tr = X_tr.copy()
    X_te = X_te.copy()
    if actual:
        X_tr[actual] = scaler.fit_transform(X_tr[actual])
        X_te[actual] = scaler.transform(X_te[actual])
    else:
        scaler.fit(pd.DataFrame([[0]]))
    joblib.dump(scaler, model_dir / "scaler.joblib")
    return X_tr, X_te


def save_split(X_tr, X_te, y_tr, y_te, model_dir):
    model_dir.mkdir(parents=True, exist_ok=True)
    X_tr.reset_index(drop=True).to_csv(model_dir / "X_train.csv", index=False)
    X_te.reset_index(drop=True).to_csv(model_dir / "X_test.csv",  index=False)
    if isinstance(y_tr, pd.DataFrame):
        y_tr.reset_index(drop=True).to_csv(model_dir / "y_train.csv", index=False)
        y_te.reset_index(drop=True).to_csv(model_dir / "y_test.csv",  index=False)
    else:
        y_tr.reset_index(drop=True).to_frame().to_csv(
            model_dir / "y_train.csv", index=False)
        y_te.reset_index(drop=True).to_frame().to_csv(
            model_dir / "y_test.csv",  index=False)


def print_split_summary(name, X_tr, X_te, y_tr, y_te, is_cox=False):
    print(f"    Train: {len(X_tr):,} × {X_tr.shape[1]}  "
          f"Test: {len(X_te):,} × {X_te.shape[1]}")
    if is_cox:
        ev = int(y_tr["event"].sum())
        print(f"    Events — train: {ev} ({ev/len(y_tr)*100:.1f}%)")
    else:
        ys = y_tr if isinstance(y_tr, pd.Series) else y_tr.iloc[:, 0]
        ye = y_te if isinstance(y_te, pd.Series) else y_te.iloc[:, 0]
        for cls in sorted(ys.unique()):
            print(f"    Class {str(cls):<10} "
                  f"train={( ys==cls).mean():.1%}  "
                  f"test={(ye==cls).mean():.1%}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    args = parse_args()
    CFG["test_size"]   = args.test_size
    CFG["random_state"] = args.random_state

    if not args.input.exists():
        raise FileNotFoundError(f"Input not found: {args.input}")

    out         = args.output_dir
    splits_dir  = out / "splits"
    nmf_dir     = out / "nmf"
    meta_dir    = out / "metadata"
    for d in [splits_dir, nmf_dir, meta_dir]:
        d.mkdir(parents=True, exist_ok=True)

    imputation_log = {}   # records all training-fitted imputation values

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE 1 — Load and validate
    # ══════════════════════════════════════════════════════════════════════════
    print("=" * 65)
    print("STAGE 1 — Load")
    print("=" * 65)
    raw = pd.read_csv(args.input)
    gene_cols = [c for c in raw.columns if c not in CLINICAL]
    print(f"  Raw shape : {raw.shape[0]:,} × {raw.shape[1]}")
    print(f"  Clinical  : {len(CLINICAL)}  Gene: {len(gene_cols)}")
    dup_n = int(raw["patient_id"].duplicated().sum())
    if dup_n == 0:
        print("  No duplicate patient IDs — OK")
    else:
        print(f"  Duplicate patient IDs detected: {dup_n}")
        dup_rows = raw.loc[raw["patient_id"].duplicated(keep=False)].copy()
        dup_path = meta_dir / "duplicate_patient_ids.csv"
        dup_rows.to_csv(dup_path, index=False)
        raw = (raw.sort_values("patient_id")
                  .drop_duplicates(subset=["patient_id"], keep="first")
                  .reset_index(drop=True))
        print(f"  Dropped duplicate patient_id rows keeping first occurrence → {dup_path}")
        print(f"  Deduplicated shape : {raw.shape[0]:,} × {raw.shape[1]}")

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE 2 — Deterministic fixes (whole-dataset, safe — no statistics used)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 65)
    print("STAGE 2 — Deterministic fixes")
    print("=" * 65)
    df = raw.copy()

    # Strip whitespace from all strings
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()
    print("  Whitespace stripped from all string columns")

    # Fix cancer_type for patient 284 (Sarcoma → Breast Cancer)
    mask = df["cancer_type"] == "Breast Sarcoma"
    df.loc[mask, "cancer_type"] = "Breast Cancer"
    print(f"  cancer_type: 'Breast Sarcoma' → 'Breast Cancer' ({mask.sum()} row)")

    # Fix typo in er_status_measured_by_ihc
    mask = df["er_status_measured_by_ihc"].isin(["Posyte", "Positve"])
    df.loc[mask, "er_status_measured_by_ihc"] = "Positive"
    print(f"  er_status_measured_by_ihc: typo → 'Positive' ({int(mask.sum())} rows)")

    # Truncated cancer_type_detailed entries → NaN
    mask = df["cancer_type_detailed"] == "Breast"
    df.loc[mask, "cancer_type_detailed"] = np.nan
    print(f"  cancer_type_detailed: truncated 'Breast' → NaN ({mask.sum()} rows)")

    # geo_location_id = 0 → NaN
    mask = df["geo_location_id"] == 0
    df.loc[mask, "geo_location_id"] = np.nan
    print(f"  geo_location_id: 0 → NaN ({mask.sum()} rows)")

    # Corrupt age values → NaN (will be imputed post-split)
    mask = ~df["age_at_diagnosis"].between(CFG["age_min"], CFG["age_max"])
    df.loc[mask, "age_at_diagnosis"] = np.nan
    print(f"  age_at_diagnosis: {mask.sum()} values outside "
          f"[{CFG['age_min']},{CFG['age_max']}] → NaN")

    # Recode death_from_cancer to binary
    df["death_from_cancer"] = df["death_from_cancer"].map(
        {"Died of Disease": 1, "Died of Other Causes": 0, "Living": 0})
    print("  death_from_cancer recoded to binary (1=Died of Disease)")

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE 3 — Shared stratified split
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 65)
    print("STAGE 3 — Shared stratified split")
    print("=" * 65)
    # Some rows may still have missing survival labels or times in the raw file.
    # These rows cannot be used for supervised train/test splitting, so drop them
    # deterministically here and record them for auditability.
    missing_target_mask = (
        df["overall_survival"].isna() |
        df["overall_survival_months"].isna()
    )
    if int(missing_target_mask.sum()) > 0:
        miss_df = df.loc[missing_target_mask, [
            "patient_id", "overall_survival", "overall_survival_months"
        ]].copy()
        miss_df.to_csv(meta_dir / "dropped_missing_survival_rows.csv", index=False)
        print(
            f"  Dropping {int(missing_target_mask.sum())} rows with missing "
            f"overall_survival and/or overall_survival_months "
            f"→ {meta_dir / 'dropped_missing_survival_rows.csv'}"
        )
        df = df.loc[~missing_target_mask].reset_index(drop=True)
        print(f"  Shape after dropping missing survival rows: {df.shape[0]:,} × {df.shape[1]}")

    sss = StratifiedShuffleSplit(
        n_splits=1,
        test_size=CFG["test_size"],
        random_state=CFG["random_state"],
    )
    strat = df["overall_survival"].astype(int)
    train_idx, test_idx = next(sss.split(df, strat))
    train_idx = sorted(train_idx.tolist())
    test_idx  = sorted(test_idx.tolist())

    tr_pos = strat.iloc[train_idx].mean()
    te_pos = strat.iloc[test_idx].mean()
    print(f"  Train: {len(train_idx):,}  Test: {len(test_idx):,}")
    print(f"  Survival rate — train: {tr_pos:.3f}  test: {te_pos:.3f}")
    assert set(train_idx) & set(test_idx) == set(), "Train/test overlap!"
    assert len(train_idx) + len(test_idx) == len(df), "Missing rows!"

    pd.Series(train_idx, name="row_index").to_csv(
        meta_dir / "shared_train_indices.csv", index=False)
    pd.Series(test_idx, name="row_index").to_csv(
        meta_dir / "shared_test_indices.csv", index=False)
    print(f"  Indices saved → {meta_dir}/")

    # Snapshot missingness AFTER all row filtering but BEFORE imputation.
    # This keeps row alignment with df/master while preserving pre-imputation flags.
    df_pre_impute = df.copy()

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE 4 — Imputation (fitted on training rows only)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 65)
    print("STAGE 4 — Imputation (fitted on training rows only)")
    print("=" * 65)

    df_tr = df.iloc[train_idx]   # training view — fit stats here only

    # 4a. age_at_diagnosis — median impute
    age_median = df_tr["age_at_diagnosis"].median()
    df["age_at_diagnosis"] = df["age_at_diagnosis"].fillna(age_median)
    imputation_log["age_at_diagnosis_median"] = round(age_median, 3)
    n_filled = df["age_at_diagnosis"].isna().sum()
    print(f"  age_at_diagnosis: median={age_median:.1f}  "
          f"remaining NaN={n_filled}")

    # 4b. mutation_count — median impute
    mc_median = df_tr["mutation_count"].median()
    df["mutation_count"] = df["mutation_count"].fillna(mc_median)
    imputation_log["mutation_count_median"] = round(mc_median, 3)
    print(f"  mutation_count: median={mc_median:.0f}")

    # 4c. tumor_size — stage-stratified median (using train distribution)
    ts_global = df_tr["tumor_size"].median()
    df["tumor_size_imputed_flag"] = df["tumor_size"].isna().astype(int)
    df["tumor_size"] = df["tumor_size"].copy()
    for stage in df_tr["tumor_stage"].dropna().unique():
        tr_mask = df_tr["tumor_stage"] == stage
        stage_med = df_tr.loc[tr_mask, "tumor_size"].median()
        all_mask = (df["tumor_stage"] == stage) & df["tumor_size"].isna()
        df.loc[all_mask, "tumor_size"] = stage_med
        imputation_log[f"tumor_size_median_stage{int(stage)}"] = round(stage_med, 3)
    df["tumor_size"] = df["tumor_size"].fillna(ts_global)
    imputation_log["tumor_size_global_median"] = round(ts_global, 3)
    print(f"  tumor_size: stage-stratified median (global fallback={ts_global:.1f})")

    # 4d. tumor_stage — Random Forest (fitted on training non-missing rows)
    stage_features = [
        "tumor_size", "lymph_nodes_examined_positive",
        "neoplasm_histologic_grade", "age_at_diagnosis",
    ]
    df["tumor_stage_missing_flag"] = df["tumor_stage"].isna().astype(int)
    stage_known_tr = df_tr[stage_features + ["tumor_stage"]].dropna()
    rf_stage = RandomForestRegressor(
        n_estimators=300, random_state=CFG["random_state"],
        n_jobs=-1, min_samples_leaf=2)
    rf_stage.fit(stage_known_tr[stage_features], stage_known_tr["tumor_stage"])
    joblib.dump(rf_stage, nmf_dir / "rf_stage_imputer.joblib")

    stage_missing = df[df["tumor_stage"].isna()]
    valid_missing  = stage_missing[stage_features].dropna()
    if len(valid_missing) > 0:
        preds = np.clip(np.round(
            rf_stage.predict(valid_missing[stage_features])), 0, 4)
        df.loc[valid_missing.index, "tumor_stage"] = preds
    # Fallback: mode of training set for any remaining NaN
    stage_mode = df_tr["tumor_stage"].mode()[0]
    df["tumor_stage"] = df["tumor_stage"].fillna(stage_mode)
    imputation_log["tumor_stage_rf_fitted_on_n"] = len(stage_known_tr)
    imputation_log["tumor_stage_mode_fallback"]  = float(stage_mode)
    n_stage = df["tumor_stage_missing_flag"].sum()
    print(f"  tumor_stage: RF imputed {n_stage} rows "
          f"(mode fallback={stage_mode:.0f})")

    # 4e. neoplasm_histologic_grade — RF classifier
    grade_num_feat = ["tumor_size", "lymph_nodes_examined_positive"]
    grade_cat_feat = ["er_status", "her2_status", "pr_status", "cellularity",
                      "pam50_+_claudin-low_subtype"]
    grade_all_feat = grade_num_feat + grade_cat_feat

    df["neoplasm_histologic_grade_missing_flag"] = \
        df["neoplasm_histologic_grade"].isna().astype(int)

    # Encode categoricals for RF (label-encode, train-fitted)
    le_store = {}
    df_grade = df[grade_all_feat + ["neoplasm_histologic_grade"]].copy()
    for c in grade_cat_feat:
        le = LabelEncoder()
        # Fit on training values only
        train_vals = df_tr[c].fillna("Unknown").astype(str)
        le.fit(train_vals)
        # Handle unseen categories in full dataset
        if "Unknown" not in le.classes_:
            le.classes_ = np.append(le.classes_, "Unknown")
        full_vals = df[c].fillna("Unknown").astype(str)
        full_vals = full_vals.where(full_vals.isin(le.classes_), "Unknown")
        df_grade[c + "_le"] = le.transform(full_vals)
        le_store[c] = le

    grade_feat_encoded = grade_num_feat + [c + "_le" for c in grade_cat_feat]
    grade_known_tr = df_tr.copy()
    for c in grade_cat_feat:
        grade_known_tr[c + "_le"] = le_store[c].transform(
            df_tr[c].fillna("Unknown").astype(str))

    grade_known_tr = grade_known_tr[
        grade_feat_encoded + ["neoplasm_histologic_grade"]].dropna()

    rf_grade = RandomForestClassifier(
        n_estimators=300, random_state=CFG["random_state"],
        n_jobs=-1, min_samples_leaf=2)
    rf_grade.fit(grade_known_tr[grade_feat_encoded],
                 grade_known_tr["neoplasm_histologic_grade"])
    joblib.dump(rf_grade, nmf_dir / "rf_grade_imputer.joblib")

    grade_missing = df[df["neoplasm_histologic_grade"].isna()]
    valid_g = df_grade.loc[grade_missing.index, grade_feat_encoded].dropna()
    if len(valid_g) > 0:
        preds_g = rf_grade.predict(valid_g)
        df.loc[valid_g.index, "neoplasm_histologic_grade"] = preds_g
    grade_mode = df_tr["neoplasm_histologic_grade"].mode()[0]
    df["neoplasm_histologic_grade"] = \
        df["neoplasm_histologic_grade"].fillna(grade_mode)
    imputation_log["neoplasm_histologic_grade_rf_n"] = len(grade_known_tr)
    imputation_log["neoplasm_histologic_grade_mode_fallback"] = float(grade_mode)
    n_grade = df["neoplasm_histologic_grade_missing_flag"].sum()
    print(f"  neoplasm_histologic_grade: RF imputed {n_grade} rows")

    # 4f. Remaining categoricals — mode impute (fitted on training)
    cat_mode_cols = [
        "cellularity", "type_of_breast_surgery", "inferred_menopausal_state",
        "primary_tumor_laterality", "er_status", "her2_status", "pr_status",
        "her2_status_measured_by_snp6", "integrative_cluster",
        "pam50_+_claudin-low_subtype", "tumor_other_histologic_subtype",
        "3-gene_classifier_subtype", "oncotree_code",
        "er_status_measured_by_ihc", "cancer_type_detailed",
    ]
    for col in cat_mode_cols:
        if col not in df.columns:
            continue
        n_miss = df[col].isna().sum()
        if n_miss == 0:
            continue
        mode_val = df_tr[col].mode()[0]
        df[col] = df[col].fillna(mode_val)
        imputation_log[f"{col}_mode"] = str(mode_val)
        print(f"  {col}: mode='{mode_val}' ({n_miss} rows filled)")

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE 5 — Feature engineering
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 65)
    print("STAGE 5 — Feature engineering")
    print("=" * 65)

    master = pd.DataFrame(index=df.index)

    # ── Targets (carried through for model assembly) ───────────────────────────
    master["overall_survival"]              = df["overall_survival"].astype(int)
    master["overall_survival_months"]       = df["overall_survival_months"]
    master["death_from_cancer"]             = df["death_from_cancer"].fillna(0).astype(int)
    master["tumor_stage_ord"]               = df["tumor_stage"].astype(int)
    master["neoplasm_histologic_grade_ord"] = \
        df["neoplasm_histologic_grade"].astype(int)
    master["chemotherapy"]   = df["chemotherapy"].astype(int)
    master["hormone_therapy"]= df["hormone_therapy"].astype(int)
    master["radio_therapy"]  = df["radio_therapy"].astype(int)

    # ── PAM50 target ───────────────────────────────────────────────────────────
    pam_raw = df["pam50_+_claudin-low_subtype"].str.strip().fillna("Unknown")
    le_pam  = LabelEncoder()
    # Fit on training values only so test classes map consistently
    le_pam.fit(df_tr["pam50_+_claudin-low_subtype"].str.strip()
               .fillna("Unknown").unique())
    pam_codes = []
    for v in pam_raw:
        if v in le_pam.classes_:
            pam_codes.append(le_pam.transform([v])[0])
        else:
            pam_codes.append(-1)
    master["pam50_target"] = pam_codes
    pam_mapping = {l: int(c) for l, c in
                   zip(le_pam.classes_, le_pam.transform(le_pam.classes_))}
    rare_codes = [pam_mapping[l] for l in CFG["pam50_rare"]
                  if l in pam_mapping]
    kept_pam = {l: c for l, c in pam_mapping.items()
                if l not in CFG["pam50_rare"]}
    pd.DataFrame(sorted(kept_pam.items(), key=lambda x: x[1]),
                 columns=["label","code"]).to_csv(
        out / "pam50_label_mapping.csv", index=False)
    print(f"  PAM50 target: {len(kept_pam)} classes kept, "
          f"{len(CFG['pam50_rare'])} rare dropped from M3")

    # ── Missing flags (carry through as features) ──────────────────────────────
    for col, flag in [
        ("age_at_diagnosis",          "age_at_diagnosis_was_missing"),
        ("mutation_count",            "mutation_count_was_missing"),
        ("tumor_size",                "tumor_size_was_missing"),
        ("tumor_stage",               "tumor_stage_was_missing"),
        ("neoplasm_histologic_grade", "neoplasm_histologic_grade_was_missing"),
        ("cellularity",               "cellularity_was_missing"),
    ]:
        # Use original raw missingness (before imputation)
        master[flag] = df_pre_impute[col].isna().astype(int)
    print("  Missing flags created from raw data")

    # ── Continuous features ────────────────────────────────────────────────────
    # Note: these use the already-imputed df values
    master["age_at_diagnosis_imputed"]          = df["age_at_diagnosis"]
    master["lymph_nodes_examined_positive_log"] = log1p_safe(
        df["lymph_nodes_examined_positive"])
    master["mutation_count_log"]                = log1p_safe(df["mutation_count"])
    master["tumor_size_log"]                    = log1p_safe(df["tumor_size"])
    # overall_survival_months: skew=0.38, no transform needed
    print("  Continuous: imputed values + log1p transforms for skewed cols")

    # ── Ordinal features ───────────────────────────────────────────────────────
    # tumor_stage_ord and neoplasm_histologic_grade_ord already in targets section
    master["cellularity_ord"] = df["cellularity"].map(
        CFG["ordinal_maps"]["cellularity"])
    # Fallback for any remaining NaN
    cell_mode = df_tr["cellularity"].map(
        CFG["ordinal_maps"]["cellularity"]).mode()[0]
    master["cellularity_ord"] = master["cellularity_ord"].fillna(cell_mode).astype(int)
    print("  Ordinal: cellularity Low=1 Moderate=2 High=3")

    # ── Binary features ────────────────────────────────────────────────────────
    for col, mapping in CFG["binary_maps"].items():
        new_col = f"{col}_bin"
        master[f"{new_col}_was_missing"] = df[col].isna().astype(int)
        mode_val = df_tr[col].mode()[0]
        encoded = df[col].fillna(mode_val).map(mapping)
        master[new_col] = encoded.fillna(0).astype(int)
    # Treatment flags
    for col in ["chemotherapy","hormone_therapy","radio_therapy","death_from_cancer"]:
        master[col] = df[col].fillna(0).astype(int)
    print("  Binary: receptor status, surgery type, treatments encoded 0/1")

    # ── One-hot encoding ───────────────────────────────────────────────────────
    # Ethnicity (strip + group rare on training distribution)
    eth = df["ethnicity"].str.strip()
    eth_mode = df_tr["ethnicity"].str.strip().mode()[0]
    eth = eth.fillna(eth_mode)
    for cat in sorted(eth.unique()):
        cn = f"ohe_ethnicity_{cat.lower().replace(' ','_')}"
        master[cn] = (eth == cat).astype(int)

    # PAM50 (one-hot for use as features in non-M3 models)
    pam_feat = df["pam50_+_claudin-low_subtype"].str.strip().fillna("Unknown")
    pam_mode = df_tr["pam50_+_claudin-low_subtype"].str.strip().mode()[0]
    pam_feat = pam_feat.fillna(pam_mode)
    for cat in sorted(pam_feat.unique()):
        cn = ("ohe_pam50_" + cat.lower()
              .replace("+","plus").replace("-","neg")
              .replace(" ","_").replace("/","_"))
        master[cn] = (pam_feat == cat).astype(int)

    # Integrative cluster
    ic = df["integrative_cluster"].fillna("Unknown")
    for cat in sorted(ic.unique()):
        cn = f"ohe_integrative_cluster_{str(cat).replace('+','plus')}"
        master[cn] = (ic == cat).astype(int)

    # 3-gene classifier
    g3 = df["3-gene_classifier_subtype"].fillna("Unknown")
    for cat in sorted(g3.unique()):
        cn = ("ohe_3gene_" + cat.lower()
              .replace("+","plus").replace("-","neg")
              .replace(" ","_").replace("/","_"))
        master[cn] = (g3 == cat).astype(int)

    # Cohort (integer 1-5, treat as nominal — ANOVA p<0.001 with survival)
    for c in sorted(df["cohort"].unique()):
        master[f"ohe_cohort_{int(c)}"] = (df["cohort"] == c).astype(int)

    # Tumor histologic subtype (group rare < 1%)
    hist = df["tumor_other_histologic_subtype"].fillna("Unknown")
    hist = group_rare(hist, CFG["rare_threshold"])
    for cat in sorted(hist.unique()):
        cn = ("ohe_histology_" + cat.lower()
              .replace("/","_").replace(" ","_"))
        master[cn] = (hist == cat).astype(int)

    # Oncotree code (group rare < 1%)
    onco = df["oncotree_code"].fillna("Unknown")
    onco = group_rare(onco, CFG["rare_threshold"])
    for cat in sorted(onco.unique()):
        cn = f"ohe_oncotree_{cat.lower()}"
        master[cn] = (onco == cat).astype(int)

    # HER2 SNP6 (group UNDEF → Other)
    snp = df["her2_status_measured_by_snp6"].fillna("Unknown")
    snp = snp.replace("UNDEF","Other")
    for cat in sorted(snp.unique()):
        master[f"ohe_her2_snp6_{cat.lower()}"] = (snp == cat).astype(int)

    ohe_count = sum(1 for c in master.columns if c.startswith("ohe_"))
    print(f"  One-hot: {ohe_count} columns "
          "(ethnicity, PAM50, integrative cluster, 3-gene, cohort, "
          "histology, oncotree, HER2 SNP6)")

    n_feat = len([c for c in master.columns if c not in
                  list(MODEL_CFG.keys()) + ALL_TARGETS])
    print(f"  Master shape: {master.shape}")

    signature_cols = []

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE 6 — NMF gene clustering (fitted on training rows only)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 65)
    print("STAGE 6 — NMF gene clustering (training rows only)")
    print("=" * 65)

    gene_df = df[gene_cols].clip(lower=-10, upper=10).copy()

    # Impute any gene NaNs using training-row medians only
    gene_nan_total = int(gene_df.isna().sum().sum())
    if gene_nan_total > 0:
        gene_train = gene_df.iloc[train_idx]
        gene_medians = gene_train.median(axis=0)
        gene_df = gene_df.fillna(gene_medians)
        gene_df = gene_df.fillna(0.0)
        print(f"  Gene NaNs detected: {gene_nan_total} cells → "
              "filled with training medians (fallback 0.0)")
    else:
        print("  Gene matrix complete: no NaNs detected")

    # Expression-derived pathway and microenvironment scores
    signature_scores, signature_manifest = compute_expression_signature_scores(
        gene_df,
        BIO_SIGNATURES,
    )
    signature_cols = signature_scores.columns.tolist()
    if signature_cols:
        for col in signature_cols:
            master[col] = signature_scores[col].astype(float)
        signature_out = signature_scores.copy()
        if "patient_id" in df.columns:
            signature_out.insert(0, "patient_id", df["patient_id"].values)
        signature_out.to_csv(out / "biology_signature_scores.csv", index=False)
        signature_manifest.to_csv(out / "biology_signature_manifest.csv", index=False)
        print(f"  Biology signatures added to master ({len(signature_cols)} columns)")
    else:
        signature_manifest = pd.DataFrame(columns=["signature", "family", "n_genes_used",
                                                   "positive_genes_used", "negative_genes_used"])
        print("  Biology signatures skipped — too few marker genes matched the expression matrix")

    gene_matrix = gene_df.values

    # NZV filter — variance computed on training rows only
    train_var  = gene_matrix[train_idx].var(axis=0)
    keep_mask  = train_var >= CFG["nzv_threshold"]
    kept_genes = [gene_cols[i] for i, k in enumerate(keep_mask) if k]
    dropped_genes = [gene_cols[i] for i, k in enumerate(keep_mask) if not k]
    gene_filt  = gene_matrix[:, keep_mask]
    print(f"  NZV filter (train var < {CFG['nzv_threshold']}): "
          f"removed {(~keep_mask).sum()}, kept {keep_mask.sum()}")

    # MinMaxScaler fitted on training genes only
    nmf_scaler = MinMaxScaler()
    train_scaled = nmf_scaler.fit_transform(gene_filt[train_idx])
    full_scaled  = np.clip(nmf_scaler.transform(gene_filt), 0.0, 1.0)

    # Fit NMF on training rows only
    nmf = NMF(
        n_components=CFG["nmf_k"],
        random_state=CFG["nmf_seed"],
        max_iter=CFG["nmf_max_iter"],
        tol=CFG["nmf_tol"],
    )
    W_train = nmf.fit_transform(train_scaled)
    W_full  = nmf.transform(full_scaled)

    converged = nmf.n_iter_ < CFG["nmf_max_iter"]
    print(f"  NMF k={CFG['nmf_k']}  n_iter={nmf.n_iter_}  "
          f"converged={converged}  err={nmf.reconstruction_err_:.4f}")
    assert converged, \
        f"NMF did not converge! Increase nmf_max_iter (current={CFG['nmf_max_iter']})"

    # Save NMF artefacts
    joblib.dump(nmf,        nmf_dir / "nmf_model.joblib")
    joblib.dump(nmf_scaler, nmf_dir / "nmf_minmax_scaler.joblib")

    prog_labels = [f"gene_programme_{i+1:02d}" for i in range(CFG["nmf_k"])]
    kept_gene_names = [gene_cols[i] for i, k in enumerate(keep_mask) if k]
    H_df = pd.DataFrame(nmf.components_,
                         index=prog_labels, columns=kept_gene_names)
    H_df.to_csv(out / "METABRIC_nmf_components.csv")

    # Add programmes to master
    for i, col in enumerate(prog_labels):
        master[col] = W_full[:, i]

    with open(meta_dir / "kept_genes.txt", "w") as f:
        f.write("\n".join(kept_gene_names))
    with open(meta_dir / "dropped_genes.txt", "w") as f:
        f.write("\n".join(dropped_genes))
    print(f"  Gene programmes added to master  "
          f"({len(prog_labels)} columns)")

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE 7 — Per-model splits
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 65)
    print("STAGE 7 — Per-model splits (shared indices)")
    print("=" * 65)

    scale_cols = [c for c in CFG["scale_cols"] if c in master.columns]
    scale_cols += prog_labels   # always scale gene programmes
    scale_cols += [c for c in signature_cols if c in master.columns]

    model_results = {}

    for model_key, mcfg in MODEL_CFG.items():
        print(f"\n  ── {model_key} ──")
        feat  = get_feature_cols(master, model_key)
        mdir  = splits_dir / model_key
        mdir.mkdir(parents=True, exist_ok=True)

        # Feature sanity checks
        assert "nottingham_prognostic_index" not in feat, \
            f"NPI leaked into {model_key}!"
        if model_key == "M4_histologic_grade":
            assert "neoplasm_histologic_grade_was_missing" not in feat, \
                "grade_was_missing leaked into M4!"

        # Build target
        task = mcfg["task"]
        if task == "cox":
            event_col = mcfg.get("cox_event_col", "overall_survival")
            y = master[["overall_survival_months", event_col]].rename(
                columns={event_col: "event",
                         "overall_survival_months": "time"})
        else:
            y = master[mcfg["target"]]

        # Apply shared indices
        X_tr = master[feat].iloc[train_idx]
        X_te = master[feat].iloc[test_idx]
        y_tr = y.iloc[train_idx]
        y_te = y.iloc[test_idx]

        # M3: mask rare PAM50 classes AFTER applying shared indices
        if mcfg.get("rare_mask"):
            tr_keep = ~y_tr.isin(rare_codes)
            te_keep = ~y_te.isin(rare_codes)
            X_tr = X_tr[tr_keep.values]
            X_te = X_te[te_keep.values]
            y_tr = y_tr[tr_keep]
            y_te = y_te[te_keep]
            n_drop_tr = (~tr_keep).sum()
            n_drop_te = (~te_keep).sum()
            print(f"    Rare PAM50 dropped — train: {n_drop_tr}  test: {n_drop_te}")
            model_results[model_key] = {
                "n_rare_dropped_train": int(n_drop_tr),
                "n_rare_dropped_test":  int(n_drop_te),
            }

        # Scale (fitted on train only)
        X_tr, X_te = scale_and_save(X_tr, X_te, scale_cols, mdir)

        # Save
        save_split(X_tr, X_te, y_tr, y_te, mdir)
        print_split_summary(model_key, X_tr, X_te, y_tr, y_te,
                             is_cox=(task == "cox"))
        print(f"    Features: {len(feat)}")

        if "note" in mcfg:
            print(f"    NOTE: {mcfg['note']}")

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE 8 — Save metadata
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 65)
    print("STAGE 8 — Metadata")
    print("=" * 65)

    config_out = {
        "pipeline_version": "v4_m1ab_m2ab_clean",
        "script":           "pipeline.py",
        "design":           "Option A — one shared stratified split",
        "split": {
            "test_size":    CFG["test_size"],
            "random_state": CFG["random_state"],
            "n_train":      len(train_idx),
            "n_test":       len(test_idx),
            "stratify_col": "overall_survival",
        },
        "nmf": {
            "k":                CFG["nmf_k"],
            "max_iter":         CFG["nmf_max_iter"],
            "tol":              CFG["nmf_tol"],
            "seed":             CFG["nmf_seed"],
            "converged":        bool(converged),
            "n_iter":           int(nmf.n_iter_),
            "reconstruction_err": round(nmf.reconstruction_err_, 4),
            "fitted_on":        "training rows only",
        },
        "gene_filtering": {
            "nzv_threshold":  CFG["nzv_threshold"],
            "genes_total":    len(gene_cols),
            "genes_kept":     int(keep_mask.sum()),
            "genes_dropped":  int((~keep_mask).sum()),
        },
        "biology_signatures": {
            "n_signatures": int(len(signature_cols)),
            "signatures": signature_cols,
            "manifest_file": "biology_signature_manifest.csv" if len(signature_cols) > 0 else None,
        },
        "imputation": imputation_log,
        "pam50_rare_dropped":   CFG["pam50_rare"],
        "npi_excluded":         True,
        "npi_reason":           "NPI = 0.2*size + lymph_node_stage + grade. "
                                "Structural redundancy (VIF=15.7). "
                                "Direct leakage into M4 (grade prediction, r=0.75).",
        "ohe_3gene_excluded_from": ["M1a_overall_survival", "M1b_cancer_specific_survival", "M2a_overall_survival_cox", "M2b_cancer_specific_cox"],
        "ohe_3gene_reason":     "ohe_3gene_her2plus r=0.80 with her2_status_bin",
        "m5_dropped":           True,
        "m5_reason":            "Confounded by indication — models prescribing "
                                "behaviour not treatment efficacy.",
        "models": {
            mk: {
                "task":     mc["task"],
                "n_features": len(get_feature_cols(master, mk)),
                **model_results.get(mk, {}),
            }
            for mk, mc in MODEL_CFG.items()
        },
    }

    with open(meta_dir / "pipeline_config.json", "w") as f:
        json.dump(config_out, f, indent=2)
    with open(meta_dir / "imputation_values.json", "w") as f:
        json.dump(imputation_log, f, indent=2)

    print(f"  pipeline_config.json  saved")
    print(f"  imputation_values.json saved")

    # ── Final readiness table ──────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("PIPELINE COMPLETE — READINESS REPORT")
    print("=" * 65)
    print(f"\n  {'Model':<30} {'Train':>6} {'Test':>6} {'Features':>9}")
    print("  " + "─" * 55)
    for mk in MODEL_CFG:
        d  = splits_dir / mk
        Xt = pd.read_csv(d / "X_train.csv")
        Xe = pd.read_csv(d / "X_test.csv")
        print(f"  {mk:<30} {len(Xt):>6} {len(Xe):>6} {Xt.shape[1]:>9}")

    print("\n  All imputation fitted on training rows only  ✓")
    print("  NMF fitted on training rows only            ✓")
    print(f"  NMF converged ({nmf.n_iter_} iterations)          ✓")
    print("  NPI excluded from all models                ✓")
    print("  ohe_3gene excluded from M1a/M1b/M2a/M2b     ✓")
    print("  grade_was_missing excluded from M4          ✓")
    print("  PAM50 rare classes masked in M3             ✓")
    print("  Train/test overlap                          0 rows ✓")
    print("\n✓ Ready for feature selection.")


if __name__ == "__main__":
    main()