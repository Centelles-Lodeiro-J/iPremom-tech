import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

base = Path(__file__).parent
candidates = [
    base / "04c_notebook_product_support_plots.patched.py",
    base / "04c_notebook_product_support_plots_advanced.py",
]

MODULE_PATH = next((p for p in candidates if p.exists()), None)
if MODULE_PATH is None:
    raise FileNotFoundError(
        "Could not find 04c_notebook_product_support_plots.patched.py "
        "or 04c_notebook_product_support_plots_advanced.py"
    )

spec = importlib.util.spec_from_file_location("notebook04c_module", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_compute_calibration_summary_basic_invariants():
    probs = np.array([0.05, 0.10, 0.20, 0.30, 0.55, 0.60, 0.75, 0.85, 0.90, 0.95])
    y = np.array([0, 0, 0, 0, 1, 0, 1, 1, 1, 1])
    cal = module.compute_calibration_summary(probs, y, max_bins=5)
    assert int(cal["n"].sum()) == len(y)
    assert cal[["pred", "obs"]].notna().all().all()
    assert ((cal["pred"] >= 0) & (cal["pred"] <= 1)).all()
    assert ((cal["obs"] >= 0) & (cal["obs"] <= 1)).all()


def test_compute_binary_metrics_summary_basic_invariants():
    probs = np.array([0.05, 0.10, 0.20, 0.30, 0.55, 0.60, 0.75, 0.85, 0.90, 0.95])
    y = np.array([0, 0, 0, 0, 1, 0, 1, 1, 1, 1])
    metrics = module.compute_binary_metrics_summary(probs, y)
    assert metrics["n"] == len(y)
    assert 0 <= metrics["brier_score"] <= 1
    assert 0 <= metrics["ece"] <= 1
    assert 0 <= metrics["prevalence"] <= 1
    assert 0 <= metrics["roc_auc"] <= 1
    assert 0 <= metrics["pr_auc"] <= 1


def test_compute_threshold_metrics_basic_invariants():
    probs = np.array([0.05, 0.10, 0.20, 0.30, 0.55, 0.60, 0.75, 0.85, 0.90, 0.95])
    y = np.array([0, 0, 0, 0, 1, 0, 1, 1, 1, 1])
    thr = module.compute_threshold_metrics(probs, y, thresholds=(0.1, 0.3, 0.5))
    assert set(["threshold", "tp", "fp", "tn", "fn", "sensitivity", "specificity", "ppv", "npv"]).issubset(thr.columns)
    assert ((thr["threshold"] >= 0) & (thr["threshold"] <= 1)).all()
    assert (thr[["tp", "fp", "tn", "fn"]].sum(axis=1) == len(y)).all()
    for col in ["sensitivity", "specificity", "ppv", "npv", "predicted_positive_rate", "accuracy"]:
        assert ((thr[col] >= 0) & (thr[col] <= 1) | thr[col].isna()).all()


def test_compute_risk_group_summary_basic_invariants():
    risk = np.array([0.1, 0.2, 0.3, 0.4, 0.8, 0.9, 1.1, 1.3, 1.6])
    times = np.array([50, 48, 47, 44, 40, 35, 30, 25, 20])
    events = np.array([0, 0, 0, 1, 0, 1, 1, 1, 1])
    summary_df, groups = module.compute_risk_group_summary(risk, times, events, q=3)
    assert int(summary_df["n"].sum()) == len(risk)
    assert list(summary_df["group"]) == ["Low risk", "Medium risk", "High risk"]
    median_risk = summary_df["median_risk"].to_numpy()
    assert np.all(np.diff(median_risk) > 0)
    assert pd.Series(groups).notna().all()
