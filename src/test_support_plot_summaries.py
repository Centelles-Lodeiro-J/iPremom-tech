import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

MODULE_PATH = Path(__file__).with_name("04c_notebook_product_support_plots.patched.py")
spec = importlib.util.spec_from_file_location("notebook04c_patched", MODULE_PATH)
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
