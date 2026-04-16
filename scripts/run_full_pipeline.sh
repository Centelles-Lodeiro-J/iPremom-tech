#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"

PY="$REPO_DIR/.venv/Scripts/python.exe"
if [ ! -f "$PY" ]; then
  echo "Python interpreter not found at $PY" >&2
  exit 1
fi

run_step() {
  echo "
=== $1 ==="
  shift
  "$PY" "$@"
}

run_step '00 capture environment' ./src/00_capture_environment_and_run_manifest.py
run_step '01 data understanding' ./src/01_notebook_data_understanding.py
run_step '02 pipeline preparation (biology enhanced)' ./src/02_pipeline_data_preparation.biology_enhanced.py
run_step '02b pipeline support plots' ./src/02b_notebook_pipeline_support_plots.py
run_step '03 feature selection (patched)' ./src/03_feature_selection_cox_and_classification.patched.py
run_step '03b feature-selection support plots' ./src/03b_notebook_feature_selection_support_plots.py
run_step '04 data cleaning validation' ./src/04_notebook_data_cleaning_validation.py
run_step '04c product-support plots (patched)' ./src/04c_notebook_product_support_plots.patched.py
run_step '05 clinical and molecular validation (biology enhanced)' ./src/05_notebook_clinical_molecular_validation.biology_enhanced.fixed.py
run_step '06 baseline models' ./src/06_notebook_baseline_models.py
run_step '07 model comparison (patched)' ./src/07_notebook_models_and_algorithm_comparison.patched.py
run_step '07b winners summary (patched)' ./src/07b_notebook_model_winners_summary.patched.py
run_step '08 permutation importance (patched)' ./src/08_precompute_permutation_importance.patched.py
run_step '09 model interpretation (patched)' ./src/09_notebook_model_interpretation.patched.py
run_step '10 clinical discussion (patched)' ./src/10_notebook_clinical_discussion.patched.py
run_step '11 survival sensitivity (advanced)' ./src/11_notebook_survival_sensitivity_advanced.py
run_step 'pytest support-plot tests' -m pytest ./src/test_support_plot_summaries.py -q

echo "
Pipeline run completed successfully."
