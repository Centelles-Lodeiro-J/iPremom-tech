# METABRIC Multi-Task Pipeline

A reproducible, leakage-aware modelling pipeline built on the METABRIC cohort for six linked tasks:

- M1a: overall-survival binary classification
- M1b: cancer-specific binary classification
- M2a: overall-survival Cox modelling
- M2b: cancer-specific Cox modelling
- M3: PAM50 subtype prediction
- M4: histologic-grade prediction

The repository is organised as a notebook-style pipeline implemented as Python scripts. The public version focuses on source code, documentation, and reproducible execution order, while keeping local environment details and raw data outside version control.

## What the pipeline includes

- deterministic data cleaning and train-only imputation
- one shared train/test split across tasks
- train-only NMF programme extraction
- task-specific feature selection
- locked final test evaluation for classification tasks
- calibration and threshold reporting for binary tasks
- descriptive transport and repeated-validation sensitivity analyses
- a separate competing-risks sensitivity summary for M2b

## Repository structure

```text
src/                     core scripts
technical_reports/       technical Word reports for each stage
outputs/                 generated artefacts (kept out of version control)
data/                    local input data only (kept out of version control)
```

## Environment policy

This repository does **not** track:

- `.venv/`
- local Conda environments
- private lockfiles
- machine-specific package snapshots

The recommended public setup is:

1. create a local virtual environment named `.venv`
2. prepare the local Python environment using your private/internal dependency notes
3. keep that environment out of Git

The VS Code launch and task files in this repository assume the interpreter path:

```text
.venv/Scripts/python.exe
```

## Expected local inputs

The pipeline expects a local METABRIC-derived input file. Raw data are not included in the public repository.

Default expected path:

```text
data/FCS_ml_test_input_data_rna_mutation.csv
```

If your input file is stored elsewhere, edit the relevant script arguments or run the scripts with explicit `--input` paths.

## Canonical run order

Use the VS Code task **Run full METABRIC notebook pipeline** or the PowerShell script in `scripts/run_full_pipeline.ps1`.

The canonical order is:

1. `00_capture_environment_and_run_manifest.py`
2. `01_notebook_data_understanding.py`
3. `02_pipeline_data_preparation.biology_enhanced.py`
4. `02b_notebook_pipeline_support_plots.py`
5. `03_feature_selection_cox_and_classification.patched.py`
6. `03b_notebook_feature_selection_support_plots.py`
7. `04_notebook_data_cleaning_validation.py`
8. `04c_notebook_product_support_plots.patched.py`
9. `05_notebook_clinical_molecular_validation.biology_enhanced.fixed.py`
10. `06_notebook_baseline_models.py`
11. `07_notebook_models_and_algorithm_comparison.patched.py`
12. `07b_notebook_model_winners_summary.patched.py`
13. `08_precompute_permutation_importance.patched.py`
14. `09_notebook_model_interpretation.patched.py`
15. `10_notebook_clinical_discussion.patched.py`
16. `11_notebook_survival_sensitivity_advanced.py`
17. `pytest src/test_support_plot_summaries.py -q`

## Running from VS Code

### Run directly without helper scripts

From the repository root, run the scripts one by one in canonical order:

```powershell
.\.venv\Scripts\python.exe .\src\00_capture_environment_and_run_manifest.py
.\.venv\Scripts\python.exe .\src\01_notebook_data_understanding.py
.\.venv\Scripts\python.exe .\src\02_pipeline_data_preparation.biology_enhanced.py
.\.venv\Scripts\python.exe .\src\02b_notebook_pipeline_support_plots.py
.\.venv\Scripts\python.exe .\src\03_feature_selection_cox_and_classification.patched.py
.\.venv\Scripts\python.exe .\src\03b_notebook_feature_selection_support_plots.py
.\.venv\Scripts\python.exe .\src\04_notebook_data_cleaning_validation.py
.\.venv\Scripts\python.exe .\src\04c_notebook_product_support_plots.patched.py
.\.venv\Scripts\python.exe .\src\05_notebook_clinical_molecular_validation.biology_enhanced.fixed.py
.\.venv\Scripts\python.exe .\src\06_notebook_baseline_models.py
.\.venv\Scripts\python.exe .\src\07_notebook_models_and_algorithm_comparison.patched.py
.\.venv\Scripts\python.exe .\src\07b_notebook_model_winners_summary.patched.py
.\.venv\Scripts\python.exe .\src\08_precompute_permutation_importance.patched.py
.\.venv\Scripts\python.exe .\src\09_notebook_model_interpretation.patched.py
.\.venv\Scripts\python.exe .\src\10_notebook_clinical_discussion.patched.py
.\.venv\Scripts\python.exe .\src\11_notebook_survival_sensitivity_advanced.py
.\.venv\Scripts\python.exe -m pytest .\src\test_support_plot_summaries.py -q
```

This direct option is the simplest route if you prefer not to use the helper runners in `scripts/`.

## Outputs

The pipeline writes local artefacts to `outputs/`, including:

- per-task train/test splits
- feature-selection outputs
- NMF models and programme matrices
- model summaries
- calibration and threshold tables
- interpretation artefacts
- figure panels for each notebook
- technical validation summaries

## Documentation

The technical Word reports in `technical_reports/` describe the role, logic, and outputs of each script. They are intended as project documentation for readers of the repository.


## Notes

This repository is designed to be reproducible **within a prepared local environment**.