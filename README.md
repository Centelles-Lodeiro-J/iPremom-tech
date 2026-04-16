# iPremom-tech: METABRIC multi-task modelling pipeline

A reproducible, leakage-aware modelling pipeline built on the METABRIC cohort for six linked tasks:

- **M1a** — overall-survival binary classification
- **M1b** — cancer-specific binary classification
- **M2a** — overall-survival Cox modelling
- **M2b** — cancer-specific Cox modelling
- **M3** — PAM50 subtype prediction
- **M4** — histologic-grade prediction

The repository is organised as a notebook-style pipeline implemented as Python scripts under `src/`. The public repository contains source code and documentation. Local environments, temporary outputs, and machine-specific editor settings are intentionally excluded from version control.

---

## Pipeline design

The pipeline follows a shared, leakage-aware design:

1. **Deterministic cleaning before modelling**
   - string normalization
   - explicit typo and encoding fixes
   - cross-field consistency repairs where justified

2. **One shared train/test split across tasks**
   - enables coherent comparison across binary, survival, subtype, and grade tasks

3. **Train-only preprocessing**
   - imputation fitted on training rows only
   - NMF programme extraction fitted on training rows only
   - task-specific feature selection performed on training data only

4. **Task-specific modelling**
   - binary classification for M1a and M1b
   - Cox survival modelling for M2a and M2b
   - multiclass classification for M3
   - ordinal/grade prediction workflow for M4

5. **Locked final evaluation**
   - classification winners selected by training-only CV
   - final test-set metrics computed once after model selection is fixed

---

## Repository structure

```text
.
├── src/                     Python scripts for the full pipeline
├── docs/                    Technical documentation and supporting reports
├── README.md
├── .gitignore
├── requirements.txt         Optional dependency snapshot
├── requirements-dev.txt     Optional development/test snapshot
└── requirements_notes.txt   Optional environment notes
```

If you still have a folder named `techical_reports/`, rename it to `docs/` for consistency.

---

## Expected local data layout

The scripts expect the METABRIC input table to be available locally. The default expected location is:

```text
data/FCS_ml_test_input_data_rna_mutation.csv
```

If your file is stored elsewhere, either:
- edit the relevant script defaults, or
- run the affected script with explicit command-line arguments where supported.

---

## Canonical run order

Run the scripts from the repository root in this order:

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

Optional test step:

17. `python -m pytest .\src\test_support_plot_summaries.py -q`

---

## Running the full pipeline

From the repository root in Windows PowerShell:

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
```

Optional test step:

```powershell
.\.venv\Scripts\python.exe -m pytest .\src\test_support_plot_summaries.py -q
```

---

## Main outputs

The pipeline writes local artefacts into `outputs/`, including:

- per-task train/test splits
- feature-selection outputs
- NMF models and programme matrices
- model summaries
- calibration and threshold tables
- interpretation artefacts
- notebook figure panels
- validation summaries

These files are generated locally and usually should not all be tracked in Git.

---

## Documentation

Project documentation lives in `docs/`.

Recommended contents:
- pipeline overview
- per-script technical reports
- selected curated figures or summary tables
- notes on methodological decisions and limitations

---

## Environment policy

This repository should **not** track:

- `.venv/`
- local Conda environments
- editor-specific local folders such as `.vscode/`
- temporary outputs under `outputs/`
- raw local datasets under `data/`
- machine-specific package snapshots unless you explicitly want to publish them

If you keep `requirements.txt` and `requirements-dev.txt`, treat them as optional reproducibility aids rather than as proof of a fully portable environment.

---

## Public repository checklist

Before pushing, check that the public repository contains:
- source code under `src/`
- documentation under `docs/`
- top-level project metadata (`README.md`, `.gitignore`, optional dependency notes)

Before pushing, check that the public repository does **not** contain:
- virtual environments
- cached notebook artefacts
- temporary outputs
- machine-specific editor state
- private or redistributable-restricted data you do not want in Git history

---

## Notes

This repository is intended to document the final modelling workflow and its outputs in a form that is readable, inspectable, and reproducible within a prepared local environment.
