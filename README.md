# iPremom-tech: METABRIC multi-task modelling pipeline

A reproducible, leakage-aware, multi-task modelling pipeline built on the METABRIC cohort. The repository implements the full workflow as sequential Python scripts under `src/` and documents the methodology, results, and interpretation in `docs/`.

The pipeline addresses six linked tasks derived from the same processed dataset:

- **M1a** — overall-survival binary classification
- **M1b** — cancer-specific binary classification
- **M2a** — overall-survival Cox modelling
- **M2b** — cancer-specific Cox modelling
- **M3** — PAM50 subtype prediction
- **M4** — histologic-grade prediction

---

## Repository structure

```text
.
├── src/         Python scripts for the full pipeline
├── docs/        Technical documentation and supporting reports
├── README.md
└── .gitignore
```

The public repository is intended to contain:
- source code
- documentation
- project-level summaries

The public repository is **not** intended to contain:
- local virtual environments
- editor-specific local configuration
- raw local datasets
- temporary outputs generated during execution

---

## Project summary

### Pipeline overview

This repository contains a leakage-aware, multi-task modelling pipeline built on the METABRIC cohort. The workflow is implemented as sequential Python scripts under `src/` and addresses six linked prediction problems derived from the same processed dataset.

The design philosophy is to use one coherent preprocessing backbone for all tasks while preserving task-specific modelling and evaluation choices. This enables consistent comparisons across endpoints without fitting separate, incompatible data-preparation pipelines for each model family.

### Methodology

#### 1. Data preparation and deterministic cleaning

The workflow begins with a deterministic cleaning stage applied before modelling. This stage standardizes strings, resolves known typos, fixes a small number of explicit cross-field inconsistencies where the evidence is strong, and recodes variables into modelling-ready formats.

Examples include:
- whitespace normalization in categorical variables
- explicit typo correction in ER IHC labels
- resolution of a single `Breast Sarcoma` inconsistency where the surrounding fields strongly support invasive breast carcinoma
- recoding of the cancer-specific outcome into a binary event indicator

This deterministic layer separates data repair from statistical imputation and makes the rationale for each edit inspectable.

#### 2. Shared split design

The repository uses a **single shared train/test split** across the six tasks. The split is stratified on overall survival so that all downstream tasks are evaluated on aligned patients whenever possible. This improves coherence across the project and avoids subtle inconsistencies introduced by per-task random splits.

#### 3. Train-only preprocessing

All fitted preprocessing steps are estimated on **training rows only** and then applied to the test set. This applies to:
- numeric and categorical imputation
- random-forest imputation for stage and grade where used
- NMF factorization of the expression matrix
- feature selection and model selection

This design minimizes information leakage and preserves the role of the test set as a true held-out evaluation set.

#### 4. Feature engineering

The prepared feature space combines:
- cleaned clinical variables
- ordinal encodings for ordered clinical factors
- binary encodings for receptor and treatment variables
- one-hot encodings for nominal clinical categories
- missingness indicators for selected variables
- log-transformed continuous variables where skewness is substantial
- NMF-derived gene programmes from the expression matrix
- pathway and tumour-microenvironment signature scores derived from expression

This produces a hybrid representation that preserves clinically interpretable variables while adding lower-dimensional molecular summaries.

#### 5. Molecular dimension reduction and biological scoring

The repository uses **NMF** rather than PCA as the main transcriptomic compression strategy. The main motivation is interpretability: non-negative decomposition yields additive gene programmes that can be manually and quantitatively annotated against known biological themes.

In the final version, those programmes are cross-checked against formal pathway and microenvironment scores, which strengthens the claim that the latent structure captures recognizable tumour biology.

#### 6. Task-specific modelling

Each task uses an evaluation metric aligned to its modelling objective:

- **M1a, M1b**: classification tasks evaluated primarily with **AUC-ROC**, with additional reporting of PR-AUC, calibration, Brier score, ECE, and threshold trade-offs
- **M2a, M2b**: survival tasks evaluated with **C-index**
- **M3**: multiclass subtype task evaluated with **Macro-F1**
- **M4**: histologic-grade task evaluated with **quadratic weighted kappa**

This avoids forcing all tasks into an accuracy-only reporting scheme and instead uses measures that better match imbalance, calibration, censoring, and ordinal structure.

#### 7. Training-only model selection and locked final evaluation

For the final classification workflow, algorithm choice is made using **training-only cross-validation** rather than test-set comparison. Once the winning algorithm is selected, it is refit on the full training set and then evaluated once on the locked test set.

#### 8. Interpretation and sensitivity layers

The later-stage notebooks add:
- permutation-importance summaries
- linear or Cox coefficient summaries where appropriate
- threshold-based reporting for the main cancer-specific binary task
- risk-group survival summaries for the cancer-specific Cox task
- repeated outer-validation and leave-one-cohort-out sensitivity analyses
- competing-risks sensitivity analysis for M2b using Aalen–Johansen cumulative-incidence summaries

These layers do not replace the main modelling pipeline; they contextualize it.

### Final model results

| Task | Final model | Primary metric | Locked test result |
|---|---|---:|---:|
| M1a overall survival | Random forest | AUC-ROC | **0.766** |
| M1b cancer-specific survival | Logistic regression (L2) | AUC-ROC | **0.705** |
| M2a overall survival Cox | Cox proportional hazards | C-index | **0.756** |
| M2b cancer-specific Cox | Cox proportional hazards | C-index | **0.724** |
| M3 PAM50 subtype | Logistic regression (L2) | Macro-F1 | **0.747** |
| M4 histologic grade | Elastic net (L1+L2) | Quadratic weighted kappa | **0.527** |

Additional binary-task diagnostics retained in the final reporting include **PR-AUC**, **Brier score**, **ECE**, and **threshold trade-off tables**.

### Results

#### Overall performance profile

The results suggest that the repository performs best on tasks where strong clinical and molecular structure is expected to exist in the data. Overall-survival classification and survival modelling perform solidly, PAM50 subtype prediction is strong, the cancer-specific tasks remain useful but more difficult, and histologic-grade prediction is the weakest of the six tasks.

#### Binary tasks

The binary tasks should not be interpreted through AUC alone. The final reporting retains:
- **PR-AUC** to reflect positive-class difficulty
- **Brier score** and **ECE** to quantify probabilistic calibration
- **threshold trade-off tables** for operational interpretation

This is especially important for **M1b**, where the repository is not only ranking risk but also framing cancer-specific event probability.

#### Survival tasks

The survival branch is methodologically coherent. By using Cox modelling with censoring handled explicitly, M2a and M2b avoid the information loss that occurs when survival is reduced to a binary label. The cancer-specific survival task is further strengthened by a separate competing-risks sensitivity analysis.

#### Molecular subtype task

The **M3 PAM50** task performs well and is biologically informative beyond its headline metric. Supporting analyses show that the learned molecular programmes and formal pathway scores separate known PAM50 subtypes in expected directions.

#### Histologic-grade task

The **M4** result is the most modest. The task is still useful as a structured pathology-support problem, but the performance is clearly lower than the subtype and survival tasks.

### Final methodological implementation choices

The final pipeline reflects a set of deliberate implementation choices that define how the repository should be read and reproduced:

1. **One shared split across tasks**  
   A single split is used to keep evaluation aligned across the six problem settings.

2. **Fitted preprocessing restricted to training data**  
   Imputation, NMF, and downstream selection steps are estimated on training rows only.

3. **Task-appropriate feature exclusions to avoid leakage**  
   Variables such as Nottingham Prognostic Index are excluded where they would create structural redundancy or target leakage. Integrative-cluster encodings are excluded from the PAM50 prediction task because they are too closely tied to subtype biology.

4. **Training-only cross-validation for classification model selection**  
   Classification algorithms are selected using CV on the training set rather than by comparing held-out test performance.

5. **Refit-after-selection workflow**  
   Once a winning classification algorithm is chosen, it is refit on the full training set before a single locked test evaluation.

6. **Separate storage of CV results and final test results**  
   Training/CV summaries and final held-out summaries are treated as conceptually distinct outputs.

7. **Expanded binary-task reporting**  
   Final binary-task reporting retains calibration and threshold information alongside discrimination metrics.

8. **Biological validation layered onto molecular compression**  
   NMF programmes are not left as unlabeled latent factors; they are checked against pathway, subtype, and microenvironment evidence.

9. **Sensitivity analyses reported as contextual diagnostics**  
   Repeated outer validation, cohort transport checks, and competing-risks summaries are used to probe robustness without redefining the primary estimand of the main models.

### Main biological findings

1. **Expression-derived biology is recoverable from the current data**  
   Proliferation, luminal, HER2, and immune/IFN scores separate PAM50 subtypes in sensible directions.

2. **The NMF layer is biologically interpretable**  
   The main programmes align with recognizable axes such as luminal/hormone signalling, proliferation/cell cycle, HER2/epithelial signalling, and immune/interferon activity.

3. **Clinical and molecular structure tell a consistent story**  
   Survival differences by receptor status, grade, and subtype remain directionally compatible with established breast-cancer literature.

4. **Cohort effects are real and visible**  
   Cohort structure is evident in both clinical variables and expression-derived features, which explains why cohort encoding and cohort sensitivity analyses are useful.

5. **Cancer-specific modelling benefits from separating all-cause and cancer-specific risk**  
   The distinction between M1a/M2a and M1b/M2b is clinically meaningful rather than cosmetic.

### Key learnings

1. **Leakage control materially changes how believable the results are.**  
   The final repository is easier to trust because preprocessing and model selection are clearly restricted to training data.

2. **A shared preprocessing backbone can still support multiple prediction paradigms effectively.**  
   Binary classification, Cox modelling, multiclass subtype prediction, and grade prediction can coexist in one coherent structure.

3. **Expression compression becomes more valuable when it is biologically annotated.**  
   NMF alone gives dimensionality reduction; NMF plus pathway validation gives interpretable molecular structure.

4. **Cancer-specific endpoints improve the clinical relevance of the project.**  
   The repository is stronger because it does not collapse everything into all-cause mortality.

5. **Sensitivity analyses improve transparency.**  
   Cohort transport and competing-risks analyses make the repository more honest about where results are stable and where caution is needed.

6. **Not all tasks should be presented with the same level of confidence.**  
   PAM50 and survival tasks are stronger; grade prediction is useful but clearly weaker.

### Limitations

- **No external validation yet**
- **Biology not fully multi-omic**
- **M4 remains the weakest predictive task**
- **Public reproducibility depends on local data access and a prepared local environment**

### Bottom line

The repository is best understood as a **leakage-aware, multi-task METABRIC modelling pipeline** that combines:
- disciplined preprocessing,
- training-only model selection,
- final locked held-out evaluation,
- biologically interpretable expression programmes,
- pathway and microenvironment validation,
- and a structured technical documentation layer.

It is strongest as a **well-documented modelling and analysis pipeline**. It is not yet a fully externalized clinical deployment package, but it provides a coherent and methodologically careful end-to-end workflow.

---

## Running the full pipeline

Run the scripts from the repository root in this order:

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

These files are usually generated locally and do not need to be fully tracked in Git.

---

## Documentation

Project documentation lives in `docs/`.

Recommended contents:
- per-script technical reports
- selected curated figures or summary tables
- methodology notes
- limitations and interpretation notes
- this project-level summary as a short entry point

---

## Environment policy

This repository should **not** track:
- `.venv/`
- local Conda environments
- editor-specific local folders such as `.vscode/`
- temporary outputs under `outputs/`
- raw local datasets under `data/`
- machine-specific package snapshots unless you explicitly want to publish them

---

## Public repository checklist

Before pushing, check that the public repository contains:
- source code under `src/`
- documentation under `docs/`
- top-level project metadata such as `README.md` and `.gitignore`

Before pushing, check that the public repository does **not** contain:
- virtual environments
- cached notebook artefacts
- temporary outputs
- machine-specific editor state
- private or redistribution-restricted data you do not want in Git history
