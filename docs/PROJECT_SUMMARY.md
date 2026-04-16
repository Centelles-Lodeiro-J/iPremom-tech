# Project summary

## Pipeline overview

This repository contains a leakage-aware, multi-task modelling pipeline built on the METABRIC cohort. The workflow is implemented as sequential Python scripts under `src/` and addresses six linked prediction problems derived from the same processed dataset:

- **M1a**: overall-survival binary classification
- **M1b**: cancer-specific binary classification
- **M2a**: overall-survival Cox modelling
- **M2b**: cancer-specific Cox modelling
- **M3**: PAM50 subtype prediction
- **M4**: histologic-grade prediction

The design philosophy is to use one coherent preprocessing backbone for all tasks while preserving task-specific modelling and evaluation choices. This enables consistent comparisons across endpoints without fitting separate, incompatible data-preparation pipelines for each model family.

## Methodology

### 1. Data preparation and deterministic cleaning

The workflow begins with a deterministic cleaning stage applied before modelling. This stage standardizes strings, resolves known typos, fixes a small number of explicit cross-field inconsistencies where the evidence is strong, and recodes variables into modelling-ready formats. The objective is to correct clearly documented data-quality problems while avoiding ad hoc manual interventions that would be difficult to reproduce.

Examples of this stage include:
- whitespace normalization in categorical variables
- explicit typo correction in ER IHC labels
- resolution of the single `Breast Sarcoma` inconsistency where the surrounding fields strongly support invasive breast carcinoma
- recoding of the cancer-specific outcome into a binary event indicator

This deterministic layer is important because it separates data repair from statistical imputation and makes the rationale for each edit inspectable.

### 2. Shared split design

The repository uses a **single shared train/test split** across the six tasks. The split is stratified on overall survival so that all downstream tasks are evaluated on aligned patients whenever possible. This design improves coherence across the project: the same held-out set is used to assess binary, survival, subtype, and grade tasks, which makes downstream comparison easier and avoids subtle inconsistencies introduced by per-task random splits.

### 3. Train-only preprocessing

A central methodological choice is that all fitted preprocessing steps are estimated on **training rows only** and then applied to the test set. This applies to:
- numeric and categorical imputation
- random-forest imputation for stage and grade where used
- NMF factorization of the expression matrix
- feature selection and model selection

This design minimizes information leakage and ensures that the test set functions as a true held-out evaluation set rather than a source of indirect training information.

### 4. Feature engineering

The prepared feature space combines:
- cleaned clinical variables
- ordinal encodings for ordered clinical factors
- binary encodings for receptor and treatment variables
- one-hot encodings for nominal clinical categories
- missingness indicators for selected variables
- log-transformed continuous variables where skewness is substantial
- NMF-derived gene programmes from the expression matrix
- additional pathway and tumour-microenvironment signature scores derived from expression

This produces a hybrid representation that preserves clinically interpretable variables while adding lower-dimensional molecular summaries.

### 5. Molecular dimension reduction and biological scoring

The repository uses **NMF** rather than PCA as the main transcriptomic compression strategy. The motivation is interpretability: non-negative decomposition yields additive gene programmes that can be manually and quantitatively annotated against known biological themes. In the final version, those programmes are further cross-checked against formal pathway and microenvironment scores, which strengthens the claim that the latent structure captures recognizable tumour biology.

### 6. Task-specific modelling

Each task uses an evaluation metric aligned to its modelling objective:

- **M1a, M1b**: classification tasks evaluated primarily with **AUC-ROC**, with additional reporting of PR-AUC, calibration, Brier score, ECE, and threshold trade-offs
- **M2a, M2b**: survival tasks evaluated with **C-index**
- **M3**: multiclass subtype task evaluated with **Macro-F1**
- **M4**: histologic-grade task evaluated with **quadratic weighted kappa**

This metric design avoids forcing all tasks into an accuracy-only reporting scheme and instead uses measures that better match imbalance, calibration, censoring, and ordinal structure.

### 7. Training-only model selection and locked final evaluation

For the final classification workflow, algorithm choice is made using **training-only cross-validation** rather than test-set comparison. Once the winning algorithm is selected, it is refit on the full training set and then evaluated once on the locked test set. This is one of the most important methodological implementation choices in the repository because it makes the reported final test metrics easier to interpret as honest held-out estimates.

### 8. Interpretation and sensitivity layers

The later-stage notebooks add:
- permutation-importance summaries
- linear or Cox coefficient summaries where appropriate
- threshold-based reporting for the main cancer-specific binary task
- risk-group survival summaries for the cancer-specific Cox task
- repeated outer-validation and leave-one-cohort-out sensitivity analyses
- competing-risks sensitivity analysis for M2b using Aalen–Johansen cumulative-incidence summaries

These layers do not replace the main modelling pipeline; they contextualize it and make its strengths and weaknesses more transparent.

## Final model results

| Task | Final model | Primary metric | Locked test result |
|---|---|---:|---:|
| M1a overall survival | Random forest | AUC-ROC | **0.766** |
| M1b cancer-specific survival | Logistic regression (L2) | AUC-ROC | **0.705** |
| M2a overall survival Cox | Cox proportional hazards | C-index | **0.756** |
| M2b cancer-specific Cox | Cox proportional hazards | C-index | **0.724** |
| M3 PAM50 subtype | Logistic regression (L2) | Macro-F1 | **0.747** |
| M4 histologic grade | Elastic net (L1+L2) | Quadratic weighted kappa | **0.527** |

Additional binary-task diagnostics retained in the final reporting include **PR-AUC**, **Brier score**, **ECE**, and **threshold trade-off tables**.

## Results

### Overall performance profile

The results suggest that the repository performs best on the tasks where strong clinical and molecular structure is expected to exist in the data:
- the binary overall-survival task performs solidly
- the cancer-specific binary task remains useful, though more difficult
- the Cox survival tasks show stable discriminatory performance
- PAM50 subtype prediction performs well relative to the difficulty of the task
- histologic-grade prediction remains the weakest of the six tasks

This pattern is internally coherent. The more direct and biologically structured targets tend to perform better, while the grade task appears noisier and less cleanly recoverable from the available feature space.

### Binary tasks

The two binary tasks should not be interpreted through AUC alone. The final reporting retains:
- **PR-AUC** to reflect positive-class difficulty
- **Brier score** and **ECE** to quantify probabilistic calibration
- **threshold trade-off tables** for operational interpretation

This is particularly important for **M1b**, where the repository is not just ranking risk but explicitly framing cancer-specific event probability. The threshold analysis shows the expected trade-off between sensitivity and specificity, and the calibration reporting makes clear that discrimination and calibration are related but distinct properties.

### Survival tasks

The survival branch is one of the more methodologically coherent parts of the project. By using Cox modelling with censoring handled explicitly, M2a and M2b avoid the information loss that occurs when survival is reduced to a simple binary label. The cancer-specific survival task is further strengthened by a separate competing-risks sensitivity analysis, which makes it easier to discuss how other-cause death interacts with cancer-specific incidence.

### Molecular subtype task

The **M3 PAM50** task performs strongly and is biologically useful beyond its headline metric. The supporting analyses show that the learned molecular programmes and formal pathway scores separate known PAM50 subtypes in expected directions, which suggests that the classifier is relying on recognizable tumour biology rather than arbitrary high-dimensional noise.

### Histologic-grade task

The **M4** result is the most modest, and that should be stated plainly. The task is still useful as a structured pathology-support problem, but the performance is clearly lower than the subtype and survival tasks. That makes the grade model more appropriate as a supportive signal than as a strong stand-alone predictor.

## Final methodological implementation choices

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

## Main biological findings

### 1. Expression-derived biology is recoverable from the current data

The pathway and microenvironment scoring layer shows that the expression matrix contains enough signal to recover familiar breast-tumour biology. Proliferation, luminal, HER2, and immune/IFN scores all separate PAM50 subtypes in sensible directions.

### 2. The NMF layer is biologically interpretable

The main NMF programmes are not arbitrary components. They align with recognizable axes such as:
- luminal / hormone signalling
- proliferation / cell cycle
- HER2 / epithelial signalling
- immune / interferon activity

This is strengthened by the concordance analyses between NMF programmes and formal pathway signatures.

### 3. Clinical and molecular structure tell a consistent story

Survival differences by receptor status, grade, and subtype remain directionally compatible with established breast-cancer literature. That consistency matters because it suggests that the modelling layer is operating on a biologically coherent representation rather than on heavily distorted preprocessing artefacts.

### 4. Cohort effects are real and visible

The repository finds substantial cohort structure in both clinical variables and expression-derived features. This is an important finding rather than a nuisance detail: it explains why cohort encoding is included and why sensitivity analyses around cohort transport are useful.

### 5. Cancer-specific modelling benefits from separating all-cause and cancer-specific risk

The distinction between M1a/M2a and M1b/M2b is not cosmetic. The cancer-specific tasks provide a more clinically targeted framing than all-cause mortality alone, and the results support keeping that distinction explicit throughout the project.

## Key learnings

1. **Leakage control is not a cosmetic improvement; it changes how believable the results are.**  
   The final repository is much easier to defend because preprocessing and model selection are clearly restricted to training data.

2. **A shared preprocessing backbone can still support multiple prediction paradigms effectively.**  
   Binary classification, Cox modelling, multiclass subtype prediction, and grade prediction can coexist in one consistent project structure when the evaluation logic is kept explicit.

3. **Expression compression becomes much more valuable when it is biologically annotated.**  
   NMF alone gives dimensionality reduction; NMF plus pathway validation gives interpretable molecular structure.

4. **Cancer-specific endpoints materially improve the clinical relevance of the pipeline.**  
   The project is stronger because it does not collapse everything into all-cause mortality.

5. **Calibration and threshold reporting are necessary for clinically framed binary tasks.**  
   AUC alone is not sufficient when the output is intended to support risk communication or action thresholds.

6. **Sensitivity analyses improve transparency even when they do not redefine the main model.**  
   The cohort transport and competing-risks analyses make the repository more honest about where results are stable and where caution is still needed.

7. **Not all tasks should be presented with the same level of confidence.**  
   PAM50 and survival tasks are stronger; grade prediction is useful but clearly weaker.

## Limitations

- **No external validation yet**  
  The current results come from internal train/test evaluation within the prepared METABRIC-based dataset.

- **Biology is stronger than before, but not yet fully multi-omic**  
  The current repository improves expression-based biological interpretation, but it still does not include a full copy-number or gene-level mutation integration layer.

- **Competing-risks modelling is still a sensitivity layer, not the primary survival estimand**  
  M2b includes a competing-risks sensitivity analysis, but the main model remains cause-specific Cox rather than a full Fine–Gray framework.

- **M4 remains the weakest predictive task**  
  Histologic-grade prediction shows only moderate performance relative to the other tasks.

- **Public reproducibility depends on a prepared local environment and local data access**  
  The repository documents the full workflow, but users still need local access to the input dataset and a compatible Python environment.

## Bottom line

The repository is best understood as a **reproducible, leakage-aware, multi-task METABRIC modelling pipeline** that combines:
- disciplined preprocessing,
- training-only model selection,
- final locked held-out evaluation,
- biologically interpretable expression programmes,
- pathway and microenvironment validation,
- and a structured technical documentation layer.

It is strongest as a **well-documented modelling and analysis pipeline**. It is not yet a fully externalized clinical deployment package, and it is not yet a full multi-omic discovery platform, but it does provide a coherent and methodologically careful end-to-end workflow.
