param(
    [switch]$InstallDeps = $false,
    [switch]$RunTests = $true
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

$py = Join-Path $repo '.venv\Scripts\python.exe'
if (-not (Test-Path $py)) {
    throw "Python interpreter not found at $py"
}

function Run-Step {
    param([string]$Label, [string[]]$Args)
    Write-Host "`n=== $Label ===" -ForegroundColor Cyan
    & $py @Args
    if ($LASTEXITCODE -ne 0) {
        throw "$Label failed with exit code $LASTEXITCODE"
    }
}

Run-Step '00 capture environment' @('.\src\00_capture_environment_and_run_manifest.py')
Run-Step '01 data understanding' @('.\src\01_notebook_data_understanding.py')
Run-Step '02 pipeline preparation (biology enhanced)' @('.\src\02_pipeline_data_preparation.biology_enhanced.py')
Run-Step '02b pipeline support plots' @('.\src\02b_notebook_pipeline_support_plots.py')
Run-Step '03 feature selection (patched)' @('.\src\03_feature_selection_cox_and_classification.patched.py')
Run-Step '03b feature-selection support plots' @('.\src\03b_notebook_feature_selection_support_plots.py')
Run-Step '04 data cleaning validation' @('.\src\04_notebook_data_cleaning_validation.py')
Run-Step '04c product-support plots (patched)' @('.\src\04c_notebook_product_support_plots.patched.py')
Run-Step '05 clinical and molecular validation (biology enhanced)' @('.\src\05_notebook_clinical_molecular_validation.biology_enhanced.fixed.py')
Run-Step '06 baseline models' @('.\src\06_notebook_baseline_models.py')
Run-Step '07 model comparison (patched)' @('.\src\07_notebook_models_and_algorithm_comparison.patched.py')
Run-Step '07b winners summary (patched)' @('.\src\07b_notebook_model_winners_summary.patched.py')
Run-Step '08 permutation importance (patched)' @('.\src\08_precompute_permutation_importance.patched.py')
Run-Step '09 model interpretation (patched)' @('.\src\09_notebook_model_interpretation.patched.py')
Run-Step '10 clinical discussion (patched)' @('.\src\10_notebook_clinical_discussion.patched.py')
Run-Step '11 survival sensitivity (advanced)' @('.\src\11_notebook_survival_sensitivity_advanced.py')

if ($RunTests) {
    Run-Step 'pytest support-plot tests' @('-m', 'pytest', '.\src\test_support_plot_summaries.py', '-q')
}

Write-Host "`nPipeline run completed successfully." -ForegroundColor Green
