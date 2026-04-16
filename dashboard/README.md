# Dashboard

This dashboard is an optional frontend for visualising exported summaries from the Python pipeline.

## **SEE dashboard_demo.gif for the final output without the need to run the code** 

## Purpose

The app is designed to present:
- locked final model results
- M1b binary metrics and threshold trade-offs
- M2b risk-group and competing-risks summaries
- feature-representation comparisons
- repeated outer-validation and cohort transport summaries
- an illustrative clinician-facing patient sandbox

The patient sandbox is intentionally **illustrative**. It does not run the trained Python models directly in the browser. The benchmark panels, however, are designed to load exported JSON files produced from the pipeline outputs.

## Expected JSON location

Place exported JSON files in:

```text
dashboard/public/pipeline-data/
```

Expected files:

- `30_model_summary.json`
- `28c_m1b_binary_metrics_summary.json`
- `28c_m1b_threshold_metrics.json`
- `28d_m2b_risk_group_summary.json`
- `54_m2b_competing_risk_summary.json`

Optional files:

- `29_feature_representation_scores.json`
- `31_cohort_transport_sensitivity.json`
- `32_repeated_outer_validation.json`

## Local run

From the `dashboard/` directory:

```bash
npm install
npm run dev
```

Then open:

```text
http://localhost:3000
```

## Notes

- The dashboard is a presentation layer for the pipeline, not the analytical core of the project.
