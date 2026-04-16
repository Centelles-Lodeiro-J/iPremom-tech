#!/usr/bin/env python3
"""
Convert selected pipeline CSV outputs into JSON files expected by the dashboard.

Example:
    python dashboard/scripts/export_dashboard_json.py \
        --outputs outputs \
        --out dashboard/public/pipeline-data
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


DEFAULT_MAP = {
    "30_model_summary.csv": "30_model_summary.json",
    "28c_m1b_binary_metrics_summary.csv": "28c_m1b_binary_metrics_summary.json",
    "28c_m1b_threshold_metrics.csv": "28c_m1b_threshold_metrics.json",
    "28d_m2b_risk_group_summary.csv": "28d_m2b_risk_group_summary.json",
    "54_m2b_competing_risk_summary.csv": "54_m2b_competing_risk_summary.json",
    "29_feature_representation_scores.csv": "29_feature_representation_scores.json",
    "31_cohort_transport_sensitivity.csv": "31_cohort_transport_sensitivity.json",
    "32_repeated_outer_validation.csv": "32_repeated_outer_validation.json",
}


def convert_csv_to_json(csv_path: Path, json_path: Path) -> None:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", type=Path, default=Path("outputs"))
    parser.add_argument("--out", type=Path, default=Path("dashboard/public/pipeline-data"))
    args = parser.parse_args()

    converted = []
    missing = []
    for csv_name, json_name in DEFAULT_MAP.items():
        csv_path = args.outputs / csv_name
        json_path = args.out / json_name
        if csv_path.exists():
            convert_csv_to_json(csv_path, json_path)
            converted.append((csv_path, json_path))
        else:
            missing.append(csv_path)

    print("Converted files:")
    for src, dst in converted:
        print(f"  {src} -> {dst}")

    if missing:
        print("\nMissing files:")
        for path in missing:
            print(f"  {path}")


if __name__ == "__main__":
    main()
