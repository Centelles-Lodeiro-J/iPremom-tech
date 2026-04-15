"""Capture runtime environment and write a reproducibility manifest.

Outputs to outputs/metadata/
- run_manifest.json
- input_file_hashes.csv
- script_hashes.csv
- requirements_runtime_frozen.txt
"""
import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

KEY_PACKAGES = [
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "matplotlib",
    "seaborn",
    "joblib",
    "lifelines",
    "pytest",
]

MAIN_SCRIPTS = [
    Path("src") / "02_pipeline_data_preparation.py",
    Path("src_patched") / "03_feature_selection_cox_and_classification.patched.py",
    Path("src_patched") / "07_notebook_models_and_algorithm_comparison.patched.py",
    Path("src_patched") / "04c_notebook_product_support_plots.patched.py",
    Path("src_patched") / "07b_notebook_model_winners_summary.patched.py",
    Path("src_patched") / "08_precompute_permutation_importance.patched.py",
    Path("src_patched") / "09_notebook_model_interpretation.patched.py",
    Path("src_patched") / "10_notebook_clinical_discussion.patched.py",
]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_version(pkg):
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return None


def main():
    p = argparse.ArgumentParser(description="Capture environment and write run manifest")
    p.add_argument("--input", type=Path, nargs="*", default=[Path("data") / "FCS_ml_test_input_data_rna_mutation.csv"])
    p.add_argument("--output-dir", type=Path, default=Path("outputs") / "metadata")
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
        },
        "cwd": str(Path.cwd()),
        "package_versions": {pkg: safe_version(pkg) for pkg in KEY_PACKAGES},
    }

    script_rows = []
    for path in MAIN_SCRIPTS:
        if path.exists():
            script_rows.append({"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size})
    with open(args.output_dir / "script_hashes.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "sha256", "bytes"])
        writer.writeheader()
        writer.writerows(script_rows)

    input_rows = []
    for path in args.input:
        if path.exists():
            input_rows.append({"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size})
    with open(args.output_dir / "input_file_hashes.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "sha256", "bytes"])
        writer.writeheader()
        writer.writerows(input_rows)

    try:
        freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], check=True, capture_output=True, text=True)
        (args.output_dir / "requirements_runtime_frozen.txt").write_text(freeze.stdout, encoding="utf-8")
        manifest["pip_freeze_file"] = str(args.output_dir / "requirements_runtime_frozen.txt")
    except Exception as e:
        manifest["pip_freeze_error"] = str(e)

    manifest["script_hash_file"] = str(args.output_dir / "script_hashes.csv")
    manifest["input_hash_file"] = str(args.output_dir / "input_file_hashes.csv")
    with open(args.output_dir / "run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved run manifest → {args.output_dir / 'run_manifest.json'}")


if __name__ == "__main__":
    main()
