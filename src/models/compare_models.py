"""Compare paired validation scenarios and save the selected model family."""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.masking import PATTERNS, RATES

MODELS = ("random_forest", "xgboost", "autoencoder")


def compare(
    baseline_dir: Path, autoencoder_dir: Path, model_root: Path, output_dir: Path
) -> pd.DataFrame:
    baseline_meta = json.loads((baseline_dir / "run.json").read_text(encoding="utf-8"))
    auto_meta = json.loads((autoencoder_dir / "run.json").read_text(encoding="utf-8"))
    for key in ("input_sha256", "seed", "rows", "split"):
        if baseline_meta[key] != auto_meta[key]:
            raise ValueError(f"Runs differ in {key}; retrain before comparing")
    if baseline_meta["split"] != "validation":
        raise ValueError("Model selection must use validation data only")
    baseline = pd.read_csv(baseline_dir / "baseline_validation.csv")
    autoencoder = pd.read_csv(autoencoder_dir / "autoencoder_validation.csv")
    if set(baseline["model"]) != {"random_forest", "xgboost"}:
        raise ValueError("Baseline report must contain both tree model families")
    if set(autoencoder["model"]) != {"autoencoder"}:
        raise ValueError("Autoencoder report contains unexpected models")
    results = pd.concat([baseline, autoencoder], ignore_index=True)
    expected = {
        (pattern, rate, model) for pattern in PATTERNS for rate in RATES for model in MODELS
    }
    actual = list(zip(results["pattern"], results["rate"], results["model"]))
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("Expected exactly one result per masking scenario and model")
    if not results["split"].eq("validation").all():
        raise ValueError("Comparison contains non-validation results")
    if not np.isfinite(results["mae"]).all():
        raise ValueError("Validation MAE must be finite")
    for column in ("validation_targets", "mask_sha256"):
        counts = results.groupby(["pattern", "rate"])[column].nunique()
        if results[column].isna().any() or counts.ne(1).any():
            raise ValueError(f"Models must share validation masks and targets: {column}")
    ranking = results.groupby("model")["mae"].mean().sort_values(kind="stable")
    winner = str(ranking.index[0])

    sources = []
    for pattern in PATTERNS:
        for rate in RATES:
            label = f"{pattern}_{int(rate * 100)}_{winner}"
            if winner == "autoencoder":
                source = model_root / "autoencoders" / f"{label}.pt"
            else:
                source = model_root / "baselines" / f"{label}.joblib"
            if not source.is_file():
                raise FileNotFoundError(f"Selected model artifact not found: {source}")
            sources.append(source)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Selection directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    for source in sources:
        shutil.copy2(source, output_dir / source.name)
    results.to_csv(output_dir / "comparison.csv", index=False)
    metadata = {
        "selection_metric": "unweighted mean validation MAE across the nine masking scenarios",
        "selected_family": winner, "mean_mae_by_family": ranking.to_dict(),
        "input_sha256": baseline_meta["input_sha256"], "seed": baseline_meta["seed"],
        "artifact_files": [path.name for path in sources],
        "test_evaluated": False,
    }
    (output_dir / "selection.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-report", type=Path, default=Path("reports/baselines"))
    parser.add_argument("--autoencoder-report", type=Path, default=Path("reports/autoencoders"))
    parser.add_argument("--model-root", type=Path, default=Path("models"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/selected"))
    args = parser.parse_args()
    compare(args.baseline_report, args.autoencoder_report, args.model_root, args.output_dir)
    selection = json.loads((args.output_dir / "selection.json").read_text(encoding="utf-8"))
    print(f"Selected {selection['selected_family']}: {selection['mean_mae_by_family']}")


if __name__ == "__main__":
    main()
