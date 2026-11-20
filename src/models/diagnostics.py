"""Evaluate selected reconstruction models and illustrative asymmetric error costs."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.features.engineer import ATOMIC_COLUMNS
from src.features.masking import PATTERNS, RATES, simulate_mask
from src.features.split import split_counties
from src.models.autoencoder import load_autoencoder, predict, sha256_file
from src.models.baselines import examples
from src.models.evaluation import mask_fingerprint
from src.models.metrics import reconstruction_metrics


def error_costs(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if actual.ndim != 1 or actual.shape != predicted.shape or not len(actual):
        raise ValueError("Actual and predicted values must be nonempty aligned vectors")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Actual and predicted values must be finite")
    errors = predicted - actual
    under = np.maximum(-errors, 0)
    over = np.maximum(errors, 0)
    return {
        "bias": float(errors.mean()),
        "mean_underestimate": float(under.mean()),
        "mean_overestimate": float(over.mean()),
        "underestimate_fraction": float((errors < 0).mean()),
        "cost_equal": float((under + over).mean()),
        "cost_under_double": float((2 * under + over).mean()),
        "cost_over_double": float((under + 2 * over).mean()),
    }


def evaluate_selected(
    processed: Path, model_dir: Path, split_name: str = "validation"
) -> pd.DataFrame:
    if split_name not in {"validation", "test"}:
        raise ValueError("Only validation or test evaluation is supported")
    metadata = json.loads((model_dir / "selection.json").read_text(encoding="utf-8"))
    if sha256_file(processed) != metadata["input_sha256"]:
        raise ValueError("Processed input does not match the selected models")
    family, seed = metadata["selected_family"], metadata["seed"]
    if family not in {"random_forest", "xgboost", "autoencoder"}:
        raise ValueError(f"Unsupported selected model family: {family}")
    comparison = pd.read_csv(model_dir / "comparison.csv")
    if not comparison["split"].eq("validation").all():
        raise ValueError("Model comparison must contain validation results only")
    profiles = pd.read_csv(processed)
    splits = split_counties(profiles, seed)
    split = splits[split_name]
    medians = splits["train"][ATOMIC_COLUMNS].median().to_numpy(dtype=float)
    if not np.isfinite(medians).all():
        raise ValueError("Training counties must have an observed value for every atomic column")
    results = []
    for pattern in PATTERNS:
        for rate in RATES:
            inputs, mask, targets = simulate_mask(split, pattern, rate, seed)
            mask_hash = mask_fingerprint(split["admin1_code"], mask)
            matching = comparison.loc[
                comparison["model"].eq(family) & comparison["pattern"].eq(pattern)
                & np.isclose(comparison["rate"], rate)
            ]
            if len(matching) != 1:
                raise ValueError("Expected one selected validation result per scenario")
            reference = matching.iloc[0]
            if split_name == "validation" and reference["mask_sha256"] != mask_hash:
                raise ValueError("Validation mask differs from model selection")
            label = f"{pattern}_{int(rate * 100)}_{family}"
            if family == "autoencoder":
                model, scale = load_autoencoder(model_dir / f"{label}.pt")
                predicted = predict(model, scale, inputs)[mask.to_numpy(dtype=bool)]
                actual = targets.to_numpy(dtype=float)[mask.to_numpy(dtype=bool)]
            else:
                artifact = joblib.load(model_dir / f"{label}.joblib")
                X, actual, _, _, _ = examples(inputs, mask, targets, split["admin1_code"])
                if list(X.columns) != artifact["feature_columns"]:
                    raise ValueError("Artifact features do not match evaluation features")
                predicted = artifact["model"].predict(X)
            metrics = reconstruction_metrics(actual, predicted)
            naive_predictions = np.broadcast_to(medians, mask.shape)[mask.to_numpy(dtype=bool)]
            naive_metrics = reconstruction_metrics(actual, naive_predictions)
            if split_name == "validation" and not np.isclose(
                metrics["mae"], reference["mae"], rtol=1e-6, atol=1e-6
            ):
                raise ValueError("Validation predictions do not match the selection report")
            results.append({
                "split": split_name, "pattern": pattern, "rate": rate, "model": family,
                "counties": len(split), "targets": len(actual), "mask_sha256": mask_hash,
                "cv_mae": float(reference["cv_mae"]), **metrics,
                **{f"median_baseline_{key}": value for key, value in naive_metrics.items()},
                **error_costs(actual, predicted),
            })
    return pd.DataFrame(results)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Diagnostics output already exists: {args.output}")
    results = evaluate_selected(args.input, args.model_dir, args.split)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False)
    print(f"Saved {args.split} diagnostics to {args.output}")


if __name__ == "__main__":
    main()
