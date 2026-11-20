"""Train grouped-CV tree baselines on simulated masked county profiles."""

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, GroupKFold
from xgboost import XGBRegressor

from src.data.clean import PROCESSED_PATH
from src.features.audit import audit_masked_inputs, audit_splits
from src.features.engineer import ATOMIC_COLUMNS, engineer_features
from src.features.masking import PATTERNS, RATES, simulate_mask
from src.features.split import split_counties
from src.models.evaluation import mask_fingerprint
from src.models.metrics import consistency_metrics, reconstruction_metrics

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FEATURE_COLUMNS = ATOMIC_COLUMNS + ["target_age_index", "target_is_male"]


def examples(
    inputs: pd.DataFrame, mask: pd.DataFrame, targets: pd.DataFrame, groups: pd.Series
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create one example per held-out cell; other held-out cells stay hidden."""
    if not inputs.index.equals(groups.index):
        raise ValueError("County groups must align with inputs")
    row, col = np.where(mask.to_numpy(dtype=bool))
    if not len(row):
        raise ValueError("No masked targets to model")
    matrix = inputs.to_numpy(dtype=float)
    if np.isinf(matrix).any():
        raise ValueError("Inputs must be finite or missing")
    predictors = np.nan_to_num(matrix[row], nan=-1.0)
    X = pd.DataFrame(predictors, columns=ATOMIC_COLUMNS)
    X["target_age_index"] = col // 2
    X["target_is_male"] = col % 2
    y = targets.to_numpy(dtype=float)[row, col]
    if not np.isfinite(y).all():
        raise ValueError("Every masked cell must have a finite held-out target")
    return X, y, groups.to_numpy()[row], row, col


def model_grid(name: str, seed: int):
    if name == "random_forest":
        return RandomForestRegressor(random_state=seed, n_jobs=1), {
            "n_estimators": [100], "min_samples_leaf": [1, 3], "max_features": [0.7, 1.0]
        }
    if name == "xgboost":
        return XGBRegressor(
            objective="reg:squarederror", random_state=seed, n_jobs=1, verbosity=0
        ), {"n_estimators": [100], "max_depth": [2, 4], "learning_rate": [0.05, 0.1]}
    raise ValueError(f"Unknown model: {name}")


def fit_baseline(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, name: str, seed: int):
    unique = np.unique(groups)
    if len(unique) < 3:
        raise ValueError("At least three masked training counties are needed for grouped CV")
    estimator, grid = model_grid(name, seed)
    search = GridSearchCV(
        estimator, grid, scoring="neg_mean_absolute_error",
        cv=GroupKFold(n_splits=min(3, len(unique))), n_jobs=1, error_score="raise",
    )
    search.fit(X, y, groups=groups)
    return search


def save_shap_plot(model, X: pd.DataFrame, path: Path) -> None:
    sample = X.iloc[: min(len(X), 200)]
    values = np.asarray(shap.TreeExplainer(model).shap_values(sample))
    if values.shape != sample.shape or not np.isfinite(values).all():
        raise ValueError("Unexpected or non-finite SHAP values")
    importance = pd.Series(np.abs(values).mean(axis=0), index=sample.columns).nlargest(15)
    fig, axis = plt.subplots(figsize=(9, 6))
    importance.sort_values().plot.barh(ax=axis, color="#7C3AED")
    axis.set_xlabel("Mean absolute SHAP value")
    axis.set_title("Training-sample feature importance")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def run_baselines(
    profiles: pd.DataFrame, model_dir: Path, report_dir: Path, seed: int = 42
) -> pd.DataFrame:
    if profiles["admin1_code"].duplicated().any():
        raise ValueError("Expected one profile per county")
    splits = split_counties(profiles, seed)
    audit_splits(splits)
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for pattern in PATTERNS:
        for rate in RATES:
            data = {}
            for name in ("train", "validation"):
                split = splits[name]
                inputs, mask, targets = simulate_mask(split, pattern, rate, seed)
                audit_masked_inputs(split, inputs, mask, targets, engineer_features(inputs))
                data[name] = (
                    inputs, mask, targets,
                    examples(inputs, mask, targets, split["admin1_code"]),
                )
            train_X, train_y, train_groups, _, _ = data["train"][3]
            validation_X, validation_y, _, row, col = data["validation"][3]
            for name in ("random_forest", "xgboost"):
                search = fit_baseline(train_X, train_y, train_groups, name, seed)
                predictions = search.predict(validation_X)
                inputs, _, _, _ = data["validation"]
                complete_values = inputs.to_numpy(dtype=float, copy=True)
                complete_values[row, col] = predictions
                complete = pd.DataFrame(complete_values, columns=inputs.columns, index=inputs.index)
                reconstructed = pd.concat(
                    [splits["validation"].drop(columns=ATOMIC_COLUMNS), complete], axis=1
                )
                result = {
                    "pattern": pattern, "rate": rate, "model": name,
                    "split": "validation",
                    "train_counties_with_targets": len(np.unique(train_groups)),
                    "train_targets": len(train_y), "validation_targets": len(validation_y),
                    "mask_sha256": mask_fingerprint(
                        splits["validation"]["admin1_code"], data["validation"][1]
                    ),
                    "cv_mae": -search.best_score_, "best_params": json.dumps(search.best_params_),
                    **reconstruction_metrics(validation_y, predictions),
                    **consistency_metrics(reconstructed),
                }
                results.append(result)
                label = f"{pattern}_{int(rate * 100)}_{name}"
                joblib.dump({
                    "model": search.best_estimator_, "feature_columns": FEATURE_COLUMNS,
                    "pattern": pattern, "rate": rate, "seed": seed,
                    "best_params": search.best_params_,
                }, model_dir / f"{label}.joblib")
                save_shap_plot(search.best_estimator_, train_X, report_dir / f"{label}_shap.png")
                print(f"{label}: validation MAE={result['mae']:.2f}", flush=True)
    table = pd.DataFrame(results)
    table.to_csv(report_dir / "baseline_validation.csv", index=False)
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=PROCESSED_PATH)
    parser.add_argument("--model-dir", type=Path, default=Path("models/baselines"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports/baselines"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    digest = hashlib.sha256()
    with args.input.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    checksum = digest.hexdigest()
    profiles = pd.read_csv(args.input)
    table = run_baselines(profiles, args.model_dir, args.report_dir, args.seed)
    metadata = {
        "input_sha256": checksum, "seed": args.seed, "rows": len(profiles),
        "results": len(table), "split": "validation",
    }
    (args.report_dir / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved {len(table)} validation results to {args.report_dir}")


if __name__ == "__main__":
    main()
