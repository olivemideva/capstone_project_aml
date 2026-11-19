"""Metrics for held-out cells and consistency with reported source totals."""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.data.clean import AGE_RANGES, GENDERS, population_column
from src.features.engineer import ATOMIC_COLUMNS


def reconstruction_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if actual.ndim != 1 or actual.shape != predicted.shape or not len(actual):
        raise ValueError("Actual and predicted values must be nonempty aligned vectors")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Actual and predicted values must be finite")
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": (
            float(r2_score(actual, predicted, force_finite=False)) if len(actual) >= 2 else np.nan
        ),
    }


def consistency_metrics(profiles: pd.DataFrame) -> dict[str, float | int]:
    values = profiles[ATOMIC_COLUMNS].to_numpy(dtype=float)
    observed = np.isfinite(values)
    if np.isinf(values).any():
        raise ValueError("Population values must not be infinite")
    checks: dict[str, list[float]] = {"gender": [], "age": []}

    def compare(kind: str, total: str, components: list[str]) -> None:
        missing = {total, *components} - set(profiles.columns)
        if missing:
            raise ValueError(f"Missing consistency columns: {sorted(missing)}")
        subset = profiles[[total, *components]].apply(pd.to_numeric, errors="raise")
        if np.isinf(subset.to_numpy(dtype=float)).any():
            raise ValueError("Population values must not be infinite")
        complete = subset.dropna()
        difference = complete[total] - complete[components].sum(axis=1)
        checks[kind].extend(difference.tolist())

    for age in AGE_RANGES:
        compare("gender", population_column("all", age), [
            population_column("f", age), population_column("m", age)
        ])
    for gender in GENDERS:
        compare("age", population_column(gender, "all"), [
            population_column(gender, age) for age in AGE_RANGES if age != "all"
        ])
    gender = np.asarray(checks["gender"])
    age = np.asarray(checks["age"])
    return {
        "negative_cells": int((values[observed] < 0).sum()),
        "observed_cells": int(observed.sum()),
        "gender_checks": len(gender),
        "gender_inconsistent": int(np.count_nonzero(gender)),
        "gender_absolute_difference": float(np.abs(gender).sum()),
        "age_checks": len(age),
        "age_inconsistent": int(np.count_nonzero(age)),
        "age_absolute_difference": float(np.abs(age).sum()),
    }
