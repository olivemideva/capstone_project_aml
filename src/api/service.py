"""Load trusted artifacts and perform reconstruction and prioritisation."""

import json
import threading
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.features.engineer import ATOMIC_COLUMNS
from src.features.masking import PATTERNS, RATES
from src.optimization.pareto import pareto_solutions
from src.optimization.problem import OBJECTIVES, CountyProblem
from src.optimization.weighted import weighted_sum

FEATURE_COLUMNS = ATOMIC_COLUMNS + ["target_age_index", "target_is_male"]
SOURCE = "observed 2019 HDX HAPI age-sex population"


class Service:
    def __init__(self, models: dict, indicators: pd.DataFrame, family: str):
        self.models = models
        self.indicators = indicators
        self.family = family
        self.optimization_lock = threading.BoundedSemaphore(1)

    @classmethod
    def load(cls, model_dir: Path, indicator_path: Path):
        metadata = json.loads((model_dir / "selection.json").read_text(encoding="utf-8"))
        family = metadata["selected_family"]
        if family not in {"random_forest", "xgboost"}:
            raise ValueError(f"Unsupported selected model family: {family}")
        allowed = {
            f"{pattern}_{int(rate * 100)}_{family}.joblib" for pattern in PATTERNS for rate in RATES
        }
        files = metadata["artifact_files"]
        if len(files) != len(allowed) or set(files) != allowed:
            raise ValueError("Selected artifact list must contain exactly the nine tree models")
        models = {}
        for pattern in PATTERNS:
            for rate in RATES:
                percent = int(rate * 100)
                artifact = joblib.load(model_dir / f"{pattern}_{percent}_{family}.joblib")
                if (
                    artifact["feature_columns"] != FEATURE_COLUMNS
                    or artifact["pattern"] != pattern
                    or artifact["rate"] != rate
                    or not callable(getattr(artifact["model"], "predict", None))
                ):
                    raise ValueError(f"Invalid model artifact for {pattern}/{percent}")
                models[(pattern, percent)] = artifact["model"]
        indicators = pd.read_csv(indicator_path, dtype={"admin1_code": str})
        required = [
            "admin1_code",
            "admin1_name",
            "children",
            "working_age",
            "elderly",
            "total",
            "child_share",
            "elderly_share",
            "dependency_ratio",
        ]
        if list(indicators.columns) != required or indicators.empty:
            raise ValueError("County indicator artifact has an invalid schema or is empty")
        for column in ("admin1_code", "admin1_name"):
            if (
                indicators[column].isna().any()
                or indicators[column].astype(str).str.strip().eq("").any()
            ):
                raise ValueError(f"County indicator artifact has missing {column}")
        if indicators["admin1_code"].duplicated().any():
            raise ValueError("County indicator artifact contains duplicate county codes")
        numbers = indicators[required[2:]].apply(pd.to_numeric, errors="raise")
        if not np.isfinite(numbers.to_numpy()).all() or (numbers < 0).any().any():
            raise ValueError("County indicators must be finite and non-negative")
        if (numbers["working_age"] <= 0).any() or (numbers["total"] <= 0).any():
            raise ValueError("County population and working-age counts must be positive")
        indicators[required[2:]] = numbers
        CountyProblem(indicators, 1)
        return cls(models, indicators, family)

    def reconstruct(self, population: dict, pattern: str, rate: int) -> tuple[dict, list[str]]:
        positions = [i for i, col in enumerate(ATOMIC_COLUMNS) if population[col] is None]
        values = np.array(
            [-1.0 if population[col] is None else float(population[col]) for col in ATOMIC_COLUMNS]
        )
        X = pd.DataFrame(np.tile(values, (len(positions), 1)), columns=ATOMIC_COLUMNS)
        X["target_age_index"] = np.array(positions) // 2
        X["target_is_male"] = np.array(positions) % 2
        predictions = np.asarray(self.models[(pattern, rate)].predict(X), dtype=float)
        if predictions.shape != (len(positions),) or not np.isfinite(predictions).all():
            raise ValueError("Model returned invalid predictions")
        if (predictions < 0).any():
            raise ValueError("Model returned negative population predictions")
        completed = {
            key: float(population[key]) for key in ATOMIC_COLUMNS if population[key] is not None
        }
        predicted_fields = [ATOMIC_COLUMNS[pos] for pos in positions]
        completed.update(zip(predicted_fields, predictions.tolist()))
        return {key: completed[key] for key in ATOMIC_COLUMNS}, predicted_fields

    def prioritise(self, budget: int, weights: tuple, seed: int, generations: int) -> list[dict]:
        problem = CountyProblem(self.indicators, budget)
        baseline = weighted_sum(self.indicators, budget, np.asarray(weights))
        pareto = pareto_solutions(self.indicators, budget, seed, generations)
        result = []
        for method, choices in (("weighted_sum", [baseline]), ("nsga3", pareto)):
            for choice in choices:
                scores = np.asarray(choice, dtype=float) @ problem.weights
                counties = self.indicators.loc[choice, ["admin1_code", "admin1_name"]]
                result.append(
                    {
                        "method": method,
                        "selected_counties": counties.to_dict(orient="records"),
                        "selected_count": len(counties),
                        **dict(zip(OBJECTIVES, scores.tolist())),
                    }
                )
        return result
