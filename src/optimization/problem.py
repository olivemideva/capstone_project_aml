"""Binary county selection under a simulated equal-cost budget."""

import numpy as np
import pandas as pd
from pymoo.core.problem import Problem

OBJECTIVES = ("child_coverage", "elderly_coverage", "dependency_burden_coverage")


def objective_weights(indicators: pd.DataFrame) -> np.ndarray:
    columns = ["children", "elderly", "dependency_ratio"]
    missing = set(columns) - set(indicators.columns)
    if missing:
        raise ValueError(f"Missing indicator columns: {sorted(missing)}")
    values = indicators[columns].to_numpy(dtype=float)
    if not len(values) or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Indicators must be nonempty, finite and non-negative")
    totals = values.sum(axis=0)
    if (totals <= 0).any():
        raise ValueError("Each objective must have a positive national total")
    return values / totals


class CountyProblem(Problem):
    def __init__(self, indicators: pd.DataFrame, budget: int):
        self.weights = objective_weights(indicators)
        if (
            isinstance(budget, bool)
            or not isinstance(budget, int)
            or not 1 <= budget <= len(indicators)
        ):
            raise ValueError("Budget must be a whole number between 1 and the number of counties")
        self.budget = budget
        super().__init__(n_var=len(indicators), n_obj=3, n_ieq_constr=1, xl=0, xu=1, vtype=bool)

    def _evaluate(self, x, out, *args, **kwargs):
        selection = np.asarray(x, dtype=bool)
        out["F"] = -(selection.astype(float) @ self.weights)
        out["G"] = selection.sum(axis=1, keepdims=True) - self.budget


def evaluate_selection(selection: np.ndarray, indicators: pd.DataFrame, budget: int) -> np.ndarray:
    problem = CountyProblem(indicators, budget)
    vector = np.asarray(selection)
    if vector.shape != (len(indicators),) or not np.isin(vector, [0, 1]).all():
        raise ValueError("Selection must be a binary vector aligned with the counties")
    if vector.sum() > budget:
        raise ValueError("Selection exceeds the simulated budget")
    return vector.astype(float) @ problem.weights
