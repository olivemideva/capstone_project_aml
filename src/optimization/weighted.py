"""Exact weighted-sum solutions for equal-cost county selections."""

import numpy as np
import pandas as pd

from src.optimization.problem import CountyProblem, evaluate_selection


def weighted_sum(indicators: pd.DataFrame, budget: int, weights: np.ndarray) -> np.ndarray:
    problem = CountyProblem(indicators, budget)
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (3,) or not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("Objective weights must contain three finite non-negative values")
    if weights.sum() <= 0:
        raise ValueError("At least one objective weight must be positive")
    score = problem.weights @ (weights / weights.sum())
    order = np.argsort(-score, kind="stable")
    selection = np.zeros(len(indicators), dtype=bool)
    selection[order[:budget]] = True
    evaluate_selection(selection, indicators, budget)
    return selection
