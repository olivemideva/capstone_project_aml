"""Approximate a feasible Pareto front using binary NSGA-III."""

import numpy as np
import pandas as pd
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.core.repair import Repair
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.optimize import minimize
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.ref_dirs import get_reference_directions

from src.optimization.problem import CountyProblem


class BudgetRepair(Repair):
    def __init__(self, budget: int, seed: int):
        super().__init__()
        self.budget = budget
        self.rng = np.random.default_rng(seed)

    def _do(self, problem, X, **kwargs):
        repaired = np.asarray(X, dtype=bool).copy()
        for row in repaired:
            chosen = np.flatnonzero(row)
            if len(chosen) > self.budget:
                row[self.rng.choice(chosen, len(chosen) - self.budget, replace=False)] = False
        return repaired


def pareto_solutions(
    indicators: pd.DataFrame, budget: int = 10, seed: int = 42, generations: int = 200
) -> np.ndarray:
    if isinstance(generations, bool) or not isinstance(generations, int) or generations < 1:
        raise ValueError("Generations must be a positive integer")
    problem = CountyProblem(indicators, budget)
    ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=8)
    algorithm = NSGA3(
        ref_dirs=ref_dirs,
        pop_size=len(ref_dirs),
        sampling=BinaryRandomSampling(),
        crossover=TwoPointCrossover(),
        mutation=BitflipMutation(),
        repair=BudgetRepair(budget, seed),
        eliminate_duplicates=True,
    )
    result = minimize(problem, algorithm, ("n_gen", generations), seed=seed, verbose=False)
    if result.X is None:
        raise RuntimeError("NSGA-III did not return any feasible solutions")
    choices = np.atleast_2d(result.X).astype(bool)
    if (choices.sum(axis=1) > budget).any():
        raise RuntimeError("NSGA-III returned an infeasible selection")
    values = -(choices.astype(float) @ problem.weights)
    indices = NonDominatedSorting().do(values, only_non_dominated_front=True)
    return choices[indices]
