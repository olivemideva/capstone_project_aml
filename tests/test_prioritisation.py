import itertools
import json

import numpy as np
import pandas as pd
import pytest

from src.data.clean import POPULATION_COLUMNS
from src.optimization.compare import compare_methods
from src.optimization.indicators import county_indicators
from src.optimization.pareto import pareto_solutions
from src.optimization.problem import CountyProblem, evaluate_selection
from src.optimization.weighted import weighted_sum


def profiles(count=8):
    rows = []
    for index in range(count):
        row = dict.fromkeys(POPULATION_COLUMNS, index + 1)
        row["admin1_code"] = f"KE{index + 1:03}"
        row["admin1_name"] = f"County {index + 1}"
        rows.append(row)
    return pd.DataFrame(rows)


def test_indicators_observed_age_groups_and_missing():
    data = profiles()
    indicators = county_indicators(data)
    assert indicators.loc[0, "children"] == 6
    assert indicators.loc[0, "working_age"] == 20
    assert indicators.loc[0, "elderly"] == 16
    assert indicators.loc[0, "dependency_ratio"] == pytest.approx(22 / 20)
    data.loc[0, "population_f_0_4"] = np.nan
    with pytest.raises(ValueError, match="complete observed"):
        county_indicators(data)


def test_weighted_sum_is_exact_and_respects_budget():
    indicators = county_indicators(profiles())
    weights = np.array([0.2, 0.3, 0.5])
    chosen = weighted_sum(indicators, 3, weights)
    score = evaluate_selection(chosen, indicators, 3) @ weights
    brute = []
    for combo in itertools.combinations(range(8), 3):
        vector = np.zeros(8, dtype=bool)
        vector[list(combo)] = True
        brute.append(evaluate_selection(vector, indicators, 3) @ weights)
    assert chosen.sum() == 3
    assert score == pytest.approx(max(brute))
    with pytest.raises(ValueError, match="budget"):
        evaluate_selection(np.ones(8), indicators, 3)
    with pytest.raises(ValueError, match="weight"):
        weighted_sum(indicators, 3, np.zeros(3))


def test_nsga3_feasibility_and_seed():
    indicators = county_indicators(profiles())
    first = pareto_solutions(indicators, 3, 42, 8)
    second = pareto_solutions(indicators, 3, 42, 8)
    np.testing.assert_array_equal(first, second)
    assert first.ndim == 2 and first.shape[1] == 8
    assert (first.sum(axis=1) <= 3).all()
    values = first @ CountyProblem(indicators, 3).weights
    for a, b in itertools.permutations(values, 2):
        assert not ((a >= b).all() and (a > b).any())


def test_comparison_outputs_and_hypervolume(tmp_path):
    summary = compare_methods(profiles(), tmp_path, budget=3, generations=8)
    results = pd.read_csv(tmp_path / "tradeoffs.csv")
    assert set(results.method) == {"nsga3", "weighted_sum"}
    assert results.selected_count.le(3).all()
    for _, row in results.iterrows():
        assert len(json.loads(row.selected_counties)) == row.selected_count
    assert all(value > 0 for value in summary["hypervolume"].values())
    assert (tmp_path / "tradeoffs.html").is_file()
    assert (tmp_path / "indicators.csv").is_file()
    assert (tmp_path / "summary.json").is_file()
