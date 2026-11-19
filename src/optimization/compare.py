"""Compare weighted-sum and NSGA-III prioritisation on observed profiles."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
from pymoo.indicators.hv import HV
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from src.data.clean import PROCESSED_PATH
from src.optimization.indicators import county_indicators
from src.optimization.pareto import pareto_solutions
from src.optimization.problem import OBJECTIVES, CountyProblem
from src.optimization.weighted import weighted_sum


def non_dominated(choices: np.ndarray, problem: CountyProblem) -> np.ndarray:
    choices = np.unique(np.atleast_2d(choices).astype(bool), axis=0)
    objectives = -(choices.astype(float) @ problem.weights)
    indices = NonDominatedSorting().do(objectives, only_non_dominated_front=True)
    return choices[indices]


def summarize_choices(
    choices: np.ndarray, indicators: pd.DataFrame, problem: CountyProblem, method: str
) -> pd.DataFrame:
    rows = []
    for choice in choices:
        selected = indicators.loc[choice, "admin1_code"].tolist()
        scores = choice.astype(float) @ problem.weights
        rows.append({
            "method": method,
            "selected_count": len(selected),
            "selected_counties": json.dumps(selected),
            **dict(zip(OBJECTIVES, scores)),
        })
    return pd.DataFrame(rows)


def compare_methods(
    profiles: pd.DataFrame, output_dir: Path, budget: int = 10,
    seed: int = 42, generations: int = 200,
) -> dict:
    indicators = county_indicators(profiles)
    problem = CountyProblem(indicators, budget)
    weighted_choices = []
    for first in range(11):
        for second in range(11 - first):
            weights = np.array([first, second, 10 - first - second], dtype=float)
            weighted_choices.append(weighted_sum(indicators, budget, weights))
    weighted_front = non_dominated(np.array(weighted_choices), problem)
    evolutionary_front = non_dominated(
        pareto_solutions(indicators, budget, seed, generations), problem
    )
    results = pd.concat([
        summarize_choices(weighted_front, indicators, problem, "weighted_sum"),
        summarize_choices(evolutionary_front, indicators, problem, "nsga3"),
    ], ignore_index=True)
    reference = np.array([0.01, 0.01, 0.01])
    hv = HV(ref_point=reference)
    hypervolume = {
        method: float(hv(-group[list(OBJECTIVES)].to_numpy()))
        for method, group in results.groupby("method")
    }
    summary = {
        "profile_source": "observed age-sex population",
        "simulated_budget_units": budget,
        "cost_per_county_units": 1,
        "seed": seed,
        "generations": generations,
        "weighted_sum_weight_vectors": 66,
        "hypervolume_reference_minimization": reference.tolist(),
        "hypervolume": hypervolume,
        "front_sizes": results.groupby("method").size().to_dict(),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    indicators.to_csv(output_dir / "indicators.csv", index=False)
    results.to_csv(output_dir / "tradeoffs.csv", index=False)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    figure = px.scatter_3d(
        results, x="child_coverage", y="elderly_coverage", z="dependency_burden_coverage",
        color="method", hover_data=["selected_count", "selected_counties"],
        color_discrete_sequence=["#7C3AED", "#A78BFA"],
        title="Simulated county prioritisation trade-offs",
    )
    figure.write_html(output_dir / "tradeoffs.html", include_plotlyjs=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=PROCESSED_PATH)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/prioritisation"))
    parser.add_argument("--budget", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--generations", type=int, default=200)
    args = parser.parse_args()
    digest = hashlib.sha256()
    with args.input.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    summary = compare_methods(
        pd.read_csv(args.input), args.output_dir, args.budget, args.seed, args.generations
    )
    summary["input_sha256"] = digest.hexdigest()
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
