"""Generate interactive exploratory figures for county demographics."""

import argparse
from pathlib import Path

import pandas as pd
import plotly.express as px

from src.data.clean import PROCESSED_PATH
from src.data.missingness import missingness_reports


def write_figures(profiles: pd.DataFrame, output_dir: Path) -> None:
    county_missingness, _ = missingness_reports(profiles)
    totals = profiles[["admin1_name", "population_all_all"]].copy()
    totals = totals.dropna(subset=["population_all_all"]).sort_values("population_all_all")
    if totals.empty:
        raise ValueError("No observed county population totals available for EDA")
    output_dir.mkdir(parents=True, exist_ok=True)
    total_figure = px.bar(
        totals, x="population_all_all", y="admin1_name", orientation="h",
        title="Observed county populations",
        labels={"population_all_all": "Population", "admin1_name": "County"},
        color_discrete_sequence=["#7C3AED"],
    )
    total_figure.update_layout(height=1200)
    total_figure.write_html(output_dir / "county_population.html", include_plotlyjs=True)
    missing_figure = px.bar(
        county_missingness, x="admin1_name", y="missing_count",
        title="Missing demographic values by county",
        labels={"admin1_name": "County", "missing_count": "Missing values"},
        color_discrete_sequence=["#7C3AED"],
    )
    missing_figure.write_html(output_dir / "county_missingness.html", include_plotlyjs=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=PROCESSED_PATH)
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()
    write_figures(pd.read_csv(args.input), args.output_dir)
    print(f"Saved interactive EDA figures to {args.output_dir}")


if __name__ == "__main__":
    main()