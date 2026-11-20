"""Reusable exploratory summaries and charts for cleaned county profiles."""

import argparse
from pathlib import Path

import pandas as pd
import plotly.express as px

from src.data.clean import PROCESSED_PATH
from src.data.missingness import missingness_reports
from src.data.validate import validate_profiles
from src.features.engineer import ATOMIC_COLUMNS
from src.optimization.indicators import county_indicators


def exploratory_tables(profiles: pd.DataFrame) -> dict[str, pd.DataFrame]:
    county_missing, column_missing = missingness_reports(profiles)
    checks = validate_profiles(profiles)
    atomic = profiles[ATOMIC_COLUMNS]
    county_totals = profiles[["admin1_code", "admin1_name", "population_all_all"]].copy()
    long = profiles.melt(
        id_vars=["admin1_code", "admin1_name"], value_vars=ATOMIC_COLUMNS,
        var_name="age_sex", value_name="population",
    )
    demographics = long.groupby("age_sex", sort=False)["population"].agg(
        count="count", mean="mean", median="median", minimum="min", maximum="max"
    ).reset_index()
    overview = pd.DataFrame([{
        "counties": len(profiles),
        "profile_columns": len(profiles.columns),
        "atomic_cells": atomic.size,
        "missing_atomic_cells": int(atomic.isna().sum().sum()),
        "duplicate_county_codes": int(profiles["admin1_code"].duplicated().sum()),
    }])
    return {
        "overview": overview,
        "county_totals": county_totals.sort_values("population_all_all", ascending=False),
        "atomic_distribution": atomic.describe().T.reset_index(names="age_sex"),
        "demographics": demographics,
        "missing_by_county": county_missing,
        "missing_by_column": column_missing,
        "consistency_checks": checks,
        "consistency_summary": checks.groupby(["check", "status"]).size().reset_index(name="count"),
        "indicators": county_indicators(profiles),
    }


def exploratory_figures(tables: dict[str, pd.DataFrame]) -> dict:
    colors = ["#7C3AED", "#A78BFA"]
    return {
        "county_population": px.bar(
            tables["county_totals"], x="admin1_name", y="population_all_all",
            title="Reported population by county", color_discrete_sequence=colors,
        ),
        "atomic_missingness": px.bar(
            tables["missing_by_column"], x="column", y="missing_fraction",
            title="Missing fraction by population field", color_discrete_sequence=colors,
        ),
        "age_sex": px.bar(
            tables["demographics"], x="age_sex", y="median",
            title="Median observed age-sex population across counties",
            color_discrete_sequence=colors,
        ),
        "county_distribution": px.histogram(
            tables["county_totals"], x="population_all_all", nbins=15,
            title="Distribution of reported county populations", color_discrete_sequence=colors,
        ),
        "dependency": px.bar(
            tables["indicators"], x="admin1_name", y="dependency_ratio",
            title="Observed county dependency ratios", color_discrete_sequence=colors,
        ),
    }


def write_exploration(profiles: pd.DataFrame, output_dir: Path) -> dict[str, pd.DataFrame]:
    tables = exploratory_tables(profiles)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(output_dir / f"{name}.csv", index=False)
    for name, figure in exploratory_figures(tables).items():
        figure.write_html(output_dir / f"{name}.html", include_plotlyjs="cdn")
    return tables


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=PROCESSED_PATH)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/exploration"))
    args = parser.parse_args()
    write_exploration(pd.read_csv(args.input), args.output_dir)
    print(f"Exploratory reports saved to {args.output_dir}")


if __name__ == "__main__":
    main()
