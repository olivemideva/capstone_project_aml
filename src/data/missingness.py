"""Summarise missing population values by county and demographic column."""

import argparse
from pathlib import Path

import pandas as pd

from src.data.clean import POPULATION_COLUMNS, PROCESSED_PATH


def missingness_reports(profiles: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"admin1_code", "admin1_name", *POPULATION_COLUMNS}
    missing = required - set(profiles.columns)
    if missing:
        raise ValueError(f"Missing profile columns: {sorted(missing)}")
    if profiles.empty:
        raise ValueError("Cannot report missingness for an empty dataset")
    mask = profiles[POPULATION_COLUMNS].isna()
    by_county = profiles[["admin1_code", "admin1_name"]].copy()
    by_county["missing_count"] = mask.sum(axis=1)
    by_county["missing_fraction"] = mask.mean(axis=1)
    by_column = pd.DataFrame({
        "column": POPULATION_COLUMNS,
        "missing_count": mask.sum(axis=0).to_numpy(),
        "missing_fraction": mask.mean(axis=0).to_numpy(),
    })
    return by_county, by_column


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=PROCESSED_PATH)
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()
    county, column = missingness_reports(pd.read_csv(args.input))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    county.to_csv(args.output_dir / "missingness_by_county.csv", index=False)
    column.to_csv(args.output_dir / "missingness_by_column.csv", index=False)
    print(f"Saved missingness reports to {args.output_dir}")
    print(f"Missing population cells: {int(county['missing_count'].sum())}")


if __name__ == "__main__":
    main()