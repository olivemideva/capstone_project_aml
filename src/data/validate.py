"""Check county population sums without assuming missing values are zero."""

import argparse
from pathlib import Path

import pandas as pd

from src.data.clean import (
    AGE_RANGES,
    GENDERS,
    POPULATION_COLUMNS,
    PROCESSED_PATH,
    population_column,
)


def validate_profiles(profiles: pd.DataFrame) -> pd.DataFrame:
    required = {"admin1_code", "admin1_name", *POPULATION_COLUMNS}
    missing = required - set(profiles.columns)
    if missing:
        raise ValueError(f"Missing profile columns: {sorted(missing)}")
    if profiles["admin1_code"].isna().any() or profiles["admin1_code"].duplicated().any():
        raise ValueError("County codes must be present and unique")
    values = profiles[POPULATION_COLUMNS].apply(pd.to_numeric, errors="raise")
    if (values < 0).any().any():
        raise ValueError("Population values must be non-negative")
    checks = []

    def add_check(name: str, total: str, components: list[str]) -> None:
        complete = values[[total, *components]].notna().all(axis=1)
        difference = values[total] - values[components].sum(axis=1, min_count=len(components))
        result = profiles[["admin1_code", "admin1_name"]].copy()
        result["check"] = name
        result["status"] = "incomplete"
        result.loc[complete & difference.eq(0), "status"] = "consistent"
        result.loc[complete & difference.ne(0), "status"] = "inconsistent"
        result["difference"] = difference.where(complete)
        checks.append(result)

    for age in AGE_RANGES:
        add_check(f"gender_{age}", population_column("all", age), [
            population_column("f", age), population_column("m", age)
        ])
    for gender in GENDERS:
        add_check(f"age_{gender}", population_column(gender, "all"), [
            population_column(gender, age) for age in AGE_RANGES if age != "all"
        ])
    return pd.concat(checks, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=PROCESSED_PATH)
    parser.add_argument("--output", type=Path, default=Path("reports/consistency_checks.csv"))
    args = parser.parse_args()
    result = validate_profiles(pd.read_csv(args.input))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Saved {len(result)} checks to {args.output}")
    print(result["status"].value_counts().to_string())
    if result["status"].eq("inconsistent").any():
        raise SystemExit("Inconsistent totals detected; inspect the report.")


if __name__ == "__main__":
    main()