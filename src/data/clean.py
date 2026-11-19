"""Clean a local HDX HAPI population extract into Kenya county profiles."""

import argparse
import logging
from pathlib import Path

import pandas as pd

from src.data.load import COUNTY_CODES, RAW_PATH, REQUIRED_COLUMNS, load_kenya

PROCESSED_PATH = Path("data/processed/kenya_county_population.csv")
AGE_RANGES = [f"{age}-{age + 4}" for age in range(0, 100, 5)] + ["100+", "all"]
GENDERS = ["f", "m", "all"]


def population_column(gender: str, age: str) -> str:
    suffix = age.replace("-", "_").replace("+", "_plus")
    return f"population_{gender}_{suffix}"


POPULATION_COLUMNS = [
    population_column(gender, age) for gender in GENDERS for age in AGE_RANGES
]


def clean_population(rows: pd.DataFrame) -> pd.DataFrame:
    """Pivot one reference period without filling or aggregating observations."""
    missing = set(REQUIRED_COLUMNS) - set(rows.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if rows.empty:
        raise ValueError("No Kenya admin-1 population records found")
    rows = rows.copy()
    identifiers = [
        "location_code", "admin_level", "admin1_code", "admin1_name", "gender", "age_range"
    ]
    for column in identifiers:
        rows[column] = rows[column].astype("string").str.strip()
    rows["location_code"] = rows["location_code"].str.upper()
    rows["admin1_code"] = rows["admin1_code"].str.upper()
    rows["admin1_name"] = rows["admin1_name"].str.replace(r"\s+", " ", regex=True)
    rows["gender"] = rows["gender"].str.lower()
    rows["age_range"] = rows["age_range"].str.lower().str.replace(r"\s+", "", regex=True)
    if rows[identifiers].isna().any().any() or rows[identifiers].eq("").any().any():
        raise ValueError("County, location, gender and age identifiers must not be missing")
    if not ((rows["location_code"] == "KEN") & (rows["admin_level"] == "1")).all():
        raise ValueError("Input must contain only Kenya admin-1 records")
    codes = set(rows["admin1_code"])
    if codes != COUNTY_CODES:
        raise ValueError(
            f"Expected 47 county codes; missing={sorted(COUNTY_CODES - codes)}, "
            f"unexpected={sorted(codes - COUNTY_CODES)}"
        )
    if rows.groupby("admin1_code")["admin1_name"].nunique().ne(1).any():
        raise ValueError("Each county code must have exactly one name")
    if rows.groupby("admin1_name")["admin1_code"].nunique().ne(1).any():
        raise ValueError("Each county name must have exactly one code")
    for column, valid in (("gender", GENDERS), ("age_range", AGE_RANGES)):
        unexpected = sorted(set(rows[column]) - set(valid))
        if unexpected:
            raise ValueError(f"Unexpected {column} values: {unexpected}")
    period = rows[["reference_period_start", "reference_period_end"]].astype("string")
    period = period.apply(lambda col: col.str.strip())
    if period.isna().any().any() or period.eq("").any().any() or len(period.drop_duplicates()) != 1:
        raise ValueError("Input must contain exactly one complete reference period")
    if rows.duplicated(["admin1_code", "gender", "age_range"]).any():
        raise ValueError(
            "Duplicate county, gender and age records; resolve source duplicates first"
        )
    rows["population"] = pd.to_numeric(rows["population"], errors="raise")
    observed = rows["population"].dropna()
    if (observed < 0).any() or (observed % 1 != 0).any():
        raise ValueError("Population values must be non-negative whole numbers")
    rows["column"] = [
        population_column(gender, age)
        for gender, age in zip(rows["gender"], rows["age_range"])
    ]
    profiles = rows.pivot(
        index=["admin1_code", "admin1_name"], columns="column", values="population"
    )
    profiles = profiles.reindex(columns=POPULATION_COLUMNS).reset_index()
    profiles.columns.name = None
    return profiles.sort_values("admin1_code").reset_index(drop=True)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=RAW_PATH)
    parser.add_argument("--output", type=Path, default=PROCESSED_PATH)
    args = parser.parse_args()
    profiles = clean_population(load_kenya(args.input))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    profiles.to_csv(args.output, index=False)
    print(f"Saved {len(profiles)} county profiles to {args.output}")


if __name__ == "__main__":
    main()