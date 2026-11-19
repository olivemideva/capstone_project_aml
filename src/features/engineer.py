"""Derive demographic ratios exclusively from the available age-sex cells."""

import numpy as np
import pandas as pd

from src.data.clean import AGE_RANGES, population_column

AGES = [age for age in AGE_RANGES if age != "all"]
ATOMIC_COLUMNS = [population_column(sex, age) for age in AGES for sex in ("f", "m")]


def atomic_population(profiles: pd.DataFrame) -> pd.DataFrame:
    """Select non-aggregate cells, excluding redundant source totals."""
    missing = set(ATOMIC_COLUMNS) - set(profiles.columns)
    if missing:
        raise ValueError(f"Missing age-sex columns: {sorted(missing)}")
    values = profiles.loc[:, ATOMIC_COLUMNS].apply(pd.to_numeric, errors="raise")
    array = values.to_numpy(dtype=float, na_value=np.nan)
    if np.isinf(array).any() or np.any(array[~np.isnan(array)] < 0):
        raise ValueError("Population values must be finite and non-negative")
    return values.astype(float)


def engineer_features(values: pd.DataFrame) -> pd.DataFrame:
    """Return ratios computed only from observed cells; incomplete groups stay missing."""
    if list(values.columns) != ATOMIC_COLUMNS:
        raise ValueError("Expected the ordered atomic age-sex columns only")
    values = atomic_population(values)
    result = pd.DataFrame(index=values.index)

    def sum_complete(columns: list[str]) -> pd.Series:
        return values[columns].sum(axis=1, min_count=len(columns))

    def ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
        return numerator.div(denominator.where(denominator > 0))

    female = sum_complete([population_column("f", age) for age in AGES])
    male = sum_complete([population_column("m", age) for age in AGES])
    total = sum_complete(ATOMIC_COLUMNS)
    result["sex_ratio_male_per_female"] = ratio(male, female)

    groups = {
        "children": AGES[:3],
        "working_age": AGES[3:13],
        "older": AGES[13:],
    }
    group_totals = {}
    for name, ages in groups.items():
        columns = [population_column(sex, age) for age in ages for sex in ("f", "m")]
        group_totals[name] = sum_complete(columns)
        result[f"{name}_share"] = ratio(group_totals[name], total)
    result["dependency_ratio"] = ratio(
        group_totals["children"] + group_totals["older"], group_totals["working_age"]
    )
    return result
