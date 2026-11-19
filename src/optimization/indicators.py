"""Derive county demographic indicators from complete observed age-sex profiles."""

import numpy as np
import pandas as pd

from src.features.engineer import atomic_population


def county_indicators(profiles: pd.DataFrame) -> pd.DataFrame:
    required = {"admin1_code", "admin1_name"}
    if not required.issubset(profiles.columns):
        raise ValueError(f"Missing county identifiers: {sorted(required - set(profiles.columns))}")
    if profiles.empty or profiles["admin1_code"].isna().any():
        raise ValueError("Profiles must include nonempty county identifiers")
    codes = profiles["admin1_code"].astype("string").str.strip()
    names = profiles["admin1_name"].astype("string").str.strip()
    if codes.eq("").any() or codes.duplicated().any() or names.isna().any() or names.eq("").any():
        raise ValueError("County codes must be unique and county names must be nonempty")
    values = atomic_population(profiles)
    if values.isna().any().any():
        raise ValueError("Indicators require complete observed age-sex profiles")
    children = values.iloc[:, :6].sum(axis=1)
    working = values.iloc[:, 6:26].sum(axis=1)
    elderly = values.iloc[:, 26:].sum(axis=1)
    if (working <= 0).any():
        raise ValueError("Each county must have a positive working-age population")
    total = children + working + elderly
    result = pd.DataFrame({
        "admin1_code": codes,
        "admin1_name": names,
        "children": children,
        "working_age": working,
        "elderly": elderly,
        "total": total,
        "child_share": children / total,
        "elderly_share": elderly / total,
        "dependency_ratio": (children + elderly) / working,
    })
    if not np.isfinite(result.select_dtypes(include="number").to_numpy()).all():
        raise ValueError("Computed indicators must be finite")
    return result.reset_index(drop=True)
