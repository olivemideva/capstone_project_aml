"""Create deterministic, disjoint county-grouped dataset splits."""

import numpy as np
import pandas as pd


def split_counties(profiles: pd.DataFrame, seed: int = 42) -> dict[str, pd.DataFrame]:
    """Split unique counties approximately 70/15/15, keeping all their rows together."""
    if "admin1_code" not in profiles:
        raise ValueError("Missing admin1_code")
    codes = profiles["admin1_code"].astype("string").str.strip()
    if codes.isna().any() or codes.eq("").any():
        raise ValueError("County codes must not be missing")
    counties = sorted(codes.unique().tolist())
    if len(counties) < 3:
        raise ValueError("At least three unique counties are required")
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(counties)
    n_validation = max(1, round(len(counties) * 0.15))
    n_test = max(1, round(len(counties) * 0.15))
    n_train = len(counties) - n_validation - n_test
    if n_train < 1:
        raise ValueError("Not enough counties to populate all splits")
    groups = {
        "train": shuffled[:n_train],
        "validation": shuffled[n_train:n_train + n_validation],
        "test": shuffled[n_train + n_validation:],
    }
    return {name: profiles.loc[codes.isin(group)].copy() for name, group in groups.items()}
