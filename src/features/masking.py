"""Reproducibly mask observed age-sex cells for reconstruction experiments."""

import numpy as np
import pandas as pd

from src.data.clean import population_column
from src.features.engineer import AGES, ATOMIC_COLUMNS, atomic_population

RATES = (0.1, 0.2, 0.4)
PATTERNS = ("random", "age-block", "county-block")


def simulate_mask(
    profiles: pd.DataFrame, pattern: str, rate: float, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return masked inputs, a boolean mask, and held-out values in matching frames."""
    if pattern not in PATTERNS:
        raise ValueError(f"Unknown masking pattern: {pattern}")
    if rate not in RATES:
        raise ValueError(f"Masking rate must be one of {RATES}")
    if not profiles.index.is_unique:
        raise ValueError("Profile index must be unique")
    values = atomic_population(profiles)
    observed = values.notna().to_numpy()
    rng = np.random.default_rng(seed)
    mask = np.zeros(observed.shape, dtype=bool)

    if pattern == "random":
        candidates = np.flatnonzero(observed)
        if not len(candidates):
            raise ValueError("No observed population cells are available to mask")
        count = round(rate * len(candidates))
        if count == 0:
            raise ValueError("Requested rate rounds to zero masked cells")
        chosen = rng.choice(candidates, size=count, replace=False)
        mask.flat[chosen] = True
    elif pattern == "age-block":
        pairs = [
            [ATOMIC_COLUMNS.index(population_column(sex, age)) for sex in ("f", "m")]
            for age in AGES
        ]
        candidates = [
            (row, pair) for row in range(len(values)) for pair in pairs
            if observed[row, pair].any()
        ]
        if not candidates:
            raise ValueError("No observed age blocks are available to mask")
        count = round(rate * len(candidates))
        if count == 0:
            raise ValueError("Requested rate rounds to zero masked age blocks")
        for choice in rng.choice(len(candidates), size=count, replace=False):
            row, pair = candidates[choice]
            mask[row, pair] = observed[row, pair]
    else:
        candidates = np.flatnonzero(observed.any(axis=1))
        if not len(candidates):
            raise ValueError("No observed counties are available to mask")
        count = round(rate * len(candidates))
        if count == 0:
            raise ValueError("Requested rate rounds to zero masked counties")
        for row in rng.choice(candidates, size=count, replace=False):
            mask[row] = observed[row]

    mask_frame = pd.DataFrame(mask, index=values.index, columns=values.columns)
    return values.mask(mask_frame), mask_frame, values.where(mask_frame)
