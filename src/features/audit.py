"""Check split isolation and that inputs and ratios do not expose held-out cells."""

import pandas as pd
from pandas.api.types import is_bool_dtype
from pandas.testing import assert_frame_equal

from src.features.engineer import ATOMIC_COLUMNS, atomic_population, engineer_features


def audit_splits(splits: dict[str, pd.DataFrame]) -> None:
    if set(splits) != {"train", "validation", "test"}:
        raise ValueError("Expected train, validation and test splits")
    seen: set[str] = set()
    for name, frame in splits.items():
        if frame.empty or frame["admin1_code"].isna().any():
            raise ValueError(f"{name} must contain county codes")
        codes = set(frame["admin1_code"].astype(str).str.strip())
        if "" in codes or seen.intersection(codes):
            raise ValueError("County codes must be nonempty and disjoint between splits")
        seen.update(codes)


def audit_masked_inputs(
    original: pd.DataFrame,
    inputs: pd.DataFrame,
    mask: pd.DataFrame,
    targets: pd.DataFrame,
    features: pd.DataFrame,
) -> None:
    values = atomic_population(original)
    if list(inputs.columns) != ATOMIC_COLUMNS:
        raise ValueError("Inputs must contain only atomic age-sex columns, not aggregate totals")
    if (
        not inputs.index.equals(values.index)
        or not mask.index.equals(values.index)
        or not targets.index.equals(values.index)
        or not mask.columns.equals(values.columns)
        or not targets.columns.equals(values.columns)
        or mask.isna().any().any()
        or not all(is_bool_dtype(dtype) for dtype in mask.dtypes)
    ):
        raise ValueError("Mask, inputs and targets must align and mask must be boolean")
    if (mask & values.isna()).any().any():
        raise ValueError("Mask includes an originally missing value")
    try:
        assert_frame_equal(inputs, values.mask(mask), check_dtype=False)
        assert_frame_equal(targets, values.where(mask), check_dtype=False)
        assert_frame_equal(features, engineer_features(inputs), check_dtype=False)
    except AssertionError as exc:
        raise ValueError(
            "Leakage audit failed: inputs, targets or features are inconsistent"
        ) from exc
