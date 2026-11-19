import pandas as pd
import pytest

from src.features.audit import audit_masked_inputs, audit_splits
from src.features.engineer import ATOMIC_COLUMNS, engineer_features
from src.features.masking import simulate_mask
from src.features.prepare import prepare_experiments
from src.features.split import split_counties


def population():
    result = pd.DataFrame(1.0, index=range(47), columns=ATOMIC_COLUMNS)
    result.insert(0, "admin1_name", [f"County {i}" for i in range(47)])
    result.insert(0, "admin1_code", [f"KE{i + 1:03}" for i in range(47)])
    result["population_all_all"] = 42.0
    return result


def test_county_group_split_and_reproducibility():
    rows = pd.concat([population(), population()], ignore_index=True)
    splits = split_counties(rows)
    audit_splits(splits)
    assert [len(frame) // 2 for frame in splits.values()] == [33, 7, 7]
    again = split_counties(rows)
    for name in splits:
        pd.testing.assert_frame_equal(splits[name], again[name])
    splits["test"] = pd.concat([splits["test"], splits["train"].iloc[[0]]])
    with pytest.raises(ValueError, match="disjoint"):
        audit_splits(splits)


def test_leakage_audit_rejects_targets_and_full_data_features():
    original = population()
    inputs, mask, targets = simulate_mask(original, "random", 0.1)
    features = engineer_features(inputs)
    audit_masked_inputs(original, inputs, mask, targets, features)
    with pytest.raises(ValueError, match="Leakage audit"):
        features_from_full_data = engineer_features(original[ATOMIC_COLUMNS])
        audit_masked_inputs(original, inputs, mask, targets, features_from_full_data)
    leaky = inputs.copy()
    leaky["population_all_all"] = original["population_all_all"]
    with pytest.raises(ValueError, match="aggregate"):
        audit_masked_inputs(original, leaky, mask, targets, features)
    with pytest.raises(ValueError, match="Leakage audit"):
        audit_masked_inputs(original, original[ATOMIC_COLUMNS], mask, targets, features)


def test_prepare_writes_separate_experiments(tmp_path):
    summary = prepare_experiments(population(), tmp_path, seed=42)
    assert len(summary) == 27
    assert (tmp_path / "summary.json").is_file()
    path = tmp_path / "county-block" / "20" / "test"
    inputs = pd.read_csv(path / "inputs.csv")
    targets = pd.read_csv(path / "targets.csv")
    mask = pd.read_csv(path / "mask.csv")
    assert "population_all_all" not in inputs
    assert mask[ATOMIC_COLUMNS].to_numpy().sum() == 42
    assert inputs.loc[mask[ATOMIC_COLUMNS].any(axis=1), ATOMIC_COLUMNS].isna().all().all()
    assert targets[ATOMIC_COLUMNS].notna().equals(mask[ATOMIC_COLUMNS])
