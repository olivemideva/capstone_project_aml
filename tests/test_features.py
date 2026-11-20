import numpy as np
import pandas as pd
import pytest

from src.features.engineer import ATOMIC_COLUMNS, atomic_population, engineer_features
from src.features.masking import PATTERNS, RATES, simulate_mask


def profiles(count=10):
    rows = pd.DataFrame(1.0, index=range(count), columns=ATOMIC_COLUMNS)
    rows.insert(0, "admin1_code", [f"KE{i + 1:03}" for i in range(count)])
    rows["population_all_all"] = 42.0
    return rows


def test_ratios_and_missing_values():
    values = atomic_population(profiles(1))
    features = engineer_features(values)
    assert features.loc[0, "sex_ratio_male_per_female"] == 1
    assert features.loc[0, "children_share"] == pytest.approx(3 / 21)
    assert features.loc[0, "older_share"] == pytest.approx(8 / 21)
    assert features.loc[0, "dependency_ratio"] == pytest.approx(11 / 10)
    values.iloc[0, 0] = np.nan
    assert engineer_features(values).loc[0].isna().all()
    values = values.fillna(0) * 0
    assert engineer_features(values).loc[0].isna().all()


@pytest.mark.parametrize("pattern", PATTERNS)
@pytest.mark.parametrize("rate", RATES)
def test_mask_reproducibility_and_shape(pattern, rate):
    frame = profiles()
    frame.loc[0, ATOMIC_COLUMNS[0]] = np.nan
    before = frame.copy(deep=True)
    inputs, mask, targets = simulate_mask(frame, pattern, rate, seed=7)
    inputs2, mask2, targets2 = simulate_mask(frame, pattern, rate, seed=7)
    pd.testing.assert_frame_equal(mask, mask2)
    pd.testing.assert_frame_equal(inputs, inputs2)
    pd.testing.assert_frame_equal(targets, targets2)
    pd.testing.assert_frame_equal(frame, before)
    assert not mask.loc[0, ATOMIC_COLUMNS[0]]
    assert mask.to_numpy().sum() > 0
    assert targets.notna().equals(mask)
    assert inputs.notna().equals(atomic_population(frame).notna() & ~mask)
    if pattern == "random":
        assert mask.to_numpy().sum() == round(rate * 419)
    elif pattern == "age-block":
        for pos in range(0, len(ATOMIC_COLUMNS), 2):
            both_observed = frame[ATOMIC_COLUMNS[pos:pos + 2]].notna().all(axis=1)
            assert (mask.iloc[both_observed.to_numpy(), pos]
                    == mask.iloc[both_observed.to_numpy(), pos + 1]).all()
    else:
        for row in range(len(frame)):
            assert not mask.iloc[row].any() or mask.iloc[row].equals(
                atomic_population(frame).iloc[row].notna()
            )


def test_invalid_mask_requests():
    with pytest.raises(ValueError, match="pattern"):
        simulate_mask(profiles(), "bad", 0.1)
    with pytest.raises(ValueError, match="rate"):
        simulate_mask(profiles(), "random", 0.3)
    with pytest.raises(ValueError, match="rounds to zero"):
        simulate_mask(profiles(1), "county-block", 0.1)
