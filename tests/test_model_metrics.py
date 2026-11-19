import numpy as np
import pandas as pd
import pytest

from src.data.clean import POPULATION_COLUMNS
from src.models.metrics import consistency_metrics, reconstruction_metrics


def test_reconstruction_metrics_and_invalid_inputs():
    result = reconstruction_metrics(np.array([1, 3]), np.array([2, 1]))
    assert result["mae"] == 1.5
    assert result["rmse"] == pytest.approx(np.sqrt(2.5))
    assert result["r2"] == -1.5
    with pytest.raises(ValueError, match="aligned"):
        reconstruction_metrics(np.array([]), np.array([]))
    with pytest.raises(ValueError, match="finite"):
        reconstruction_metrics(np.array([1]), np.array([np.nan]))


def test_consistency_against_reported_totals_including_negative():
    row = {column: 0.0 for column in POPULATION_COLUMNS}
    row.update({
        "population_f_0_4": 4.0, "population_m_0_4": 6.0,
        "population_all_0_4": 10.0, "population_f_all": 4.0,
        "population_m_all": 6.0, "population_all_all": 10.0,
    })
    profiles = pd.DataFrame([row])
    assert consistency_metrics(profiles)["gender_inconsistent"] == 0
    assert consistency_metrics(profiles)["age_inconsistent"] == 0
    profiles.loc[0, "population_f_0_4"] = -1
    metrics = consistency_metrics(profiles)
    assert metrics["negative_cells"] == 1
    assert metrics["gender_inconsistent"] == 1
    assert metrics["age_inconsistent"] == 1
    assert metrics["gender_absolute_difference"] == 5
    profiles.loc[0, "population_f_0_4"] = np.nan
    assert consistency_metrics(profiles)["gender_checks"] == 21
