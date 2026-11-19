import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import GroupKFold

from src.data.clean import POPULATION_COLUMNS
from src.features.engineer import ATOMIC_COLUMNS
from src.features.masking import simulate_mask
from src.models import baselines
from src.models.baselines import examples, fit_baseline


def test_examples_never_include_held_out_values():
    frame = pd.DataFrame(
        np.arange(10 * len(ATOMIC_COLUMNS), dtype=float).reshape(10, -1) + 1,
        columns=ATOMIC_COLUMNS,
    )
    inputs, mask, targets = simulate_mask(frame, "random", 0.2, seed=7)
    groups = pd.Series([f"county{i}" for i in range(len(frame))])
    X, y, repeated_groups, row, col = examples(inputs, mask, targets, groups)
    assert (X.to_numpy()[np.arange(len(col)), col] == -1).all()
    np.testing.assert_array_equal(y, frame.to_numpy()[row, col])
    for train, validation in GroupKFold(n_splits=3).split(X, y, groups=repeated_groups):
        assert set(repeated_groups[train]).isdisjoint(repeated_groups[validation])


@pytest.mark.parametrize("name", ["random_forest", "xgboost"])
def test_fit_baseline_with_county_groups(name):
    frame = pd.DataFrame(
        np.arange(10 * len(ATOMIC_COLUMNS), dtype=float).reshape(10, -1) + 1,
        columns=ATOMIC_COLUMNS,
    )
    inputs, mask, targets = simulate_mask(frame, "random", 0.4, seed=7)
    X, y, groups, _, _ = examples(inputs, mask, targets, pd.Series(range(10)))
    trained = fit_baseline(X, y, groups, name, 42)
    assert np.isfinite(trained.predict(X)).all()
    assert trained.best_params_


def test_training_pipeline_with_integer_source(tmp_path, monkeypatch):
    rows = []
    for i in range(20):
        row = dict.fromkeys(POPULATION_COLUMNS, 1)
        row.update({"admin1_code": f"KE{i + 1:03}", "admin1_name": f"County {i}"})
        rows.append(row)
    monkeypatch.setattr(baselines, "PATTERNS", ("random",))
    monkeypatch.setattr(baselines, "RATES", (0.1,))
    result = baselines.run_baselines(
        pd.DataFrame(rows), tmp_path / "models", tmp_path / "reports"
    )
    assert len(result) == 2
    assert set(result["split"]) == {"validation"}
    assert len(list((tmp_path / "models").glob("*.joblib"))) == 2
    assert len(list((tmp_path / "reports").glob("*_shap.png"))) == 2
