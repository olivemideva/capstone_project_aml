import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.features.engineer import ATOMIC_COLUMNS
from src.models import diagnostics
from src.models.diagnostics import error_costs
from src.models.evaluation import mask_fingerprint


def test_asymmetric_costs_distinguish_under_and_over_estimates():
    costs = error_costs(np.array([10, 10, 10]), np.array([7, 12, 10]))
    assert costs["bias"] == pytest.approx(-1 / 3)
    assert costs["mean_underestimate"] == pytest.approx(1)
    assert costs["mean_overestimate"] == pytest.approx(2 / 3)
    assert costs["cost_equal"] == pytest.approx(5 / 3)
    assert costs["cost_under_double"] == pytest.approx(8 / 3)
    assert costs["cost_over_double"] == pytest.approx(7 / 3)
    assert costs["underestimate_fraction"] == pytest.approx(1 / 3)


def test_error_costs_reject_invalid_values():
    with pytest.raises(ValueError, match="finite"):
        error_costs(np.array([1]), np.array([np.nan]))
    with pytest.raises(ValueError, match="aligned"):
        error_costs(np.array([1]), np.array([1, 2]))


def test_selected_evaluation_uses_training_only_medians(tmp_path, monkeypatch):
    def frame(codes, values):
        rows = pd.DataFrame(
            np.repeat(np.array(values)[:, None], len(ATOMIC_COLUMNS), axis=1),
            columns=ATOMIC_COLUMNS,
        )
        rows["admin1_code"] = codes
        return rows

    train = frame(["train1", "train2"], [10, 30])
    validation = frame(["val1", "val2"], [100, 200])

    def mask_frame(split, pattern, rate, seed):
        inputs = split[ATOMIC_COLUMNS].copy()
        mask = pd.DataFrame(False, index=split.index, columns=ATOMIC_COLUMNS)
        mask.iloc[:, 0] = True
        targets = inputs.where(mask)
        return inputs.mask(mask), mask, targets

    _, mask, _ = mask_frame(validation, "random", 0.1, 42)
    model_dir = tmp_path / "selected"
    model_dir.mkdir()
    (model_dir / "selection.json").write_text(json.dumps({
        "input_sha256": "hash", "selected_family": "random_forest", "seed": 42,
    }))
    pd.DataFrame([{
        "model": "random_forest", "pattern": "random", "rate": 0.1, "split": "validation",
        "mask_sha256": mask_fingerprint(validation["admin1_code"], mask),
        "mae": 0.0, "cv_mae": 1.0,
    }]).to_csv(model_dir / "comparison.csv", index=False)
    processed = tmp_path / "processed.csv"
    train.to_csv(processed, index=False)
    monkeypatch.setattr(diagnostics, "sha256_file", lambda path: "hash")
    monkeypatch.setattr(diagnostics, "PATTERNS", ("random",))
    monkeypatch.setattr(diagnostics, "RATES", (0.1,))
    monkeypatch.setattr(diagnostics, "simulate_mask", mask_frame)
    monkeypatch.setattr(
        diagnostics, "split_counties",
        lambda profiles, seed: {"train": train, "validation": validation, "test": validation},
    )
    monkeypatch.setattr(
        diagnostics, "examples",
        lambda inputs, mask, targets, groups: (
            pd.DataFrame({"feature": [0, 0]}), np.array([100, 200]), None, None, None
        ),
    )
    monkeypatch.setattr(diagnostics.joblib, "load", lambda path: {
        "feature_columns": ["feature"],
        "model": SimpleNamespace(predict=lambda X: np.array([100, 200])),
    })
    result = diagnostics.evaluate_selected(processed, model_dir).iloc[0]
    assert result["mae"] == 0
    assert result["median_baseline_mae"] == 130
    assert result["cost_equal"] == 0
    test_result = diagnostics.evaluate_selected(processed, model_dir, "test").iloc[0]
    assert test_result["split"] == "test"
    assert test_result["median_baseline_mae"] == 130
