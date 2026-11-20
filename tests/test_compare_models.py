import json

import pandas as pd
import pytest

from src.features.masking import PATTERNS, RATES
from src.models.compare_models import compare
from src.models.evaluation import mask_fingerprint


def setup_reports(root):
    baseline = root / "baseline"
    auto = root / "auto"
    models = root / "models"
    baseline.mkdir()
    auto.mkdir()
    (models / "autoencoders").mkdir(parents=True)
    meta = {"input_sha256": "abc", "seed": 42, "rows": 47, "split": "validation"}
    for directory in (baseline, auto):
        (directory / "run.json").write_text(json.dumps(meta))
    rows = []
    for pattern in PATTERNS:
        for rate in RATES:
            for model, error in (("random_forest", 3), ("xgboost", 2), ("autoencoder", 1)):
                rows.append({
                    "pattern": pattern, "rate": rate, "model": model, "split": "validation",
                    "mae": error, "validation_targets": 10, "mask_sha256": "same",
                })
            (models / "autoencoders" / f"{pattern}_{int(rate * 100)}_autoencoder.pt").write_bytes(
                b"artifact"
            )
    results = pd.DataFrame(rows)
    results[results.model != "autoencoder"].to_csv(
        baseline / "baseline_validation.csv", index=False
    )
    results[results.model == "autoencoder"].to_csv(auto / "autoencoder_validation.csv", index=False)
    return baseline, auto, models


def test_compare_selects_family_and_preserves_scenarios(tmp_path):
    baseline, auto, models = setup_reports(tmp_path)
    output = tmp_path / "selected"
    result = compare(baseline, auto, models, output)
    selected = json.loads((output / "selection.json").read_text())
    assert len(result) == 27
    assert selected["selected_family"] == "autoencoder"
    assert len(selected["artifact_files"]) == 9
    assert selected["test_evaluated"] is False
    with pytest.raises(FileExistsError):
        compare(baseline, auto, models, output)


def test_comparison_rejects_mismatched_mask(tmp_path):
    baseline, auto, models = setup_reports(tmp_path)
    path = auto / "autoencoder_validation.csv"
    data = pd.read_csv(path)
    data.loc[0, "mask_sha256"] = "different"
    data.to_csv(path, index=False)
    with pytest.raises(ValueError, match="validation masks"):
        compare(baseline, auto, models, tmp_path / "selected")


def test_mask_fingerprint_changes_with_county_and_mask():
    mask = pd.DataFrame({"a": [True, False], "b": [False, True]})
    codes = pd.Series(["a", "b"])
    fingerprint = mask_fingerprint(codes, mask)
    assert fingerprint != mask_fingerprint(codes, ~mask)
    assert fingerprint != mask_fingerprint(pd.Series(["b", "a"]), mask)
