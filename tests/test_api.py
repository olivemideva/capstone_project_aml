import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api.export_indicators import export
from src.api.main import create_app
from src.api.service import FEATURE_COLUMNS
from src.features.engineer import ATOMIC_COLUMNS

FIXTURE = Path("tests/fixtures/api_indicators.csv")


class FixtureModel:
    def predict(self, X):
        assert list(X.columns) == FEATURE_COLUMNS
        return np.full(len(X), 12.5)


@pytest.fixture
def client(monkeypatch, tmp_path):
    root = tmp_path / "models"
    root.mkdir()
    files = [
        f"{pattern}_{rate}_random_forest.joblib"
        for pattern in ("random", "age-block", "county-block")
        for rate in (10, 20, 40)
    ]
    (root / "selection.json").write_text(
        json.dumps({"selected_family": "random_forest", "artifact_files": files})
    )

    def fake_load(path):
        pattern, rate, _ = path.name.split("_", 2)
        return {
            "model": FixtureModel(),
            "feature_columns": FEATURE_COLUMNS,
            "pattern": pattern,
            "rate": int(rate) / 100,
        }

    monkeypatch.setattr(joblib, "load", fake_load)
    monkeypatch.setenv("AML_CORS_ORIGINS", "http://localhost:8501")
    with TestClient(create_app(root, FIXTURE, expected_counties=None)) as test_client:
        yield test_client


def payload():
    population = dict.fromkeys(ATOMIC_COLUMNS, 4)
    population[ATOMIC_COLUMNS[0]] = None
    population[ATOMIC_COLUMNS[5]] = None
    return {"population": population, "pattern": "random", "rate": 20}


def test_health_counties_and_indicators(client):
    assert client.get("/health").json() == {"status": "ok", "model_family": "random_forest"}
    assert len(client.get("/counties").json()) == 4
    data = client.get("/indicators/ke001").json()
    assert data["children"] == 100 and data["source"].startswith("observed")
    assert client.get("/indicators/missing").status_code == 404


def test_reconstruction_validation_and_metrics(client):
    request = payload()
    response = client.post("/reconstruct", json=request)
    assert response.status_code == 200
    result = response.json()
    assert set(result["predicted_fields"]) == {ATOMIC_COLUMNS[0], ATOMIC_COLUMNS[5]}
    assert result["population"][ATOMIC_COLUMNS[0]] == 12.5
    assert result["population"][ATOMIC_COLUMNS[1]] == 4
    request["population"][ATOMIC_COLUMNS[1]] = -1
    assert client.post("/reconstruct", json=request).status_code == 422
    request["population"].pop(ATOMIC_COLUMNS[1])
    assert client.post("/reconstruct", json=request).status_code == 422
    metrics = client.get("/metrics").text
    assert "api_predictions_total 1.0" in metrics
    assert 'route="/reconstruct",status="422"' in metrics
    assert "api_request_duration_seconds" in metrics


def test_prioritisation_budget_and_methods(client):
    result = client.post("/prioritise", json={"budget": 2, "generations": 2, "weights": [1, 1, 1]})
    assert result.status_code == 200, result.text
    data = result.json()
    assert {s["method"] for s in data["solutions"]} == {"weighted_sum", "nsga3"}
    assert all(s["selected_count"] <= 2 for s in data["solutions"])
    assert client.post("/prioritise", json={"budget": 5}).status_code == 422
    assert client.post("/prioritise", json={"budget": 2, "weights": [0, 0, 0]}).status_code == 422


def test_cors_is_configurable(client):
    response = client.options(
        "/reconstruct",
        headers={"Origin": "http://localhost:8501", "Access-Control-Request-Method": "POST"},
    )
    assert response.headers["access-control-allow-origin"] == "http://localhost:8501"
    response = client.options(
        "/reconstruct",
        headers={"Origin": "https://untrusted.invalid", "Access-Control-Request-Method": "POST"},
    )
    assert "access-control-allow-origin" not in response.headers


def test_startup_missing_artifacts_fails(tmp_path):
    with pytest.raises(FileNotFoundError):
        with TestClient(create_app(tmp_path, FIXTURE)):
            pass


def test_fixture_indicator_shape():
    assert len(pd.read_csv(FIXTURE)) == 4


def test_export_rejects_input_checksum_mismatch(tmp_path):
    source = tmp_path / "source.csv"
    source.write_text("wrong input", encoding="utf-8")
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({"input_sha256": "wrong"}), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match"):
        export(source, selection, tmp_path / "indicators.csv")


def test_real_artifact_inference():
    app = create_app()
    with TestClient(app) as real:
        assert len(real.get("/counties").json()) == 47
        response = real.post("/reconstruct", json=payload())
        assert response.status_code == 200, response.text
        predictions = response.json()
        assert predictions["model_family"] == "random_forest"
        assert all(predictions["population"][key] >= 0 for key in predictions["predicted_fields"])
