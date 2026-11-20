from types import SimpleNamespace

import pytest

from app import api_client
from app.metrics_parser import parse_snapshot
from app.pages.indicators import _extract_values


def test_base_url_adds_http_for_render_private_hostport(monkeypatch):
    monkeypatch.setenv("AML_API_URL", "aml-capstone-api:10000")
    api_client.base_url.cache_clear()
    assert api_client.base_url() == "http://aml-capstone-api:10000"


def test_base_url_adds_https_for_render_public_hostname(monkeypatch):
    monkeypatch.setenv("AML_API_URL", "aml-capstone-api.onrender.com")
    api_client.base_url.cache_clear()
    assert api_client.base_url() == "https://aml-capstone-api.onrender.com"


def test_base_url_rejects_non_http_scheme(monkeypatch):
    monkeypatch.setenv("AML_API_URL", "ftp://api.example.com")
    api_client.base_url.cache_clear()
    with pytest.raises(ValueError, match="HTTP"):
        api_client.base_url()


def test_request_raises_for_http_error(monkeypatch):
    api_client.base_url.cache_clear()

    def fake_request(*args, **kwargs):
        return SimpleNamespace(status_code=422, json=lambda: {"detail": "bad payload"})

    monkeypatch.setattr(api_client.requests, "request", fake_request)
    with pytest.raises(RuntimeError, match=r"422 /reconstruct: bad payload"):
        api_client.request("POST", "/reconstruct", payload={"invalid": True})


def test_metrics_raw_raises_for_network_error(monkeypatch):
    api_client.base_url.cache_clear()

    def fake_get(*args, **kwargs):
        raise api_client.requests.RequestException("connection reset")

    monkeypatch.setattr(api_client.requests, "get", fake_get)
    with pytest.raises(RuntimeError, match="metrics request failed"):
        api_client.metrics_raw()


def test_parse_snapshot_extracts_expected_totals():
    raw = """
    # HELP api_requests_total total requests
    api_requests_total{method="GET",route="/health",status="200"} 5
    api_requests_total{method="POST",route="/reconstruct",status="200"} 2
    api_errors_total{method="POST",route="/reconstruct"} 1
    api_predictions_total 2
    api_request_duration_seconds_sum{method="GET",route="/health"} 0.5
    api_request_duration_seconds_count{method="GET",route="/health"} 5
    """
    snapshot = parse_snapshot(raw)
    assert snapshot.request_total == 7
    assert snapshot.error_total == 1
    assert snapshot.prediction_total == 2
    assert snapshot.average_latency_seconds == 0.1


def test_extract_values_reads_dot_path():
    features = [
        {"properties": {"COUNTY_COD": "KE001"}},
        {"properties": {"COUNTY_COD": "KE002"}},
    ]
    assert _extract_values(features, "properties.COUNTY_COD") == {"KE001", "KE002"}
    with pytest.raises(ValueError, match="not found"):
        _extract_values(features, "properties.UNKNOWN")
