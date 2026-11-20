from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from app import api_client
from app.metrics_parser import parse_snapshot
from app.pages.indicators import _extract_values
from app.population_csv import read_population_csv, reconstruct_rows
from src.features.engineer import ATOMIC_COLUMNS


def test_streamlit_entry_point_renders_without_exceptions():
    from streamlit.testing.v1 import AppTest

    script = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"
    app = AppTest.from_file(script).run(timeout=30)
    assert not app.exception


def test_streamlit_switches_between_light_and_dark_modes():
    from streamlit.testing.v1 import AppTest

    script = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"
    app = AppTest.from_file(script).run(timeout=30)
    assert app.session_state["theme_mode"] == "light"
    app.button(key="theme_button").click().run(timeout=30)
    assert not app.exception
    assert app.session_state["theme_mode"] == "dark"
    assert any("--ui-background: #130C22" in item.value for item in app.markdown)
    app.button(key="theme_button").click().run(timeout=30)
    assert not app.exception
    assert app.session_state["theme_mode"] == "light"
    assert any("--ui-background: #FAF9FF" in item.value for item in app.markdown)


def test_system_theme_changes_until_manual_override(monkeypatch):
    from streamlit.testing.v1 import AppTest

    from app import theme

    device = {"mode": "dark"}
    monkeypatch.setattr(theme, "device_theme", lambda **kwargs: device["mode"])
    script = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"
    app = AppTest.from_file(script).run(timeout=30)
    assert app.session_state["theme_mode"] == "dark"
    device["mode"] = "light"
    app.run(timeout=30)
    assert app.session_state["theme_mode"] == "light"
    app.button(key="theme_button").click().run(timeout=30)
    assert app.session_state["theme_mode"] == "dark"
    device["mode"] = "light"
    app.run(timeout=30)
    assert app.session_state["theme_mode"] == "dark"


def test_palette_text_contrast():
    from app.theme import PALETTES

    def luminance(color):
        rgb = [int(color[pos:pos + 2], 16) / 255 for pos in (1, 3, 5)]
        linear = [v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4 for v in rgb]
        return sum(v * weight for v, weight in zip(linear, (0.2126, 0.7152, 0.0722)))

    for palette in PALETTES.values():
        for foreground, background in (
            ("text", "background"), ("text", "surface"), ("text", "surface_alt"),
            ("muted", "surface"), ("primary", "surface"),
            ("button_text", "button"), ("button_text", "button_hover"),
        ):
            a, b = luminance(palette[foreground]), luminance(palette[background])
            assert (max(a, b) + 0.05) / (min(a, b) + 0.05) >= 4.5


def test_csv_reconstruction_preserves_known_values_and_extra_columns(monkeypatch):
    calls = []
    def reconstruct(payload):
        calls.append(payload)
        return {
            "population": {
                column: 250.0 if value is None else value
                for column, value in payload["population"].items()
            },
        }

    monkeypatch.setattr(api_client, "reconstruct", reconstruct)
    data = {column: [1000, 0] for column in ATOMIC_COLUMNS}
    data[ATOMIC_COLUMNS[0]] = ["", 0]
    data["admin1_code"] = ["001", "002"]
    frame = read_population_csv(pd.DataFrame(data).to_csv(index=False).encode())
    completed, changes = reconstruct_rows(frame, "random", 20)
    assert len(calls) == 1
    assert calls[0]["pattern"] == "random"
    assert calls[0]["rate"] == 20
    assert calls[0]["population"][ATOMIC_COLUMNS[0]] is None
    assert completed.loc[0, ATOMIC_COLUMNS[0]] == 250
    assert completed.loc[0, ATOMIC_COLUMNS[1]] == 1000
    assert completed.loc[1, ATOMIC_COLUMNS[0]] == 0
    assert completed["admin1_code"].tolist() == ["001", "002"]
    assert changes == [(0, ATOMIC_COLUMNS[0], 250.0)]


@pytest.mark.parametrize("value", ["not a number", "-1", "inf", "1000000000001"])
def test_csv_rejects_invalid_values(value):
    data = {column: [1] for column in ATOMIC_COLUMNS}
    data[ATOMIC_COLUMNS[0]] = [value]
    with pytest.raises(ValueError, match="Data row 1"):
        read_population_csv(pd.DataFrame(data).to_csv(index=False).encode())


def test_csv_rejects_missing_duplicate_and_short_columns():
    with pytest.raises(ValueError, match="missing 42"):
        read_population_csv(b"county\nExample\n")
    header = ",".join(ATOMIC_COLUMNS)
    with pytest.raises(ValueError, match="duplicate"):
        read_population_csv(f"{header},{ATOMIC_COLUMNS[0]}\n".encode())
    with pytest.raises(ValueError, match="wrong number"):
        read_population_csv(f"{header}\n1,2\n".encode())


def test_csv_api_failure_identifies_row(monkeypatch):
    data = {column: [1] for column in ATOMIC_COLUMNS}
    data[ATOMIC_COLUMNS[0]] = [""]
    frame = read_population_csv(pd.DataFrame(data).to_csv(index=False).encode())
    def fail(_payload):
        raise RuntimeError("service unavailable")
    monkeypatch.setattr(api_client, "reconstruct", fail)
    with pytest.raises(RuntimeError, match="data row 1.*service unavailable"):
        reconstruct_rows(frame, "random", 20)


def test_csv_row_and_size_limits():
    header = ",".join(ATOMIC_COLUMNS)
    row = ",".join(["1"] * len(ATOMIC_COLUMNS))
    assert len(read_population_csv((header + "\n" + (row + "\n") * 100).encode())) == 100
    with pytest.raises(ValueError, match="at most 100"):
        read_population_csv((header + "\n" + (row + "\n") * 101).encode())
    with pytest.raises(ValueError, match="5 MB"):
        read_population_csv(b"x" * (5 * 1024 * 1024 + 1))


def test_plans_show_readable_results_without_changing_request(monkeypatch):
    from streamlit.testing.v1 import AppTest

    calls = []

    def prioritise(payload):
        calls.append(payload)
        return {"solutions": [{
            "method": "weighted_sum",
            "selected_counties": [{"admin1_code": "KE001", "admin1_name": "Example"}],
            "selected_count": 1,
            "child_coverage": 0.1,
            "elderly_coverage": 0.2,
            "dependency_burden_coverage": 0.3,
        }]}

    monkeypatch.setattr(api_client, "prioritise", prioritise)
    script = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"
    app = AppTest.from_file(script).run(timeout=30)
    app.radio[0].set_value("Compare plans").run(timeout=30)
    next(button for button in app.button if button.label == "Compare plans").click().run(timeout=30)
    assert not app.exception
    assert calls[0] == {
        "budget": 10, "weights": (1.0, 1.0, 1.0), "generations": 50, "seed": 42,
    }
    result = app.dataframe[0].value
    assert result.loc[0, "Plan type"] == "Priority-weighted selection"
    assert result.loc[0, "Share of children (%)"] == 10.0
    assert result.loc[0, "Selected counties"] == "Example"


def test_county_comparison_converts_shares_for_display(monkeypatch):
    from streamlit.testing.v1 import AppTest

    from app.pages.indicators import fetch_indicator_table

    monkeypatch.setattr(api_client, "counties", lambda: [{"admin1_code": "KE001"}])
    monkeypatch.setattr(api_client, "indicators", lambda code: {
        "admin1_code": code, "admin1_name": "Example", "children": 40,
        "working_age": 50, "elderly": 10, "total": 100, "child_share": 0.4,
        "elderly_share": 0.1, "dependency_ratio": 1.0, "source": "2019",
    })
    fetch_indicator_table.clear()
    try:
        script = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"
        app = AppTest.from_file(script).run(timeout=30)
        app.radio[0].set_value("Explore counties").run(timeout=30)
        assert not app.exception
        assert app.dataframe[0].value.loc[0, "Share of population aged 0–14 (%)"] == 40.0
    finally:
        fetch_indicator_table.clear()


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
