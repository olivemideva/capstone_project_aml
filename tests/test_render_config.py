from pathlib import Path

import yaml

from app import api_client


def test_free_render_ui_uses_public_api_url(monkeypatch):
    blueprint = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "render.yaml").read_text(encoding="utf-8")
    )
    services = {service["name"]: service for service in blueprint["services"]}
    assert services["aml-capstone-api"]["plan"] == "free"
    ui = services["aml-capstone-ui"]
    api_env = next(item for item in ui["envVars"] if item["key"] == "AML_API_URL")
    assert api_env["value"] == "https://aml-capstone-api-98go.onrender.com"
    assert "fromService" not in api_env

    monkeypatch.setenv("AML_API_URL", api_env["value"])
    api_client.base_url.cache_clear()
    try:
        assert api_client.base_url() == api_env["value"]
    finally:
        api_client.base_url.cache_clear()
