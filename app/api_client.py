"""HTTP client with explicit API URL configuration and timeout handling."""

import os
from functools import lru_cache
from urllib.parse import urlsplit

import requests

API_READ_TIMEOUT = 90.0


def _check_response(response, path: str) -> None:
    challenge = response.headers.get("cf-mitigated", "").lower() == "challenge"
    if challenge:
        raise RuntimeError(
            f"API request for {path} was blocked by the hosting provider's security check "
            f"(HTTP {response.status_code}). Please try again later; if this persists, "
            "contact the service administrator."
        )
    if response.status_code == 429:
        raise RuntimeError(
            f"API request for {path} was rate limited (HTTP 429). "
            "Please wait before trying again; contact the service administrator if it persists."
        )
    content_type = response.headers.get("content-type", "").lower()
    if "text/html" in content_type or response.text.lstrip().lower().startswith(
        ("<!doctype html", "<html")
    ):
        raise RuntimeError(
            f"API request for {path} returned an unexpected HTML page "
            f"(HTTP {response.status_code}). Please try again later or contact "
            "the service administrator."
        )


@lru_cache(maxsize=1)
def base_url() -> str:
    value = os.environ.get("AML_API_URL", "http://127.0.0.1:8000").strip().rstrip("/")
    if "://" not in value:
        scheme = "https" if value.endswith(".onrender.com") else "http"
        value = f"{scheme}://{value}"
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("AML_API_URL must be an HTTP(S) URL or a hostname")
    return value


def request(method: str, path: str, payload=None, timeout: float = 30.0):
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    try:
        response = requests.request(method, f"{base_url()}{path}", json=payload, timeout=timeout)
    except requests.RequestException as exc:
        raise RuntimeError(f"API request failed for {path}: {exc}") from exc
    _check_response(response, path)
    try:
        body = response.json()
    except ValueError:
        raise RuntimeError(
            f"API request for {path} returned an invalid JSON response "
            f"(HTTP {response.status_code})."
        ) from None
    if response.status_code >= 400:
        detail = body["detail"] if isinstance(body, dict) and "detail" in body else body
        raise RuntimeError(f"{response.status_code} {path}: {detail}")
    return body


def health():
    return request("GET", "/health", timeout=API_READ_TIMEOUT)


def counties():
    return request("GET", "/counties", timeout=API_READ_TIMEOUT)


def indicators(code: str):
    return request("GET", f"/indicators/{code}", timeout=API_READ_TIMEOUT)


def all_indicators():
    return request("GET", "/indicators", timeout=API_READ_TIMEOUT)


def reconstruct(payload: dict):
    return request("POST", "/reconstruct", payload=payload, timeout=API_READ_TIMEOUT)


def prioritise(payload: dict):
    return request("POST", "/prioritise", payload=payload, timeout=120)


def metrics_raw():
    try:
        response = requests.get(f"{base_url()}/metrics", timeout=API_READ_TIMEOUT)
    except requests.RequestException as exc:
        raise RuntimeError(f"API metrics request failed: {exc}") from exc
    _check_response(response, "/metrics")
    if response.status_code >= 400:
        raise RuntimeError(f"{response.status_code} /metrics: {response.text.strip()}")
    return response.text
