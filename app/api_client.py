"""HTTP client with explicit API URL configuration and timeout handling."""

import os
from functools import lru_cache
from urllib.parse import urlsplit

import requests


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
    try:
        body = response.json()
    except ValueError:
        body = {"detail": response.text.strip() or "No response body"}
    if response.status_code >= 400:
        detail = body["detail"] if isinstance(body, dict) and "detail" in body else body
        raise RuntimeError(f"{response.status_code} {path}: {detail}")
    return body


def health():
    return request("GET", "/health", timeout=10)


def counties():
    return request("GET", "/counties", timeout=10)


def indicators(code: str):
    return request("GET", f"/indicators/{code}", timeout=10)


def reconstruct(payload: dict):
    return request("POST", "/reconstruct", payload=payload, timeout=30)


def prioritise(payload: dict):
    return request("POST", "/prioritise", payload=payload, timeout=120)


def metrics_raw():
    try:
        response = requests.get(f"{base_url()}/metrics", timeout=10)
    except requests.RequestException as exc:
        raise RuntimeError(f"API metrics request failed: {exc}") from exc
    if response.status_code >= 400:
        raise RuntimeError(f"{response.status_code} /metrics: {response.text.strip()}")
    return response.text
