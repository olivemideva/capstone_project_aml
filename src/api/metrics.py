"""Per-application Prometheus instrumentation with bounded route labels."""

import time

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Histogram,
    generate_latest,
)
from starlette.responses import Response


class ApiMetrics:
    def __init__(self):
        self.registry = CollectorRegistry()
        self.requests = Counter(
            "api_requests_total",
            "HTTP requests",
            ("method", "route", "status"),
            registry=self.registry,
        )
        self.errors = Counter(
            "api_errors_total",
            "HTTP errors",
            ("method", "route", "status"),
            registry=self.registry,
        )
        self.latency = Histogram(
            "api_request_duration_seconds",
            "HTTP request duration",
            ("method", "route"),
            registry=self.registry,
        )
        self.predictions = Counter(
            "api_predictions_total",
            "Successful reconstruction requests",
            registry=self.registry,
        )

    async def middleware(self, request, call_next):
        start = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            return response
        finally:
            route = request.scope.get("route")
            label = getattr(route, "path", "unmatched")
            self.requests.labels(request.method, label, str(status)).inc()
            self.latency.labels(request.method, label).observe(time.perf_counter() - start)
            if status >= 400:
                self.errors.labels(request.method, label, str(status)).inc()

    def response(self) -> Response:
        return Response(
            generate_latest(self.registry), headers={"Content-Type": CONTENT_TYPE_LATEST}
        )
