"""Prometheus text parsing and summary helpers for monitoring visuals."""

from __future__ import annotations

from dataclasses import dataclass

from prometheus_client.parser import text_string_to_metric_families


@dataclass(frozen=True)
class MetricSnapshot:
    request_total: float
    error_total: float
    prediction_total: float
    duration_sum: float
    duration_count: float

    @property
    def average_latency_seconds(self) -> float:
        if self.duration_count <= 0:
            return 0.0
        return self.duration_sum / self.duration_count


def parse_snapshot(raw: str) -> MetricSnapshot:
    request_total = 0.0
    error_total = 0.0
    prediction_total = 0.0
    duration_sum = 0.0
    duration_count = 0.0
    for family in text_string_to_metric_families(raw):
        for sample in family.samples:
            if sample.name == "api_requests_total":
                request_total += float(sample.value)
            elif sample.name == "api_errors_total":
                error_total += float(sample.value)
            elif sample.name == "api_predictions_total":
                prediction_total += float(sample.value)
            elif sample.name == "api_request_duration_seconds_sum":
                duration_sum += float(sample.value)
            elif sample.name == "api_request_duration_seconds_count":
                duration_count += float(sample.value)
    return MetricSnapshot(
        request_total=request_total,
        error_total=error_total,
        prediction_total=prediction_total,
        duration_sum=duration_sum,
        duration_count=duration_count,
    )
