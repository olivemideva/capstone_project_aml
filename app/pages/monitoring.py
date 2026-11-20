"""Monitoring page for health, metrics trends, and optional Evidently drift report."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import plotly.express as px
import streamlit as st

from app import api_client
from app.metrics_parser import parse_snapshot
from app.theme import plotly_layout


def _sample_metrics() -> None:
    raw = api_client.metrics_raw()
    snapshot = parse_snapshot(raw)
    history = st.session_state.setdefault("metrics_history", [])
    history.append(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "request_total": snapshot.request_total,
            "error_total": snapshot.error_total,
            "prediction_total": snapshot.prediction_total,
            "average_latency_seconds": snapshot.average_latency_seconds,
        }
    )
    st.session_state["metrics_raw"] = raw


def _render_history(colors: dict) -> None:
    history = st.session_state.get("metrics_history", [])
    if not history:
        st.info("No metric samples yet. Use 'Sample metrics now' to add points.")
        return
    frame = pd.DataFrame(history)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    chart = px.line(
        frame,
        x="timestamp",
        y=["request_total", "error_total", "prediction_total"],
        title="Cumulative counters over this Streamlit session",
    )
    chart.update_traces(line={"width": 2})
    chart.update_layout(colorway=[colors["primary"], colors["accent"], "#EC4899"])
    plotly_layout(chart)
    st.plotly_chart(chart, use_container_width=True)
    latency = px.line(
        frame,
        x="timestamp",
        y="average_latency_seconds",
        title="Average API latency (seconds)",
        markers=True,
    )
    latency.update_layout(colorway=[colors["primary"]])
    plotly_layout(latency)
    st.plotly_chart(latency, use_container_width=True)


def _render_drift_report() -> None:
    st.subheader("Evidently drift report")
    st.caption(
        "Upload both datasets to generate a local report. "
        "This is on-demand and not an automatic live monitor."
    )
    reference_file = st.file_uploader("Reference CSV", type=["csv"], key="drift_reference")
    current_file = st.file_uploader("Current CSV", type=["csv"], key="drift_current")
    if not reference_file or not current_file:
        return
    if st.button("Generate drift report", use_container_width=True):
        try:
            reference = pd.read_csv(reference_file)
            current = pd.read_csv(current_file)
            overlap = sorted(set(reference.columns) & set(current.columns))
            if not overlap:
                raise ValueError("Reference and current CSV files must share at least one column")
            if reference.empty or current.empty:
                raise ValueError("Reference and current CSV files must both contain rows")
            from evidently import Report
            from evidently.presets import DataDriftPreset

            report = Report(metrics=[DataDriftPreset()])
            report.run(
                reference_data=reference[overlap],
                current_data=current[overlap],
            )
            st.components.v1.html(report.get_html(), height=900, scrolling=True)
        except Exception as exc:
            st.error(f"Unable to generate Evidently report: {exc}")


def render(colors: dict) -> None:
    st.title("Monitoring")
    st.caption("Metrics are sampled from /metrics and kept only in this Streamlit session.")
    left, right = st.columns(2)
    try:
        health = api_client.health()
    except Exception as exc:
        st.error(str(exc))
        return
    left.metric("Service status", health["status"])
    right.metric("Model family", health["model_family"])

    action_left, action_right = st.columns(2)
    if action_left.button("Sample metrics now", type="primary", use_container_width=True):
        try:
            _sample_metrics()
        except Exception as exc:
            st.error(str(exc))
    if action_right.button("Reset sampled history", use_container_width=True):
        st.session_state.pop("metrics_history", None)
        st.session_state.pop("metrics_raw", None)
    _render_history(colors)
    raw = st.session_state.get("metrics_raw")
    if raw:
        with st.expander("Latest raw /metrics payload"):
            st.code(raw, language="text")
    _render_drift_report()
