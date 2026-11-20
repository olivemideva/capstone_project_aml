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
        st.info("No readings yet. Select 'Take a reading' to start the charts.")
        return
    frame = pd.DataFrame(history)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame = frame.rename(columns={
        "timestamp": "Time (UTC)",
        "request_total": "Requests",
        "error_total": "Errors",
        "prediction_total": "Estimates",
        "average_latency_seconds": "Average response time (seconds)",
    })
    chart = px.line(
        frame, x="Time (UTC)", y=["Requests", "Errors", "Estimates"],
        labels={"value": "Total since service started", "variable": "Reading"},
        title="Service activity over time",
    )
    chart.update_traces(line={"width": 2})
    chart.update_layout(colorway=[colors["primary"], colors["accent"], "#EC4899"])
    plotly_layout(chart)
    st.plotly_chart(chart, use_container_width=True)
    latency = px.line(
        frame,
        x="Time (UTC)",
        y="Average response time (seconds)",
        title="Average time taken to respond",
        markers=True,
    )
    latency.update_layout(colorway=[colors["primary"]])
    plotly_layout(latency)
    st.plotly_chart(latency, use_container_width=True)


def _render_drift_report() -> None:
    st.subheader("Compare two datasets")
    st.caption(
        "Optional: upload an earlier dataset and a newer dataset to check for changes "
        "in their shared columns. The report is created on request, not continuously."
    )
    reference_file = st.file_uploader("Earlier dataset (CSV)", type=["csv"], key="drift_reference")
    current_file = st.file_uploader("Newer dataset (CSV)", type=["csv"], key="drift_current")
    if not reference_file or not current_file:
        return
    if st.button("Compare datasets", use_container_width=True):
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
            st.error(f"Unable to compare the datasets: {exc}")


def render(colors: dict) -> None:
    st.title("Check how the service is doing")
    st.markdown("See whether the service is responding and follow its activity over time.")
    st.caption(
        "Readings are taken when you select the button and are kept only during this "
        "browser session. Service totals may reset when the service restarts."
    )
    left, right = st.columns(2)
    try:
        health = api_client.health()
    except Exception as exc:
        st.error(str(exc))
        return
    status = "Responding" if health["status"] == "ok" else str(health["status"])
    models = {"random_forest": "Random forest", "xgboost": "XGBoost"}
    left.metric("Service status", status)
    right.metric("Estimation method", models.get(health["model_family"], health["model_family"]))
    st.caption(
        "The estimation method is the type of model used to fill missing population numbers."
    )

    action_left, action_right = st.columns(2)
    if action_left.button("Take a reading", type="primary", use_container_width=True):
        try:
            _sample_metrics()
        except Exception as exc:
            st.error(str(exc))
    if action_right.button("Clear readings", use_container_width=True):
        st.session_state.pop("metrics_history", None)
        st.session_state.pop("metrics_raw", None)
    _render_history(colors)
    raw = st.session_state.get("metrics_raw")
    if raw:
        with st.expander("Technical details: latest raw service readings"):
            st.code(raw, language="text")
    _render_drift_report()
