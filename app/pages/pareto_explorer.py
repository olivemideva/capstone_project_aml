"""Pareto explorer page for weighted and NSGA-III simulated selections."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from app import api_client
from app.presentation import METHOD_LABELS
from app.theme import plotly_layout


def render(colors: dict) -> None:
    st.title("Compare possible county plans")
    st.markdown(
        "Try different priorities and see how choosing some counties affects who is included."
    )
    st.info(
        "This is a simulation, not a real funding decision. It assumes every county costs "
        "the same to include. The age-based measure is not a validated measure of fairness."
    )
    st.subheader("1. Set your priorities")
    st.caption(
        "Larger numbers give a priority more influence in the priority-weighted plan. "
        "The alternative plans explore several possible balances."
    )
    budget_col, child_col, elderly_col, dep_col = st.columns(4)
    payload = {
        "budget": int(budget_col.number_input(
            "Maximum counties", min_value=1, max_value=47, value=10,
            help="The simulated budget is the maximum number of counties, not an amount of money.",
        )),
        "weights": (
            child_col.number_input("Priority: children", min_value=0.0, value=1.0),
            elderly_col.number_input("Priority: older people", min_value=0.0, value=1.0),
            dep_col.number_input(
                "Priority: age balance", min_value=0.0, value=1.0,
                help="Based on the age-based dependency ratio, not actual support needs.",
            ),
        ),
    }
    with st.expander("Advanced search settings"):
        payload["generations"] = int(st.slider(
            "Search effort", min_value=1, max_value=200, value=50,
            help="More rounds may find different alternatives but can take longer.",
        ))
        payload["seed"] = int(st.number_input(
            "Repeatability number", min_value=0, max_value=2**32 - 1, value=42,
            help="Use the same number and settings to repeat a search.",
        ))
    if st.button("Compare plans", type="primary", use_container_width=True):
        if sum(payload["weights"]) <= 0:
            st.warning("Give at least one priority a value greater than zero.")
            return
        try:
            response = api_client.prioritise(payload)
        except Exception as exc:
            st.error(str(exc))
            return
        table = pd.DataFrame(response["solutions"])
        table["method"] = table["method"].map(lambda method: METHOD_LABELS.get(method, method))
        st.subheader("2. Explore the results")
        st.caption(
            "Each point is a possible plan. The first two axes show the share of all children "
            "and older people living in selected counties. The third shows the selected "
            "counties' share of the sum of county age-based dependency ratios; it is not "
            "a share of people. Higher values mean a larger share of that measure."
        )
        labels = {
            "child_coverage": "Share of children",
            "elderly_coverage": "Share of older people",
            "dependency_burden_coverage": "Share of county age-balance measure",
            "selected_count": "Counties selected",
            "selected_counties": "Selected counties",
            "method": "Plan type",
        }
        scatter = px.scatter_3d(
            table,
            x="child_coverage",
            y="elderly_coverage",
            z="dependency_burden_coverage",
            color="method",
            hover_data=["selected_count"],
            labels=labels,
            color_discrete_map={
                METHOD_LABELS["weighted_sum"]: colors["primary"],
                METHOD_LABELS["nsga3"]: colors["accent"],
            },
            title="How the possible plans compare",
        )
        scatter.update_scenes(
            xaxis_tickformat=".0%", yaxis_tickformat=".0%", zaxis_tickformat=".0%"
        )
        scatter.update_traces(
            hovertemplate=(
                "Plan: %{fullData.name}<br>"
                "Share of children: %{x:.1%}<br>"
                "Share of older people: %{y:.1%}<br>"
                "Share of county age-balance measure: %{z:.1%}<br>"
                "Counties selected: %{customdata[0]}<extra></extra>"
            )
        )
        plotly_layout(scatter)
        st.plotly_chart(scatter, use_container_width=True)
        display = table.assign(
            selected_counties=table["selected_counties"].apply(
                lambda rows: ", ".join(item["admin1_name"] for item in rows)
            )
        )
        for column in ("child_coverage", "elderly_coverage", "dependency_burden_coverage"):
            display[column] = display[column] * 100
            labels[column] += " (%)"
        st.dataframe(display.rename(columns=labels), use_container_width=True, hide_index=True)
