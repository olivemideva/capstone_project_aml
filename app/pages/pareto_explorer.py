"""Pareto explorer page for weighted and NSGA-III simulated selections."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from app import api_client
from app.theme import plotly_layout


def render(colors: dict) -> None:
    st.title("Explore planning trade-offs")
    st.markdown("Compare how county selections balance children, older people and dependency.")
    st.caption(
        "Outputs are simulated equal-cost county selections, not actual budget allocations. "
        "Dependency is an illustrative proxy, not a validated measure of fairness."
    )
    budget_col, child_col, elderly_col, dep_col = st.columns(4)
    payload = {
        "budget": int(budget_col.number_input("Budget", min_value=1, max_value=47, value=10)),
        "weights": (
            child_col.number_input("Child weight", min_value=0.0, value=1.0),
            elderly_col.number_input("Elderly weight", min_value=0.0, value=1.0),
            dep_col.number_input("Dependency weight", min_value=0.0, value=1.0),
        ),
        "generations": int(
            st.slider("NSGA-III generations", min_value=1, max_value=200, value=50)
        ),
        "seed": int(st.number_input("Seed", min_value=0, max_value=2**32 - 1, value=42)),
    }
    if st.button("Run prioritisation", type="primary", use_container_width=True):
        try:
            response = api_client.prioritise(payload)
        except Exception as exc:
            st.error(str(exc))
            return
        st.info(response["note"])
        table = pd.DataFrame(response["solutions"])
        scatter = px.scatter_3d(
            table,
            x="child_coverage",
            y="elderly_coverage",
            z="dependency_burden_coverage",
            color="method",
            hover_data=["selected_count"],
            color_discrete_map={"weighted_sum": colors["primary"], "nsga3": colors["accent"]},
            title="Coverage trade-offs",
        )
        plotly_layout(scatter)
        st.plotly_chart(scatter, use_container_width=True)
        display = table.assign(
            selected_counties=table["selected_counties"].apply(
                lambda rows: ", ".join(item["admin1_name"] for item in rows)
            )
        )
        st.dataframe(display, use_container_width=True, hide_index=True)
