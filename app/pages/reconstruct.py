"""Reconstruction page for age-sex profile completion."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from app import api_client
from app.theme import plotly_layout
from src.features.engineer import ATOMIC_COLUMNS


def _default_values() -> dict[str, float]:
    return {column: 1_000.0 for column in ATOMIC_COLUMNS}


def _collect_population() -> dict[str, float | None]:
    defaults = _default_values()
    st.caption("Set values for observed fields and mark missing fields as null.")
    missing = st.multiselect(
        "Fields to mask as missing",
        options=ATOMIC_COLUMNS,
        default=ATOMIC_COLUMNS[:2],
        help="The API requires at least one missing age-sex field.",
    )
    values: dict[str, float | None] = {}
    left, right = st.columns(2)
    for index, column in enumerate(ATOMIC_COLUMNS):
        container = left if index % 2 == 0 else right
        if column in missing:
            container.text_input(column, value="null", disabled=True)
            values[column] = None
        else:
            values[column] = container.number_input(
                column,
                min_value=0.0,
                value=defaults[column],
                key=f"population_{column}",
            )
    return values


def render(colors: dict) -> None:
    st.title("Reconstruct masked age-sex values")
    st.caption("Predictions are experimental and should not be treated as official county totals.")
    population = _collect_population()
    scenario_left, scenario_right = st.columns(2)
    pattern = scenario_left.selectbox("Mask pattern", ["random", "age-block", "county-block"])
    rate = scenario_right.selectbox("Mask rate (%)", [10, 20, 40], index=1)
    if st.button("Run reconstruction", type="primary", use_container_width=True):
        try:
            result = api_client.reconstruct(
                {"population": population, "pattern": pattern, "rate": rate}
            )
        except Exception as exc:
            st.error(str(exc))
            return
        st.success(f"Model family: {result['model_family']}")
        st.info(result["note"])
        table = pd.DataFrame(
            {
                "field": ATOMIC_COLUMNS,
                "value": [result["population"][column] for column in ATOMIC_COLUMNS],
                "source": [
                    "predicted" if column in result["predicted_fields"] else "observed"
                    for column in ATOMIC_COLUMNS
                ],
            }
        )
        chart = px.bar(
            table,
            x="field",
            y="value",
            color="source",
            color_discrete_map={"predicted": colors["primary"], "observed": colors["accent"]},
            title="Observed vs predicted values",
        )
        chart.update_xaxes(tickangle=-70)
        plotly_layout(chart)
        st.plotly_chart(chart, use_container_width=True)
        st.dataframe(table, use_container_width=True, hide_index=True)
