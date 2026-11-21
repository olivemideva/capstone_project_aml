"""County indicators page with charts and optional GeoJSON choropleth."""

from __future__ import annotations

import json
from collections.abc import Iterable

import pandas as pd
import plotly.express as px
import streamlit as st

from app import api_client
from app.presentation import INDICATOR_LABELS
from app.theme import plotly_layout

METRIC_EXPLANATIONS = {
    "child_share": "The percentage of residents aged 0 to 14.",
    "elderly_share": "The percentage of residents aged 65 and over.",
    "dependency_ratio": (
        "The number of people aged 0 to 14 or 65 and over for every person aged 15 to 64. "
        "This is an age-based comparison, not a count of people who actually need support."
    ),
    "children": "The number of residents aged 0 to 14.",
    "elderly": "The number of residents aged 65 and over.",
    "total": "The total number of residents across the age groups in the data.",
}


@st.cache_data(ttl=600)
def fetch_indicator_table() -> pd.DataFrame:
    rows = api_client.all_indicators()
    return pd.DataFrame(rows).sort_values("admin1_name").reset_index(drop=True)


def _extract_values(features: Iterable[dict], key_path: str) -> set[str]:
    parts = key_path.split(".")
    if not parts:
        raise ValueError("Property path cannot be empty")
    values: set[str] = set()
    for feature in features:
        current = feature
        for part in parts:
            if not isinstance(current, dict) or part not in current:
                raise ValueError(f"Property path '{key_path}' was not found in every feature")
            current = current[part]
        values.add(str(current))
    return values


def render(colors: dict) -> None:
    st.title("Get to know Kenya's counties")
    st.markdown("See how population sizes and age groups compare across counties.")
    st.caption("These figures come from the observed 2019 data, not model estimates.")
    if st.button("Refresh county data", use_container_width=True):
        fetch_indicator_table.clear()
    try:
        frame = fetch_indicator_table()
    except Exception as exc:
        st.error(str(exc))
        return
    metric = st.selectbox(
        "What would you like to compare?", list(METRIC_EXPLANATIONS),
        format_func=INDICATOR_LABELS.get,
    )
    st.info(METRIC_EXPLANATIONS[metric])
    display_frame = frame.copy()
    for column in ("child_share", "elderly_share"):
        display_frame[column] = display_frame[column] * 100
    labels = dict(INDICATOR_LABELS)
    labels["child_share"] += " (%)"
    labels["elderly_share"] += " (%)"
    highest = display_frame.loc[display_frame[metric].idxmax()]
    summary_left, summary_right = st.columns(2)
    summary_left.metric("Counties compared", len(display_frame))
    summary_right.metric("Highest in this comparison", highest["admin1_name"])
    st.caption("Hover over a bar to see its county and value.")
    chart = px.bar(
        display_frame,
        x="admin1_name",
        y=metric,
        color=metric,
        labels=labels,
        color_continuous_scale=[colors["accent"], colors["primary"]],
        title=f"{labels[metric]} by county",
    )
    chart.update_xaxes(tickangle=-55)
    plotly_layout(chart)
    st.plotly_chart(chart, use_container_width=True)

    st.subheader("See the data on a map")
    st.caption(
        "Optional: upload a county boundary file in GeoJSON format. "
        "The file must include county codes matching this dataset."
    )
    uploaded = st.file_uploader("County boundary file", type=["json", "geojson"])
    if uploaded is not None:
        key_path = st.text_input(
            "Where to find the county code in the file",
            value="properties.COUNTY_COD",
            help="Use dot notation, e.g. properties.COUNTY_COD",
        )
        try:
            geojson = json.load(uploaded)
            features = geojson.get("features", [])
            if not isinstance(features, list) or not features:
                raise ValueError("GeoJSON must contain a non-empty 'features' list")
            map_codes = _extract_values(features, key_path)
            api_codes = set(frame["admin1_code"].astype(str))
            missing_in_map = sorted(api_codes - map_codes)
            if missing_in_map:
                st.warning(
                    "GeoJSON is missing county codes present in API data: "
                    + ", ".join(missing_in_map)
                )
            missing_in_api = sorted(map_codes - api_codes)
            if missing_in_api:
                st.info(
                    "GeoJSON contains extra county codes not present in API data: "
                    + ", ".join(missing_in_api[:12])
                    + ("..." if len(missing_in_api) > 12 else "")
                )
            choropleth = px.choropleth(
                display_frame,
                geojson=geojson,
                locations="admin1_code",
                featureidkey=key_path,
                color=metric,
                hover_name="admin1_name",
                labels=labels,
                color_continuous_scale=[colors["accent"], colors["primary"]],
                title=f"{labels[metric]} by county",
            )
            choropleth.update_geos(fitbounds="locations", visible=False)
            plotly_layout(choropleth)
            st.plotly_chart(choropleth, use_container_width=True)
        except Exception as exc:
            st.error(f"Unable to render GeoJSON map: {exc}")

    st.subheader("Explore the numbers")
    st.dataframe(display_frame.rename(columns=labels), use_container_width=True, hide_index=True)
