"""County indicators page with charts and optional GeoJSON choropleth."""

from __future__ import annotations

import json
from collections.abc import Iterable

import pandas as pd
import plotly.express as px
import streamlit as st

from app import api_client
from app.theme import plotly_layout


@st.cache_data(ttl=600)
def fetch_indicator_table() -> pd.DataFrame:
    counties = api_client.counties()
    rows = [api_client.indicators(item["admin1_code"]) for item in counties]
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
    st.title("Explore county indicators")
    st.markdown("Compare observed population and dependency measures across Kenya's counties.")
    st.caption("Based on observed 2019 data, not reconstructed predictions.")
    if st.button("Refresh API indicator cache", use_container_width=True):
        fetch_indicator_table.clear()
    try:
        frame = fetch_indicator_table()
    except Exception as exc:
        st.error(str(exc))
        return
    metric = st.selectbox(
        "Indicator",
        ["child_share", "elderly_share", "dependency_ratio", "children", "elderly", "total"],
    )
    chart = px.bar(
        frame,
        x="admin1_name",
        y=metric,
        color=metric,
        color_continuous_scale=[colors["accent"], colors["primary"]],
        title=f"{metric} by county",
    )
    chart.update_xaxes(tickangle=-55)
    plotly_layout(chart)
    st.plotly_chart(chart, use_container_width=True)

    st.subheader("Map the results")
    st.caption("Upload licensed county GeoJSON to activate the optional map overlay.")
    uploaded = st.file_uploader("County GeoJSON", type=["json", "geojson"])
    if uploaded is not None:
        key_path = st.text_input(
            "GeoJSON property path for county code",
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
                frame,
                geojson=geojson,
                locations="admin1_code",
                featureidkey=key_path,
                color=metric,
                hover_name="admin1_name",
                color_continuous_scale=[colors["accent"], colors["primary"]],
                title=f"{metric} choropleth",
            )
            choropleth.update_geos(fitbounds="locations", visible=False)
            plotly_layout(choropleth)
            st.plotly_chart(choropleth, use_container_width=True)
        except Exception as exc:
            st.error(f"Unable to render GeoJSON map: {exc}")

    st.dataframe(frame, use_container_width=True, hide_index=True)
