"""Upload a county population table and estimate missing age-sex values."""

import hashlib

import pandas as pd
import plotly.express as px
import streamlit as st

from app.population_csv import MAX_ROWS, read_population_csv, reconstruct_rows
from app.presentation import POPULATION_LABELS
from app.theme import plotly_layout
from src.features.engineer import ATOMIC_COLUMNS


def render(colors: dict) -> None:
    st.title("Fill in missing population numbers")
    st.markdown(
        "Upload a county population table and let the model estimate the blank age-group numbers."
    )
    st.info(
        "These estimates are experimental, not official figures. Review them before using them "
        "for decisions; some estimates can be inaccurate."
    )
    st.subheader("1. Prepare and upload your file")
    st.markdown(
        "Each row should describe one county. Include all 42 female and male age-group columns; "
        "leave a cell empty when its number is missing. Other columns, such as county name, "
        "are kept in your download."
    )
    template = pd.DataFrame(columns=["admin1_code", "admin1_name", *ATOMIC_COLUMNS])
    st.download_button(
        "Download a blank CSV template", template.to_csv(index=False),
        file_name="county_population_template.csv", mime="text/csv",
    )
    with st.expander("How to prepare the file"):
        st.markdown(
            "- Use the template or a processed county population CSV from this project.\n"
            "- Enter the numbers you know under the matching age and sex columns.\n"
            "- Leave missing numbers blank, or use `NA` or `null`. Zero means a known zero.\n"
            f"- Upload at most {MAX_ROWS} rows and a file no larger than 5 MB.\n"
            "- Existing totals and other extra columns are retained, not recalculated; "
            "they may no longer agree after gaps are filled."
        )
    uploaded = st.file_uploader("Choose a population CSV", type="csv", max_upload_size=5)
    if uploaded is None:
        return
    content = uploaded.getvalue()
    try:
        frame = read_population_csv(content)
    except ValueError as exc:
        st.error(str(exc))
        return
    missing_count = int(frame[ATOMIC_COLUMNS].isna().sum().sum())
    left, right = st.columns(2)
    left.metric("County rows", len(frame))
    right.metric("Numbers to estimate", missing_count)
    if not missing_count:
        st.info("There are no missing age-group numbers in this file.")
        return
    st.caption(
        "Age-group numbers are sent to the configured estimation service one row at a time. "
        "County names and other columns are not sent. Avoid uploading sensitive information."
    )
    st.subheader("2. Choose the closest training example")
    st.caption(
        "The model was trained using different kinds of gaps. These settings choose which model "
        "to use; they do not change which cells in your file are missing."
    )
    patterns = {
        "Scattered missing numbers": "random",
        "Whole age groups missing": "age-block",
        "Whole county profiles missing": "county-block",
    }
    scenario_left, scenario_right = st.columns(2)
    pattern_label = scenario_left.selectbox("How is information missing?", list(patterns))
    rate = scenario_right.selectbox(
        "Approximate amount missing in the training examples", [10, 20, 40], index=1,
        format_func=lambda value: f"{value}%",
    )
    pattern = patterns[pattern_label]
    request_key = (hashlib.sha256(content).hexdigest(), pattern, rate)
    if st.button("Estimate missing numbers", type="primary", use_container_width=True):
        st.session_state.pop("reconstruction_result", None)
        try:
            with st.spinner("Estimating missing numbers..."):
                completed, changes = reconstruct_rows(frame, pattern, rate)
        except RuntimeError as exc:
            st.error(str(exc))
            return
        st.session_state["reconstruction_result"] = (request_key, completed, changes)
    saved = st.session_state.get("reconstruction_result")
    if not saved or saved[0] != request_key:
        return
    _, completed, changes = saved
    st.subheader("3. Review and download")
    st.success(f"Estimated {len(changes)} missing numbers across your file.")
    st.caption(
        "Only missing age-group cells were filled. Known numbers and extra columns are retained. "
        "Estimates may be fractional and totals are not recalculated."
    )
    details = pd.DataFrame([
        {
            "Data row": index + 1,
            "County": frame.at[index, "admin1_name"] if "admin1_name" in frame else "",
            "Population group": POPULATION_LABELS[column],
            "Estimated number of people": value,
        }
        for index, column, value in changes
    ])
    st.dataframe(details, use_container_width=True, hide_index=True)
    chart = px.bar(
        details, x="Population group", y="Estimated number of people", color="County",
        hover_data=["Data row"], title="Estimated values by population group",
        color_discrete_sequence=[colors["primary"], colors["accent"], "#EC4899"],
    )
    chart.update_xaxes(tickangle=-70)
    plotly_layout(chart)
    st.plotly_chart(chart, use_container_width=True)
    st.download_button(
        "Download completed CSV", completed.to_csv(index=False),
        file_name="completed_population.csv", mime="text/csv",
    )
