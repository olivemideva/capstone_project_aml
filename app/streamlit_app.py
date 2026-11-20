"""Entry point for the purple-themed Streamlit dashboard."""

import streamlit as st

from app import api_client
from app.pages import indicators, monitoring, pareto_explorer, reconstruct
from app.theme import css, palette, toggle

st.set_page_config(
    page_title="Kenya Demographic Decision Support", page_icon="app/assets/icon.svg", layout="wide"
)
toggle()
css()
colors = palette()

st.sidebar.title("Navigation")
selected_page = st.sidebar.radio(
    "Page", ["Reconstruct", "Indicators", "Pareto Explorer", "Monitoring"]
)
st.sidebar.caption(f"API: {api_client.base_url()}")

if selected_page == "Reconstruct":
    reconstruct.render(colors)
elif selected_page == "Indicators":
    indicators.render(colors)
elif selected_page == "Pareto Explorer":
    pareto_explorer.render(colors)
else:
    monitoring.render(colors)
