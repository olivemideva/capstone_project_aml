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

st.sidebar.markdown(
    '<div class="project-brand"><div class="eyebrow">Kenya demographic data</div>'
    '<div class="brand-title">County Insights</div>'
    '<div class="brand-copy">Explore profiles and planning trade-offs</div></div>',
    unsafe_allow_html=True,
)
st.sidebar.subheader("Explore")
selected_page = st.sidebar.radio(
    "Page", ["Reconstruct", "Indicators", "Pareto Explorer", "Monitoring"]
)
st.sidebar.caption("Experimental decision support using 2019 county data.")
st.sidebar.caption(f"API connection: {api_client.base_url()}")

if selected_page == "Reconstruct":
    reconstruct.render(colors)
elif selected_page == "Indicators":
    indicators.render(colors)
elif selected_page == "Pareto Explorer":
    pareto_explorer.render(colors)
else:
    monitoring.render(colors)
