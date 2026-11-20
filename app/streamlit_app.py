"""Entry point for the purple-themed Streamlit dashboard."""

import streamlit as st

from app.pages import indicators, monitoring, pareto_explorer, reconstruct
from app.theme import css, palette, toggle

st.set_page_config(
    page_title="Kenya Demographic Decision Support", page_icon="app/assets/icon.svg", layout="wide"
)
st.logo("app/assets/logo.svg", icon_image="app/assets/icon.svg", size="large")
toggle()
css()
colors = palette()

st.sidebar.markdown(
    '<div class="project-brand"><div class="eyebrow">Kenya demographic data</div>'
    '<div class="brand-title">County Insights</div>'
    '<div class="brand-copy">Understand populations and explore planning choices</div></div>',
    unsafe_allow_html=True,
)
st.sidebar.subheader("Explore")
selected_page = st.sidebar.radio(
    "Choose a page", ["Fill missing numbers", "Explore counties", "Compare plans", "Service status"]
)
st.sidebar.caption("Explore 2019 county data and try experimental planning tools.")

if selected_page == "Fill missing numbers":
    reconstruct.render(colors)
elif selected_page == "Explore counties":
    indicators.render(colors)
elif selected_page == "Compare plans":
    pareto_explorer.render(colors)
else:
    monitoring.render(colors)
