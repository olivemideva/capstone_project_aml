"""Theme state and CSS for a purple light/dark Streamlit interface."""

import streamlit as st

PALETTES = {
    "light": {
        "name": "light", "primary": "#7C3AED", "accent": "#A78BFA", "text": "#1F1135",
        "background": "#FFFFFF", "surface": "#F5F3FF", "surface_alt": "#EDE9FE", "grid": "#D6BCFA",
    },
    "dark": {
        "name": "dark", "primary": "#A78BFA", "accent": "#C4B5FD", "text": "#F3F0FF",
        "background": "#160A2B", "surface": "#24123D", "surface_alt": "#31195C", "grid": "#5B3A96",
    },
}


def palette() -> dict:
    if "theme_mode" not in st.session_state:
        st.session_state.theme_mode = "light"
    return PALETTES[st.session_state.theme_mode]


def toggle() -> None:
    if "theme_mode" not in st.session_state:
        st.session_state.theme_mode = "light"
    selected = st.sidebar.toggle(
        "Dark mode", value=st.session_state.theme_mode == "dark", key="theme_toggle"
    )
    st.session_state.theme_mode = "dark" if selected else "light"


def css() -> None:
    colors = palette()
    st.markdown(
        f"""
        <style>
        :root {{
            --ui-primary: {colors["primary"]};
            --ui-accent: {colors["accent"]};
            --ui-text: {colors["text"]};
            --ui-background: {colors["background"]};
            --ui-surface: {colors["surface"]};
            --ui-surface-alt: {colors["surface_alt"]};
            --ui-grid: {colors["grid"]};
        }}
        [data-testid="stAppViewContainer"] {{
            background: var(--ui-background);
            color: var(--ui-text);
        }}
        [data-testid="stSidebar"] {{
            background: var(--ui-surface);
        }}
        [data-testid="stMetricValue"] {{
            color: var(--ui-primary);
        }}
        .stButton>button, .stDownloadButton>button {{
            background: var(--ui-primary);
            color: white;
            border: 0;
        }}
        .stButton>button:hover, .stDownloadButton>button:hover {{
            background: var(--ui-accent);
            color: white;
        }}
        .stTabs [data-baseweb="tab"] {{
            background: var(--ui-surface-alt);
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def plotly_layout(figure):
    colors = palette()
    return figure.update_layout(
        paper_bgcolor=colors["background"],
        plot_bgcolor=colors["surface"],
        font={"color": colors["text"]},
        title_font={"color": colors["primary"]},
        legend={"bgcolor": colors["surface_alt"]},
        xaxis={"gridcolor": colors["grid"]},
        yaxis={"gridcolor": colors["grid"]},
    )
