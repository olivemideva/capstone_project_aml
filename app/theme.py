"""Theme state and CSS for a purple light/dark Streamlit interface."""

import streamlit as st

PALETTES = {
    "light": {
        "name": "light", "primary": "#7C3AED", "accent": "#A78BFA", "text": "#1F1135",
        "background": "#FAF9FF", "surface": "#FFFFFF", "surface_alt": "#F0EBFF",
        "grid": "#D6BCFA", "muted": "#615572", "border": "#DFD5F5",
        "button": "#6D28D9", "button_hover": "#5720AE", "button_text": "#FFFFFF",
    },
    "dark": {
        "name": "dark", "primary": "#BBA3FF", "accent": "#DDC5FF", "text": "#F6F1FF",
        "background": "#130C22", "surface": "#211530", "surface_alt": "#302044",
        "grid": "#58436F", "muted": "#C7B9D9", "border": "#55406D",
        "button": "#8052D9", "button_hover": "#9868EC", "button_text": "#FFFFFF",
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
            --ui-muted: {colors["muted"]};
            --ui-border: {colors["border"]};
            --ui-button: {colors["button"]};
            --ui-button-hover: {colors["button_hover"]};
            --ui-button-text: {colors["button_text"]};
        }}
        [data-testid="stAppViewContainer"], [data-testid="stMain"] {{
            background: var(--ui-background);
            color: var(--ui-text);
        }}
        [data-testid="stHeader"] {{
            background: var(--ui-background);
        }}
        [data-testid="stSidebar"], [data-testid="stSidebarContent"] {{
            background: var(--ui-surface);
            color: var(--ui-text);
        }}
        [data-testid="stMain"] .block-container {{
            max-width: 1200px;
            padding-top: 2rem;
            padding-bottom: 4rem;
        }}
        [data-testid="stAppViewContainer"] h1,
        [data-testid="stAppViewContainer"] h2,
        [data-testid="stAppViewContainer"] h3,
        [data-testid="stAppViewContainer"] p,
        [data-testid="stAppViewContainer"] label,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] label {{
            color: var(--ui-text);
        }}
        .project-brand {{
            border: 1px solid var(--ui-border);
            border-radius: 20px;
            background: var(--ui-surface-alt);
            padding: 1.15rem 1.3rem;
            margin-bottom: 1.3rem;
        }}
        .project-brand .eyebrow {{
            color: var(--ui-primary);
            font-size: 0.75rem;
            font-weight: 700;
            letter-spacing: 0.12em;
            text-transform: uppercase;
        }}
        .project-brand .brand-title {{
            color: var(--ui-text);
            font-size: 1.15rem;
            font-weight: 700;
            margin-top: 0.3rem;
        }}
        .project-brand .brand-copy {{
            color: var(--ui-muted);
            font-size: 0.88rem;
            margin-top: 0.3rem;
        }}
        [data-testid="stNumberInput"] input,
        [data-testid="stTextInput"] input,
        [data-testid="stSelectbox"] [role="group"],
        [data-testid="stMultiSelect"] [data-testid="stMultiSelectTagsContainer"],
        [data-baseweb="select"] > div {{
            background-color: var(--ui-surface-alt);
            color: var(--ui-text);
            border-color: var(--ui-border);
        }}
        [data-testid="stSelectbox"] input,
        [data-testid="stSelectbox"] button,
        [data-testid="stMultiSelect"] input {{
            color: var(--ui-text);
        }}
        [data-testid="stTextInput"] input:disabled {{
            -webkit-text-fill-color: var(--ui-muted);
            opacity: 1;
        }}
        [data-testid="stNumberInput"] button {{
            background: var(--ui-surface-alt);
            color: var(--ui-text);
            border-color: var(--ui-border);
        }}
        [data-testid="stExpander"] {{
            border-color: var(--ui-border);
            background: var(--ui-surface);
            border-radius: 14px;
        }}
        [data-testid="stMetricValue"] {{
            color: var(--ui-primary);
        }}
        .stButton>button, .stDownloadButton>button {{
            background: var(--ui-button);
            color: var(--ui-button-text);
            border: 1px solid var(--ui-button);
            border-radius: 10px;
            font-weight: 600;
            transition: background-color 160ms ease, box-shadow 160ms ease;
        }}
        .stButton>button:hover, .stDownloadButton>button:hover {{
            background: var(--ui-button-hover);
            color: var(--ui-button-text);
            border-color: var(--ui-button-hover);
            box-shadow: 0 4px 14px rgba(109, 40, 217, 0.25);
        }}
        .stButton>button:focus-visible, .stDownloadButton>button:focus-visible {{
            outline: 3px solid var(--ui-accent);
            outline-offset: 3px;
        }}
        [data-testid="stSidebar"] [role="radiogroup"] label {{
            border: 1px solid transparent;
            border-radius: 10px;
            padding: 0.45rem 0.65rem;
            transition: background-color 160ms ease, border-color 160ms ease;
        }}
        [data-testid="stSidebar"] [role="radiogroup"] label:hover {{
            background: var(--ui-surface-alt);
            border-color: var(--ui-border);
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
        scene={
            "xaxis": {"gridcolor": colors["grid"], "backgroundcolor": colors["surface"]},
            "yaxis": {"gridcolor": colors["grid"], "backgroundcolor": colors["surface"]},
            "zaxis": {"gridcolor": colors["grid"], "backgroundcolor": colors["surface"]},
        },
    )
