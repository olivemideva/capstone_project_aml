"""Theme state and CSS for a purple light/dark Streamlit interface."""

from pathlib import Path
from urllib.parse import quote

import streamlit as st
import streamlit.components.v1 as components

device_theme = components.declare_component(
    "device_theme", path=str(Path(__file__).parent / "theme_detector")
)

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
        "button": "#8052D9", "button_hover": "#6835B9", "button_text": "#FFFFFF",
    },
}


def palette() -> dict:
    if "theme_mode" not in st.session_state:
        st.session_state.theme_mode = "light"
    return PALETTES[st.session_state.theme_mode]


def toggle() -> None:
    detected = device_theme(key="device_theme")
    if not st.session_state.get("theme_manual", False):
        st.session_state.theme_mode = detected if detected in ("light", "dark") else "light"
    target = "light" if st.session_state.theme_mode == "dark" else "dark"
    if st.button(
        f"Switch to {target} mode", key="theme_button", help=f"Switch to {target} mode"
    ):
        st.session_state.theme_mode = target
        st.session_state.theme_manual = True
        st.rerun()


def css() -> None:
    colors = palette()
    icon_path = (
        '<circle cx="12" cy="12" r="4"/><path d="M12 2v2m0 16v2M4.93 4.93l1.42 '
        '1.42m11.3 11.3 1.42 1.42M2 12h2m16 0h2M4.93 19.07l1.42-1.42M17.65 '
        '6.35l1.42-1.42"/>'
        if colors["name"] == "dark"
        else '<path d="M20.9 13A9 9 0 0 1 11 3.1 9 9 0 1 0 20.9 13Z"/>'
    )
    icon = quote(
        '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" '
        f'viewBox="0 0 24 24" fill="none" stroke="{colors["text"]}" '
        f'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">{icon_path}</svg>'
    )
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
            padding-top: 5rem;
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
        [data-testid="stMetric"] {{
            background: var(--ui-surface);
            border: 1px solid var(--ui-border);
            border-radius: 16px;
            padding: 1rem 1.15rem;
            transition: border-color 160ms ease, box-shadow 160ms ease;
        }}
        [data-testid="stMetric"]:hover {{
            border-color: var(--ui-primary);
            box-shadow: 0 6px 20px rgba(109, 40, 217, 0.12);
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
        .stButton>button p, .stDownloadButton>button p {{
            color: var(--ui-button-text) !important;
        }}
        .stButton>button:focus-visible, .stDownloadButton>button:focus-visible {{
            outline: 3px solid var(--ui-accent);
            outline-offset: 3px;
        }}
        .stButton>button:active, .stDownloadButton>button:active {{
            transform: scale(0.98);
        }}
        .st-key-theme_button button {{
            position: relative;
            min-width: 48px;
            width: 48px;
            height: 48px;
            padding: 0;
            background: var(--ui-surface-alt);
            border: 1px solid var(--ui-border);
        }}
        .st-key-theme_button button p {{
            font-size: 0;
        }}
        .st-key-theme_button button::after {{
            content: "";
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            width: 24px;
            height: 24px;
            background: url("data:image/svg+xml,{icon}") center / contain no-repeat;
        }}
        .st-key-theme_button button:hover {{
            background: var(--ui-surface);
            border-color: var(--ui-primary);
        }}
        @media (prefers-reduced-motion: reduce) {{
            .stButton>button, .stDownloadButton>button, [data-testid="stMetric"] {{
                transition: none;
            }}
        }}
        [data-testid="stSidebar"] [role="radiogroup"] label {{
            border: 1px solid transparent;
            border-radius: 10px;
            padding: 0.45rem 0.65rem;
            transition: background-color 160ms ease, border-color 160ms ease, transform 160ms ease;
        }}
        [data-testid="stSidebar"] [role="radiogroup"] label:hover {{
            background: var(--ui-surface-alt);
            border-color: var(--ui-border);
            transform: translateX(4px);
        }}
        [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {{
            background: var(--ui-surface-alt);
            border-color: var(--ui-primary);
        }}
        [data-testid="stSidebar"] [role="radiogroup"] label:focus-within {{
            outline: 2px solid var(--ui-primary);
            outline-offset: 2px;
        }}
        [data-testid="stAlert"] {{
            color: var(--ui-text);
        }}
        [data-testid="stAlertContainer"] {{
            background: var(--ui-surface-alt);
            border: 1px solid var(--ui-border);
            color: var(--ui-text);
        }}
        [data-testid="stAlertContainer"] [data-testid^="stAlertContent"] {{
            color: var(--ui-text);
        }}
        [data-testid="stAlert"] p {{
            color: var(--ui-text) !important;
        }}
        [data-testid="stFileUploaderDropzone"] {{
            background: var(--ui-surface-alt);
            border: 1px dashed var(--ui-border);
            color: var(--ui-text);
        }}
        [data-testid="stFileUploaderDropzoneInstructions"],
        [data-testid="stFileUploaderDropzoneInstructions"] span {{
            color: var(--ui-text);
        }}
        [data-testid="stFileUploaderDropzone"] button {{
            background: var(--ui-button);
            color: var(--ui-button-text);
            border: 1px solid var(--ui-button);
        }}
        [data-testid="stFileUploaderDropzone"] button p {{
            color: var(--ui-button-text) !important;
        }}
        [data-testid="stFileUploaderDropzone"] button:hover {{
            background: var(--ui-button-hover);
            border-color: var(--ui-button-hover);
        }}
        [data-testid="stFileUploaderDropzone"] button:focus-visible {{
            outline: 3px solid var(--ui-primary);
            outline-offset: 3px;
        }}
        @media (prefers-reduced-motion: reduce) {{
            [data-testid="stSidebar"] [role="radiogroup"] label {{
                transition: none;
            }}
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
