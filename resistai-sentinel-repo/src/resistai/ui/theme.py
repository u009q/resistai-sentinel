"""
Visual identity: colour tokens, CSS and a shared Plotly template.

Every chart in the app renders through `PLOTLY_TEMPLATE`, so axes, fonts and
grid weight stay consistent instead of each view inventing its own look.
"""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

from resistai.config import APP_NAME, RISK_COLORS

# --------------------------------------------------------------------------
# Design tokens
# --------------------------------------------------------------------------
COLORS: dict[str, str] = {
    "ink": "#0f2b46",
    "ink_muted": "#5b7183",
    "surface": "#ffffff",
    "surface_alt": "#f4f7fa",
    "border": "#dde5ec",
    "accent": "#0b6e99",
    "high": RISK_COLORS["High"],
    "moderate": RISK_COLORS["Moderate"],
    "low": RISK_COLORS["Low"],
}

#: Ordered, colour-blind-safe sequence for categorical series.
CATEGORICAL_SEQUENCE: list[str] = [
    "#0b6e99",
    "#2a9d57",
    "#f4a900",
    "#d7263d",
    "#6a4c93",
    "#5b7183",
]

#: Green -> amber -> red, used wherever a value encodes risk.
RISK_SCALE: list[list] = [
    [0.0, "#2a9d57"],
    [0.5, "#f4a900"],
    [1.0, "#d7263d"],
]

PLOTLY_TEMPLATE = "resistai"

_CUSTOM_CSS = f"""
<style>
    .block-container {{ padding-top: 2.2rem; max-width: 1400px; }}
    h1, h2, h3 {{ color: {COLORS["ink"]}; letter-spacing: -0.01em; }}

    /* KPI cards */
    div[data-testid="stMetric"] {{
        background: {COLORS["surface"]};
        border: 1px solid {COLORS["border"]};
        border-radius: 10px;
        padding: 14px 16px;
    }}
    div[data-testid="stMetricLabel"] p {{
        color: {COLORS["ink_muted"]};
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }}
    div[data-testid="stMetricValue"] {{ color: {COLORS["ink"]}; }}

    /* Tabs */
    button[data-baseweb="tab"] {{ font-weight: 600; }}

    /* Risk badge */
    .resistai-badge {{
        display: inline-block;
        padding: 6px 14px;
        border-radius: 999px;
        color: #ffffff;
        font-weight: 700;
        font-size: 0.85rem;
        letter-spacing: 0.02em;
    }}

    /* Driver row */
    .resistai-driver {{
        display: flex;
        justify-content: space-between;
        padding: 8px 12px;
        margin-bottom: 6px;
        background: {COLORS["surface_alt"]};
        border-left: 3px solid {COLORS["accent"]};
        border-radius: 4px;
        font-size: 0.9rem;
    }}
    .resistai-driver span:last-child {{
        color: {COLORS["ink_muted"]};
        font-variant-numeric: tabular-nums;
    }}

    footer, #MainMenu {{ visibility: hidden; }}
</style>
"""


def _register_template() -> None:
    """Register the shared Plotly template once per process."""
    if PLOTLY_TEMPLATE in pio.templates:
        return

    pio.templates[PLOTLY_TEMPLATE] = go.layout.Template(
        layout=go.Layout(
            font=dict(family="Inter, Segoe UI, sans-serif", size=13, color=COLORS["ink"]),
            paper_bgcolor=COLORS["surface"],
            plot_bgcolor=COLORS["surface"],
            colorway=CATEGORICAL_SEQUENCE,
            margin=dict(l=10, r=10, t=48, b=10),
            title=dict(font=dict(size=15), x=0.0, xanchor="left"),
            xaxis=dict(
                gridcolor=COLORS["border"],
                zerolinecolor=COLORS["border"],
                linecolor=COLORS["border"],
            ),
            yaxis=dict(
                gridcolor=COLORS["border"],
                zerolinecolor=COLORS["border"],
                linecolor=COLORS["border"],
            ),
            legend=dict(orientation="h", y=-0.18, x=0),
            hoverlabel=dict(font_size=12),
        )
    )


def apply_theme() -> None:
    """Install the page config, CSS and Plotly template. Call once, first."""
    _register_template()
    st.markdown(_CUSTOM_CSS, unsafe_allow_html=True)


def risk_badge(level: str) -> str:
    """
    Return the HTML for a coloured risk pill.

    `level` is validated against the known bands before being interpolated,
    so no caller can inject markup through this helper.
    """
    safe_level = level if level in RISK_COLORS else "Low"
    color = RISK_COLORS[safe_level]
    return f'<span class="resistai-badge" style="background:{color};">{safe_level} risk</span>'


def driver_row(label: str, points: int) -> str:
    """Return the HTML for one rule-driver row, with the label escaped."""
    from html import escape

    return (
        f'<div class="resistai-driver"><span>{escape(str(label))}</span>'
        f"<span>+{int(points)}</span></div>"
    )


def page_header() -> None:
    """Render the application title block."""
    st.markdown(f"## {APP_NAME}")
