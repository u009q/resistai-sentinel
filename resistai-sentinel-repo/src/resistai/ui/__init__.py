"""Streamlit presentation layer."""

from resistai.ui.state import get_dataset, get_predictor, get_scored_cohort
from resistai.ui.theme import PLOTLY_TEMPLATE, apply_theme, risk_badge

__all__ = [
    "PLOTLY_TEMPLATE",
    "apply_theme",
    "get_dataset",
    "get_predictor",
    "get_scored_cohort",
    "risk_badge",
]
