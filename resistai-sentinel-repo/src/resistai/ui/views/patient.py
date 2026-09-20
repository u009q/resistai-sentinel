"""Single-patient clinical panel."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from resistai.config import FEATURE_COLUMNS, RISK_COLORS
from resistai.models.inference import AMRPredictor
from resistai.scoring.notes import note_flags
from resistai.scoring.rules import MAX_RULE_SCORE, clinical_drivers, rule_level, rule_score
from resistai.ui.theme import PLOTLY_TEMPLATE, driver_row, risk_badge


def _gauge(probability: float, level: str) -> go.Figure:
    """A single-value gauge -- the one place a dial beats a bar."""
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%", "font": {"size": 34}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": RISK_COLORS[level], "thickness": 0.7},
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 30], "color": "#e8f5ed"},
                    {"range": [30, 55], "color": "#fdf3dd"},
                    {"range": [55, 100], "color": "#fbe4e8"},
                ],
            },
        )
    )
    fig.update_layout(template=PLOTLY_TEMPLATE, height=220, margin=dict(t=10, b=10))
    return fig


def render(patient: pd.Series, predictor: AMRPredictor) -> None:
    """Render the prediction, the rule score and the note findings."""
    probability = float(patient["risk_probability"])
    level = str(patient["risk_level"])

    score = rule_score(patient)
    rules_level = rule_level(score)

    left, middle, right = st.columns([2, 2, 3])

    with left:
        st.markdown("###### Model prediction")
        st.plotly_chart(_gauge(probability, level), width="stretch")
        st.markdown(risk_badge(level), unsafe_allow_html=True)

    with middle:
        st.markdown("###### Rule-based score")
        st.metric("Points", f"{score} / {MAX_RULE_SCORE}", delta=rules_level, delta_color="off")
        st.progress(score / MAX_RULE_SCORE)
        if rules_level != level:
            # Divergence between the two tracks is itself clinically useful.
            st.info(
                f"Rule engine says **{rules_level}**, model says **{level}**. "
                "Review manually."
            )
        else:
            st.success("Rule engine and model agree.")

    with right:
        st.markdown("###### Key risk drivers")
        drivers = clinical_drivers(patient)
        if drivers:
            for label, points in drivers:
                st.markdown(driver_row(label, points), unsafe_allow_html=True)
        else:
            st.markdown(
                "_No rule-based risk drivers fired for this patient._"
            )

    st.divider()

    notes_col, vitals_col = st.columns([3, 2])

    with notes_col:
        st.markdown("###### Clinician note")
        st.write(str(patient["doctor_note"]))
        flags = note_flags(patient["doctor_note"])
        if flags:
            for flag in flags:
                st.warning(flag, icon=":material/warning:")
        else:
            st.caption("No resistance-related phrases detected in the note.")

    with vitals_col:
        st.markdown("###### Clinical record")
        record = pd.DataFrame(
            {
                "Signal": [name.replace("_", " ").title() for name in FEATURE_COLUMNS],
                "Value": [patient[name] for name in FEATURE_COLUMNS],
            }
        )
        st.dataframe(record, width="stretch", hide_index=True)
