"""What-if simulator and projected risk trajectory."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from resistai.config import RISK_COLORS
from resistai.models.inference import AMRPredictor
from resistai.ui.theme import PLOTLY_TEMPLATE, risk_badge


def render(patient: pd.Series, predictor: AMRPredictor) -> None:
    """Let the user toggle interventions and see the model respond."""
    baseline_probability = float(patient["risk_probability"])

    st.subheader("What-if simulator")
    st.caption(
        "Adjust the intervention below to see how the model's predicted risk "
        "responds. Nothing is written back to the cohort."
    )

    controls = st.columns(4)
    broad = controls[0].toggle(
        "Broad-spectrum antibiotic", value=bool(patient["broad_spectrum_used"])
    )
    reserved = controls[1].toggle(
        "Reserved antibiotic", value=bool(patient["reserved_abx_used"])
    )
    icu = controls[2].toggle("ICU admission", value=bool(patient["icu_admission"]))
    switches = controls[3].number_input(
        "Antibiotic switches",
        min_value=0,
        max_value=10,
        value=int(patient["antibiotic_switches"]),
        step=1,
    )

    stay = st.slider(
        "Projected length of stay (days)",
        min_value=1,
        max_value=45,
        value=int(patient["length_of_stay_days"]),
    )

    overrides = {
        "broad_spectrum_used": int(broad),
        "reserved_abx_used": int(reserved),
        "icu_admission": int(icu),
        "antibiotic_switches": int(switches),
        "length_of_stay_days": int(stay),
    }
    simulated_probability, simulated_level = predictor.counterfactual(patient, overrides)
    delta = simulated_probability - baseline_probability

    st.divider()

    left, right = st.columns([2, 3])

    with left:
        st.metric(
            "Simulated risk",
            f"{simulated_probability:.1%}",
            delta=f"{delta:+.1%} vs current",
            delta_color="inverse",
        )
        st.markdown(risk_badge(simulated_level), unsafe_allow_html=True)

        if delta <= -0.05:
            st.success("This intervention materially lowers predicted risk.")
        elif delta >= 0.05:
            st.error("This change materially raises predicted risk.")
        else:
            st.info("Predicted risk is largely unchanged.")

    with right:
        st.markdown("###### Projected trajectory")
        st.caption(
            "The model's response as the admission lengthens — a projection, "
            "not a time-series forecast."
        )
        trajectory = predictor.risk_trajectory({**patient.to_dict(), **overrides}, days=7)
        frame = pd.DataFrame(
            {"day": range(1, len(trajectory) + 1), "risk": list(trajectory)}
        )

        fig = px.line(
            frame,
            x="day",
            y="risk",
            markers=True,
            labels={"day": "Day of admission", "risk": "Predicted risk"},
        )
        fig.update_traces(line_color="#0b6e99", line_width=2.5)
        fig.add_hline(
            y=0.55,
            line_dash="dot",
            line_color=RISK_COLORS["High"],
            annotation_text="High-risk threshold",
            annotation_position="top left",
        )
        fig.update_layout(template=PLOTLY_TEMPLATE, height=320)
        fig.update_yaxes(tickformat=".0%", range=[0, 1])
        st.plotly_chart(fig, width="stretch")
