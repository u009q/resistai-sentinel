"""Hospital-wide executive overview."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from resistai.config import DEPARTMENT_ORDER, RISK_COLORS
from resistai.features.hospital import department_summary
from resistai.ui.theme import PLOTLY_TEMPLATE


def render(scored: pd.DataFrame) -> None:
    """Render KPI cards plus cohort-level risk distribution."""
    st.subheader("Executive overview")

    high = int((scored["risk_level"] == "High").sum())
    moderate = int((scored["risk_level"] == "Moderate").sum())
    total = len(scored)

    kpi = st.columns(4)
    kpi[0].metric("Patients monitored", f"{total:,}")
    kpi[1].metric(
        "High risk", f"{high:,}", delta=f"{high / total:.1%} of cohort", delta_color="off"
    )
    kpi[2].metric("Moderate risk", f"{moderate:,}")
    kpi[3].metric("Mean predicted risk", f"{scored['risk_probability'].mean():.1%}")

    st.divider()

    left, right = st.columns([3, 2])

    with left:
        summary = department_summary(scored)
        # Keep a stable clinical ordering rather than whatever groupby returns.
        summary["department"] = pd.Categorical(
            summary["department"], categories=DEPARTMENT_ORDER, ordered=True
        )
        summary = summary.sort_values("department")

        fig = px.bar(
            summary,
            x="avg_risk",
            y="department",
            orientation="h",
            color="avg_risk",
            color_continuous_scale=[c[1] for c in _risk_scale()],
            range_color=(0.0, 1.0),
            title="Average predicted AMR risk by department",
            labels={"avg_risk": "Average risk", "department": ""},
            text=summary["avg_risk"].map(lambda v: f"{v:.0%}"),
        )
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_layout(
            template=PLOTLY_TEMPLATE, coloraxis_showscale=False, height=360
        )
        fig.update_xaxes(tickformat=".0%", range=[0, 1])
        st.plotly_chart(fig, width="stretch")

    with right:
        distribution = (
            scored["risk_level"]
            .value_counts()
            .reindex(["High", "Moderate", "Low"])
            .fillna(0)
            .reset_index()
        )
        distribution.columns = ["risk_level", "patients"]

        fig = px.bar(
            distribution,
            x="risk_level",
            y="patients",
            color="risk_level",
            color_discrete_map=RISK_COLORS,
            title="Cohort risk distribution",
            labels={"risk_level": "", "patients": "Patients"},
            text="patients",
        )
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_layout(
            template=PLOTLY_TEMPLATE, showlegend=False, height=360
        )
        st.plotly_chart(fig, width="stretch")

    with st.expander("Department detail"):
        st.dataframe(
            department_summary(scored),
            width="stretch",
            hide_index=True,
            column_config={
                "avg_risk": st.column_config.ProgressColumn(
                    "Average risk", format="%.2f", min_value=0.0, max_value=1.0
                ),
                "max_risk": st.column_config.NumberColumn("Peak risk", format="%.2f"),
                "high_risk_cases": st.column_config.NumberColumn("High-risk cases"),
                "patients": st.column_config.NumberColumn("Patients"),
            },
        )


def _risk_scale() -> list[list]:
    from resistai.ui.theme import RISK_SCALE

    return RISK_SCALE
