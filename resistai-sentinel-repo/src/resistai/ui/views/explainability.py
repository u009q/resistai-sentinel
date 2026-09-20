"""Model transparency: global importance, local contribution and metrics."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from resistai.models.inference import AMRPredictor
from resistai.ui.theme import PLOTLY_TEMPLATE


def _local_contributions(patient: pd.Series, predictor: AMRPredictor) -> pd.DataFrame:
    """
    Estimate each feature's local effect by ablation.

    For every feature, re-score the patient with that feature set to zero and
    record the change in predicted probability. This is a cheap, honest
    approximation -- not SHAP, and labelled as such in the UI.
    """
    baseline, _ = predictor.predict_one(patient)
    rows: list[dict[str, object]] = []

    for feature in predictor.feature_names:
        if float(patient[feature]) == 0.0:
            continue
        counterfactual, _ = predictor.counterfactual(patient, {feature: 0})
        rows.append(
            {
                "feature": feature.replace("_", " ").title(),
                "contribution": round(baseline - counterfactual, 4),
            }
        )

    if not rows:
        return pd.DataFrame(columns=["feature", "contribution"])
    return pd.DataFrame(rows).sort_values("contribution")


def render(patient: pd.Series, predictor: AMRPredictor) -> None:
    """Render global and patient-level explanations side by side."""
    left, right = st.columns(2)

    with left:
        st.markdown("###### Global feature importance")
        st.caption("How much each signal drives the model across the whole cohort.")
        importance = predictor.feature_importance().copy()
        importance["feature"] = importance["feature"].str.replace("_", " ").str.title()
        importance = importance.sort_values("importance")

        fig = px.bar(
            importance,
            x="importance",
            y="feature",
            orientation="h",
            labels={"importance": "Importance", "feature": ""},
            text=importance["importance"].map(lambda v: f"{v:.2f}"),
        )
        fig.update_traces(marker_color="#0b6e99", textposition="outside", cliponaxis=False)
        fig.update_layout(template=PLOTLY_TEMPLATE, height=380)
        st.plotly_chart(fig, width="stretch")

    with right:
        st.markdown("###### This patient's drivers")
        st.caption(
            "Change in predicted risk if each active signal were removed "
            "(ablation estimate, not SHAP)."
        )
        contributions = _local_contributions(patient, predictor)
        if contributions.empty:
            st.info("This patient has no active risk signals to ablate.")
        else:
            fig = px.bar(
                contributions,
                x="contribution",
                y="feature",
                orientation="h",
                labels={"contribution": "Δ predicted risk", "feature": ""},
                text=contributions["contribution"].map(lambda v: f"{v:+.2f}"),
            )
            fig.update_traces(
                marker_color="#d7263d", textposition="outside", cliponaxis=False
            )
            fig.update_layout(template=PLOTLY_TEMPLATE, height=380)
            st.plotly_chart(fig, width="stretch")

    st.divider()
    st.markdown("###### Model card")

    metrics = predictor.metrics
    cols = st.columns(4)
    cols[0].metric("ROC AUC", f"{metrics.get('roc_auc', 0):.3f}")
    cols[1].metric("Average precision", f"{metrics.get('average_precision', 0):.3f}")
    cols[2].metric(
        "Brier score",
        f"{metrics.get('brier_score', 0):.3f}",
        help="Calibration error; lower is better.",
    )
    cols[3].metric("Positive rate", f"{metrics.get('positive_rate', 0):.1%}")

    st.caption(
        f"Trained {predictor.trained_at} on {metrics.get('n_train', 0):,} patients, "
        f"evaluated on {metrics.get('n_test', 0):,} held out. "
        "Synthetic cohort — metrics are illustrative, not clinical validation."
    )
