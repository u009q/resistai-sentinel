"""
ResistAI Sentinel — Streamlit entry point.

Run with:
    streamlit run app.py

This module does layout and wiring only. Every piece of domain logic lives
in `src/resistai/`, so it can be tested and reused without Streamlit.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make `src/` importable without requiring an editable install.
_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import streamlit as st  # noqa: E402

from resistai import __version__  # noqa: E402
from resistai.config import APP_NAME, APP_TAGLINE  # noqa: E402
from resistai.ui.state import bootstrap, selected_patient  # noqa: E402
from resistai.ui.theme import apply_theme, risk_badge  # noqa: E402
from resistai.ui.views import (  # noqa: E402
    actions,
    explainability,
    hospital,
    overview,
    patient as patient_view,
    simulator,
)

st.set_page_config(
    page_title=APP_NAME,
    page_icon=":material/health_and_safety:",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _sidebar(scored) -> str:
    """Render the sidebar and return the selected medical record number."""
    with st.sidebar:
        st.markdown(f"### {APP_NAME}")
        st.caption(APP_TAGLINE)
        st.divider()

        st.markdown("**Patient**")

        departments = ["All departments", *sorted(scored["department"].unique())]
        department = st.selectbox("Department", departments, index=0)

        pool = (
            scored
            if department == "All departments"
            else scored.loc[scored["department"] == department]
        )

        risk_filter = st.multiselect(
            "Risk level",
            options=["High", "Moderate", "Low"],
            default=["High", "Moderate", "Low"],
        )
        if risk_filter:
            pool = pool.loc[pool["risk_level"].isin(risk_filter)]

        if pool.empty:
            st.warning("No patients match these filters.")
            st.stop()

        # Highest-risk patients first: that is the triage order clinicians want.
        pool = pool.sort_values("risk_probability", ascending=False)
        options = pool["patient_uid"].tolist()

        patient_uid = st.selectbox(
            "Medical record",
            options=options,
            format_func=lambda uid: (
                f"{uid} · {pool.loc[pool['patient_uid'] == uid, 'risk_probability'].iloc[0]:.0%}"
            ),
        )

        st.divider()
        st.caption(f"{len(pool):,} patients in selection")
        st.caption(f"ResistAI Sentinel v{__version__}")
        st.caption("Synthetic data — decision support only.")

    return patient_uid


def main() -> None:
    """Compose the dashboard."""
    apply_theme()

    loaded = bootstrap()
    if loaded is None:
        return
    scored, predictor = loaded

    patient_uid = _sidebar(scored)
    record = selected_patient(scored, patient_uid)

    header, badge = st.columns([4, 1])
    with header:
        st.title(APP_NAME)
        st.caption(
            f"{record['patient_uid']} · {record['department']} · Room {record['room']}"
        )
    with badge:
        st.markdown("<div style='padding-top:28px'></div>", unsafe_allow_html=True)
        st.markdown(risk_badge(str(record["risk_level"])), unsafe_allow_html=True)

    tabs = st.tabs(
        [
            "Overview",
            "Patient",
            "Hospital map",
            "Explainability",
            "Simulator",
            "Actions",
        ]
    )

    with tabs[0]:
        overview.render(scored)
    with tabs[1]:
        patient_view.render(record, predictor)
    with tabs[2]:
        hospital.render(scored)
    with tabs[3]:
        explainability.render(record, predictor)
    with tabs[4]:
        simulator.render(record, predictor)
    with tabs[5]:
        actions.render(record, scored, predictor)


if __name__ == "__main__":
    main()
