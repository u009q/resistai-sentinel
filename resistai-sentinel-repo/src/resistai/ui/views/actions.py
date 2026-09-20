"""Infection-control actions, recommendations and the PDF export."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from resistai.features.hospital import room_summary
from resistai.models.inference import AMRPredictor
from resistai.reporting.pdf import build_patient_report
from resistai.scoring.notes import note_flags
from resistai.scoring.rules import clinical_drivers

_RECOMMENDATIONS: dict[str, str] = {
    "High": (
        "High likelihood of antimicrobial resistance. Initiate contact isolation, "
        "obtain culture and sensitivity before any further escalation, notify the "
        "infection control team, and schedule an antimicrobial stewardship review "
        "within 24 hours."
    ),
    "Moderate": (
        "Moderate AMR risk. Continue close monitoring, review the current antibiotic "
        "regimen against local antibiogram data, and repeat inflammatory markers "
        "within 24 hours."
    ),
    "Low": (
        "Low AMR risk. Standard precautions apply. Continue the current treatment plan "
        "and reassess if the clinical picture changes."
    ),
}

_CHECKLISTS: dict[str, list[str]] = {
    "High": [
        "Isolate the patient (contact precautions)",
        "Obtain culture and sensitivity",
        "Notify infection control team",
        "Antimicrobial stewardship review within 24h",
        "Environmental disinfection of the room",
    ],
    "Moderate": [
        "Close clinical monitoring",
        "Review antibiotic regimen against local antibiogram",
        "Repeat inflammatory markers within 24h",
    ],
    "Low": [
        "Standard precautions",
        "Continue current treatment plan",
    ],
}


def render(patient: pd.Series, scored: pd.DataFrame, predictor: AMRPredictor) -> None:
    """Render the action checklist, unit-level alerts and the report export."""
    level = str(patient["risk_level"])
    probability = float(patient["risk_probability"])
    recommendation = _RECOMMENDATIONS[level]

    left, right = st.columns([3, 2])

    with left:
        st.markdown("###### Clinical recommendation")
        renderer = {"High": st.error, "Moderate": st.warning, "Low": st.success}[level]
        renderer(recommendation)

        st.markdown("###### Action checklist")
        for index, item in enumerate(_CHECKLISTS[level]):
            st.checkbox(item, key=f"action_{patient['patient_uid']}_{index}")

    with right:
        st.markdown("###### Unit-level alerts")

        department = str(patient["department"])
        peers = scored.loc[scored["department"] == department]
        spread = (peers["risk_level"] == "High").mean() if len(peers) else 0.0
        st.metric(
            f"High-risk share in {department}",
            f"{spread:.1%}",
            help="Proportion of patients in this department the model scores High.",
        )

        hotspots = room_summary(scored)
        critical = hotspots.loc[hotspots["avg_risk"] >= 0.55]
        if not critical.empty:
            st.error(f"{len(critical)} room(s) above the high-risk threshold.")
            for _, room in critical.head(5).iterrows():
                st.caption(f"{room['room']} — {room['avg_risk']:.0%} average risk")
        else:
            st.success("No room is currently above the high-risk threshold.")

    st.divider()
    st.markdown("###### Export")

    # The PDF is built in memory; nothing touches the filesystem.
    pdf_bytes = build_patient_report(
        patient=patient.to_dict(),
        probability=probability,
        risk_level=level,
        drivers=clinical_drivers(patient),
        recommendation=recommendation,
        note_flags=note_flags(patient["doctor_note"]),
    )

    st.download_button(
        "Download AMR risk report (PDF)",
        data=pdf_bytes,
        file_name=f"{patient['patient_uid']}_AMR_report.pdf",
        mime="application/pdf",
        type="primary",
    )
