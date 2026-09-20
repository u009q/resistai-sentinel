"""Hospital topology: floor plan, room hotspots and the risk treemap."""

from __future__ import annotations

from html import escape

import pandas as pd
import plotly.express as px
import streamlit as st

from resistai.config import DEPARTMENT_ORDER, HOSPITAL_MAP
from resistai.features.hospital import department_summary, room_summary
from resistai.ui.theme import PLOTLY_TEMPLATE, RISK_SCALE

_FLOOR_PLAN_SLOTS: list[str] = ["ER", "ICU", "Infectious Diseases", "General Ward", "Outpatient"]


def _risk_color(value: float) -> str:
    """Map a risk value to the floor-plan fill colour."""
    if value >= 0.55:
        return "#d7263d"
    if value >= 0.30:
        return "#f4a900"
    return "#2a9d57"


def _floor_plan_svg(dept_risk: dict[str, float]) -> str:
    """
    Build the floor-plan SVG.

    Only department names from `HOSPITAL_MAP` and numeric risk values reach
    the markup, and both are escaped -- the original notebook interpolated
    values straight into an `unsafe_allow_html` block.
    """
    block_width, gap, height, top = 168, 16, 96, 24
    blocks: list[str] = []

    for index, department in enumerate(_FLOOR_PLAN_SLOTS):
        if department not in HOSPITAL_MAP:
            continue
        risk = float(dept_risk.get(department, 0.0))
        x = index * (block_width + gap)
        label = escape(department)
        blocks.append(
            f'<rect x="{x}" y="{top}" width="{block_width}" height="{height}" '
            f'rx="10" fill="{_risk_color(risk)}" opacity="0.92"/>'
            f'<text x="{x + block_width / 2}" y="{top + 40}" text-anchor="middle" '
            f'fill="#ffffff" font-size="14" font-weight="700" '
            f'font-family="Inter, Segoe UI, sans-serif">{label}</text>'
            f'<text x="{x + block_width / 2}" y="{top + 64}" text-anchor="middle" '
            f'fill="#ffffff" font-size="13" opacity="0.9" '
            f'font-family="Inter, Segoe UI, sans-serif">{risk:.0%} avg risk</text>'
        )

    total_width = len(_FLOOR_PLAN_SLOTS) * (block_width + gap) - gap
    corridor_y = top + height + 14
    blocks.append(
        f'<rect x="0" y="{corridor_y}" width="{total_width}" height="30" rx="6" '
        f'fill="#eef3f7"/>'
        f'<text x="{total_width / 2}" y="{corridor_y + 20}" text-anchor="middle" '
        f'fill="#5b7183" font-size="12" '
        f'font-family="Inter, Segoe UI, sans-serif">Main corridor</text>'
    )

    return (
        f'<svg width="100%" viewBox="0 0 {total_width} {corridor_y + 40}" '
        f'xmlns="http://www.w3.org/2000/svg" role="img" '
        f'aria-label="Hospital floor plan coloured by average AMR risk">'
        f'{"".join(blocks)}</svg>'
    )


def render(scored: pd.DataFrame) -> None:
    """Render the floor plan, treemap and room hotspot table."""
    dept_summary = department_summary(scored)
    dept_risk = dict(zip(dept_summary["department"], dept_summary["avg_risk"]))

    st.subheader("Hospital floor plan")
    st.caption("Departments coloured by average predicted AMR risk.")
    st.markdown(_floor_plan_svg(dept_risk), unsafe_allow_html=True)

    st.divider()

    left, right = st.columns([3, 2])

    with left:
        st.markdown("###### Risk concentration by department and room")
        treemap_source = scored.copy()
        treemap_source["department"] = pd.Categorical(
            treemap_source["department"], categories=DEPARTMENT_ORDER, ordered=True
        )
        fig = px.treemap(
            treemap_source,
            path=["department", "room"],
            values="risk_probability",
            color="risk_probability",
            color_continuous_scale=[c[1] for c in RISK_SCALE],
            range_color=(0.0, 1.0),
        )
        fig.update_layout(template=PLOTLY_TEMPLATE, height=420)
        fig.update_traces(
            hovertemplate="<b>%{label}</b><br>Total risk mass: %{value:.1f}<extra></extra>"
        )
        st.plotly_chart(fig, width="stretch")

    with right:
        st.markdown("###### Room hotspots")
        rooms = room_summary(scored).head(12)
        st.dataframe(
            rooms,
            width="stretch",
            hide_index=True,
            column_config={
                "department": "Department",
                "room": "Room",
                "avg_risk": st.column_config.ProgressColumn(
                    "Average risk", format="%.2f", min_value=0.0, max_value=1.0
                ),
                "patients": st.column_config.NumberColumn("Patients"),
            },
        )
