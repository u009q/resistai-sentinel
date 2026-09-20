"""
Patient PDF report generation.

The report is rendered entirely into an in-memory buffer and handed to the
caller as bytes. The original notebook wrote to `/tmp/{patient_uid}_...pdf`
using a value that came from the UI -- a path built from untrusted input,
plus a file left on disk containing clinical data. Neither is acceptable in
a system that might one day see real records.
"""

from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO
from typing import Iterable, Mapping

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas

from resistai.config import APP_NAME, RISK_COLORS

_PAGE_WIDTH, _PAGE_HEIGHT = A4
_MARGIN = 20 * mm
_LINE_HEIGHT = 6 * mm


def _sanitize(value: object, limit: int = 300) -> str:
    """Flatten a value to a single printable line of bounded length."""
    text = " ".join(str(value).split())
    return text[:limit]


def _wrap(text: str, width: int = 88) -> list[str]:
    """Greedy word wrap -- enough for a single-column clinical summary."""
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) <= width:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


class _ReportWriter:
    """Small helper that tracks the vertical cursor and handles page breaks."""

    def __init__(self, pdf: canvas.Canvas) -> None:
        self._pdf = pdf
        self._y = _PAGE_HEIGHT - _MARGIN

    def _ensure_space(self, needed: float = _LINE_HEIGHT) -> None:
        if self._y - needed < _MARGIN:
            self._pdf.showPage()
            self._y = _PAGE_HEIGHT - _MARGIN

    def heading(self, text: str, size: int = 16) -> None:
        self._ensure_space(size + 4)
        self._pdf.setFillColor(colors.HexColor("#0f2b46"))
        self._pdf.setFont("Helvetica-Bold", size)
        self._pdf.drawString(_MARGIN, self._y, text)
        self._y -= size + 4

    def subheading(self, text: str) -> None:
        self._ensure_space()
        self._pdf.setFillColor(colors.HexColor("#0f2b46"))
        self._pdf.setFont("Helvetica-Bold", 11)
        self._pdf.drawString(_MARGIN, self._y, text)
        self._y -= _LINE_HEIGHT

    def body(self, text: str, indent: float = 0.0) -> None:
        self._pdf.setFillColor(colors.black)
        self._pdf.setFont("Helvetica", 10)
        for line in _wrap(text):
            self._ensure_space()
            self._pdf.drawString(_MARGIN + indent, self._y, line)
            self._y -= _LINE_HEIGHT

    def badge(self, label: str, value: str, hex_color: str) -> None:
        self._ensure_space(10 * mm)
        self._pdf.setFillColor(colors.HexColor(hex_color))
        self._pdf.roundRect(_MARGIN, self._y - 7 * mm, 60 * mm, 9 * mm, 2 * mm, fill=1, stroke=0)
        self._pdf.setFillColor(colors.white)
        self._pdf.setFont("Helvetica-Bold", 10)
        self._pdf.drawString(_MARGIN + 4 * mm, self._y - 4.5 * mm, f"{label}: {value}")
        self._y -= 13 * mm

    def rule(self) -> None:
        self._ensure_space(4 * mm)
        self._pdf.setStrokeColor(colors.HexColor("#d4dde6"))
        self._pdf.line(_MARGIN, self._y, _PAGE_WIDTH - _MARGIN, self._y)
        self._y -= 5 * mm

    def spacer(self, height: float = 3 * mm) -> None:
        self._y -= height


def build_patient_report(
    patient: Mapping[str, object],
    probability: float,
    risk_level: str,
    drivers: Iterable[tuple[str, int]],
    recommendation: str,
    note_flags: Iterable[str] = (),
) -> bytes:
    """
    Render a one-page AMR risk report.

    Args:
        patient: The patient row (mapping of column -> value).
        probability: Model-predicted probability of confirmed AMR.
        risk_level: Risk band derived from `probability`.
        drivers: `(label, points)` pairs from the rule engine.
        recommendation: Clinical recommendation text.
        note_flags: Findings extracted from the clinician note.

    Returns:
        The PDF document as bytes, ready for `st.download_button`.
    """
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    pdf.setTitle(f"{APP_NAME} - AMR Risk Report")
    writer = _ReportWriter(pdf)

    writer.heading(f"{APP_NAME} - AMR Risk Report")
    writer.body(
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
    )
    writer.rule()

    writer.subheading("Patient")
    writer.body(f"Medical record: {_sanitize(patient.get('patient_uid'), 40)}")
    writer.body(f"Department: {_sanitize(patient.get('department'), 40)}")
    writer.body(f"Room: {_sanitize(patient.get('room'), 40)}")
    writer.body(f"Length of stay: {_sanitize(patient.get('length_of_stay_days'), 10)} days")
    writer.spacer()

    writer.badge(
        "Predicted AMR risk",
        f"{risk_level} ({probability:.0%})",
        RISK_COLORS.get(risk_level, "#0f2b46"),
    )

    writer.subheading("Key risk drivers")
    driver_list = list(drivers)
    if driver_list:
        for label, points in driver_list:
            writer.body(f"- {label}  (+{points})", indent=4 * mm)
    else:
        writer.body("- No major rule-based risk drivers identified.", indent=4 * mm)
    writer.spacer()

    flags = list(note_flags)
    if flags:
        writer.subheading("Clinician note findings")
        for flag in flags:
            writer.body(f"- {_sanitize(flag, 120)}", indent=4 * mm)
        writer.spacer()

    writer.subheading("Recommendation")
    writer.body(_sanitize(recommendation, 600))
    writer.spacer()

    writer.rule()
    writer.body(
        "Decision support only. This report does not replace clinical judgement, "
        "microbiological culture, or local antimicrobial stewardship policy."
    )

    pdf.showPage()
    pdf.save()
    return buffer.getvalue()
