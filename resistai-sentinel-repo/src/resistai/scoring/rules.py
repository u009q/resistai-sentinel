"""
Transparent rule-based AMR scoring.

This runs alongside the ML model rather than inside it. Clinicians can audit
a points table; they cannot audit 300 decision trees. When the two tracks
disagree, that disagreement is itself a signal worth surfacing.
"""

from __future__ import annotations

from typing import Mapping

#: Points awarded per clinical signal. Sum of all weights = 10.
RULE_WEIGHTS: dict[str, int] = {
    "reserved_abx_used": 3,
    "broad_spectrum_used": 2,
    "icu_admission": 2,
    "prior_hospitalization": 1,
    "antibiotic_switches": 1,   # awarded when >= 1 switch
    "active_infection": 1,      # fever AND elevated WBC
}

MAX_RULE_SCORE: int = sum(RULE_WEIGHTS.values())

#: Human-readable justification for each rule that fires.
_DRIVER_LABELS: dict[str, str] = {
    "reserved_abx_used": "Reserved (last-line) antibiotic exposure",
    "broad_spectrum_used": "Broad-spectrum antibiotic therapy",
    "icu_admission": "ICU admission",
    "prior_hospitalization": "Prior hospitalization within 90 days",
    "antibiotic_switches": "Multiple antibiotic switches",
    "active_infection": "Fever with elevated WBC (active infection)",
}


def _fired_rules(row: Mapping[str, object]) -> list[str]:
    """Return the names of the rules that fire for this patient."""
    fired: list[str] = []
    if row.get("reserved_abx_used"):
        fired.append("reserved_abx_used")
    if row.get("broad_spectrum_used"):
        fired.append("broad_spectrum_used")
    if row.get("icu_admission"):
        fired.append("icu_admission")
    if row.get("prior_hospitalization"):
        fired.append("prior_hospitalization")
    if int(row.get("antibiotic_switches", 0) or 0) >= 1:
        fired.append("antibiotic_switches")
    if row.get("fever") and row.get("wbc_high"):
        fired.append("active_infection")
    return fired


def rule_score(row: Mapping[str, object]) -> int:
    """Total rule points for a patient, in the range [0, MAX_RULE_SCORE]."""
    return sum(RULE_WEIGHTS[name] for name in _fired_rules(row))


def rule_level(score: int) -> str:
    """Map a rule score to a risk band."""
    if score >= 6:
        return "High"
    if score >= 3:
        return "Moderate"
    return "Low"


def clinical_drivers(row: Mapping[str, object]) -> list[tuple[str, int]]:
    """
    Explain the rule score.

    Returns:
        `(label, points)` pairs for every rule that fired, heaviest first.
    """
    drivers = [(_DRIVER_LABELS[name], RULE_WEIGHTS[name]) for name in _fired_rules(row)]
    return sorted(drivers, key=lambda pair: pair[1], reverse=True)
