"""
Keyword screening of free-text clinician notes.

Intentionally simple and auditable: a curated keyword list, matched against
a normalised copy of the note. A transformer would score better and explain
worse, and the note is a supporting signal here, not the decision.
"""

from __future__ import annotations

import re

#: Phrase -> clinical meaning surfaced to the user.
RESISTANCE_KEYWORDS: dict[str, str] = {
    "no improvement": "No clinical improvement documented",
    "suspected antimicrobial resistance": "Resistance explicitly suspected",
    "empiric therapy failed": "Empiric therapy documented as failed",
    "culture": "Culture requested or pending",
    "review antibiotic": "Antibiotic review recommended",
    "antibiotic review": "Antibiotic review recommended",
    "escalation": "Therapy escalation discussed",
}

_WHITESPACE = re.compile(r"\s+")


def _normalize(note: str) -> str:
    """Lowercase and collapse whitespace so matching is robust to formatting."""
    return _WHITESPACE.sub(" ", str(note or "")).strip().lower()


def note_flags(note: str) -> list[str]:
    """Return the clinical meanings of every keyword found in the note."""
    normalized = _normalize(note)
    seen: list[str] = []
    for keyword, meaning in RESISTANCE_KEYWORDS.items():
        if keyword in normalized and meaning not in seen:
            seen.append(meaning)
    return seen


def note_suggests_resistance(note: str) -> bool:
    """True when the note contains at least one resistance-related phrase."""
    return bool(note_flags(note))
