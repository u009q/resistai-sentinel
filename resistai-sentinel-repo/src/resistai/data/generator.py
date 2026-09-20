"""
Synthetic AMR cohort generator.

The cohort is simulated from an explicit logistic model: each clinical
signal contributes a known log-odds weight, the latent probability is the
sigmoid of that sum, and the observed outcome is a Bernoulli draw from it.

This matters. In the original notebook the label was a hard threshold on
the same probability that was also fed to the model as a feature, so the
classifier only had to learn `prob > 0.45` and scored ~100% -- a textbook
target leak. Sampling the outcome instead leaves genuine, irreducible
uncertainty for the model to work against.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from resistai.config import (
    FEATURE_COLUMNS,
    GENERATOR,
    ID_COLUMN,
    RISK_THRESHOLDS,
    TARGET_COLUMN,
    GeneratorSettings,
)
from resistai.features.hospital import assign_departments, assign_rooms, make_patient_uid

_BASE_NOTE_FRAGMENTS: tuple[str, ...] = (
    "Patient under evaluation.",
    "Vitals monitored.",
    "Empiric antibiotics started.",
)

_RESISTANCE_NOTE_FRAGMENTS: tuple[str, ...] = (
    "No improvement despite antibiotics.",
    "Suspected antimicrobial resistance.",
    "Empiric therapy failed.",
    "Recommend culture and antibiotic review.",
    "Escalation discussed with infectious diseases team.",
)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable logistic function."""
    return 1.0 / (1.0 + np.exp(-x))


def _build_notes(
    resistance_flag: np.ndarray, rng: np.random.Generator
) -> list[str]:
    """Compose a free-text clinician note for each patient."""
    picks = rng.integers(0, len(_RESISTANCE_NOTE_FRAGMENTS), size=len(resistance_flag))
    base = " ".join(_BASE_NOTE_FRAGMENTS)
    return [
        f"{base} {_RESISTANCE_NOTE_FRAGMENTS[pick]}" if flag else base
        for flag, pick in zip(resistance_flag, picks)
    ]


def generate_cohort(settings: GeneratorSettings | None = None) -> pd.DataFrame:
    """
    Generate a synthetic patient cohort with AMR risk labels.

    Args:
        settings: Generator parameters. Defaults to `config.GENERATOR`.

    Returns:
        A dataframe conforming to `data.schema.DATASET_COLUMNS`.
    """
    cfg = settings or GENERATOR
    rng = np.random.default_rng(cfg.seed)
    n = cfg.n_patients

    broad = rng.random(n) < cfg.p_broad_spectrum
    reserved_prob = np.where(broad, cfg.p_reserved_given_broad, cfg.p_reserved_baseline)
    reserved = rng.random(n) < reserved_prob
    switches = rng.poisson(1.0, size=n)
    icu = rng.random(n) < cfg.p_icu
    fever = rng.random(n) < cfg.p_fever
    wbc_high = rng.random(n) < cfg.p_wbc_high
    prior_admission = rng.random(n) < cfg.p_prior_hospitalization
    # Length of stay is longer for ICU patients; clipped to a plausible range.
    length_of_stay = np.clip(
        rng.gamma(shape=2.0, scale=2.5, size=n) + icu * 4.0, 1.0, 45.0
    ).round(0)

    frame = pd.DataFrame(
        {
            "broad_spectrum_used": broad.astype(int),
            "reserved_abx_used": reserved.astype(int),
            "antibiotic_switches": switches.astype(int),
            "icu_admission": icu.astype(int),
            "fever": fever.astype(int),
            "wbc_high": wbc_high.astype(int),
            "prior_hospitalization": prior_admission.astype(int),
            "length_of_stay_days": length_of_stay.astype(int),
        }
    )

    # Latent risk = sigmoid(intercept + sum of weighted signals)
    log_odds = np.full(n, cfg.intercept, dtype=float)
    for column, weight in cfg.weights.items():
        log_odds += weight * frame[column].to_numpy(dtype=float)
    # Interaction: fever together with elevated WBC signals active infection.
    log_odds += 0.50 * (frame["fever"] & frame["wbc_high"]).to_numpy(dtype=float)

    probability = _sigmoid(log_odds)
    outcome = (rng.random(n) < probability).astype(int)

    frame["patient_id"] = [f"P{i + 1:05d}" for i in range(n)]
    frame[ID_COLUMN] = [make_patient_uid(pid) for pid in frame["patient_id"]]
    frame["amr_risk_prob"] = probability.round(4)
    frame["amr_risk_level"] = [RISK_THRESHOLDS.level(p) for p in probability]
    frame[TARGET_COLUMN] = outcome
    frame["doctor_note"] = _build_notes(outcome.astype(bool), rng)

    frame["department"] = assign_departments(frame)
    frame["room"] = assign_rooms(frame)

    ordered = [
        ID_COLUMN,
        "patient_id",
        *FEATURE_COLUMNS,
        "amr_risk_prob",
        "amr_risk_level",
        TARGET_COLUMN,
        "doctor_note",
        "department",
        "room",
    ]
    return frame[ordered]
