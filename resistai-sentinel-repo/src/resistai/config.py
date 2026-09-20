"""
Central configuration.

All paths, thresholds and tunables live here so that no module hardcodes
a magic number. Secrets are read from the environment only -- never from
source control.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
# PROJECT_ROOT = <repo>/  (this file lives at <repo>/src/resistai/config.py)
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

DATA_DIR: Path = Path(os.getenv("RESISTAI_DATA_DIR", PROJECT_ROOT / "data"))
MODEL_DIR: Path = Path(os.getenv("RESISTAI_MODEL_DIR", PROJECT_ROOT / "models"))

DATASET_PATH: Path = DATA_DIR / "amr_dataset.csv"
MODEL_PATH: Path = MODEL_DIR / "amr_model.joblib"

# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
RANDOM_SEED: int = 42

# --------------------------------------------------------------------------
# Feature contract
# --------------------------------------------------------------------------
# NOTE: `amr_risk_prob` is deliberately NOT a feature.
# In the original notebook it was both an input feature and the source of the
# target label, which leaked the answer into the model and inflated accuracy
# to ~100%. The model now learns from clinical signals only.
FEATURE_COLUMNS: list[str] = [
    "broad_spectrum_used",
    "reserved_abx_used",
    "antibiotic_switches",
    "icu_admission",
    "fever",
    "wbc_high",
    "prior_hospitalization",
    "length_of_stay_days",
]

TARGET_COLUMN: str = "amr_confirmed"
ID_COLUMN: str = "patient_uid"
NOTE_COLUMN: str = "doctor_note"


# --------------------------------------------------------------------------
# Clinical thresholds
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class RiskThresholds:
    """Probability cut-offs that map a predicted probability to a risk band."""

    high: float = 0.55
    moderate: float = 0.30

    def level(self, probability: float) -> str:
        if probability >= self.high:
            return "High"
        if probability >= self.moderate:
            return "Moderate"
        return "Low"


RISK_THRESHOLDS = RiskThresholds()

RISK_COLORS: dict[str, str] = {
    "High": "#d7263d",
    "Moderate": "#f4a900",
    "Low": "#2a9d57",
}


# --------------------------------------------------------------------------
# Hospital topology
# --------------------------------------------------------------------------
HOSPITAL_MAP: dict[str, list[str]] = {
    "ER": ["ER-1", "ER-2", "ER-Corridor"],
    "ICU": ["ICU-1", "ICU-2", "ICU-Corridor"],
    "Infectious Diseases": ["ID-1", "ID-2", "ID-Corridor"],
    "General Ward": ["GW-1", "GW-2", "GW-Corridor"],
    "Outpatient": ["OP-1", "OP-2", "OP-Corridor"],
}

DEPARTMENT_ORDER: list[str] = list(HOSPITAL_MAP.keys())


# --------------------------------------------------------------------------
# Dataset generation
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class GeneratorSettings:
    """Parameters for the synthetic cohort generator."""

    n_patients: int = 4000
    seed: int = RANDOM_SEED
    # Prevalence of each clinical signal in the cohort
    p_broad_spectrum: float = 0.40
    p_reserved_given_broad: float = 0.20
    p_reserved_baseline: float = 0.05
    p_icu: float = 0.20
    p_fever: float = 0.25
    p_wbc_high: float = 0.22
    p_prior_hospitalization: float = 0.30
    # Log-odds contributions used to build the latent AMR probability
    intercept: float = -2.40
    weights: dict[str, float] = field(
        default_factory=lambda: {
            "broad_spectrum_used": 0.85,
            "reserved_abx_used": 1.35,
            "antibiotic_switches": 0.45,
            "icu_admission": 0.95,
            "fever": 0.35,
            "wbc_high": 0.40,
            "prior_hospitalization": 0.60,
            "length_of_stay_days": 0.06,
        }
    )


GENERATOR = GeneratorSettings()


# --------------------------------------------------------------------------
# Model training
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class TrainingSettings:
    n_estimators: int = 300
    max_depth: int | None = 8
    min_samples_leaf: int = 20
    class_weight: str | None = "balanced"
    test_size: float = 0.2
    seed: int = RANDOM_SEED


TRAINING = TrainingSettings()


# --------------------------------------------------------------------------
# Application metadata
# --------------------------------------------------------------------------
APP_NAME: str = "ResistAI Sentinel"
APP_TAGLINE: str = "Explainable AMR Risk Intelligence for Hospitals"
APP_ICON: str = "shield"


def ensure_directories() -> None:
    """Create the runtime directories if they do not exist yet."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
