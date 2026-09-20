"""
Dataset schema and validation.

A single source of truth for what a valid AMR cohort file looks like.
Validating on load turns a silent `KeyError` deep inside the UI into one
clear error message at the boundary.
"""

from __future__ import annotations

import pandas as pd

from resistai.config import FEATURE_COLUMNS, ID_COLUMN, NOTE_COLUMN, TARGET_COLUMN

#: Columns every generated dataset must contain.
DATASET_COLUMNS: list[str] = [
    ID_COLUMN,
    "patient_id",
    *FEATURE_COLUMNS,
    "amr_risk_prob",      # latent ground-truth probability (simulation only)
    "amr_risk_level",     # band derived from the latent probability
    TARGET_COLUMN,        # observed binary outcome -- the training target
    NOTE_COLUMN,
    "department",
    "room",
]

#: Columns that must be non-negative integers.
_INTEGER_COLUMNS: list[str] = [
    "broad_spectrum_used",
    "reserved_abx_used",
    "antibiotic_switches",
    "icu_admission",
    "fever",
    "wbc_high",
    "prior_hospitalization",
    "length_of_stay_days",
    TARGET_COLUMN,
]


class DatasetValidationError(ValueError):
    """Raised when a dataset does not satisfy the expected schema."""


def validate_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Check the dataframe against the expected schema.

    Args:
        df: The loaded cohort.

    Returns:
        The same dataframe, unchanged, when valid.

    Raises:
        DatasetValidationError: If columns are missing or values are invalid.
    """
    missing = [col for col in DATASET_COLUMNS if col not in df.columns]
    if missing:
        raise DatasetValidationError(
            f"Dataset is missing required columns: {', '.join(missing)}. "
            "Regenerate it with `python scripts/generate_data.py`."
        )

    if df.empty:
        raise DatasetValidationError("Dataset is empty.")

    if df[ID_COLUMN].duplicated().any():
        raise DatasetValidationError(f"Duplicate values found in '{ID_COLUMN}'.")

    for column in _INTEGER_COLUMNS:
        series = pd.to_numeric(df[column], errors="coerce")
        if series.isna().any() or (series < 0).any():
            raise DatasetValidationError(
                f"Column '{column}' must contain non-negative numeric values."
            )

    if not df["amr_risk_prob"].between(0.0, 1.0).all():
        raise DatasetValidationError("'amr_risk_prob' must lie within [0, 1].")

    return df
