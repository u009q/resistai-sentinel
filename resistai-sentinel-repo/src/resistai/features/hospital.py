"""
Hospital topology: patient identifiers, departments, rooms and aggregates.

Every identifier here is *deterministic*. The original notebook used
`uuid.uuid4()` for the medical record number, which produced a new set of
IDs on every Streamlit rerun and broke the patient selector, and
`hash(str)` for room assignment, which is salted per process and therefore
unstable between runs. Both are now derived from a stable SHA-256 digest.
"""

from __future__ import annotations

import hashlib

import pandas as pd

from resistai.config import HOSPITAL_MAP, TARGET_COLUMN


def _stable_digest(value: str) -> int:
    """Return a process-independent integer digest for `value`."""
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest(), 16)


def make_patient_uid(patient_id: str) -> str:
    """
    Build a deterministic, pseudonymous medical record number.

    The digest is one-way, so the MRN cannot be reversed back to the source
    identifier -- the property you want if this ever points at real records.
    """
    digest = hashlib.sha256(patient_id.encode("utf-8")).hexdigest()
    return f"MRN-{digest[:8].upper()}"


def assign_departments(df: pd.DataFrame) -> pd.Series:
    """
    Route each patient to a department from their clinical state.

    Ordering matters: ICU admission dominates, then confirmed resistance,
    then acute infection markers.
    """
    icu = df["icu_admission"].astype(bool)
    confirmed = df.get(TARGET_COLUMN, pd.Series(0, index=df.index)).astype(bool)
    acute = (df["fever"].astype(bool) & df["wbc_high"].astype(bool)) | (
        df["antibiotic_switches"] >= 2
    )

    department = pd.Series("Outpatient", index=df.index, dtype="object")
    department[acute] = "General Ward"
    department[confirmed] = "Infectious Diseases"
    department[icu] = "ICU"
    department[~icu & ~confirmed & ~acute & df["broad_spectrum_used"].astype(bool)] = "ER"
    return department


def assign_rooms(df: pd.DataFrame) -> pd.Series:
    """Place each patient in a room within their department, deterministically."""

    def _room(row: pd.Series) -> str:
        rooms = HOSPITAL_MAP.get(row["department"], HOSPITAL_MAP["ER"])
        return rooms[_stable_digest(str(row["patient_id"])) % len(rooms)]

    return df.apply(_room, axis=1)


def department_summary(df: pd.DataFrame, risk_column: str = "risk_probability") -> pd.DataFrame:
    """
    Aggregate risk by department.

    Args:
        df: Cohort with a per-patient risk column.
        risk_column: Name of the column holding the predicted probability.

    Returns:
        One row per department, sorted by average risk descending.
    """
    summary = (
        df.groupby("department", observed=True)
        .agg(
            avg_risk=(risk_column, "mean"),
            max_risk=(risk_column, "max"),
            high_risk_cases=(risk_column, lambda s: int((s >= 0.55).sum())),
            patients=("patient_id", "count"),
        )
        .reset_index()
        .sort_values("avg_risk", ascending=False)
    )
    summary["avg_risk"] = summary["avg_risk"].round(3)
    summary["max_risk"] = summary["max_risk"].round(3)
    return summary


def room_summary(df: pd.DataFrame, risk_column: str = "risk_probability") -> pd.DataFrame:
    """Aggregate risk by room, for infection-control hotspot detection."""
    summary = (
        df.groupby(["department", "room"], observed=True)
        .agg(
            avg_risk=(risk_column, "mean"),
            patients=("patient_id", "count"),
        )
        .reset_index()
        .sort_values("avg_risk", ascending=False)
    )
    summary["avg_risk"] = summary["avg_risk"].round(3)
    return summary
