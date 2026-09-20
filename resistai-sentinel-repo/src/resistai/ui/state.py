"""
Cached resource access for the Streamlit app.

Streamlit reruns the whole script on every widget interaction. Without
caching, that meant re-reading a 4,000-row CSV and re-loading the model on
every click. `cache_data` / `cache_resource` reduce that to once per process.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from resistai.bootstrap import ensure_artefacts
from resistai.data.loader import DatasetNotFoundError, load_dataset
from resistai.data.schema import DatasetValidationError
from resistai.models.inference import AMRPredictor, ModelNotFoundError, load_predictor


@st.cache_resource(show_spinner="Preparing data and model (first run only)…")
def prepare_artefacts() -> bool:
    """
    Build the cohort and model on first boot if they are missing.

    Cached as a resource so the work happens once per process, not once per
    session — on a hosted deployment several viewers can arrive at the same
    time and none of them should trigger a second training run.
    """
    ensure_artefacts()
    return True


@st.cache_data(show_spinner="Loading cohort…")
def get_dataset() -> pd.DataFrame:
    """Load and validate the cohort. Cached across reruns."""
    prepare_artefacts()
    return load_dataset()


@st.cache_resource(show_spinner="Loading model…")
def get_predictor() -> AMRPredictor:
    """Load the trained model bundle. Cached across reruns and sessions."""
    prepare_artefacts()
    return load_predictor()


@st.cache_data(show_spinner="Scoring cohort…")
def get_scored_cohort() -> pd.DataFrame:
    """Score every patient once, so hospital-level views are instant."""
    return get_predictor().score_cohort(get_dataset())


def bootstrap() -> tuple[pd.DataFrame, AMRPredictor] | None:
    """
    Load everything the app needs, reporting setup problems clearly.

    Returns:
        `(scored_cohort, predictor)`, or `None` when a prerequisite is
        missing -- in which case the error has already been rendered.
    """
    try:
        predictor = get_predictor()
        scored = get_scored_cohort()
    except (DatasetNotFoundError, ModelNotFoundError, RuntimeError) as exc:
        st.error(str(exc))
        st.code(
            "python scripts/generate_data.py\npython scripts/train_model.py",
            language="bash",
        )
        return None
    except DatasetValidationError as exc:
        st.error(f"Dataset failed validation: {exc}")
        return None

    return scored, predictor


def selected_patient(scored: pd.DataFrame, patient_uid: str) -> pd.Series:
    """
    Fetch one patient by medical record number.

    The lookup is an exact match against known IDs, so a crafted value in the
    query string cannot reach the dataframe as a filter expression.
    """
    matches = scored.loc[scored["patient_uid"] == patient_uid]
    if matches.empty:
        raise KeyError(f"Unknown patient: {patient_uid}")
    return matches.iloc[0]
