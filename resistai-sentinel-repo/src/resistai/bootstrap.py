"""
First-run artefact bootstrap.

Locally you run `scripts/generate_data.py` and `scripts/train_model.py` once
and the artefacts persist. On a hosted platform such as Streamlit Community
Cloud the checkout is fresh and `data/` and `models/` are gitignored, so the
app has to build them on first boot.

Both steps are deterministic (`config.RANDOM_SEED`), so every deployment
produces byte-identical artefacts from the same commit.
"""

from __future__ import annotations

import os
from pathlib import Path

from resistai.config import DATASET_PATH, MODEL_PATH, ensure_directories


def _enabled() -> bool:
    """Auto-bootstrap is on unless explicitly disabled."""
    return os.getenv("RESISTAI_AUTO_BOOTSTRAP", "1").strip().lower() not in {
        "0",
        "false",
        "no",
    }


def artefacts_present() -> bool:
    """True when both the cohort and the trained model already exist."""
    return DATASET_PATH.exists() and MODEL_PATH.exists()


def ensure_artefacts(force: bool = False) -> dict[str, Path]:
    """
    Generate the cohort and train the model if either is missing.

    Args:
        force: Rebuild both artefacts even if they already exist.

    Returns:
        A mapping of artefact name to path.

    Raises:
        RuntimeError: If artefacts are missing and auto-bootstrap is disabled.
    """
    if artefacts_present() and not force:
        return {"dataset": DATASET_PATH, "model": MODEL_PATH}

    if not _enabled():
        raise RuntimeError(
            "Artefacts are missing and RESISTAI_AUTO_BOOTSTRAP is disabled. "
            "Run scripts/generate_data.py and scripts/train_model.py."
        )

    # Imported here rather than at module scope: this pulls in scikit-learn,
    # which is a slow import to pay for on every process that only wants to
    # check whether the files exist.
    from resistai.data.generator import generate_cohort
    from resistai.data.loader import load_dataset
    from resistai.data.schema import validate_dataset
    from resistai.models.train import train_model

    ensure_directories()

    if force or not DATASET_PATH.exists():
        cohort = validate_dataset(generate_cohort())
        cohort.to_csv(DATASET_PATH, index=False)

    if force or not MODEL_PATH.exists():
        train_model(load_dataset()).save(MODEL_PATH)

    return {"dataset": DATASET_PATH, "model": MODEL_PATH}
