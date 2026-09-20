"""Dataset loading with schema validation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from resistai.config import DATASET_PATH
from resistai.data.schema import validate_dataset


class DatasetNotFoundError(FileNotFoundError):
    """Raised when the cohort file has not been generated yet."""


def load_dataset(path: Path | None = None) -> pd.DataFrame:
    """
    Read and validate the AMR cohort.

    Args:
        path: Optional override for the dataset location.

    Returns:
        A validated dataframe.

    Raises:
        DatasetNotFoundError: If the file does not exist.
        DatasetValidationError: If the file does not match the schema.
    """
    target = Path(path) if path is not None else DATASET_PATH
    if not target.exists():
        raise DatasetNotFoundError(
            f"Dataset not found at {target}. "
            "Run `python scripts/generate_data.py` first."
        )

    df = pd.read_csv(target)
    return validate_dataset(df)
