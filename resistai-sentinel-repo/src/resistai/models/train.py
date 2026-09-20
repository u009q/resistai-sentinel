"""
Train and persist the AMR risk classifier.

The artefact written to disk is a *bundle*, not a bare estimator: it carries
the exact feature order, the library version and the held-out metrics. A
lone `.pkl` tells you nothing about what it expects or how well it did, and
that is how a model silently drifts out of sync with its caller.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from resistai import __version__
from resistai.config import (
    FEATURE_COLUMNS,
    MODEL_PATH,
    TARGET_COLUMN,
    TRAINING,
    TrainingSettings,
    ensure_directories,
)


@dataclass
class ModelBundle:
    """A trained estimator together with everything needed to use it safely."""

    estimator: RandomForestClassifier
    feature_names: list[str]
    positive_class_index: int
    metrics: dict[str, Any] = field(default_factory=dict)
    app_version: str = __version__
    sklearn_version: str = sklearn.__version__
    trained_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )

    def save(self, path: Path | None = None) -> Path:
        """Persist the bundle to disk and return where it landed."""
        ensure_directories()
        target = Path(path) if path is not None else MODEL_PATH
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, target)
        return target


def train_model(
    df: pd.DataFrame, settings: TrainingSettings | None = None
) -> ModelBundle:
    """
    Fit a random forest on the clinical features.

    Args:
        df: A validated cohort dataframe.
        settings: Training hyperparameters. Defaults to `config.TRAINING`.

    Returns:
        A `ModelBundle` holding the fitted estimator and its held-out metrics.
    """
    cfg = settings or TRAINING

    X = df[FEATURE_COLUMNS].astype(float)
    y = df[TARGET_COLUMN].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=cfg.test_size,
        random_state=cfg.seed,
        stratify=y,
    )

    estimator = RandomForestClassifier(
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        min_samples_leaf=cfg.min_samples_leaf,
        class_weight=cfg.class_weight,
        random_state=cfg.seed,
        n_jobs=-1,
    )
    estimator.fit(X_train, y_train)

    positive_index = int(list(estimator.classes_).index(1))
    test_probabilities = estimator.predict_proba(X_test)[:, positive_index]

    metrics: dict[str, Any] = {
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "positive_rate": round(float(y.mean()), 4),
        # ROC AUC and average precision measure ranking quality; Brier measures
        # calibration. A model used for triage needs both to be sane.
        "roc_auc": round(float(roc_auc_score(y_test, test_probabilities)), 4),
        "average_precision": round(
            float(average_precision_score(y_test, test_probabilities)), 4
        ),
        "brier_score": round(float(brier_score_loss(y_test, test_probabilities)), 4),
        "classification_report": classification_report(
            y_test,
            (test_probabilities >= 0.5).astype(int),
            output_dict=True,
            zero_division=0,
        ),
    }

    return ModelBundle(
        estimator=estimator,
        feature_names=list(FEATURE_COLUMNS),
        positive_class_index=positive_index,
        metrics=metrics,
    )
