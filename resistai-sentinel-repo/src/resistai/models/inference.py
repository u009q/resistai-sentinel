"""
Inference wrapper around a persisted `ModelBundle`.

Every prediction path goes through `_as_frame`, which reindexes the input to
the bundle's own feature order. That one line prevents the whole class of
bug where a caller passes columns in a different order (or an extra column)
and scikit-learn either warns or, worse, silently scores garbage.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd

from resistai.config import MODEL_PATH, RISK_THRESHOLDS
from resistai.models.train import ModelBundle


class ModelNotFoundError(FileNotFoundError):
    """Raised when the model artefact has not been trained yet."""


class AMRPredictor:
    """Thin, schema-safe facade over a trained `ModelBundle`."""

    def __init__(self, bundle: ModelBundle) -> None:
        self._bundle = bundle

    # -- properties ---------------------------------------------------------
    @property
    def feature_names(self) -> list[str]:
        return list(self._bundle.feature_names)

    @property
    def metrics(self) -> dict[str, Any]:
        return dict(self._bundle.metrics)

    @property
    def trained_at(self) -> str:
        return self._bundle.trained_at

    # -- internals ----------------------------------------------------------
    def _as_frame(self, payload: Mapping[str, Any] | pd.DataFrame) -> pd.DataFrame:
        """Coerce any accepted input into a correctly ordered float frame."""
        frame = (
            payload.copy()
            if isinstance(payload, pd.DataFrame)
            else pd.DataFrame([dict(payload)])
        )
        missing = [name for name in self.feature_names if name not in frame.columns]
        if missing:
            raise ValueError(
                f"Missing required features: {', '.join(missing)}"
            )
        return frame[self.feature_names].astype(float)

    # -- public API ---------------------------------------------------------
    def predict_probability(
        self, payload: Mapping[str, Any] | pd.DataFrame
    ) -> np.ndarray:
        """Probability of confirmed AMR for each row."""
        frame = self._as_frame(payload)
        proba = self._bundle.estimator.predict_proba(frame)
        return proba[:, self._bundle.positive_class_index]

    def predict_one(self, payload: Mapping[str, Any]) -> tuple[float, str]:
        """
        Score a single patient.

        Returns:
            `(probability, risk_level)`.
        """
        probability = float(self.predict_probability(payload)[0])
        return probability, RISK_THRESHOLDS.level(probability)

    def score_cohort(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of `df` with `risk_probability` and `risk_level` added."""
        scored = df.copy()
        scored["risk_probability"] = self.predict_probability(df).round(4)
        scored["risk_level"] = [
            RISK_THRESHOLDS.level(p) for p in scored["risk_probability"]
        ]
        return scored

    def feature_importance(self) -> pd.DataFrame:
        """Global feature importance, most influential first."""
        return (
            pd.DataFrame(
                {
                    "feature": self.feature_names,
                    "importance": self._bundle.estimator.feature_importances_,
                }
            )
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    def counterfactual(
        self,
        payload: Mapping[str, Any],
        overrides: Mapping[str, Any],
    ) -> tuple[float, str]:
        """
        Re-score a patient under hypothetical changes (the what-if simulator).

        Args:
            payload: The patient's current feature values.
            overrides: Features to replace, e.g. `{"icu_admission": 1}`.
        """
        simulated = dict(payload)
        unknown = [key for key in overrides if key not in self.feature_names]
        if unknown:
            raise ValueError(f"Unknown feature(s): {', '.join(unknown)}")
        simulated.update(overrides)
        return self.predict_one(simulated)

    def risk_trajectory(
        self,
        payload: Mapping[str, Any],
        days: int = 7,
        seed: int | None = None,
    ) -> Sequence[float]:
        """
        Project a short risk trajectory by extending length of stay.

        This is a deterministic projection of the model's own response to a
        longer admission -- not a time-series forecast. The original notebook
        used `time.time() % 5` here, which made the chart change on every
        rerun for no clinical reason.
        """
        baseline = dict(payload)
        stay = float(baseline.get("length_of_stay_days", 1) or 1)
        rng = np.random.default_rng(seed if seed is not None else int(stay))
        jitter = rng.normal(0.0, 0.01, size=days)

        trajectory: list[float] = []
        for day in range(days):
            projected = dict(baseline)
            projected["length_of_stay_days"] = stay + day
            probability = float(self.predict_probability(projected)[0])
            trajectory.append(float(np.clip(probability + jitter[day], 0.0, 1.0)))
        return trajectory


def load_predictor(path: Path | None = None) -> AMRPredictor:
    """
    Load the persisted model bundle.

    Raises:
        ModelNotFoundError: If the artefact is missing.
    """
    target = Path(path) if path is not None else MODEL_PATH
    if not target.exists():
        raise ModelNotFoundError(
            f"Model not found at {target}. "
            "Run `python scripts/train_model.py` first."
        )
    bundle = joblib.load(target)
    if not isinstance(bundle, ModelBundle):
        raise ModelNotFoundError(
            f"{target} is not a ResistAI model bundle. Retrain the model."
        )
    return AMRPredictor(bundle)
