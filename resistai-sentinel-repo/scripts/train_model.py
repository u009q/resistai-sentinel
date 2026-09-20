"""
Train the AMR risk classifier and persist the model bundle.

Usage:
    python scripts/train_model.py
"""

from __future__ import annotations

import json

import _bootstrap  # noqa: F401  (side effect: extends sys.path)

from resistai.config import MODEL_PATH
from resistai.data.loader import load_dataset
from resistai.models.train import train_model


def main() -> None:
    cohort = load_dataset()
    bundle = train_model(cohort)
    path = bundle.save(MODEL_PATH)

    report = bundle.metrics["classification_report"]
    print(f"Model saved to {path}")
    print(f"scikit-learn {bundle.sklearn_version} · trained {bundle.trained_at}")
    print()
    print("Held-out performance")
    print(f"  ROC AUC           {bundle.metrics['roc_auc']:.4f}")
    print(f"  Average precision {bundle.metrics['average_precision']:.4f}")
    print(f"  Brier score       {bundle.metrics['brier_score']:.4f}")
    print(f"  Positive rate     {bundle.metrics['positive_rate']:.4f}")
    print()
    print("Per-class (threshold 0.5)")
    print(json.dumps({k: v for k, v in report.items() if k in {"0", "1"}}, indent=2))


if __name__ == "__main__":
    main()
