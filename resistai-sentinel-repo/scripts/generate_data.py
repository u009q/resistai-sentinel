"""
Generate the synthetic AMR cohort.

Usage:
    python scripts/generate_data.py [--patients 4000] [--seed 42]
"""

from __future__ import annotations

import argparse
from dataclasses import replace

import _bootstrap  # noqa: F401  (side effect: extends sys.path)

from resistai.config import DATASET_PATH, GENERATOR, ensure_directories
from resistai.data.generator import generate_cohort
from resistai.data.schema import validate_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the AMR cohort.")
    parser.add_argument(
        "--patients", type=int, default=GENERATOR.n_patients, help="Cohort size."
    )
    parser.add_argument("--seed", type=int, default=GENERATOR.seed, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_directories()

    settings = replace(GENERATOR, n_patients=args.patients, seed=args.seed)
    cohort = validate_dataset(generate_cohort(settings))
    cohort.to_csv(DATASET_PATH, index=False)

    positives = int(cohort["amr_confirmed"].sum())
    print(f"Wrote {len(cohort):,} patients to {DATASET_PATH}")
    print(f"Confirmed AMR: {positives:,} ({positives / len(cohort):.1%})")
    print("Departments:")
    for department, count in cohort["department"].value_counts().items():
        print(f"  {department:<22} {count:>6,}")


if __name__ == "__main__":
    main()
