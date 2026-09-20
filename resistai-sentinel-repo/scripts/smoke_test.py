"""
Headless render check.

Runs the full Streamlit script through `AppTest` and fails if any tab raises.
Cheap insurance against the class of break that only shows up in the browser:
a renamed column, a chart that chokes on an empty frame, a missing import.

Usage:
    python scripts/smoke_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import _bootstrap  # noqa: F401  (side effect: extends sys.path)

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app.py"

EXPECTED_TABS = 6
REQUIRED_METRICS = {"Patients monitored", "ROC AUC"}


def main() -> int:
    app = AppTest.from_file(str(APP), default_timeout=300)
    app.run()

    if app.exception:
        print("FAIL — the app raised while rendering:", file=sys.stderr)
        for exception in app.exception:
            print(f"  {exception.value}", file=sys.stderr)
        return 1

    if len(app.tabs) != EXPECTED_TABS:
        print(
            f"FAIL — expected {EXPECTED_TABS} tabs, found {len(app.tabs)}",
            file=sys.stderr,
        )
        return 1

    labels = {metric.label for metric in app.metric}
    missing = REQUIRED_METRICS - labels
    if missing:
        print(f"FAIL — missing metrics: {', '.join(sorted(missing))}", file=sys.stderr)
        return 1

    print(f"PASS — {len(app.tabs)} tabs rendered, {len(app.metric)} metrics present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
