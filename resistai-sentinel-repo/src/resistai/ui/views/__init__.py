"""Individual dashboard views. Each module exposes a single `render()`."""

from resistai.ui.views import actions, explainability, hospital, overview, patient, simulator

__all__ = ["actions", "explainability", "hospital", "overview", "patient", "simulator"]
