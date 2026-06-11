"""Canonical ball-carrier distance and touch-validation thresholds.

Pass detection, pass alternatives, freeze overlays, and debug tooling should
import from here (or helpers in ``possession`` / ``possession_touch``) rather
than duplicating literals.
"""

from __future__ import annotations

# Tight dribble at the feet — used for pass passer logic and lane-scoring freezes.
CONTROL_MAX_DISTANCE_PX = 55.0
CONTROL_MAX_DISTANCE_M = 0.8

# Looser first-touch gate — pass detection only (control is tried first).
RECEPTION_MAX_DISTANCE_PX = 120.0
RECEPTION_MAX_DISTANCE_M = 1.8

# Vertical offset in image space above/below feet that vetoes ground possession.
AERIAL_DY_THRESHOLD_PX = 20.0

# Backward-compatible aliases (formerly 80 px / 1.0 m for loose possession).
CARRIER_MAX_DISTANCE_PX = CONTROL_MAX_DISTANCE_PX
CARRIER_MAX_DISTANCE_M = CONTROL_MAX_DISTANCE_M
