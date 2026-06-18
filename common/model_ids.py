"""Canonical Roboflow Universe model ids for world_cup_projects.

Pinned defaults (CLI + local YOLO .pt + Inference when applicable):
- Players: football-players-detection-3zvbc/11
- Ball: football-ball-detection-rejhg/4
- Pitch keypoints: football-field-detection-f07vi/15 (Inference only, via pitch.py)

See README.md § Models and SPORTS_MERGE_PLAN.md §5.
"""

from __future__ import annotations

CANONICAL_FOOTBALL_PLAYERS_MODEL_ID = "football-players-detection-3zvbc/11"
CANONICAL_FOOTBALL_BALL_MODEL_ID = "football-ball-detection-rejhg/4"
CANONICAL_PITCH_MODEL_ID = "football-field-detection-f07vi/15"

DEFAULT_FOOTBALL_PLAYERS_MODEL_ID = CANONICAL_FOOTBALL_PLAYERS_MODEL_ID
DEFAULT_FOOTBALL_BALL_MODEL_ID = CANONICAL_FOOTBALL_BALL_MODEL_ID
PITCH_INFERENCE_MODEL_ID = CANONICAL_PITCH_MODEL_ID

# Optional Universe versions for experiments (--universe-best, compare scripts).
FOOTBALL_PLAYERS_INFERENCE_V11 = CANONICAL_FOOTBALL_PLAYERS_MODEL_ID
FOOTBALL_PLAYERS_INFERENCE_V19 = "football-players-detection-3zvbc/19"
FOOTBALL_PLAYERS_INFERENCE_V20 = "football-players-detection-3zvbc/20"
FOOTBALL_PLAYERS_INFERENCE_RFDETR = "football-players-detection-3zvbc/18"
# Newer Universe YOLO (yolo11m); local football-player-detection.pt is v11.
BEST_FOOTBALL_PLAYERS_YOLO_MODEL_ID = FOOTBALL_PLAYERS_INFERENCE_V19

KNOWN_FOOTBALL_PLAYER_MODELS = (
    FOOTBALL_PLAYERS_INFERENCE_V11,
    FOOTBALL_PLAYERS_INFERENCE_V19,
    FOOTBALL_PLAYERS_INFERENCE_V20,
    FOOTBALL_PLAYERS_INFERENCE_RFDETR,
)
