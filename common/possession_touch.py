"""Shared possession-touch validation for pass detection and debug overlays."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import supervision as sv

from world_cup_projects.common.possession import Carrier, ball_xy, feet_xy, player_mask
from world_cup_projects.common.possession_config import AERIAL_DY_THRESHOLD_PX


@dataclass(frozen=True)
class TouchValidationConfig:
    aerial_dy_threshold_px: float = AERIAL_DY_THRESHOLD_PX


def nearest_player_tid(dets: sv.Detections, ball: np.ndarray) -> int | None:
    pmask = player_mask(dets)
    if not pmask.any() or dets.tracker_id is None:
        return None
    feet = feet_xy(dets)[pmask]
    tids = dets.tracker_id[pmask]
    dist = np.hypot(feet[:, 0] - ball[0], feet[:, 1] - ball[1])
    tid = int(tids[int(np.argmin(dist))])
    return tid if tid >= 0 else None


def is_aerial_touch(
    dets: sv.Detections,
    carrier: Carrier,
    *,
    threshold_px: float,
) -> bool:
    ball = ball_xy(dets)
    if ball is None:
        return False
    feet = feet_xy(dets)[carrier.index]
    return abs(float(ball[1] - feet[1])) > threshold_px


def is_aerial_flyby_below_feet(
    dets: sv.Detections,
    carrier: Carrier,
    *,
    threshold_px: float,
) -> bool:
    """Ball visibly below the feet in image space (aerial fly-by, not chest reception)."""
    ball = ball_xy(dets)
    if ball is None:
        return False
    feet = feet_xy(dets)[carrier.index]
    return float(ball[1] - feet[1]) > threshold_px


def reception_aerial_veto_threshold(config: TouchValidationConfig) -> float:
    """Looser than control: chest receptions are above the feet (negative dy)."""
    return config.aerial_dy_threshold_px * 2.0


def is_valid_possession_touch(
    dets: sv.Detections,
    carrier: Carrier,
    *,
    touch_kind: str,
    config: TouchValidationConfig,
) -> bool:
    """Reject aerial fly-bys and nearest-player mismatches.

    Control uses symmetric vertical offset; reception only vetoes fly-bys below
    the feet so chest-height first contacts still count.
    """
    if touch_kind == "control" and is_aerial_touch(
        dets, carrier, threshold_px=config.aerial_dy_threshold_px
    ):
        return False
    if touch_kind == "reception" and is_aerial_flyby_below_feet(
        dets,
        carrier,
        threshold_px=reception_aerial_veto_threshold(config),
    ):
        return False
    ball = carrier.ball
    tid = int(dets.tracker_id[carrier.index]) if dets.tracker_id is not None else -1
    nearest_tid = nearest_player_tid(dets, ball)
    return nearest_tid is not None and nearest_tid == tid
