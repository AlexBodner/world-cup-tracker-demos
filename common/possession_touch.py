"""Shared possession-touch validation for pass detection and debug overlays."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import supervision as sv

from world_cup_projects.common.possession import Carrier, ball_xy, feet_xy, player_mask
from world_cup_projects.common.possession_config import (
    AERIAL_DY_THRESHOLD_PX,
    CONTROL_MAX_DISTANCE_M,
    CONTROL_MAX_DISTANCE_PX,
)


@dataclass(frozen=True)
class TouchValidationConfig:
    aerial_dy_threshold_px: float = AERIAL_DY_THRESHOLD_PX
    # Transit fly-by: fast ball through the control radius without settling at feet.
    transit_min_speed_m_s: float = 9.0
    transit_min_feet_px: float = 30.0
    transit_min_speed_px_per_frame: float = 10.0
    max_plausible_transit_speed_m_s: float = 35.0
    ball_speed_lookback_frames: int = 10
    ball_speed_min_lookback_frames: int = 3


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


def ball_instant_speed_m_s(
    prev_ball: np.ndarray,
    ball: np.ndarray,
    *,
    fps: float,
    transformer,
    frame_gap: int = 1,
    prev_transformer=None,
) -> float | None:
    """Ball speed in m/s from two image positions (optionally separated by >1 frame)."""
    if fps <= 0 or transformer is None or frame_gap < 1:
        return None
    from world_cup_projects.common.pitch import image_to_pitch_m

    prev_t = prev_transformer if prev_transformer is not None else transformer
    pitch_prev = image_to_pitch_m(np.array([prev_ball], dtype=np.float32), prev_t)
    pitch_curr = image_to_pitch_m(np.array([ball], dtype=np.float32), transformer)
    if pitch_prev is not None and pitch_curr is not None:
        dist_m = float(np.linalg.norm(pitch_curr[0] - pitch_prev[0]))
    else:
        pts = np.stack([prev_ball, ball], axis=0).astype(np.float32)
        pitch = image_to_pitch_m(pts, transformer)
        if pitch is None:
            return None
        dist_m = float(np.linalg.norm(pitch[1] - pitch[0]))
    return dist_m * fps / frame_gap


def is_transit_flyby_touch(
    dets: sv.Detections,
    carrier: Carrier,
    *,
    prev_ball: np.ndarray | None,
    config: TouchValidationConfig,
    fps: float = 25.0,
    transformer=None,
    frame_gap: int = 1,
    prev_transformer=None,
    speed_prev_ball: np.ndarray | None = None,
    speed_frame_gap: int | None = None,
    speed_prev_transformer=None,
) -> bool:
    """Fast ball through a player's zone without settling at feet (not real possession)."""
    ref_ball = speed_prev_ball if speed_prev_ball is not None else prev_ball
    if ref_ball is None:
        return False
    ball = ball_xy(dets)
    if ball is None:
        return False
    feet = feet_xy(dets)[carrier.index]
    feet_dist_px = float(np.hypot(ball[0] - feet[0], ball[1] - feet[1]))
    if feet_dist_px < config.transit_min_feet_px:
        return False

    gap = max(1, speed_frame_gap if speed_prev_ball is not None else frame_gap)
    ref_t = speed_prev_transformer if speed_prev_ball is not None else prev_transformer
    speed_m_s = ball_instant_speed_m_s(
        ref_ball,
        ball,
        fps=fps,
        transformer=transformer,
        frame_gap=gap,
        prev_transformer=ref_t,
    )
    if speed_m_s is not None and speed_m_s <= config.max_plausible_transit_speed_m_s:
        return speed_m_s >= config.transit_min_speed_m_s

    instant_px_f = float(np.linalg.norm(ball - ref_ball)) / gap
    return instant_px_f >= config.transit_min_speed_px_per_frame


def is_transit_flyby_control(
    dets: sv.Detections,
    carrier: Carrier,
    *,
    prev_ball: np.ndarray | None,
    config: TouchValidationConfig,
    fps: float = 25.0,
    transformer=None,
    frame_gap: int = 1,
    prev_transformer=None,
    speed_prev_ball: np.ndarray | None = None,
    speed_frame_gap: int | None = None,
    speed_prev_transformer=None,
) -> bool:
    """Alias for :func:`is_transit_flyby_touch` (control-path naming)."""
    return is_transit_flyby_touch(
        dets,
        carrier,
        prev_ball=prev_ball,
        config=config,
        fps=fps,
        transformer=transformer,
        frame_gap=frame_gap,
        prev_transformer=prev_transformer,
        speed_prev_ball=speed_prev_ball,
        speed_frame_gap=speed_frame_gap,
        speed_prev_transformer=speed_prev_transformer,
    )


def _reception_beyond_control_range(carrier: Carrier) -> bool:
    """Reception gate is looser than control; only vetoes transit beyond dribble range."""
    d = carrier.distance
    return d > CONTROL_MAX_DISTANCE_PX or d > CONTROL_MAX_DISTANCE_M


def is_valid_possession_touch(
    dets: sv.Detections,
    carrier: Carrier,
    *,
    touch_kind: str,
    config: TouchValidationConfig,
    prev_ball: np.ndarray | None = None,
    fps: float = 25.0,
    transformer=None,
    frame_gap: int = 1,
    prev_transformer=None,
    speed_prev_ball: np.ndarray | None = None,
    speed_frame_gap: int | None = None,
    speed_prev_transformer=None,
) -> bool:
    """Reject aerial fly-bys and nearest-player mismatches.

    Control uses symmetric vertical offset; reception only vetoes fly-bys below
    the feet so chest-height first contacts still count. Transit fly-by veto
    applies to ``control`` always; for ``reception`` only when beyond control
    range (tight one-touch receptions are unchanged).
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
    if touch_kind == "control" and is_transit_flyby_touch(
        dets,
        carrier,
        prev_ball=prev_ball,
        config=config,
        fps=fps,
        transformer=transformer,
        frame_gap=frame_gap,
        prev_transformer=prev_transformer,
        speed_prev_ball=speed_prev_ball,
        speed_frame_gap=speed_frame_gap,
        speed_prev_transformer=speed_prev_transformer,
    ):
        return False
    if (
        touch_kind == "reception"
        and _reception_beyond_control_range(carrier)
        and is_transit_flyby_touch(
            dets,
            carrier,
            prev_ball=prev_ball,
            config=config,
            fps=fps,
            transformer=transformer,
            frame_gap=frame_gap,
            prev_transformer=prev_transformer,
            speed_prev_ball=speed_prev_ball,
            speed_frame_gap=speed_frame_gap,
            speed_prev_transformer=speed_prev_transformer,
        )
    ):
        return False
    ball = carrier.ball
    tid = int(dets.tracker_id[carrier.index]) if dets.tracker_id is not None else -1
    nearest_tid = nearest_player_tid(dets, ball)
    return nearest_tid is not None and nearest_tid == tid
