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
    # Long in-flight path from a known release point: slow average inbound speed
    # means the ball is dropping through a zone, not possession at the feet.
    transit_min_release_travel_px: float = 450.0
    transit_min_release_gap_frames: int = 50
    transit_release_flyby_max_speed_px_per_frame: float = 8.0
    # Ball path redirect at a touch (one-touch kick / intercept) vs straight fly-by.
    redirect_lookback_frames: int = 5
    redirect_lookahead_frames: int = 5
    redirect_min_angle_deg: float = 28.0
    redirect_min_speed_ratio: float = 1.35
    redirect_min_segment_px: float = 10.0
    # In-flight opponent touch during a known release: no path redirect ⇒ gravity fly-by.
    gravity_flyby_min_release_gap_frames: int = 15


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


def inbound_speed_px_per_frame(
    ref_ball: np.ndarray,
    ball: np.ndarray,
    *,
    frame_gap: int,
) -> tuple[float, float]:
    """Return ``(travel_px, speed_px_per_frame)`` for an inbound ball path."""
    gap = max(1, frame_gap)
    travel_px = float(np.linalg.norm(ball - ref_ball))
    return travel_px, travel_px / gap


def is_fast_inbound_transit(
    ref_ball: np.ndarray,
    ball: np.ndarray,
    *,
    frame_gap: int,
    config: TouchValidationConfig,
    fps: float = 25.0,
    transformer=None,
    prev_transformer=None,
) -> bool:
    """Primary speed gate: fast inbound ball transit, independent of aerial dy or feet px."""
    gap = max(1, frame_gap)
    travel_px, speed_px_f = inbound_speed_px_per_frame(ref_ball, ball, frame_gap=gap)
    if speed_px_f >= config.transit_min_speed_px_per_frame:
        return True
    speed_m_s = ball_instant_speed_m_s(
        ref_ball,
        ball,
        fps=fps,
        transformer=transformer,
        frame_gap=gap,
        prev_transformer=prev_transformer,
    )
    return (
        speed_m_s is not None
        and speed_m_s <= config.max_plausible_transit_speed_m_s
        and speed_m_s >= config.transit_min_speed_m_s
    )


def is_release_inbound_flyby(
    release_ball: np.ndarray,
    ball: np.ndarray,
    *,
    release_gap_frames: int,
    config: TouchValidationConfig,
) -> bool:
    """Ball traveled far from a pass release but arrived slowly — dropping through a zone."""
    if release_gap_frames < config.transit_min_release_gap_frames:
        return False
    travel_px, speed_px_f = inbound_speed_px_per_frame(
        release_ball, ball, frame_gap=release_gap_frames
    )
    return (
        travel_px >= config.transit_min_release_travel_px
        and speed_px_f < config.transit_release_flyby_max_speed_px_per_frame
    )


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
    release_ball: np.ndarray | None = None,
    release_gap_frames: int | None = None,
) -> bool:
    """Fast ball through a player's zone without settling at feet (not real possession)."""
    ball = ball_xy(dets)
    if ball is None:
        return False

    ref_ball = speed_prev_ball if speed_prev_ball is not None else prev_ball
    if ref_ball is not None:
        gap = max(1, speed_frame_gap if speed_prev_ball is not None else frame_gap)
        ref_t = speed_prev_transformer if speed_prev_ball is not None else prev_transformer
        if is_fast_inbound_transit(
            ref_ball,
            ball,
            frame_gap=gap,
            config=config,
            fps=fps,
            transformer=transformer,
            prev_transformer=ref_t,
        ):
            return True

    if (
        release_ball is not None
        and release_gap_frames is not None
        and is_release_inbound_flyby(
            release_ball,
            ball,
            release_gap_frames=release_gap_frames,
            config=config,
        )
    ):
        return True

    if ref_ball is None:
        return False

    feet = feet_xy(dets)[carrier.index]
    feet_dist_px = float(np.hypot(ball[0] - feet[0], ball[1] - feet[1]))
    zone_px = feet_dist_px
    if transformer is None and carrier.distance <= CONTROL_MAX_DISTANCE_PX:
        zone_px = max(feet_dist_px, float(carrier.distance))
    if zone_px < config.transit_min_feet_px:
        return False

    gap = max(1, speed_frame_gap if speed_prev_ball is not None else frame_gap)
    _, speed_px_f = inbound_speed_px_per_frame(ref_ball, ball, frame_gap=gap)
    return speed_px_f >= config.transit_min_speed_px_per_frame


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


def _ball_touch_path_metrics(
    frames_by_idx: dict[int, sv.Detections],
    touch_frame: int,
    *,
    lookback: int = 5,
    lookahead: int = 5,
    min_segment_px: float = 10.0,
) -> tuple[float, float] | None:
    """Inbound angle (deg) and outbound/inbound speed ratio at ``touch_frame``."""
    from world_cup_projects.common.geometry import unit

    samples: list[tuple[int, np.ndarray]] = []
    for frame_idx in range(touch_frame - lookback, touch_frame + lookahead + 1):
        dets = frames_by_idx.get(frame_idx)
        if dets is None:
            continue
        ball = ball_xy(dets)
        if ball is not None:
            samples.append((frame_idx, np.asarray(ball, dtype=np.float64)))
    if len(samples) < 4:
        return None

    before = [(f, p) for f, p in samples if f <= touch_frame]
    after = [(f, p) for f, p in samples if f >= touch_frame]
    if len(before) < 2 or len(after) < 2:
        return None

    pivot = before[-1][1]
    for frame_idx, point in samples:
        if frame_idx == touch_frame:
            pivot = point
            break

    f_in0, p_in0 = before[-2]
    v_in = pivot - (p_in0 if f_in0 < touch_frame else before[-1][1])
    in_len = float(np.linalg.norm(v_in))
    if in_len < min_segment_px:
        return None

    after_touch = [(f, p) for f, p in after if f >= touch_frame]
    if len(after_touch) < 2:
        return None
    _, p_out1 = after_touch[1]
    v_out = p_out1 - pivot
    out_len = float(np.linalg.norm(v_out))
    if out_len < min_segment_px:
        return None

    u_in, u_out = unit(v_in), unit(v_out)
    if u_in is None or u_out is None:
        return None
    cos_angle = float(np.clip(np.dot(u_in, u_out), -1.0, 1.0))
    angle_deg = float(np.degrees(np.arccos(cos_angle)))
    speed_ratio = out_len / max(in_len, 1e-6)
    return angle_deg, speed_ratio


def ball_redirected_at_touch(
    frames_by_idx: dict[int, sv.Detections],
    touch_frame: int,
    *,
    lookback: int = 5,
    lookahead: int = 5,
    min_angle_deg: float = 28.0,
    min_speed_ratio: float = 1.35,
    min_segment_px: float = 10.0,
) -> bool:
    """True when inbound/outbound ball vectors diverge at ``touch_frame`` (kick, not fly-by)."""
    metrics = _ball_touch_path_metrics(
        frames_by_idx,
        touch_frame,
        lookback=lookback,
        lookahead=lookahead,
        min_segment_px=min_segment_px,
    )
    if metrics is None:
        return False
    angle_deg, speed_ratio = metrics
    return (
        speed_ratio >= min_speed_ratio
        or (min_angle_deg <= angle_deg <= 135.0)
    )


def is_gravity_arc_flyby_at_touch(
    frames_by_idx: dict[int, sv.Detections],
    touch_frame: int,
    *,
    lookback: int = 5,
    lookahead: int = 5,
    max_angle_deg: float = 28.0,
    max_speed_ratio: float = 1.35,
    min_segment_px: float = 10.0,
) -> bool:
    """True when the ball continues on the same arc through ``touch_frame`` (gravity only)."""
    metrics = _ball_touch_path_metrics(
        frames_by_idx,
        touch_frame,
        lookback=lookback,
        lookahead=lookahead,
        min_segment_px=min_segment_px,
    )
    if metrics is None:
        return False
    angle_deg, speed_ratio = metrics
    return angle_deg < max_angle_deg and speed_ratio < max_speed_ratio


def redirect_overrides_transit_flyby(
    frames_by_idx: dict[int, sv.Detections],
    touch_frame: int,
    *,
    config: TouchValidationConfig,
    release_gap_frames: int | None = None,
) -> bool:
    """True when a touch redirected the ball path enough to count as possession."""
    metrics = _ball_touch_path_metrics(
        frames_by_idx,
        touch_frame,
        lookback=config.redirect_lookback_frames,
        lookahead=config.redirect_lookahead_frames,
        min_segment_px=config.redirect_min_segment_px,
    )
    if metrics is None:
        return False
    angle_deg, speed_ratio = metrics
    min_angle = config.redirect_min_angle_deg
    min_ratio = config.redirect_min_speed_ratio
    if (
        release_gap_frames is not None
        and release_gap_frames >= config.gravity_flyby_min_release_gap_frames
    ):
        return (
            min_angle <= angle_deg <= 135.0
            or (speed_ratio >= min_ratio and angle_deg >= min_angle)
        )
    return (
        speed_ratio >= min_ratio
        or (min_angle <= angle_deg <= 135.0)
    )


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
    release_ball: np.ndarray | None = None,
    release_gap_frames: int | None = None,
    frames_by_idx: dict[int, sv.Detections] | None = None,
    frame_idx: int | None = None,
) -> bool:
    """Reject fly-bys and nearest-player mismatches.

    Speed-based transit fly-by is evaluated first and is independent of aerial
    dy / chest-height checks. Aerial vetoes only apply to clearly off-ground
    contacts in image space.
    """
    transit_kwargs = dict(
        prev_ball=prev_ball,
        config=config,
        fps=fps,
        transformer=transformer,
        frame_gap=frame_gap,
        prev_transformer=prev_transformer,
        speed_prev_ball=speed_prev_ball,
        speed_frame_gap=speed_frame_gap,
        speed_prev_transformer=speed_prev_transformer,
        release_ball=release_ball,
        release_gap_frames=release_gap_frames,
    )
    ball = carrier.ball
    release_inbound_flyby = (
        release_ball is not None
        and release_gap_frames is not None
        and is_release_inbound_flyby(
            release_ball,
            ball,
            release_gap_frames=release_gap_frames,
            config=config,
        )
    )
    if is_transit_flyby_touch(dets, carrier, **transit_kwargs):
        if release_inbound_flyby:
            return False
        if (
            frames_by_idx is not None
            and frame_idx is not None
            and redirect_overrides_transit_flyby(
                frames_by_idx,
                frame_idx,
                config=config,
                release_gap_frames=release_gap_frames,
            )
        ):
            pass
        else:
            return False
    if (
        release_ball is not None
        and release_gap_frames is not None
        and release_gap_frames >= config.gravity_flyby_min_release_gap_frames
        and frames_by_idx is not None
        and frame_idx is not None
        and is_gravity_arc_flyby_at_touch(
            frames_by_idx,
            frame_idx,
            lookback=config.redirect_lookback_frames,
            lookahead=config.redirect_lookahead_frames,
            max_angle_deg=config.redirect_min_angle_deg,
            max_speed_ratio=config.redirect_min_speed_ratio,
            min_segment_px=config.redirect_min_segment_px,
        )
    ):
        return False
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


def ball_departed_for_one_touch(
    touch_ball: np.ndarray,
    in_flight_ball: np.ndarray | None,
    *,
    touch_frame: int,
    in_flight_frame: int,
    depart_min_px: float,
    frame_balls: list[tuple[int, np.ndarray]] | None = None,
) -> bool:
    """True when the ball left the one-touch passer zone (not a stationary fly-by).

    When the ball is missing on every in-flight frame we allow the release anchor.
    ``frame_balls`` lists ``(frame_idx, ball_xy)`` samples after the touch so slow
    roll-outs can satisfy the departure threshold a few frames later.
    """
    max_travel = 0.0
    samples = list(frame_balls or [])
    if in_flight_ball is not None:
        samples.append((in_flight_frame, in_flight_ball))
    if not samples:
        return True
    for sample_frame, ball in samples:
        travel_px, _ = inbound_speed_px_per_frame(
            touch_ball,
            ball,
            frame_gap=max(1, sample_frame - touch_frame),
        )
        max_travel = max(max_travel, travel_px)
    return max_travel >= depart_min_px
