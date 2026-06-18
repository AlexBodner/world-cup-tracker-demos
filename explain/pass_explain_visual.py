"""Filmstrip pass-detection explain frames — consecutive frames with live state labels."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import supervision as sv

from world_cup_projects.common.video import (
    finalize_video_for_playback,
    write_gif_from_mp4,
    write_h264_video,
)
from world_cup_projects.common.pitch import (
    ViewTransformer,
    homography_from_keypoints_radar,
    render_radar_simple,
)
from world_cup_projects.common.possession import ball_xy
from world_cup_projects.common.possession_config import (
    AERIAL_DY_THRESHOLD_PX,
    CONTROL_MAX_DISTANCE_PX,
    RECEPTION_MAX_DISTANCE_PX,
)
from world_cup_projects.common.visual import (
    ROBOFLOW_PURPLE_BGR,
    annotate_ball,
    draw_branding_tag,
    draw_carrier_spotlight,
    draw_hud_bar,
    draw_radar_minimap,
    draw_text_shadow,
    ease_out_cubic,
)
from world_cup_projects.explain.pass_alternatives_visual import (
    _dim_frame,
    _fit_social_square,
    _ui_scale,
)
from world_cup_projects.common.soccernet import ROLE_GOALKEEPER
from world_cup_projects.player_stats.carrier_tracking import CarrierFrameState
from world_cup_projects.player_stats.pass_events import InferredPass, PassDetectionConfig
from world_cup_projects.player_stats.pass_network_render import (
    _draw_pass_highlights,
    _get_player_box,
    _team_color,
)

_FOCUS_DIM = 0.16
_SPOTLIGHT_RADIUS = 200
_SPOTLIGHT_STRENGTH = 0.84
_ANCHOR_SPOTLIGHT_STRENGTH = 0.42
_FLIGHT_ACCENT_BGR = (40, 220, 255)
_PANEL_BG_BGR = (18, 18, 22)
_BADGE_BG_BGR = (24, 24, 30)
_BRANDING = "Roboflow | pass detection"
_MIN_CONTROL = PassDetectionConfig().min_control_frames
_MIN_ARRIVAL = PassDetectionConfig().min_arrival_frames
# Explain lock waits for the ball to sit inside the player ellipse — tighter than
# pass-detection control range, which can credit while the ball is still above the feet.
_RECEIVER_EXPLAIN_TIGHT_PX = 12.0
_PANEL_W = 960
_PANEL_H = 540
_GUTTER_W = 152
_CROP_MIN_FRAC = 0.36
_CROP_PAD_RATIO = 0.55


@dataclass(frozen=True)
class PassStripPlan:
    """Frame indices for each filmstrip panel."""

    passer_frames: tuple[int, ...]
    flight_frames: tuple[int, ...]
    receiver_frames: tuple[int, ...]
    summary_frame: int
    passer_confirm_frame: int
    receiver_confirm_frame: int


@dataclass(frozen=True)
class PassExplainContext:
    """Frames, detections, and carrier timeline for one inferred pass."""

    pass_event: InferredPass
    frame_rate: float
    metric: bool
    min_control_frames: int
    min_arrival_frames: int
    strip_plan: PassStripPlan
    frames: dict[int, np.ndarray]
    dets_by_frame: dict[int, sv.Detections]
    timeline_by_frame: dict[int, CarrierFrameState]
    keypoints_by_frame: dict[int, sv.KeyPoints | None]
    radar_transformers: dict[int, ViewTransformer | None]


def _feet_for_tid(dets: sv.Detections, tid: int) -> tuple[int, int] | None:
    box = _get_player_box(dets, tid)
    if box is None:
        return None
    return int((box[0] + box[2]) / 2), int(box[3])


def _letterbox_frame(
    frame: np.ndarray,
    target_w: int = _PANEL_W,
    target_h: int = _PANEL_H,
) -> np.ndarray:
    """Fit into a 16:9 panel without stretching."""
    h, w = frame.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((target_h, target_w, 3), _PANEL_BG_BGR, dtype=np.uint8)
    x0 = (target_w - new_w) // 2
    y0 = (target_h - new_h) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas


def _action_focus_points(
    dets: sv.Detections,
    *tids: int,
    ball_pt: tuple[int, int] | None = None,
) -> list[tuple[int, int]]:
    points: list[tuple[int, int]] = []
    for tid in tids:
        if (feet := _feet_for_tid(dets, tid)) is not None:
            points.append(feet)
        if (box := _get_player_box(dets, tid)) is not None:
            points.append((int(box[0]), int(box[1])))
            points.append((int(box[2]), int(box[3])))
    if ball_pt is not None:
        points.append(ball_pt)
    elif (ball := ball_xy(dets)) is not None:
        points.append((int(ball[0]), int(ball[1])))
    return points


def _crop_rect_from_points(
    frame_shape: tuple[int, int, int],
    points: list[tuple[int, int]],
    *,
    aspect: float | None = None,
    min_frac: float = _CROP_MIN_FRAC,
    pad_ratio: float = _CROP_PAD_RATIO,
) -> tuple[int, int, int, int] | None:
    """Tight 16:9 crop around action; returns x0,y0,x1,y1."""
    if not points:
        return None
    fh, fw = frame_shape[:2]
    aspect = aspect if aspect is not None else _PANEL_W / _PANEL_H
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    bx0, bx1 = min(xs), max(xs)
    by0, by1 = min(ys), max(ys)
    bw, bh = max(bx1 - bx0, 80), max(by1 - by0, 80)
    pad = int(max(bw, bh) * pad_ratio)
    cx = (bx0 + bx1) // 2
    cy = (by0 + by1) // 2
    half_w = max(bw // 2 + pad, int(fw * min_frac / 2))
    half_h = max(bh // 2 + pad, int(fh * min_frac / 2))
    if half_w / max(half_h, 1) < aspect:
        half_w = int(half_h * aspect)
    else:
        half_h = int(half_w / aspect)
    x0 = max(0, cx - half_w)
    x1 = min(fw, cx + half_w)
    y0 = max(0, cy - half_h)
    y1 = min(fh, cy + half_h)
    if x1 - x0 < int(fw * min_frac):
        cx = (x0 + x1) // 2
        half = int(fw * min_frac / 2)
        x0, x1 = max(0, cx - half), min(fw, cx + half)
    if y1 - y0 < int(fh * min_frac):
        cy = (y0 + y1) // 2
        half = int(fh * min_frac / 2)
        y0, y1 = max(0, cy - half), min(fh, cy + half)
    return x0, y0, x1, y1


def _strip_crop_rect(
    ctx: PassExplainContext,
    frame_indices: tuple[int, ...],
    *tids: int,
) -> tuple[int, int, int, int] | None:
    """One crop per filmstrip so consecutive panels align."""
    points: list[tuple[int, int]] = []
    ref_shape: tuple[int, int, int] | None = None
    for fi in frame_indices:
        dets = ctx.dets_by_frame.get(fi)
        frame = ctx.frames.get(fi)
        if dets is None or frame is None:
            continue
        ref_shape = frame.shape
        points.extend(_action_focus_points(dets, *tids))
    if ref_shape is None or not points:
        return None
    return _crop_rect_from_points(ref_shape, points)


def _apply_crop(frame: np.ndarray, crop: tuple[int, int, int, int] | None) -> np.ndarray:
    if crop is None:
        return frame
    x0, y0, x1, y1 = crop
    if x1 <= x0 or y1 <= y0:
        return frame
    return frame[y0:y1, x0:x1].copy()


def _fit_panel_frame(
    frame: np.ndarray,
    *,
    crop: tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    return _letterbox_frame(_apply_crop(frame, crop))


def _passer_is_goalkeeper(
    pass_event: InferredPass, dets_by_frame: dict[int, sv.Detections]
) -> bool:
    dets = dets_by_frame.get(pass_event.frame_idx)
    if dets is None or dets.tracker_id is None:
        return False
    mask = dets.tracker_id == pass_event.passer_tid
    if not mask.any():
        return False
    idx = int(np.flatnonzero(mask)[0])
    return int(dets.class_id[idx]) == ROLE_GOALKEEPER


def _find_passer_touch_run(
    release: int,
    passer_tid: int,
    timeline_by_frame: dict[int, CarrierFrameState],
    *,
    touch_kind: Literal["control", "reception"],
    lookback: int = 24,
) -> tuple[int, ...]:
    """Longest consecutive touch run for passer ending at or before release."""
    start = max(1, release - lookback)
    best_run: list[int] = []
    current: list[int] = []
    for fi in range(start, release + 1):
        st = timeline_by_frame.get(fi)
        in_touch = (
            st is not None
            and not st.in_flight
            and st.active_tid == passer_tid
            and st.active_kind == touch_kind
        )
        if in_touch:
            current.append(fi)
        else:
            if len(current) > len(best_run):
                best_run = current
            current = []
    if len(current) > len(best_run):
        best_run = current
    return tuple(best_run)


def _find_passer_control_run(
    release: int,
    passer_tid: int,
    timeline_by_frame: dict[int, CarrierFrameState],
    *,
    min_control_frames: int,
    lookback: int = 24,
) -> tuple[int, ...]:
    """Control run for passer; falls back to reception if control is too short."""
    control = _find_passer_touch_run(
        release, passer_tid, timeline_by_frame, touch_kind="control", lookback=lookback
    )
    if len(control) >= min_control_frames:
        return control[-min_control_frames:]
    reception = _find_passer_touch_run(
        release, passer_tid, timeline_by_frame, touch_kind="reception", lookback=lookback
    )
    if len(reception) >= min_control_frames:
        return reception[:min_control_frames]
    return control or reception


def _flight_start_frame(
    release: int,
    arrival: int,
    passer_tid: int,
    timeline_by_frame: dict[int, CarrierFrameState],
) -> int:
    """First frame after release where the ball is no longer at the passer's feet."""
    for fi in range(release + 1, arrival + 1):
        st = timeline_by_frame.get(fi)
        if st is None:
            return fi
        if st.in_flight or st.active_tid != passer_tid:
            return fi
    return min(release + 1, arrival)


def _pass_anchor_points(
    pass_event: InferredPass,
    dets_by_frame: dict[int, sv.Detections],
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Pass corridor endpoints: passer feet at release, receiver feet at arrival."""
    release = pass_event.frame_idx
    arrival = release + pass_event.gap_frames
    release_dets = dets_by_frame[release]
    arrival_dets = dets_by_frame.get(arrival, release_dets)
    p0 = _feet_for_tid(release_dets, pass_event.passer_tid)
    if p0 is None:
        ball = ball_xy(release_dets)
        p0 = (int(ball[0]), int(ball[1])) if ball is not None else (0, 0)
    p1 = _feet_for_tid(arrival_dets, pass_event.receiver_tid)
    if p1 is None:
        ball = ball_xy(arrival_dets)
        p1 = (int(ball[0]), int(ball[1])) if ball is not None else p0
    return p0, p1


def _ball_along_pass_segment(
    fi: int,
    release: int,
    arrival: int,
    p0: tuple[int, int],
    p1: tuple[int, int],
) -> tuple[int, int]:
    gap = max(arrival - release, 1)
    t = float(np.clip((fi - release) / gap, 0.0, 1.0))
    x = p0[0] + t * (p1[0] - p0[0])
    y = p0[1] + t * (p1[1] - p0[1])
    return int(x), int(y)


def _pass_known_ball_positions(
    pass_event: InferredPass,
    dets_by_frame: dict[int, sv.Detections],
) -> dict[int, tuple[float, float]]:
    """Raw detector ball positions across the pass window (no feet-based filtering)."""
    release = pass_event.frame_idx
    arrival = release + pass_event.gap_frames
    known: dict[int, tuple[float, float]] = {}
    for fi in range(release, arrival + 1):
        dets = dets_by_frame.get(fi)
        if dets is None:
            continue
        ball = ball_xy(dets)
        if ball is not None:
            known[fi] = (float(ball[0]), float(ball[1]))
    return known


def _interpolate_ball_xy(
    fi: int, known: dict[int, tuple[float, float]]
) -> tuple[int, int] | None:
    if fi in known:
        x, y = known[fi]
        return int(x), int(y)
    if not known:
        return None
    before = [f for f in known if f <= fi]
    after = [f for f in known if f >= fi]
    if before and after:
        f0, f1 = max(before), min(after)
        if f0 == f1:
            x, y = known[f0]
            return int(x), int(y)
        t = (fi - f0) / (f1 - f0)
        x0, y0 = known[f0]
        x1, y1 = known[f1]
        return int(x0 + t * (x1 - x0)), int(y0 + t * (y1 - y0))
    if before:
        x, y = known[max(before)]
        return int(x), int(y)
    if after:
        x, y = known[min(after)]
        return int(x), int(y)
    return None


def _pass_ball_track_point(
    pass_event: InferredPass,
    fi: int,
    dets_by_frame: dict[int, sv.Detections],
    *,
    known_balls: dict[int, tuple[float, float]] | None = None,
) -> tuple[int, int]:
    """Raw ball when detected; else interpolate gaps; else walk the pass chord."""
    release = pass_event.frame_idx
    arrival = release + pass_event.gap_frames
    p0, p1 = _pass_anchor_points(pass_event, dets_by_frame)
    segment_pt = _ball_along_pass_segment(fi, release, arrival, p0, p1)

    dets = dets_by_frame.get(fi)
    if dets is not None:
        raw = ball_xy(dets)
        if raw is not None:
            return int(raw[0]), int(raw[1])

    known = known_balls or _pass_known_ball_positions(pass_event, dets_by_frame)
    before = [f for f in known if f <= fi]
    after = [f for f in known if f >= fi]
    if before and after and min(after) > max(before):
        bridged = _interpolate_ball_xy(fi, known)
        if bridged is not None:
            return bridged
    return segment_pt


def _pass_ball_point_for_frame(
    pass_event: InferredPass,
    fi: int,
    dets_by_frame: dict[int, sv.Detections],
    *,
    dets: sv.Detections | None = None,
    known_balls: dict[int, tuple[float, float]] | None = None,
) -> tuple[int, int] | None:
    """Ball marker — same as the main demo: trust ``ball_xy``, bridge only when missing."""
    return _pass_ball_track_point(
        pass_event, fi, dets_by_frame, known_balls=known_balls
    )


def _find_passer_demo_confirm_frame(
    pass_event: InferredPass,
    timeline_by_frame: dict[int, CarrierFrameState],
    *,
    min_control_frames: int,
) -> int:
    """Last control-touch frame before release — same streak as pass-detection demo."""
    release = pass_event.frame_idx
    control_run = _find_passer_control_run(
        release,
        pass_event.passer_tid,
        timeline_by_frame,
        min_control_frames=min_control_frames,
    )
    if control_run:
        return control_run[-1]
    return release


def _receiver_confirm_search_end(
    pass_event: InferredPass,
    dets_by_frame: dict[int, sv.Detections] | None,
    *,
    lookahead: int = 12,
) -> int:
    arrival = pass_event.frame_idx + pass_event.gap_frames
    end = arrival + lookahead
    if dets_by_frame:
        end = max(end, max(dets_by_frame))
    return end


def _find_receiver_control_run(
    release: int,
    receiver_tid: int,
    timeline_by_frame: dict[int, CarrierFrameState],
    *,
    search_end: int,
) -> tuple[int, ...]:
    """First post-release control streak at the receiver — same kind as pass demo."""
    current: list[int] = []
    for fi in range(release + 1, search_end + 1):
        st = timeline_by_frame.get(fi)
        in_control = (
            st is not None
            and not st.in_flight
            and st.active_tid == receiver_tid
            and st.active_kind == "control"
        )
        if in_control:
            current.append(fi)
        elif current:
            break
    return tuple(current)


def _find_receiver_tight_feet_frame(
    pass_event: InferredPass,
    dets_by_frame: dict[int, sv.Detections],
    *,
    search_end: int,
    start_frame: int | None = None,
    max_distance_px: float = _RECEIVER_EXPLAIN_TIGHT_PX,
) -> int | None:
    """First frame with the ball sitting inside the tight feet ellipse."""
    release = pass_event.frame_idx
    receiver_tid = pass_event.receiver_tid
    begin = release + 1 if start_frame is None else start_frame
    for fi in range(begin, search_end + 1):
        dets = dets_by_frame.get(fi)
        if dets is None:
            continue
        ball = ball_xy(dets)
        feet = _feet_for_tid(dets, receiver_tid)
        if ball is None or feet is None:
            continue
        dy = abs(float(ball[1] - feet[1]))
        if dy > AERIAL_DY_THRESHOLD_PX:
            continue
        if float(np.hypot(ball[0] - feet[0], ball[1] - feet[1])) <= max_distance_px:
            return fi
    return None


def _find_receiver_visual_confirm_frame(
    pass_event: InferredPass,
    dets_by_frame: dict[int, sv.Detections],
    *,
    timeline_by_frame: dict[int, CarrierFrameState] | None = None,
    min_control_frames: int = _MIN_CONTROL,
) -> int:
    """Explain-only receiver lock — demo control streak plus a tight feet frame.

    Requires ``min_control_frames`` consecutive demo ``control`` touches at the
    receiver (same streak gate as pass detection), then freezes on the first
    frame where the ball sits inside the tight feet ellipse (≤12px). Does not
    affect which passes are inferred for the full video.
    """
    release = pass_event.frame_idx
    arrival = release + pass_event.gap_frames
    search_end = _receiver_confirm_search_end(pass_event, dets_by_frame)

    if not timeline_by_frame:
        return arrival

    control_run = _find_receiver_control_run(
        release,
        pass_event.receiver_tid,
        timeline_by_frame,
        search_end=search_end,
    )
    if len(control_run) < min_control_frames:
        return arrival

    tight = _find_receiver_tight_feet_frame(
        pass_event,
        dets_by_frame,
        search_end=search_end,
        start_frame=control_run[0],
    )
    if tight is not None:
        return tight
    return control_run[min_control_frames - 1]


def _receiver_arrival_panels(
    release: int,
    confirm: int,
    *,
    min_arrival_frames: int,
) -> tuple[int, ...]:
    """Panels leading up to visual control confirm — tail ends on the lock frame."""
    start = max(release + 1, confirm - min_arrival_frames + 1)
    return tuple(range(start, confirm + 1))


def _pick_flight_frame(
    pass_event: InferredPass,
    release: int,
    arrival: int,
    passer_tid: int,
    timeline_by_frame: dict[int, CarrierFrameState],
    dets_by_frame: dict[int, sv.Detections] | None,
) -> int:
    """In-flight frame with bridged ball, well separated from passer."""
    start = _flight_start_frame(release, arrival, passer_tid, timeline_by_frame)
    if not dets_by_frame:
        return start
    known = _pass_known_ball_positions(pass_event, dets_by_frame)
    gap = max(arrival - start, 1)
    # Stay in mid-flight — not at reception where the receiver arrives.
    search_end = max(start, arrival - max(6, gap // 4))
    best = start
    best_score = -1.0
    for fi in range(start, search_end + 1):
        dets = dets_by_frame.get(fi)
        if dets is None:
            continue
        ball_pt = _pass_ball_point_for_frame(
            pass_event, fi, dets_by_frame, known_balls=known
        )
        if ball_pt is None:
            continue
        passer_feet = _feet_for_tid(dets, passer_tid)
        if passer_feet is None:
            dist = 120.0
        else:
            dist = float(np.hypot(ball_pt[0] - passer_feet[0], ball_pt[1] - passer_feet[1]))
        mid = start + gap * 0.45
        mid_bonus = 80.0 - min(abs(fi - mid) * 2.0, 80.0)
        score = dist + mid_bonus
        if score > best_score:
            best_score = score
            best = fi
    return best


def build_fixed_strip_plan(pass_event: InferredPass) -> PassStripPlan:
    """Anchor panels on release/arrival when carrier timeline is too sparse to infer them."""
    release = pass_event.frame_idx
    arrival = release + pass_event.gap_frames
    flight = release + max(1, pass_event.gap_frames // 2)
    passer_frames = (max(1, release - 2), max(1, release - 1), release)
    confirm = arrival
    recv = _receiver_arrival_panels(
        release, confirm, min_arrival_frames=_MIN_ARRIVAL
    )
    return PassStripPlan(
        passer_frames=passer_frames,
        flight_frames=(flight,),
        receiver_frames=recv,
        summary_frame=arrival,
        passer_confirm_frame=passer_frames[-1],
        receiver_confirm_frame=confirm,
    )


def build_strip_plan(
    pass_event: InferredPass,
    timeline_by_frame: dict[int, CarrierFrameState],
    *,
    dets_by_frame: dict[int, sv.Detections] | None = None,
    min_control_frames: int = _MIN_CONTROL,
    min_arrival_frames: int = _MIN_ARRIVAL,
) -> PassStripPlan:
    """Pick consecutive frames that show control → flight → arrival streaks."""
    release = pass_event.frame_idx
    arrival = release + pass_event.gap_frames
    passer_tid = pass_event.passer_tid
    receiver_tid = pass_event.receiver_tid

    passer_frames = list(
        _find_passer_control_run(
            release,
            passer_tid,
            timeline_by_frame,
            min_control_frames=min_control_frames,
        )
    )
    if len(passer_frames) < min_control_frames:
        raise ValueError(
            f"Pass #{passer_tid}→#{receiver_tid} has only {len(passer_frames)} "
            f"touch frames before release (need {min_control_frames})"
        )

    flight_frame = _pick_flight_frame(
        pass_event,
        passer_frames[-1],
        arrival,
        passer_tid,
        timeline_by_frame,
        dets_by_frame,
    )

    dets_map = dets_by_frame or {}
    passer_confirm = _find_passer_demo_confirm_frame(
        pass_event,
        timeline_by_frame,
        min_control_frames=min_control_frames,
    )
    confirm = _find_receiver_visual_confirm_frame(
        pass_event,
        dets_map,
        timeline_by_frame=timeline_by_frame,
        min_control_frames=min_control_frames,
    )
    receiver_frames = _receiver_arrival_panels(
        release, confirm, min_arrival_frames=min_arrival_frames
    )

    # First N touches of the run — when control is earned, not the tail of a long dribble.
    return PassStripPlan(
        passer_frames=tuple(passer_frames[:min_control_frames]),
        flight_frames=(flight_frame,),
        receiver_frames=receiver_frames,
        summary_frame=arrival,
        passer_confirm_frame=passer_confirm,
        receiver_confirm_frame=confirm,
    )


def _player_subset(dets: sv.Detections, tid: int) -> sv.Detections | None:
    if dets.tracker_id is None:
        return None
    mask = dets.tracker_id == tid
    if not mask.any():
        return None
    return dets[mask]


def _apply_focus_spotlights(
    dimmed: np.ndarray,
    original: np.ndarray,
    centers: list[tuple[int, int]],
    *,
    radius: int = _SPOTLIGHT_RADIUS,
    strength: float = _SPOTLIGHT_STRENGTH,
) -> np.ndarray:
    out = dimmed
    for center in centers:
        out = draw_carrier_spotlight(
            out, original, center, radius=radius, strength=strength
        )
    return out


def _draw_locked_outer_ellipse(image: np.ndarray, box: np.ndarray) -> None:
    """Soft white outer ring when a passer/receiver beat is locked."""
    x0, y0, x1, y1 = box
    cx, cy = int((x0 + x1) / 2), int(y1)
    w = int((x1 - x0) * 0.92)
    h = max(10, int(w / 3))
    overlay = image.copy()
    cv2.ellipse(
        overlay,
        (cx, cy),
        (w + 14, h + 8),
        0,
        0,
        360,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.addWeighted(overlay, 0.72, image, 0.28, 0, image)


def _draw_player_ellipse(
    image: np.ndarray,
    dets: sv.Detections,
    tid: int,
    color_bgr: tuple[int, int, int],
    *,
    thickness: int = 2,
) -> None:
    subset = _player_subset(dets, tid)
    if subset is None:
        return
    ellipse = sv.EllipseAnnotator(
        color=sv.Color(color_bgr[2], color_bgr[1], color_bgr[0]),
        thickness=thickness,
    )
    image[:] = ellipse.annotate(image, subset)


def _draw_focus_player(
    image: np.ndarray,
    dets: sv.Detections,
    tid: int,
    color_bgr: tuple[int, int, int],
    *,
    prominent: bool = True,
    emphasis: float | None = None,
    locked: bool = False,
) -> None:
    """Single team ellipse; optional white outer ring when locked."""
    level = 1.0 if prominent else 0.45
    if emphasis is not None:
        level = float(np.clip(emphasis, 0.0, 1.0))
    if level < 0.22:
        return
    thick = max(1, int(1 + 2 * level))
    _draw_player_ellipse(image, dets, tid, color_bgr, thickness=thick)
    if locked:
        box = _get_player_box(dets, tid)
        if box is not None:
            _draw_locked_outer_ellipse(image, box)


def _highlight_ball_at(image: np.ndarray, ball_pt: tuple[int, int]) -> None:
    """Small ring on the ball — spotlight carries the main emphasis."""
    bx, by = ball_pt
    cv2.circle(image, (bx, by), 11, (40, 220, 255), 2, cv2.LINE_AA)
    cv2.circle(image, (bx, by), 4, (255, 255, 255), -1, cv2.LINE_AA)


def _draw_pass_ball_marker(
    image: np.ndarray,
    dets: sv.Detections,
    ball_pt: tuple[int, int] | None,
) -> np.ndarray:
    """Ring + detector marker when the highlight sits on a live ball detection."""
    if ball_pt is None:
        return image
    _highlight_ball_at(image, ball_pt)
    raw = ball_xy(dets)
    if raw is None:
        return image
    rx, ry = int(raw[0]), int(raw[1])
    if abs(rx - ball_pt[0]) <= 10 and abs(ry - ball_pt[1]) <= 10:
        return annotate_ball(image, dets)
    return image


def _highlight_ball(image: np.ndarray, dets: sv.Detections) -> None:
    pos = ball_xy(dets)
    if pos is None:
        return
    _highlight_ball_at(image, (int(pos[0]), int(pos[1])))


def _draw_anchor_player(
    image: np.ndarray,
    dets: sv.Detections,
    tid: int,
    color_bgr: tuple[int, int, int],
) -> None:
    """Passer marker in the dimmed area — single ellipse only."""
    _draw_player_ellipse(image, dets, tid, color_bgr, thickness=2)


def _draw_dashed_line(
    image: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    color_bgr: tuple[int, int, int],
    *,
    alpha: float = 0.45,
) -> None:
    layer = image.copy()
    dx, dy = end[0] - start[0], end[1] - start[1]
    length = float(np.hypot(dx, dy))
    if length < 12:
        return
    ux, uy = dx / length, dy / length
    traveled = 0.0
    draw = True
    while traveled < length:
        seg = 12 if draw else 8
        seg = min(seg, length - traveled)
        x0 = int(start[0] + ux * traveled)
        y0 = int(start[1] + uy * traveled)
        x1 = int(start[0] + ux * (traveled + seg))
        y1 = int(start[1] + uy * (traveled + seg))
        if draw:
            cv2.line(layer, (x0, y0), (x1, y1), color_bgr, 2, cv2.LINE_AA)
        traveled += seg
        draw = not draw
    image[:] = cv2.addWeighted(layer, alpha, image, 1.0 - alpha, 0)


def _build_flight_frame(
    full: np.ndarray,
    dets: sv.Detections,
    passer_tid: int,
    color_bgr: tuple[int, int, int],
    *,
    ball_pt: tuple[int, int] | None,
) -> np.ndarray:
    """Ball lit and separated; passer stays annotated in the dark."""
    out = _dim_frame(full, _FOCUS_DIM)
    _draw_anchor_player(out, dets, passer_tid, color_bgr)
    passer_feet = _feet_for_tid(dets, passer_tid)
    if ball_pt is not None:
        out = draw_carrier_spotlight(
            out, full, ball_pt, radius=220, strength=0.88
        )
        out = _draw_pass_ball_marker(out, dets, ball_pt)
        if passer_feet is not None:
            _draw_dashed_line(out, passer_feet, ball_pt, color_bgr)
    return out


def _build_focus_frame(
    full: np.ndarray,
    dets: sv.Detections,
    *,
    focus_tids: tuple[int, ...],
    anchor_tids: tuple[int, ...] = (),
    color_bgr: tuple[int, int, int],
    show_ball: bool = False,
) -> np.ndarray:
    """Dim everything; relight focus players (and ball) with team-colored markers."""
    dimmed = _dim_frame(full, _FOCUS_DIM)
    centers: list[tuple[int, int]] = []
    for tid in focus_tids:
        feet = _feet_for_tid(dets, tid)
        if feet is not None:
            centers.append(feet)
    if show_ball:
        ball = ball_xy(dets)
        if ball is not None:
            centers.append((int(ball[0]), int(ball[1])))
    out = _apply_focus_spotlights(dimmed, full, centers)
    if anchor_tids:
        anchor_centers = [
            f for tid in anchor_tids if (f := _feet_for_tid(dets, tid)) is not None
        ]
        out = _apply_focus_spotlights(
            out,
            full,
            anchor_centers,
            radius=int(_SPOTLIGHT_RADIUS * 0.65),
            strength=_ANCHOR_SPOTLIGHT_STRENGTH,
        )
    for tid in anchor_tids:
        _draw_focus_player(out, dets, tid, color_bgr, prominent=False)
    for tid in focus_tids:
        _draw_focus_player(out, dets, tid, color_bgr, prominent=True, locked=False)
    if show_ball:
        _highlight_ball(out, dets)
        out = annotate_ball(out, dets)
    return out


def _draw_gutter_step_dots(
    gutter: np.ndarray,
    *,
    step: int,
    total: int,
    color_bgr: tuple[int, int, int],
    origin: tuple[int, int],
) -> None:
    """Minimal step dots."""
    x0, y0 = origin
    r = 4
    gap = 14
    dim = (48, 48, 56)
    for i in range(total):
        cx = x0 + i * gap
        fill = color_bgr if i < step else dim
        cv2.circle(gutter, (cx, y0), r, fill, -1, cv2.LINE_AA)


def _draw_gutter_progress_bar(
    gutter: np.ndarray,
    *,
    emphasis: float,
    accent_bgr: tuple[int, int, int],
) -> None:
    gh, gw = gutter.shape[:2]
    bar_x = gw - 10
    top, bottom = 68, gh - 20
    cv2.rectangle(gutter, (bar_x - 1, top), (bar_x + 1, bottom), (42, 42, 50), -1)
    fill_h = int((bottom - top) * float(np.clip(emphasis, 0.0, 1.0)))
    if fill_h > 0:
        cv2.rectangle(
            gutter,
            (bar_x - 1, bottom - fill_h),
            (bar_x + 1, bottom),
            accent_bgr,
            -1,
        )


def _draw_panel_badge(
    panel: np.ndarray,
    badge: str,
    *,
    accent_bgr: tuple[int, int, int],
    locked: bool,
    layout: Literal["talk", "social"],
) -> None:
    """Compact Roboflow-style chip (dark glass + purple rail)."""
    scale = _ui_scale(layout)
    fs = (0.50 if locked else 0.44) * scale
    thick = 1
    (tw, th), baseline = cv2.getTextSize(
        badge, cv2.FONT_HERSHEY_SIMPLEX, fs, thick
    )
    pad_x, pad_y = 12, 8
    x0, y0 = 12, 12
    x1, y1 = x0 + tw + pad_x * 2, y0 + th + baseline + pad_y * 2
    overlay = panel.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), _BADGE_BG_BGR, -1)
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (50, 50, 60), 1)
    rail = ROBOFLOW_PURPLE_BGR if locked else accent_bgr
    cv2.rectangle(overlay, (x0, y0), (x0 + 3, y1), rail, -1)
    panel[:] = cv2.addWeighted(overlay, 0.68, panel, 0.32, 0)
    draw_text_shadow(
        panel,
        badge,
        (x0 + pad_x, y0 + pad_y + th),
        font_scale=fs,
        color_bgr=(245, 245, 248) if locked else (210, 210, 218),
        thickness=thick,
    )


def _draw_panel_step_bar(
    panel: np.ndarray,
    *,
    step: int,
    total: int,
    accent_bgr: tuple[int, int, int],
) -> None:
    h, w = panel.shape[:2]
    bar_h = 3
    y0 = h - bar_h - 8
    x0, x1 = 12, w - 12
    cv2.rectangle(panel, (x0, y0), (x1, y0 + bar_h), (38, 38, 46), -1)
    if total > 0:
        fill_w = int((x1 - x0) * step / total)
        if fill_w > 0:
            cv2.rectangle(
                panel, (x0, y0), (x0 + fill_w, y0 + bar_h), accent_bgr, -1
            )


def _decorate_panel_frame(
    panel: np.ndarray,
    *,
    badge: str,
    step: int,
    total: int,
    emphasis: float,
    locked: bool,
    accent_bgr: tuple[int, int, int],
    layout: Literal["talk", "social"],
) -> np.ndarray:
    out = panel.copy()
    _draw_panel_step_bar(out, step=step, total=total, accent_bgr=accent_bgr)
    _draw_panel_badge(
        out, badge, accent_bgr=accent_bgr, locked=locked, layout=layout
    )
    if locked:
        cv2.rectangle(
            out, (0, 0), (out.shape[1] - 1, out.shape[0] - 1), ROBOFLOW_PURPLE_BGR, 1
        )
    return out


def _compose_panel_row(
    frame_img: np.ndarray,
    frame_idx: int,
    label: str,
    color_bgr: tuple[int, int, int],
    *,
    layout: Literal["talk", "social"],
    step: int | None = None,
    step_total: int | None = None,
    sublabel: str = "",
    badge: str = "",
    emphasis: float = 1.0,
    locked: bool = False,
    accent_bgr: tuple[int, int, int] | None = None,
    crop: tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    """Left gutter beside a cropped, letterboxed 16:9 panel."""
    scale = _ui_scale(layout)
    accent = accent_bgr or color_bgr
    frame_img = _fit_panel_frame(frame_img, crop=crop)
    if badge and step is not None and step_total is not None:
        frame_img = _decorate_panel_frame(
            frame_img,
            badge=badge,
            step=step,
            total=step_total,
            emphasis=emphasis,
            locked=locked,
            accent_bgr=accent,
            layout=layout,
        )
    gutter = np.full((_PANEL_H, _GUTTER_W, 3), _PANEL_BG_BGR, dtype=np.uint8)
    stripe_w = 3
    stripe_color = ROBOFLOW_PURPLE_BGR if locked else accent
    cv2.rectangle(gutter, (0, 0), (stripe_w, _PANEL_H), stripe_color, -1)
    _draw_gutter_progress_bar(gutter, emphasis=emphasis, accent_bgr=accent)
    label_scale = (0.50 if locked else 0.44) * scale
    draw_text_shadow(
        gutter,
        label,
        (14, int(36 * scale)),
        font_scale=label_scale,
        color_bgr=ROBOFLOW_PURPLE_BGR if locked else (230, 230, 235),
        thickness=1,
    )
    y_extra = int(72 * scale)
    if step is not None and step_total is not None and step_total > 1:
        _draw_gutter_step_dots(
            gutter,
            step=step,
            total=step_total,
            color_bgr=accent,
            origin=(18, y_extra),
        )
    return np.hstack([gutter, frame_img])


def _draw_strip_title(
    canvas: np.ndarray,
    title: str,
    subtitle: str,
    *,
    layout: Literal["talk", "social"],
) -> None:
    scale = _ui_scale(layout)
    pad = 14
    draw_text_shadow(
        canvas,
        title,
        (pad, pad + int(22 * scale)),
        font_scale=0.58 * scale,
        color_bgr=(255, 255, 255),
        thickness=2,
    )
    if subtitle:
        draw_text_shadow(
            canvas,
            subtitle,
            (pad, pad + int(48 * scale)),
            font_scale=0.38 * scale,
            color_bgr=(175, 175, 185),
            thickness=1,
        )


def _passer_control_emphasis(ctx: PassExplainContext, frame_idx: int) -> float:
    """0..1 ramp across the explain window (weak -> locked)."""
    panels = ctx.strip_plan.passer_frames
    if frame_idx not in panels or len(panels) <= 1:
        return 1.0
    idx = panels.index(frame_idx)
    return (idx + 1) / len(panels)


def _passer_label(ctx: PassExplainContext, frame_idx: int) -> str:
    """Label the 3-frame explain window (1/3, 2/3), not the machine's total streak."""
    panels = ctx.strip_plan.passer_frames
    min_c = ctx.min_control_frames
    if frame_idx not in panels:
        return "CONTROL"
    panel_idx = panels.index(frame_idx) + 1
    if frame_idx == panels[-1]:
        return "PASSER LOCKED"
    return f"CONTROL {panel_idx}/{min_c}"


def _flight_label(ctx: PassExplainContext, frame_idx: int) -> str:
    return "BALL TRAVELLING"


def _receiver_label(ctx: PassExplainContext, frame_idx: int) -> str:
    panels = ctx.strip_plan.receiver_frames
    min_a = ctx.min_arrival_frames
    if frame_idx not in panels:
        return "ARRIVAL"
    panel_idx = panels.index(frame_idx) + 1
    if frame_idx == panels[-1]:
        return "RECEIVER LOCKED"
    return f"ARRIVAL {panel_idx}/{min_a}"


def _streak_emphasis(panel_idx: int, total: int) -> float:
    if total <= 1:
        return 1.0
    return panel_idx / total


def _passer_panel_badge(panel_idx: int, total: int, *, locked: bool) -> str:
    if locked:
        return "PASSER LOCKED"
    return f"TOUCH {panel_idx}/{total}"


def _receiver_panel_badge(panel_idx: int, total: int, *, locked: bool) -> str:
    if locked:
        return "RECEIVER LOCKED"
    return f"ARRIVAL {panel_idx}/{total}"


def _explain_streak_sublabel(
    panel_idx: int,
    total: int,
    *,
    locked: bool,
    kind: Literal["control", "arrival"],
) -> str:
    """Sublabel matches the 3-frame explain window, not the machine's total streak."""
    if locked:
        return f"{panel_idx}/{total} touches -> locked"
    verb = "control" if kind == "control" else "arrival"
    return f"{verb} touch {panel_idx}/{total}"


def _build_passer_control_frame(
    full: np.ndarray,
    dets: sv.Detections,
    passer_tid: int,
    color_bgr: tuple[int, int, int],
    *,
    emphasis: float,
    locked: bool = False,
    ball_pt: tuple[int, int] | None = None,
) -> np.ndarray:
    """Passer strip: each step looks visibly different (dim world -> lit -> locked)."""
    dim_level = 0.30 - 0.14 * emphasis
    dimmed = _dim_frame(full, dim_level)
    passer_feet = _feet_for_tid(dets, passer_tid)
    ball = ball_pt

    if emphasis < 0.45:
        out = dimmed
        if passer_feet is not None:
            out = draw_carrier_spotlight(
                out, full, passer_feet, radius=110, strength=0.42
            )
        _draw_focus_player(out, dets, passer_tid, color_bgr, emphasis=0.22)
        return out

    centers: list[tuple[int, int]] = []
    if passer_feet is not None:
        centers.append(passer_feet)
    if emphasis >= 0.55 and ball is not None:
        centers.append(ball)
    strength = 0.40 + 0.52 * emphasis
    radius = int(120 + 100 * emphasis)
    out = _apply_focus_spotlights(
        dimmed, full, centers, radius=radius, strength=strength
    )
    show_locked = locked
    _draw_focus_player(
        out,
        dets,
        passer_tid,
        color_bgr,
        emphasis=emphasis,
        locked=show_locked,
    )
    if emphasis >= 0.55 and ball is not None and passer_feet is not None:
        out = _draw_pass_ball_marker(out, dets, ball)
        if show_locked:
            cv2.line(
                out,
                passer_feet,
                ball,
                color_bgr,
                3,
                cv2.LINE_AA,
            )
    return out


def _build_receiver_arrival_frame(
    full: np.ndarray,
    dets: sv.Detections,
    receiver_tid: int,
    passer_tid: int,
    color_bgr: tuple[int, int, int],
    *,
    emphasis: float,
    locked: bool = False,
    ball_pt: tuple[int, int] | None = None,
) -> np.ndarray:
    """Receiver strip: passer stays dim; receiver ramps up like passer control."""
    dim_level = 0.30 - 0.14 * emphasis
    dimmed = _dim_frame(full, dim_level)
    receiver_feet = _feet_for_tid(dets, receiver_tid)
    ball = ball_pt

    if emphasis < 0.45:
        out = dimmed
        _draw_anchor_player(out, dets, passer_tid, color_bgr)
        if receiver_feet is not None:
            out = draw_carrier_spotlight(
                out, full, receiver_feet, radius=110, strength=0.42
            )
        _draw_focus_player(out, dets, receiver_tid, color_bgr, emphasis=0.22)
        return out

    centers: list[tuple[int, int]] = []
    if receiver_feet is not None:
        centers.append(receiver_feet)
    if emphasis >= 0.55 and ball is not None:
        centers.append(ball)
    out = _apply_focus_spotlights(
        dimmed,
        full,
        centers,
        radius=int(120 + 100 * emphasis),
        strength=0.40 + 0.52 * emphasis,
    )
    _draw_anchor_player(out, dets, passer_tid, color_bgr)
    show_locked = locked
    _draw_focus_player(
        out,
        dets,
        receiver_tid,
        color_bgr,
        emphasis=emphasis,
        locked=show_locked,
    )
    if emphasis >= 0.55 and ball is not None and receiver_feet is not None:
        out = _draw_pass_ball_marker(out, dets, ball)
        if show_locked:
            cv2.line(
                out,
                receiver_feet,
                ball,
                color_bgr,
                3,
                cv2.LINE_AA,
            )
    return out


def _pass_ball_for_ctx(
    ctx: PassExplainContext,
    frame_idx: int,
    *,
    known_balls: dict[int, tuple[float, float]] | None = None,
) -> tuple[int, int] | None:
    return _pass_ball_point_for_frame(
        ctx.pass_event,
        frame_idx,
        ctx.dets_by_frame,
        dets=ctx.dets_by_frame.get(frame_idx),
        known_balls=known_balls,
    )


def _render_passer_panel(
    ctx: PassExplainContext,
    frame_idx: int,
    *,
    layout: Literal["talk", "social"],
    crop: tuple[int, int, int, int] | None = None,
    emphasis: float | None = None,
    show_locked: bool | None = None,
) -> np.ndarray:
    p = ctx.pass_event
    dets = ctx.dets_by_frame[frame_idx]
    color = _team_color(p.team)
    panels = ctx.strip_plan.passer_frames
    panel_idx = panels.index(frame_idx) + 1 if frame_idx in panels else 1
    level = (
        emphasis
        if emphasis is not None
        else _passer_control_emphasis(ctx, frame_idx)
    )
    confirm = ctx.strip_plan.passer_confirm_frame
    locked = (
        show_locked
        if show_locked is not None
        else frame_idx >= confirm
    )
    frame_img = _build_passer_control_frame(
        ctx.frames[frame_idx],
        dets,
        p.passer_tid,
        color,
        emphasis=level,
        locked=locked,
        ball_pt=_pass_ball_for_ctx(ctx, frame_idx),
    )
    sublabel = _explain_streak_sublabel(
        panel_idx, len(panels), locked=locked, kind="control"
    )
    return _compose_panel_row(
        frame_img,
        frame_idx,
        _passer_label(ctx, frame_idx),
        color,
        layout=layout,
        step=panel_idx,
        step_total=len(panels),
        sublabel=sublabel,
        badge=_passer_panel_badge(panel_idx, len(panels), locked=locked),
        emphasis=level,
        locked=locked,
        accent_bgr=ROBOFLOW_PURPLE_BGR if locked else color,
        crop=crop,
    )


def _render_flight_panel(
    ctx: PassExplainContext,
    frame_idx: int,
    *,
    layout: Literal["talk", "social"],
    crop: tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    p = ctx.pass_event
    dets = ctx.dets_by_frame[frame_idx]
    color = _team_color(p.team)
    release = p.frame_idx
    arrival = release + p.gap_frames
    known = _pass_known_ball_positions(p, ctx.dets_by_frame)
    ball_pt = _pass_ball_for_ctx(ctx, frame_idx, known_balls=known)
    frame_img = _build_flight_frame(
        ctx.frames[frame_idx], dets, p.passer_tid, color, ball_pt=ball_pt
    )
    return _compose_panel_row(
        frame_img,
        frame_idx,
        _flight_label(ctx, frame_idx),
        color,
        layout=layout,
        step=1,
        step_total=1,
        sublabel="ball between passer and receiver",
        badge="BALL IN FLIGHT",
        emphasis=1.0,
        locked=False,
        accent_bgr=_FLIGHT_ACCENT_BGR,
        crop=crop,
    )


def _render_receiver_panel(
    ctx: PassExplainContext,
    frame_idx: int,
    *,
    layout: Literal["talk", "social"],
    crop: tuple[int, int, int, int] | None = None,
    emphasis: float | None = None,
    show_locked: bool | None = None,
) -> np.ndarray:
    p = ctx.pass_event
    dets = ctx.dets_by_frame[frame_idx]
    color = _team_color(p.team)
    panels = ctx.strip_plan.receiver_frames
    panel_idx = panels.index(frame_idx) + 1 if frame_idx in panels else 1
    confirm = ctx.strip_plan.receiver_confirm_frame
    locked = (
        show_locked
        if show_locked is not None
        else frame_idx >= confirm
    )
    level = (
        emphasis
        if emphasis is not None
        else _streak_emphasis(panel_idx, len(panels))
    )
    frame_img = _build_receiver_arrival_frame(
        ctx.frames[frame_idx],
        dets,
        p.receiver_tid,
        p.passer_tid,
        color,
        emphasis=level,
        locked=locked,
        ball_pt=_pass_ball_for_ctx(ctx, frame_idx),
    )
    sublabel = _explain_streak_sublabel(
        panel_idx, len(panels), locked=locked, kind="arrival"
    )
    return _compose_panel_row(
        frame_img,
        frame_idx,
        _receiver_label(ctx, frame_idx),
        color,
        layout=layout,
        step=panel_idx,
        step_total=len(panels),
        sublabel=sublabel,
        badge=_receiver_panel_badge(panel_idx, len(panels), locked=locked),
        emphasis=level,
        locked=locked,
        accent_bgr=ROBOFLOW_PURPLE_BGR if locked else color,
        crop=crop,
    )


def _compose_strip(
    panels: list[np.ndarray],
    *,
    title: str,
    subtitle: str,
    layout: Literal["talk", "social"],
    accent_bgr: tuple[int, int, int] | None = None,
) -> np.ndarray:
    """Stack frames vertically — one row per frame, read top to bottom."""
    sep_h = 6
    sep = np.full(
        (sep_h, panels[0].shape[1], 3), (10, 10, 12), dtype=np.uint8
    )
    body = panels[0]
    for panel in panels[1:]:
        if panel.shape[1] != body.shape[1]:
            h = int(panel.shape[0] * (body.shape[1] / panel.shape[1]))
            panel = cv2.resize(panel, (body.shape[1], h), interpolation=cv2.INTER_AREA)
        sep_line = sep if sep.shape[1] == body.shape[1] else np.full(
            (sep_h, body.shape[1], 3), (10, 10, 12), dtype=np.uint8
        )
        body = np.vstack([body, sep_line, panel])
    header_h = int(62 * _ui_scale(layout))
    header = np.full((header_h, body.shape[1], 3), _PANEL_BG_BGR, dtype=np.uint8)
    bar = accent_bgr if accent_bgr is not None else ROBOFLOW_PURPLE_BGR
    cv2.rectangle(header, (0, 0), (3, header_h), bar, -1)
    _draw_strip_title(header, title, subtitle, layout=layout)
    return np.vstack([header, body])


def _attach_radar_summary(
    canvas: np.ndarray,
    ctx: PassExplainContext,
    frame_idx: int,
) -> np.ndarray:
    if not ctx.metric:
        return canvas
    dets = ctx.dets_by_frame[frame_idx]
    keypoints = ctx.keypoints_by_frame.get(frame_idx)
    if keypoints is None:
        return canvas
    radar_h = ctx.radar_transformers.get(frame_idx)
    if radar_h is None:
        radar_h = homography_from_keypoints_radar(keypoints, confidence=0.5)
    if radar_h is None:
        return canvas
    radar = render_radar_simple(
        dets, keypoints, confidence=0.5, transformer=radar_h, debug_keypoints=False
    )
    if radar is None:
        return canvas
    return draw_radar_minimap(canvas, dets, keypoints, prebuilt_radar=radar, scale_frac=0.28)


def render_strip_passer(
    ctx: PassExplainContext,
    *,
    layout: Literal["talk", "social"] = "talk",
) -> np.ndarray:
    p = ctx.pass_event
    crop = _strip_crop_rect(ctx, ctx.strip_plan.passer_frames, p.passer_tid)
    panels = [
        _render_passer_panel(ctx, fi, layout=layout, crop=crop)
        for fi in ctx.strip_plan.passer_frames
    ]
    out = _compose_strip(
        panels,
        title="LOCK THE PASSER",
        subtitle="3 consecutive control touches at the feet",
        layout=layout,
        accent_bgr=ROBOFLOW_PURPLE_BGR,
    )
    return draw_branding_tag(out, _BRANDING)


def render_strip_flight(
    ctx: PassExplainContext,
    *,
    layout: Literal["talk", "social"] = "talk",
) -> np.ndarray:
    fi = ctx.strip_plan.flight_frames[0]
    p = ctx.pass_event
    dets = ctx.dets_by_frame[fi]
    ball_pt = _pass_ball_for_ctx(ctx, fi)
    points = _action_focus_points(dets, p.passer_tid, ball_pt=ball_pt)
    crop = _crop_rect_from_points(ctx.frames[fi].shape, points)
    panel = _render_flight_panel(ctx, fi, layout=layout, crop=crop)
    header_h = int(62 * _ui_scale(layout))
    header = np.full((header_h, panel.shape[1], 3), _PANEL_BG_BGR, dtype=np.uint8)
    cv2.rectangle(header, (0, 0), (3, header_h), _FLIGHT_ACCENT_BGR, -1)
    _draw_strip_title(
        header,
        "BALL TRAVELLING",
        "passer stays marked in the dark - ball lit in flight",
        layout=layout,
    )
    out = np.vstack([header, panel])
    return draw_branding_tag(out, _BRANDING)


def render_strip_receiver(
    ctx: PassExplainContext,
    *,
    layout: Literal["talk", "social"] = "talk",
) -> np.ndarray:
    p = ctx.pass_event
    crop = _strip_crop_rect(
        ctx, ctx.strip_plan.receiver_frames, p.receiver_tid, p.passer_tid
    )
    panels = [
        _render_receiver_panel(ctx, fi, layout=layout, crop=crop)
        for fi in ctx.strip_plan.receiver_frames
    ]
    out = _compose_strip(
        panels,
        title="LOCK THE RECEIVER",
        subtitle="buildup then lock when pass is credited to receiver",
        layout=layout,
        accent_bgr=ROBOFLOW_PURPLE_BGR,
    )
    return draw_branding_tag(out, _BRANDING)


def render_summary(
    ctx: PassExplainContext,
    *,
    layout: Literal["talk", "social"] = "talk",
) -> np.ndarray:
    fi = ctx.strip_plan.summary_frame
    p = ctx.pass_event
    frame = ctx.frames[fi]
    dets = ctx.dets_by_frame[fi]
    color = _team_color(p.team)
    dimmed = _dim_frame(frame, 0.22)
    centers = []
    for tid in (p.passer_tid, p.receiver_tid):
        if (feet := _feet_for_tid(dets, tid)) is not None:
            centers.append(feet)
    out = _apply_focus_spotlights(dimmed, frame, centers, strength=0.75)
    for tid in (p.passer_tid, p.receiver_tid):
        _draw_focus_player(out, dets, tid, color, prominent=True, locked=True)
    out = annotate_ball(out, dets)
    _draw_pass_highlights(
        out, dets, fi, (p,), ctx.frame_rate, draw_player_halos=False
    )
    points = _action_focus_points(dets, p.passer_tid, p.receiver_tid)
    crop = _crop_rect_from_points(frame.shape, points, min_frac=0.42)
    out = _letterbox_frame(
        _apply_crop(out, crop),
        target_w=_PANEL_W + _GUTTER_W,
        target_h=_PANEL_H,
    )

    _draw_strip_title(out, "PASS CREDITED", f"#{p.passer_tid} -> #{p.receiver_tid}", layout=layout)
    out = _attach_radar_summary(out, ctx, fi)
    return draw_branding_tag(out, _BRANDING)


def render_pass_explain_strips(
    ctx: PassExplainContext,
    *,
    layout: Literal["talk", "social"] = "talk",
) -> dict[str, np.ndarray]:
    strips = {
        "strip_passer": render_strip_passer(ctx, layout=layout),
        "strip_flight": render_strip_flight(ctx, layout=layout),
        "strip_receiver": render_strip_receiver(ctx, layout=layout),
        "summary": render_summary(ctx, layout=layout),
    }
    if layout == "social":
        strips = {k: _fit_social_square(v) for k, v in strips.items()}
    return strips


def render_pass_detect_timeline(strips: dict[str, np.ndarray], *, gap: int = 8) -> np.ndarray:
    """Stack all strips vertically for one poster slide (no width stretch)."""
    order = ["strip_passer", "strip_flight", "strip_receiver", "summary"]
    parts = [strips[k] for k in order if k in strips]
    if not parts:
        raise ValueError("No strips to compose")
    target_w = max(p.shape[1] for p in parts)
    aligned = []
    for part in parts:
        if part.shape[1] < target_w:
            pad = target_w - part.shape[1]
            part = cv2.copyMakeBorder(
                part, 0, 0, 0, pad, cv2.BORDER_CONSTANT, value=(8, 8, 10)
            )
        aligned.append(part)
    sep = np.full((gap, target_w, 3), (8, 8, 10), dtype=np.uint8)
    out = aligned[0]
    for part in aligned[1:]:
        out = np.vstack([out, sep, part])
    return out


def _explain_pass_score(
    pass_event: InferredPass,
    timeline_by_frame: dict[int, CarrierFrameState],
    dets_by_frame: dict[int, sv.Detections],
    *,
    min_control_frames: int,
    min_gap_frames: int,
) -> float | None:
    """Rank passes that are explainable on film (outfield control buildup + visible gap)."""
    if pass_event.gap_frames < min_gap_frames:
        return None
    if _passer_is_goalkeeper(pass_event, dets_by_frame):
        return None
    control_run = _find_passer_control_run(
        pass_event.frame_idx,
        pass_event.passer_tid,
        timeline_by_frame,
        min_control_frames=min_control_frames,
    )
    if len(control_run) < min_control_frames:
        return None
    quality = pass_event.quality_score or 0.0
    gap_term = min(pass_event.gap_frames / 45.0, 1.0)
    control_term = min(len(control_run) / 6.0, 1.0)
    return quality * 0.55 + gap_term * 0.35 + control_term * 0.10


def pick_explain_pass(
    passes: list[InferredPass],
    *,
    pass_index: int | None = None,
    min_gap_frames: int = 10,
    timeline_by_frame: dict[int, CarrierFrameState] | None = None,
    dets_by_frame: dict[int, sv.Detections] | None = None,
    min_control_frames: int = _MIN_CONTROL,
) -> InferredPass:
    if not passes:
        raise ValueError("No passes found in scan")

    ranked: list[InferredPass]
    if timeline_by_frame is not None and dets_by_frame is not None:
        scored: list[tuple[float, InferredPass]] = []
        for p in passes:
            score = _explain_pass_score(
                p,
                timeline_by_frame,
                dets_by_frame,
                min_control_frames=min_control_frames,
                min_gap_frames=min_gap_frames,
            )
            if score is not None:
                scored.append((score, p))
        if not scored:
            raise ValueError(
                "No suitable outfield passes with a visible control buildup "
                f"(need {min_control_frames} control frames before release)"
            )
        ranked = [p for _, p in sorted(scored, key=lambda pair: pair[0], reverse=True)]
    else:
        candidates = [p for p in passes if p.gap_frames >= min_gap_frames] or list(passes)
        ranked = sorted(
            candidates,
            key=lambda p: (p.quality_score is not None, p.quality_score or 0.0),
            reverse=True,
        )

    if pass_index is not None:
        if pass_index < 0 or pass_index >= len(ranked):
            raise IndexError(
                f"--pass-index {pass_index} out of range (0..{len(ranked) - 1})"
            )
        return ranked[pass_index]
    return ranked[0]


def pick_midfield_explain_pass(
    passes: list[InferredPass],
    *,
    sequence_length: int,
    timeline_by_frame: dict[int, CarrierFrameState],
    dets_by_frame: dict[int, sv.Detections],
    min_control_frames: int = _MIN_CONTROL,
    min_gap_frames: int = 10,
    window_frac: float = 0.25,
) -> InferredPass:
    """Best explainable pass near the temporal centre of the clip."""
    mid = sequence_length // 2
    window = max(75, int(sequence_length * window_frac))
    ranked: list[InferredPass] = []
    for p in passes:
        score = _explain_pass_score(
            p,
            timeline_by_frame,
            dets_by_frame,
            min_control_frames=min_control_frames,
            min_gap_frames=min_gap_frames,
        )
        if score is not None:
            ranked.append(p)
    if not ranked:
        raise ValueError("No suitable outfield passes for midfield explain")
    ranked.sort(
        key=lambda p: _explain_pass_score(
            p,
            timeline_by_frame,
            dets_by_frame,
            min_control_frames=min_control_frames,
            min_gap_frames=min_gap_frames,
        )
        or 0.0,
        reverse=True,
    )
    in_window = [p for p in ranked if mid - window <= p.frame_idx <= mid + window]
    if in_window:
        return in_window[0]
    return min(ranked, key=lambda p: abs(p.frame_idx - mid))


def build_pass_explain_context(
    pass_event: InferredPass,
    *,
    frame_rate: float,
    frames_by_idx: dict[int, np.ndarray],
    dets_by_frame: dict[int, sv.Detections],
    timeline_by_frame: dict[int, CarrierFrameState],
    keypoints_by_frame: dict[int, sv.KeyPoints | None] | None = None,
    radar_transformers: dict[int, ViewTransformer | None] | None = None,
    metric: bool = True,
    min_control_frames: int = _MIN_CONTROL,
    min_arrival_frames: int = _MIN_ARRIVAL,
    strip_plan: PassStripPlan | None = None,
) -> PassExplainContext:
    if strip_plan is None:
        strip_plan = build_strip_plan(
            pass_event,
            timeline_by_frame,
            dets_by_frame=dets_by_frame,
            min_control_frames=min_control_frames,
            min_arrival_frames=min_arrival_frames,
        )
    needed = set(frames_by_idx.keys()) & set(dets_by_frame.keys())
    for fi in (
        *strip_plan.passer_frames,
        *strip_plan.flight_frames,
        *strip_plan.receiver_frames,
        strip_plan.summary_frame,
    ):
        needed.add(fi)

    kps = keypoints_by_frame or {}
    radar = radar_transformers or {}
    return PassExplainContext(
        pass_event=pass_event,
        frame_rate=frame_rate,
        metric=metric,
        min_control_frames=min_control_frames,
        min_arrival_frames=min_arrival_frames,
        strip_plan=strip_plan,
        frames={fi: frames_by_idx[fi] for fi in needed if fi in frames_by_idx},
        dets_by_frame={fi: dets_by_frame[fi] for fi in needed if fi in dets_by_frame},
        timeline_by_frame={
            fi: timeline_by_frame[fi] for fi in needed if fi in timeline_by_frame
        },
        keypoints_by_frame={fi: kps.get(fi) for fi in needed},
        radar_transformers={fi: radar.get(fi) for fi in needed},
    )


def explain_video_frame_range(strip_plan: PassStripPlan) -> tuple[int, int]:
    """Inclusive source-frame span — through visual receiver confirm and pass credit."""
    end = max(strip_plan.summary_frame, strip_plan.receiver_confirm_frame)
    return strip_plan.passer_frames[0], end


def _explain_lock_frames(ctx: PassExplainContext) -> tuple[int, int]:
    """When to hold the passer/receiver locked beats in the annotated video."""
    plan = ctx.strip_plan
    return plan.passer_confirm_frame, plan.receiver_confirm_frame


def frames_needed_for_explain(
    strip_plan: PassStripPlan,
    *,
    include_video: bool = False,
) -> set[int]:
    needed = set(strip_plan.passer_frames)
    needed.update(strip_plan.flight_frames)
    needed.update(strip_plan.receiver_frames)
    needed.add(strip_plan.summary_frame)
    needed.add(strip_plan.passer_confirm_frame)
    needed.add(strip_plan.receiver_confirm_frame)
    if include_video:
        start, end = explain_video_frame_range(strip_plan)
        needed.update(range(start, end + 1))
    return needed


def _video_phase(
    ctx: PassExplainContext, frame_idx: int
) -> Literal["passer", "flight", "receiver"]:
    passer_confirm = ctx.strip_plan.passer_confirm_frame
    receiver_confirm = ctx.strip_plan.receiver_confirm_frame
    if frame_idx <= passer_confirm:
        return "passer"
    if frame_idx < receiver_confirm:
        return "flight"
    return "receiver"


def _passer_emphasis_for_video(ctx: PassExplainContext, frame_idx: int) -> float:
    panels = ctx.strip_plan.passer_frames
    confirm = ctx.strip_plan.passer_confirm_frame
    if frame_idx >= confirm:
        return 1.0
    if frame_idx < panels[0]:
        return 0.35
    if frame_idx not in panels:
        return 0.85
    return (panels.index(frame_idx) + 1) / len(panels)


def _receiver_emphasis_for_video(ctx: PassExplainContext, frame_idx: int) -> float:
    panels = ctx.strip_plan.receiver_frames
    if frame_idx not in panels:
        return 0.85
    return (panels.index(frame_idx) + 1) / len(panels)


def _draw_video_status_chip(
    frame: np.ndarray,
    badge: str,
    *,
    locked: bool,
    accent_bgr: tuple[int, int, int],
) -> None:
    """HUD-adjacent state chip on full-frame video."""
    fs = 0.62 if locked else 0.54
    thick = 2 if locked else 1
    (tw, th), baseline = cv2.getTextSize(
        badge, cv2.FONT_HERSHEY_SIMPLEX, fs, thick
    )
    pad_x, pad_y = 14, 10
    x1 = frame.shape[1] - 16
    x0 = x1 - tw - pad_x * 2
    y0 = 52
    y1 = y0 + th + baseline + pad_y * 2
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), _BADGE_BG_BGR, -1)
    rail = ROBOFLOW_PURPLE_BGR if locked else accent_bgr
    cv2.rectangle(overlay, (x0, y0), (x0 + 3, y1), rail, -1)
    frame[:] = cv2.addWeighted(overlay, 0.72, frame, 0.28, 0)
    draw_text_shadow(
        frame,
        badge,
        (x0 + pad_x, y0 + pad_y + th),
        font_scale=fs,
        color_bgr=(245, 245, 248) if locked else (210, 210, 218),
        thickness=thick,
    )


def render_explain_video_frame(ctx: PassExplainContext, frame_idx: int) -> np.ndarray:
    """One annotated broadcast frame — real motion, phase-aware overlays."""
    p = ctx.pass_event
    plan = ctx.strip_plan
    full = ctx.frames[frame_idx]
    dets = ctx.dets_by_frame[frame_idx]
    color = _team_color(p.team)
    phase = _video_phase(ctx, frame_idx)
    release = p.frame_idx
    arrival = release + p.gap_frames

    passer_lock, receiver_lock = _explain_lock_frames(ctx)
    if phase == "passer":
        emph = _passer_emphasis_for_video(ctx, frame_idx)
        locked = frame_idx >= passer_lock
        out = _build_passer_control_frame(
            full,
            dets,
            p.passer_tid,
            color,
            emphasis=emph,
            locked=locked,
            ball_pt=_pass_ball_for_ctx(ctx, frame_idx),
        )
        panels = plan.passer_frames
        panel_idx = panels.index(frame_idx) + 1 if frame_idx in panels else len(panels)
        badge = _passer_panel_badge(panel_idx, len(panels), locked=locked)
        title = "LOCK THE PASSER"
    elif phase == "flight":
        ball_pt = _pass_ball_for_ctx(ctx, frame_idx)
        out = _build_flight_frame(full, dets, p.passer_tid, color, ball_pt=ball_pt)
        badge = "IN FLIGHT"
        title = "BALL TRAVELLING"
        locked = False
    else:
        emph = _receiver_emphasis_for_video(ctx, frame_idx)
        locked = frame_idx >= receiver_lock
        out = _build_receiver_arrival_frame(
            full,
            dets,
            p.receiver_tid,
            p.passer_tid,
            color,
            emphasis=emph,
            locked=locked,
            ball_pt=_pass_ball_for_ctx(ctx, frame_idx),
        )
        panels = plan.receiver_frames
        panel_idx = panels.index(frame_idx) + 1 if frame_idx in panels else len(panels)
        badge = _receiver_panel_badge(panel_idx, len(panels), locked=locked)
        title = "LOCK THE RECEIVER"

    out = draw_hud_bar(out, title)
    _draw_video_status_chip(out, badge, locked=locked, accent_bgr=color)
    return draw_branding_tag(out, _BRANDING)


def render_explain_video_summary(ctx: PassExplainContext) -> np.ndarray:
    """Full-frame pass-credited beat for the annotated clip."""
    fi = ctx.strip_plan.summary_frame
    p = ctx.pass_event
    frame = ctx.frames[fi]
    dets = ctx.dets_by_frame[fi]
    color = _team_color(p.team)
    dimmed = _dim_frame(frame, 0.22)
    centers = [
        f
        for tid in (p.passer_tid, p.receiver_tid)
        if (f := _feet_for_tid(dets, tid)) is not None
    ]
    out = _apply_focus_spotlights(dimmed, frame, centers, strength=0.75)
    for tid in (p.passer_tid, p.receiver_tid):
        _draw_focus_player(out, dets, tid, color, prominent=True, locked=True)
    out = annotate_ball(out, dets)
    _draw_pass_highlights(
        out, dets, fi, (p,), ctx.frame_rate, draw_player_halos=False
    )
    out = draw_hud_bar(out, "PASS CREDITED")
    _draw_video_status_chip(
        out,
        f"#{p.passer_tid} -> #{p.receiver_tid}",
        locked=True,
        accent_bgr=ROBOFLOW_PURPLE_BGR,
    )
    return draw_branding_tag(out, _BRANDING)


@dataclass(frozen=True)
class PassExplainVideoTiming:
    """Annotated clip pacing — walks real source frames at a low output FPS."""

    output_fps: float = 8.0
    hold_locked_seconds: float = 1.2
    summary_hold_seconds: float = 2.0
    lock_nudge_frames: int = 5
    crf: int = 16


def build_pass_explain_video_sequence(
    ctx: PassExplainContext,
    *,
    timing: PassExplainVideoTiming = PassExplainVideoTiming(),
    include_summary: bool = True,
) -> list[np.ndarray]:
    """Walk consecutive source frames with overlays — smooth slow motion."""
    plan = ctx.strip_plan
    start, end = explain_video_frame_range(plan)
    passer_lock, receiver_lock = _explain_lock_frames(ctx)
    locked_frames = {passer_lock, receiver_lock}
    hold_n = max(1, int(round(timing.hold_locked_seconds * timing.output_fps)))
    summary_n = max(1, int(round(timing.summary_hold_seconds * timing.output_fps)))
    out: list[np.ndarray] = []
    for fi in range(start, end + 1):
        frame = render_explain_video_frame(ctx, fi)
        repeats = hold_n if fi in locked_frames else 1
        out.extend([frame] * repeats)
    if include_summary:
        summary = render_explain_video_summary(ctx)
        out.extend([summary] * summary_n)
    return out


def write_pass_explain_video(
    path: str | Path,
    frames_bgr: list[np.ndarray],
    *,
    fps: float = 8.0,
    crf: int = 16,
) -> Path:
    """Write annotated explain clip as h264 MP4 (single-pass ffmpeg encode)."""
    path = Path(path)
    try:
        return write_h264_video(path, frames_bgr, fps=fps, crf=crf)
    except RuntimeError:
        h, w = frames_bgr[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
        if not writer.isOpened():
            raise RuntimeError(f"Could not open video writer: {path}")
        for frame in frames_bgr:
            writer.write(frame)
        writer.release()
        if not finalize_video_for_playback(path, crf=crf):
            import warnings

            warnings.warn(
                "ffmpeg not found — MP4 may show green in some players (install ffmpeg)",
                stacklevel=2,
            )
        return path


def write_pass_explain_gif(
    path: str | Path,
    mp4_path: str | Path,
    *,
    fps: float = 8.0,
    width: int | None = 1280,
) -> Path:
    """Write GIF from the h264 explain MP4 (ffmpeg palettegen)."""
    return write_gif_from_mp4(path, mp4_path, fps=fps, width=width)


def write_pass_explain_frames(
    out_dir: str | Path,
    strips: dict[str, np.ndarray],
    *,
    timeline: bool = True,
) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for stem, image in strips.items():
        path = out_dir / f"pass_detect_{stem}.png"
        cv2.imwrite(str(path), image)
        written.append(path)
    if timeline:
        path = out_dir / "pass_detect_timeline.png"
        cv2.imwrite(str(path), render_pass_detect_timeline(strips))
        written.append(path)
    return written
