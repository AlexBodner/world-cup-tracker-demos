"""Draw pass-lane corridors (pitch/radar space) on the video and minimap."""

from __future__ import annotations

import cv2
import numpy as np
import supervision as sv

from world_cup_projects.common.pitch import (
    PITCH_CONFIG,
    SoccerPitchConfiguration,
    ViewTransformer,
    pitch_cm_to_image,
)
from world_cup_projects.pass_alternatives.pass_options import PassOption

_RADAR_SCALE = 0.1
_RADAR_PADDING = 50
_CORRIDOR_ALPHA_RADAR = 0.45
_CORRIDOR_ALPHA_VIDEO = 0.38
_RANK_COLORS = [
    sv.Color.from_hex("#3CDC3C"),  # best
    sv.Color.from_hex("#28DCE8"),
    sv.Color.from_hex("#FF8C28"),
]
_RANK_BGR = [c.as_bgr() for c in _RANK_COLORS]
_BLOCKER_BGR = (40, 40, 255)
_BLOCKER_RING_BGR = (255, 255, 255)


def _pitch_cm_to_radar_px(
    points_cm: np.ndarray,
    *,
    config: SoccerPitchConfiguration = PITCH_CONFIG,
    scale: float = _RADAR_SCALE,
    padding: int = _RADAR_PADDING,
) -> np.ndarray:
    pts = np.asarray(points_cm, dtype=np.float64).reshape(-1, 2)
    return np.stack(
        [
            (pts[:, 0] * scale + padding).astype(np.int32),
            (pts[:, 1] * scale + padding).astype(np.int32),
        ],
        axis=1,
    )


def draw_pass_lanes_on_radar(
    radar: np.ndarray,
    options: list[PassOption],
    pitch_cm: np.ndarray,
    *,
    config: SoccerPitchConfiguration = PITCH_CONFIG,
) -> np.ndarray:
    """Overlay ranked pass corridors and highlight blocking rivals (pitch cm coords)."""
    out = radar.copy()
    for rank, option in enumerate(options):
        debug = option.lane_debug
        if debug is None:
            continue
        color = _RANK_COLORS[min(rank, len(_RANK_COLORS) - 1)]
        poly = _pitch_cm_to_radar_px(debug.corridor_polygon_cm, config=config)
        overlay = out.copy()
        cv2.fillPoly(overlay, [poly], color.as_bgr())
        out = cv2.addWeighted(overlay, _CORRIDOR_ALPHA_RADAR, out, 1.0 - _CORRIDOR_ALPHA_RADAR, 0)
        cv2.polylines(out, [poly], isClosed=True, color=color.as_bgr(), thickness=2)

        for idx in debug.blocking_rival_indices:
            if idx >= len(pitch_cm):
                continue
            pt = _pitch_cm_to_radar_px(pitch_cm[idx : idx + 1], config=config)[0]
            cv2.circle(out, tuple(pt), 14, _BLOCKER_BGR, -1)
            cv2.circle(out, tuple(pt), 16, _BLOCKER_RING_BGR, 2)
            cv2.putText(
                out,
                "!",
                (pt[0] - 5, pt[1] + 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

    return out


def _corridor_image_polygon(
    corridor_polygon_cm: np.ndarray,
    transformer: ViewTransformer,
    frame_shape: tuple[int, int],
) -> np.ndarray | None:
    """Project a pitch corridor (cm) onto image pixels, clipped to the frame."""
    img = pitch_cm_to_image(corridor_polygon_cm, transformer)
    if img is None or len(img) < 3:
        return None
    h, w = frame_shape[:2]
    poly = img.astype(np.int32)
    if not ((poly[:, 0] >= -w) & (poly[:, 0] <= 2 * w) & (poly[:, 1] >= -h) & (poly[:, 1] <= 2 * h)).any():
        return None
    return poly


def draw_pass_corridors_on_frame(
    frame: np.ndarray,
    options: list[PassOption],
    transformer: ViewTransformer,
) -> np.ndarray:
    """Draw pitch-space pass corridors on the main camera view (H^-1)."""
    out = frame
    for rank, option in enumerate(options):
        debug = option.lane_debug
        if debug is None:
            continue
        poly = _corridor_image_polygon(
            debug.corridor_polygon_cm, transformer, out.shape
        )
        if poly is None:
            continue
        color = _RANK_BGR[min(rank, len(_RANK_BGR) - 1)]
        overlay = out.copy()
        cv2.fillPoly(overlay, [poly], color)
        out = cv2.addWeighted(overlay, _CORRIDOR_ALPHA_VIDEO, out, 1.0 - _CORRIDOR_ALPHA_VIDEO, 0)
        cv2.polylines(out, [poly], isClosed=True, color=color, thickness=3)

    return out


def draw_blocking_rivals_on_frame(
    frame: np.ndarray,
    options: list[PassOption],
    *,
    feet_xy: np.ndarray,
) -> np.ndarray:
    """Mark rivals counted inside a corridor (image feet)."""
    out = frame
    drawn: set[int] = set()
    for option in options:
        debug = option.lane_debug
        if debug is None:
            continue
        for idx in debug.blocking_rival_indices:
            if idx in drawn or idx >= len(feet_xy):
                continue
            drawn.add(idx)
            x, y = int(feet_xy[idx, 0]), int(feet_xy[idx, 1])
            cv2.circle(out, (x, y), 22, _BLOCKER_BGR, -1)
            cv2.circle(out, (x, y), 24, _BLOCKER_RING_BGR, 2)
            cv2.putText(
                out,
                "!",
                (x - 7, y + 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
    return out


def draw_pass_lane_legend(frame: np.ndarray) -> np.ndarray:
    """Legend for pass-lane debug overlays."""
    lines = [
        "shaded = pass corridor (pitch/radar, ~2.5 m)",
        "green/cyan/orange = BEST / 2ND / 3RD",
        "red ! = rival inside corridor",
    ]
    y = 78
    for line in lines:
        cv2.putText(
            frame,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )
        y += 18
    return frame
