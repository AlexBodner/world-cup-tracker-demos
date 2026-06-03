"""Roboflow-style overlays shared by World Cup demos (Football AI / blog look)."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
import supervision as sv

from world_cup_projects.common.pitch import (
    PITCH_CONFIG,
    draw_pitch,
    draw_points_on_pitch,
    image_to_pitch_cm,
    pitch_keypoint_accept_mask,
    pitch_keypoint_confidence,
)
from world_cup_projects.common.possession import ball_xy, feet_xy, player_mask

ROBOFLOW_PURPLE = sv.Color.from_hex("#8315F9")
ROBOFLOW_PURPLE_BGR = ROBOFLOW_PURPLE.as_bgr()
TEAM_COLORS = [
    sv.Color.from_hex("#00BFFF"),
    sv.Color.from_hex("#FF1493"),
    sv.Color.from_hex("#FFD700"),
]
TEAM_PALETTE = sv.ColorPalette(TEAM_COLORS)
BALL_COLOR = sv.Color.from_hex("#FFD700")

_ELLIPSE = sv.EllipseAnnotator(
    color=TEAM_PALETTE, color_lookup=sv.ColorLookup.CLASS, thickness=2
)
_LABEL = sv.LabelAnnotator(
    text_position=sv.Position.BOTTOM_CENTER,
    text_scale=0.45,
    text_thickness=1,
    border_radius=4,
    color=TEAM_PALETTE,
    color_lookup=sv.ColorLookup.CLASS,
)
_BALL_TRI = sv.TriangleAnnotator(
    color=BALL_COLOR, base=14, height=18, color_lookup=sv.ColorLookup.INDEX
)


def team_class_ids(teams: np.ndarray) -> np.ndarray:
    return np.where(np.isin(teams, (0, 1)), teams, 2).astype(int)


def draw_text_shadow(
    frame: np.ndarray,
    text: str,
    org: tuple[int, int],
    *,
    font_scale: float = 0.7,
    color_bgr: tuple[int, int, int] = (255, 255, 255),
    thickness: int = 2,
    shadow_offset: tuple[int, int] = (2, 2),
) -> None:
    x, y = org
    sx, sy = shadow_offset
    cv2.putText(
        frame,
        text,
        (x + sx, y + sy),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (12, 12, 12),
        thickness + 1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color_bgr,
        thickness,
        cv2.LINE_AA,
    )


def draw_hud_bar(frame: np.ndarray, title: str, *, height: int = 44) -> np.ndarray:
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, height), (18, 18, 22), -1)
    frame[:] = cv2.addWeighted(overlay, 0.72, frame, 0.28, 0)
    draw_text_shadow(
        frame, title, (14, 30), font_scale=0.75, color_bgr=ROBOFLOW_PURPLE_BGR, thickness=2
    )
    return frame


def draw_branding_tag(frame: np.ndarray, text: str = "powered by trackers") -> np.ndarray:
    h, w = frame.shape[:2]
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
    pad = 8
    x0 = w - tw - pad * 2 - 10
    y0 = h - th - pad * 2 - 10
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (x0 - pad, y0 - pad),
        (w - 10, h - 10),
        (18, 18, 22),
        -1,
    )
    frame[:] = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)
    draw_text_shadow(
        frame,
        text,
        (x0, y0 + th),
        font_scale=0.48,
        color_bgr=ROBOFLOW_PURPLE_BGR,
        thickness=1,
    )
    return frame


def annotate_players(
    frame: np.ndarray,
    dets: sv.Detections,
    *,
    labels: list[str] | None = None,
) -> np.ndarray:
    pmask = player_mask(dets)
    if not pmask.any():
        return frame
    players = dets[pmask]
    teams = players.data.get("team", np.zeros(len(players)))
    players.class_id = team_class_ids(teams)
    frame = _ELLIPSE.annotate(frame, players)
    if labels is not None:
        frame = _LABEL.annotate(frame, players, labels=labels)
    return frame


def annotate_ball(frame: np.ndarray, dets: sv.Detections) -> np.ndarray:
    ball = ball_xy(dets)
    if ball is None:
        return frame
    x, y = float(ball[0]), float(ball[1])
    ball_dets = sv.Detections(
        xyxy=np.array([[x - 6, y - 6, x + 6, y + 6]], dtype=np.float32),
        class_id=np.array([0]),
    )
    return _BALL_TRI.annotate(frame, ball_dets)


def _valid_pitch_cm(
    xy: np.ndarray, config=PITCH_CONFIG, *, margin_cm: float = 200.0
) -> np.ndarray:
    """Mask for points that lie on the pitch (homography outliers are dropped)."""
    if xy is None or len(xy) == 0:
        return np.zeros(0, dtype=bool)
    finite = np.isfinite(xy).all(axis=1)
    return (
        finite
        & (xy[:, 0] >= margin_cm)
        & (xy[:, 0] <= config.length - margin_cm)
        & (xy[:, 1] >= margin_cm)
        & (xy[:, 1] <= config.width - margin_cm)
    )


# Hard-reset radar EMA when H flips and a dot jumps across the pitch (~25 m).
RADAR_JUMP_RESET_CM = 2500.0


@dataclass
class RadarSmoother:
    """EMA-smooth pitch (cm) positions per ``tracker_id`` for stable minimap dots."""

    alpha: float = 0.38
    _pos: dict[int, np.ndarray] = field(default_factory=dict)

    def reset(self) -> None:
        self._pos.clear()

    def update(self, track_id: int, pitch_cm: np.ndarray, *, valid: bool) -> np.ndarray | None:
        if not valid:
            return self._pos.get(track_id)
        pt = pitch_cm.astype(np.float64)
        prev = self._pos.get(track_id)
        if prev is not None:
            if float(np.linalg.norm(pt - prev)) > RADAR_JUMP_RESET_CM:
                self._pos[track_id] = pt
                return pt
        if prev is None:
            self._pos[track_id] = pt
        else:
            a = self.alpha
            self._pos[track_id] = (1.0 - a) * prev + a * pt
        return self._pos[track_id]


def draw_radar_minimap(
    frame: np.ndarray,
    dets: sv.Detections,
    transformer,
    *,
    scale_frac: float = 0.33,
    position: str = "bottom_right",
    smoother: RadarSmoother | None = None,
) -> np.ndarray:
    if transformer is None:
        return frame
    pmask = player_mask(dets)
    if not pmask.any():
        return frame

    feet = feet_xy(dets)
    teams = dets.data.get("team", np.zeros(len(dets)))
    pitch_xy = image_to_pitch_cm(feet[pmask], transformer)
    if pitch_xy is None:
        return frame

    tids = dets.tracker_id[pmask] if dets.tracker_id is not None else np.arange(len(pitch_xy))
    teams_p = teams[pmask]
    valid = _valid_pitch_cm(pitch_xy)
    smooth = smoother or RadarSmoother()

    radar = draw_pitch(config=PITCH_CONFIG)
    for team_id, color in enumerate(TEAM_COLORS[:2]):
        team_pts: list[np.ndarray] = []
        for pt, tid, ok, team in zip(pitch_xy, tids, valid, teams_p):
            if team != team_id:
                continue
            tid_i = int(tid) if tid is not None and int(tid) >= 0 else 10_000 + len(team_pts)
            smoothed = smooth.update(tid_i, pt, valid=bool(ok))
            if smoothed is not None:
                team_pts.append(smoothed)
        if team_pts:
            radar = draw_points_on_pitch(
                config=PITCH_CONFIG,
                xy=np.asarray(team_pts, dtype=np.float32),
                face_color=color,
                edge_color=sv.Color.BLACK,
                radius=10,
                thickness=2,
                pitch=radar,
            )

    ball = ball_xy(dets)
    if ball is not None:
        ball_cm = image_to_pitch_cm(np.array([ball], dtype=np.float32), transformer)
        if ball_cm is not None and _valid_pitch_cm(ball_cm).all():
            radar = draw_points_on_pitch(
                config=PITCH_CONFIG,
                xy=ball_cm,
                face_color=sv.Color.WHITE,
                edge_color=sv.Color.BLACK,
                radius=8,
                pitch=radar,
            )

    h, w = frame.shape[:2]
    rw = int(w * scale_frac)
    rh = int(radar.shape[0] * (rw / radar.shape[1]))
    radar = sv.resize_image(radar, (rw, rh))
    margin = 14
    brand_clearance = 52  # keep clear of "powered by trackers" (bottom-right)
    if position == "bottom_left":
        x, y = margin, h - rh - margin
    elif position == "bottom_center":
        x, y = (w - rw) // 2, h - rh - margin
    else:  # bottom_right (default — out of the way of play + branding)
        x, y = w - rw - margin, h - rh - margin - brand_clearance
    panel = frame.copy()
    cv2.rectangle(panel, (x - 10, y - 10), (x + rw + 10, y + rh + 10), (18, 18, 22), -1)
    frame[:] = cv2.addWeighted(panel, 0.45, frame, 0.55, 0)
    roi = frame[y : y + rh, x : x + rw]
    frame[y : y + rh, x : x + rw] = cv2.addWeighted(radar, 0.92, roi, 0.08, 0)
    cv2.rectangle(frame, (x - 2, y - 2), (x + rw + 2, y + rh + 2), (255, 255, 255), 1)
    return frame


# Back-compat alias
draw_radar_bottom_center = draw_radar_minimap


_KP_USED_BGR = (80, 220, 80)
_KP_LOW_CONF_BGR = (80, 80, 255)
_KP_INVALID_BGR = (140, 140, 140)
_KP_EDGE_BGR = (70, 70, 90)


def draw_pitch_keypoints_debug(
    frame: np.ndarray,
    keypoints: sv.KeyPoints | None,
    *,
    confidence_threshold: float = 0.5,
    draw_skeleton: bool = True,
) -> np.ndarray:
    """Overlay raw pitch-keypoint detections (index + confidence) for homography debugging.

    Labels use 0-based indices matching ``config.vertices`` / the YOLO pose head order.
    Skeleton edges only connect keypoints that pass the confidence filter (model is
    trained on broadcast football; SoccerNet angles may still look sparse or wrong).
    """
    h, w = frame.shape[:2]
    margin = 14
    legend_y = h - margin - 88

    if keypoints is None or keypoints.xy.shape[0] == 0:
        draw_text_shadow(
            frame,
            "pitch keypoints: none",
            (margin, legend_y),
            font_scale=0.5,
            color_bgr=(180, 180, 180),
            thickness=1,
        )
        return frame

    xy = keypoints.xy[0]
    n = len(PITCH_CONFIG.vertices)
    conf = pitch_keypoint_confidence(keypoints, n_vertices=n)
    accept = pitch_keypoint_accept_mask(xy, conf, confidence=confidence_threshold)

    if draw_skeleton:
        # edges are 1-based vertex ids (same as draw_pitch / roboflow sports)
        for start, end in PITCH_CONFIG.edges:
            i, j = start - 1, end - 1
            if i >= len(xy) or j >= len(xy):
                continue
            if xy[i, 0] <= 1 or xy[i, 1] <= 1 or xy[j, 0] <= 1 or xy[j, 1] <= 1:
                continue
            if not (accept[i] and accept[j]):
                continue
            p1 = (int(xy[i, 0]), int(xy[i, 1]))
            p2 = (int(xy[j, 0]), int(xy[j, 1]))
            cv2.line(frame, p1, p2, _KP_EDGE_BGR, 1, cv2.LINE_AA)

    for i in range(min(len(xy), n)):
        x, y = float(xy[i, 0]), float(xy[i, 1])
        if x <= 1 or y <= 1:
            color = _KP_INVALID_BGR
        elif accept[i]:
            color = _KP_USED_BGR
        else:
            color = _KP_LOW_CONF_BGR
        px, py = int(x), int(y)
        cv2.circle(frame, (px, py), 5, color, -1, cv2.LINE_AA)
        cv2.circle(frame, (px, py), 5, (255, 255, 255), 1, cv2.LINE_AA)
        label = f"{i}:{conf[i]:.2f}"
        draw_text_shadow(
            frame,
            label,
            (px + 7, py - 6),
            font_scale=0.38,
            color_bgr=color,
            thickness=1,
        )

    n_ok = int(accept[: min(len(xy), n)].sum())
    summary = f"pitch kp {n_ok}/{n} used (conf > {confidence_threshold:.2f})"
    draw_text_shadow(
        frame, summary, (margin, legend_y), font_scale=0.52, color_bgr=(230, 230, 230), thickness=1
    )
    for dy, text, color in (
        (22, "green = used for H", _KP_USED_BGR),
        (40, "red = low confidence", _KP_LOW_CONF_BGR),
        (58, "gray = invalid / missing", _KP_INVALID_BGR),
        (76, "lines = pitch topology (confident kps only)", _KP_EDGE_BGR),
    ):
        draw_text_shadow(
            frame, text, (margin, legend_y + dy), font_scale=0.42, color_bgr=color, thickness=1
        )
    return frame


def draw_glow_arrow(
    frame: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    color_bgr: tuple[int, int, int],
    *,
    thickness: int = 4,
) -> None:
    cv2.arrowedLine(
        frame, start, end, (20, 20, 20), thickness + 3, cv2.LINE_AA, tipLength=0.05
    )
    cv2.arrowedLine(frame, start, end, color_bgr, thickness, cv2.LINE_AA, tipLength=0.05)


def draw_score_chip(
    frame: np.ndarray,
    text: str,
    center: tuple[int, int],
    *,
    bg_bgr: tuple[int, int, int],
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thick = 0.5, 1
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thick)
    cx, cy = center
    x0, y0 = cx - tw // 2 - 8, cy - th // 2 - 6
    x1, y1 = cx + tw // 2 + 8, cy + th // 2 + baseline + 6
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), bg_bgr, -1)
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (255, 255, 255), 1)
    frame[:] = cv2.addWeighted(overlay, 0.85, frame, 0.15, 0)
    draw_text_shadow(frame, text, (x0 + 8, y0 + th + 4), font_scale=scale, color_bgr=(255, 255, 255), thickness=thick)
