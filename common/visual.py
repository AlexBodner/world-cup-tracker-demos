"""Roboflow-style overlays shared by World Cup demos (Football AI / blog look)."""

from __future__ import annotations

from typing import Literal

import cv2
import numpy as np
import supervision as sv

from world_cup_projects.common.pitch import (
    HOMOGRAPHY_RANSAC_REPROJ_THRESH,
    PITCH_CONFIG,
    ViewTransformer,
    draw_pitch,
    draw_points_on_pitch,
    image_to_pitch_cm,
    pitch_cm_to_image,
    pitch_keypoint_accept_mask,
    pitch_keypoint_confidence,
    render_radar_from_transformer,
    render_radar_sports,
)
from world_cup_projects.common.possession import ball_xy, feet_xy, player_mask
from world_cup_projects.common.soccernet import ROLE_GOALKEEPER, ROLE_PLAYER

ROBOFLOW_PURPLE = sv.Color.from_hex("#8315F9")
ROBOFLOW_PURPLE_BGR = ROBOFLOW_PURPLE.as_bgr()
TEAM_COLORS = [
    sv.Color.from_hex("#00BFFF"),
    sv.Color.from_hex("#FF1493"),
    sv.Color.from_hex("#FFD700"),
]
TEAM_PALETTE = sv.ColorPalette(TEAM_COLORS)
BALL_COLOR = sv.Color.from_hex("#FFD700")
KALMAN_FACING_BGR = (0, 200, 255)

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
_GK_ELLIPSE = sv.EllipseAnnotator(
    color=sv.Color.from_hex("#E8E8E8"), thickness=2
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


def draw_player_facing_arrows(
    frame: np.ndarray,
    dets: sv.Detections,
    facing: np.ndarray,
    *,
    style: Literal["motion", "kalman"] = "motion",
    arrow_len: int | None = None,
) -> np.ndarray:
    """Small arrow from each player ellipse showing motion/facing direction."""
    if facing is None or len(dets) == 0:
        return frame
    if arrow_len is None:
        arrow_len = 24 if style == "motion" else 20
    pmask = np.isin(dets.class_id, (ROLE_PLAYER, ROLE_GOALKEEPER))
    if not pmask.any():
        return frame
    indices = np.flatnonzero(pmask)
    anchors = dets[pmask].get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    teams = dets.data.get("team", np.full(len(dets), -1))[pmask]
    for local_i, global_i in enumerate(indices):
        direction = facing[global_i]
        if not np.isfinite(direction).all():
            continue
        ax, ay = anchors[local_i]
        if style == "kalman":
            ax, ay = ax + 5, ay - 3
        dx, dy = float(direction[0]), float(direction[1])
        tip = (int(ax + dx * arrow_len), int(ay + dy * arrow_len))
        if style == "kalman":
            color = KALMAN_FACING_BGR
        else:
            team = int(teams[local_i])
            if team in (0, 1):
                color = TEAM_COLORS[team].as_bgr()
            else:
                color = (230, 230, 230)
        cv2.arrowedLine(
            frame,
            (int(ax), int(ay)),
            tip,
            (16, 16, 20),
            4,
            cv2.LINE_AA,
            tipLength=0.42,
        )
        cv2.arrowedLine(
            frame,
            (int(ax), int(ay)),
            tip,
            color,
            2,
            cv2.LINE_AA,
            tipLength=0.42,
        )
    return frame


def draw_facing_legend(frame: np.ndarray) -> np.ndarray:
    """Legend when both motion and Kalman facing arrows are shown."""
    margin = 14
    legend_y = 52
    draw_text_shadow(
        frame,
        "motion=team color | kalman=gold",
        (margin, legend_y),
        font_scale=0.48,
        color_bgr=(210, 210, 210),
        thickness=1,
    )
    return frame


def _tracker_id_labels(dets: sv.Detections) -> list[str]:
    n = len(dets)
    tids = dets.tracker_id if dets.tracker_id is not None else np.full(n, -1, dtype=int)
    return [f"#{int(tid)}" if int(tid) >= 0 else "" for tid in tids]


def annotate_players(
    frame: np.ndarray,
    dets: sv.Detections,
    *,
    labels: list[str] | None = None,
    facing: np.ndarray | None = None,
    facing_motion: np.ndarray | None = None,
    facing_kalman: np.ndarray | None = None,
    show_tracker_ids: bool = False,
) -> np.ndarray:
    outfield = dets[dets.class_id == ROLE_PLAYER]
    if len(outfield):
        teams = outfield.data.get("team", np.zeros(len(outfield)))
        outfield_vis = sv.Detections(
            xyxy=outfield.xyxy,
            class_id=team_class_ids(teams),
            tracker_id=outfield.tracker_id,
            data=outfield.data,
        )
        frame = _ELLIPSE.annotate(frame, outfield_vis)
        player_labels = labels
        if player_labels is None and show_tracker_ids:
            player_labels = _tracker_id_labels(outfield)
        if player_labels is not None:
            frame = _LABEL.annotate(frame, outfield_vis, labels=player_labels)

    gks = dets[dets.class_id == ROLE_GOALKEEPER]
    if len(gks):
        gk_teams = gks.data.get("team", np.full(len(gks), -1))
        has_team = np.isin(gk_teams, (0, 1))
        if has_team.any():
            gk_colored = gks[has_team]
            gk_vis = sv.Detections(
                xyxy=gk_colored.xyxy,
                class_id=team_class_ids(gk_teams[has_team]),
                tracker_id=gk_colored.tracker_id,
                data=gk_colored.data,
            )
            frame = _ELLIPSE.annotate(frame, gk_vis)
            if show_tracker_ids:
                frame = _LABEL.annotate(
                    frame, gk_vis, labels=_tracker_id_labels(gk_colored)
                )
        if (~has_team).any():
            gk_neutral = gks[~has_team]
            frame = _GK_ELLIPSE.annotate(frame, gk_neutral)
            if show_tracker_ids:
                gk_vis = sv.Detections(
                    xyxy=gk_neutral.xyxy,
                    class_id=np.zeros(len(gk_neutral), dtype=int),
                    tracker_id=gk_neutral.tracker_id,
                    data=gk_neutral.data,
                )
                frame = _LABEL.annotate(
                    frame, gk_vis, labels=_tracker_id_labels(gk_neutral)
                )

    if facing_motion is None and facing is not None:
        facing_motion = facing
    if facing_motion is not None:
        frame = draw_player_facing_arrows(frame, dets, facing_motion, style="motion")
    if facing_kalman is not None:
        kalman_style = "kalman" if facing_motion is not None else "motion"
        frame = draw_player_facing_arrows(
            frame, dets, facing_kalman, style=kalman_style
        )
    if facing_motion is not None and facing_kalman is not None:
        frame = draw_facing_legend(frame)
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


def draw_radar_minimap(
    frame: np.ndarray,
    dets: sv.Detections,
    keypoints: sv.KeyPoints | None = None,
    *,
    scale_frac: float = 0.33,
    position: str = "bottom_right",
    pitch_confidence: float = 0.9,
    use_ransac: bool = False,
    ransac_thresh: float = HOMOGRAPHY_RANSAC_REPROJ_THRESH,
    transformer: ViewTransformer | None = None,
    clip_radar_transformer: ViewTransformer | None = None,
    locked_goal_defenders: tuple[int, int] | None = None,
    prebuilt_radar: np.ndarray | None = None,
    debug_keypoints: bool = False,
) -> np.ndarray:
    """Sports-style radar minimap: per-frame H from gated keypoints (no mirror lock)."""
    del clip_radar_transformer  # deprecated; minimap always fits per-frame from keypoints
    if prebuilt_radar is not None:
        radar = prebuilt_radar
    elif keypoints is not None:
        from world_cup_projects.common.pitch import render_radar_simple

        radar = render_radar_simple(
            dets,
            keypoints,
            confidence=pitch_confidence,
            locked_goal_defenders=locked_goal_defenders,
            debug_keypoints=debug_keypoints,
        )
    elif transformer is not None:
        from world_cup_projects.common.pitch import render_radar_from_transformer

        radar = render_radar_from_transformer(
            dets, transformer, locked_goal_defenders=locked_goal_defenders
        )
    else:
        return frame
    if radar is None:
        return frame

    h, w, _ = frame.shape
    rw = max(int(w * scale_frac), 120)
    rh = int(radar.shape[0] * (rw / radar.shape[1]))
    radar = sv.resize_image(radar, (rw, rh))
    margin = 14
    brand_clearance = 52
    if position == "bottom_left":
        x, y = margin, h - rh - margin
    elif position == "bottom_center":
        x, y = (w - rw) // 2, h - rh - margin
    else:
        x, y = w - rw - margin, h - rh - margin - brand_clearance
    rect = sv.Rect(x=x, y=y, width=rw, height=rh)
    return sv.draw_image(frame, radar, opacity=0.5, rect=rect)


draw_radar_bottom_center = draw_radar_minimap


_KP_USED_BGR = (80, 220, 80)
_KP_LOW_CONF_BGR = (80, 80, 255)
_KP_INVALID_BGR = (140, 140, 140)
_KP_EDGE_BGR = (70, 70, 90)
_KP_RAW_BGR = (255, 220, 80)
_KP_RADAR_SMOOTH_BGR = (220, 80, 255)
_KP_SPEED_SMOOTH_BGR = (80, 200, 255)


def draw_pitch_keypoints_compare(
    frame: np.ndarray,
    keypoints: sv.KeyPoints | None,
    *,
    radar_smooth_xy: np.ndarray | None = None,
    speed_smooth_xy: np.ndarray | None = None,
    confidence_threshold: float = 0.5,
) -> np.ndarray:
    """Overlay raw detections vs temporally smoothed points used to fit homography."""
    margin = 14
    legend_y = 52

    if keypoints is None or keypoints.xy.shape[0] == 0:
        draw_text_shadow(
            frame,
            "pitch kp: none",
            (margin, legend_y),
            font_scale=0.5,
            color_bgr=(180, 180, 180),
            thickness=1,
        )
        return frame

    xy = keypoints.xy[0]
    n = len(PITCH_CONFIG.vertices)
    conf = pitch_keypoint_confidence(keypoints, n_vertices=n)

    def _valid_pt(x: float, y: float) -> bool:
        return x > 1 and y > 1 and np.isfinite(x) and np.isfinite(y)

    def _draw_smooth(
        smooth: np.ndarray | None, color: tuple[int, int, int], radius: int
    ) -> int:
        if smooth is None:
            return 0
        count = 0
        for i in range(min(len(smooth), n)):
            x, y = float(smooth[i, 0]), float(smooth[i, 1])
            if not _valid_pt(x, y):
                continue
            px, py = int(x), int(y)
            cv2.circle(frame, (px, py), radius, color, -1, cv2.LINE_AA)
            cv2.circle(frame, (px, py), radius, (255, 255, 255), 1, cv2.LINE_AA)
            count += 1
        return count

    n_radar = _draw_smooth(radar_smooth_xy, _KP_RADAR_SMOOTH_BGR, 7)
    n_speed = _draw_smooth(speed_smooth_xy, _KP_SPEED_SMOOTH_BGR, 5)

    n_raw = 0
    for i in range(min(len(xy), n)):
        x, y = float(xy[i, 0]), float(xy[i, 1])
        if not _valid_pt(x, y):
            continue
        px, py = int(x), int(y)
        cv2.circle(frame, (px, py), 4, _KP_RAW_BGR, 1, cv2.LINE_AA)
        n_raw += 1
        if radar_smooth_xy is not None and i < len(radar_smooth_xy):
            sx, sy = float(radar_smooth_xy[i, 0]), float(radar_smooth_xy[i, 1])
            if _valid_pt(sx, sy):
                cv2.line(
                    frame,
                    (px, py),
                    (int(sx), int(sy)),
                    _KP_RADAR_SMOOTH_BGR,
                    1,
                    cv2.LINE_AA,
                )

    summary = f"pitch kp raw {n_raw}  |  smooth radar {n_radar}  speed {n_speed}"
    draw_text_shadow(
        frame, summary, (margin, legend_y), font_scale=0.5, color_bgr=(230, 230, 230), thickness=1
    )
    for dy, text, color in (
        (20, "cyan ring = raw model (this frame)", _KP_RAW_BGR),
        (38, "magenta = smoothed -> radar H", _KP_RADAR_SMOOTH_BGR),
        (56, "yellow = smoothed -> speed H (conf filter)", _KP_SPEED_SMOOTH_BGR),
    ):
        draw_text_shadow(
            frame, text, (margin, legend_y + dy), font_scale=0.42, color_bgr=color, thickness=1
        )
    return frame


def _keypoint_xy_valid(x: float, y: float) -> bool:
    return bool(np.isfinite(x) and np.isfinite(y) and x > 1 and y > 1)


def draw_pitch_keypoints_debug(
    frame: np.ndarray,
    keypoints: sv.KeyPoints | None,
    *,
    confidence_threshold: float = 0.5,
    draw_skeleton: bool = True,
    show_rejected: bool = True,
) -> np.ndarray:
    """Overlay raw pitch-keypoint detections (index + confidence) for homography debugging.

    Labels use 0-based indices matching ``config.vertices`` / the YOLO pose head order.
    Skeleton edges only connect keypoints that pass the confidence filter (model is
    trained on broadcast football; SoccerNet angles may still look sparse or wrong).
    Missing/placeholder model outputs (``x <= 1`` or ``y <= 1``) are omitted from the
    overlay so they do not stack on the frame border.
    """
    margin = 14
    legend_y = 52  # below HUD bar; avoids overlapping bottom-right radar

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
    from world_cup_projects.common.pitch import (
        pitch_keypoint_inlier_mask,
        view_transformer_from_keypoints,
    )

    conf = pitch_keypoint_confidence(keypoints, n_vertices=n)
    h_t = view_transformer_from_keypoints(
        keypoints, confidence=confidence_threshold, use_ransac=True
    )
    accept = pitch_keypoint_inlier_mask(
        xy, conf, h_t, confidence=confidence_threshold, max_reproj_px=8.0
    )

    n_invalid = 0
    if draw_skeleton:
        # edges are 1-based vertex ids (same as draw_pitch / roboflow sports)
        for start, end in PITCH_CONFIG.edges:
            i, j = start - 1, end - 1
            if i >= len(xy) or j >= len(xy):
                continue
            if not (
                _keypoint_xy_valid(float(xy[i, 0]), float(xy[i, 1]))
                and _keypoint_xy_valid(float(xy[j, 0]), float(xy[j, 1]))
            ):
                continue
            if not (accept[i] and accept[j]):
                continue
            p1 = (int(xy[i, 0]), int(xy[i, 1]))
            p2 = (int(xy[j, 0]), int(xy[j, 1]))
            cv2.line(frame, p1, p2, _KP_EDGE_BGR, 1, cv2.LINE_AA)

    for i in range(min(len(xy), n)):
        x, y = float(xy[i, 0]), float(xy[i, 1])
        if not _keypoint_xy_valid(x, y):
            n_invalid += 1
            continue
        if accept[i]:
            color = _KP_USED_BGR
        elif show_rejected:
            color = _KP_LOW_CONF_BGR
        else:
            continue
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
    summary = (
        f"pitch kp {n_ok}/{n} used (conf > {confidence_threshold:.2f}, "
        f"{n_invalid} missing)"
    )
    draw_text_shadow(
        frame, summary, (margin, legend_y), font_scale=0.52, color_bgr=(230, 230, 230), thickness=1
    )
    for dy, text, color in (
        (22, "green = used for H (tracker upgrade path)", _KP_USED_BGR),
        (40, "red = low confidence", _KP_LOW_CONF_BGR),
        (58, "gray = invalid / missing", _KP_INVALID_BGR),
        (76, "see compare overlay for smoothed H inputs", _KP_EDGE_BGR),
    ):
        draw_text_shadow(
            frame, text, (margin, legend_y + dy), font_scale=0.42, color_bgr=color, thickness=1
        )
    return frame


def draw_homography_feet_debug(
    frame: np.ndarray,
    dets: sv.Detections,
    keypoints: sv.KeyPoints | None,
    *,
    confidence_threshold: float = 0.5,
    reproj_thresh_px: float = 25.0,
) -> np.ndarray:
    """Feet warp check using the same confidence-filtered H as the radar."""
    from world_cup_projects.common.pitch import view_transformer_from_keypoints

    transformer = view_transformer_from_keypoints(
        keypoints, confidence=confidence_threshold, use_ransac=False
    )
    if transformer is None:
        return frame

    pmask = player_mask(dets)
    if not pmask.any():
        return frame
    feet = feet_xy(dets)[pmask]
    pitch_cm = image_to_pitch_cm(feet, transformer)
    if pitch_cm is None:
        return frame
    back = pitch_cm_to_image(pitch_cm, transformer)
    if back is None:
        return frame
    valid = _valid_pitch_cm(pitch_cm)
    for foot, reproj, ok in zip(feet, back, valid):
        fx, fy = int(foot[0]), int(foot[1])
        if not ok:
            cv2.circle(frame, (fx, fy), 6, (80, 80, 255), 2, cv2.LINE_AA)
            continue
        rx, ry = int(reproj[0]), int(reproj[1])
        err = float(np.hypot(reproj[0] - foot[0], reproj[1] - foot[1]))
        color = (80, 220, 80) if err <= reproj_thresh_px else (80, 80, 255)
        cv2.circle(frame, (fx, fy), 5, color, -1, cv2.LINE_AA)
        if err > 4.0:
            cv2.line(frame, (fx, fy), (rx, ry), (255, 220, 80), 1, cv2.LINE_AA)
            cv2.circle(frame, (rx, ry), 4, (255, 220, 80), 1, cv2.LINE_AA)
    draw_text_shadow(
        frame,
        "feet: green=on-pitch  orange=H warp error  red=off-pitch (sports H)",
        (14, 118),
        font_scale=0.42,
        color_bgr=(200, 200, 200),
        thickness=1,
    )
    return frame


def ease_out_cubic(t: float) -> float:
    t = float(np.clip(t, 0.0, 1.0))
    return 1.0 - (1.0 - t) ** 3


def draw_carrier_spotlight(
    dimmed: np.ndarray,
    original: np.ndarray,
    center: tuple[int, int],
    *,
    radius: int = 150,
    strength: float = 0.62,
) -> np.ndarray:
    """Keep the ball carrier readable while the rest of the frame is dimmed."""
    h, w = dimmed.shape[:2]
    cx, cy = center
    mask = np.zeros((h, w), dtype=np.float32)
    cv2.circle(mask, (cx, cy), radius, 1.0, -1, cv2.LINE_AA)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=radius * 0.38)
    mask = (mask[..., None] * strength).astype(np.float32)
    out = dimmed.astype(np.float32) * (1.0 - mask) + original.astype(np.float32) * mask
    return np.clip(out, 0, 255).astype(np.uint8)


def draw_carrier_halo(
    frame: np.ndarray,
    center: tuple[int, int],
    *,
    radius: int = 20,
    strength: float = 0.24,
) -> None:
    """Soft white glow at the ball carrier's feet (drawn in-place)."""
    h, w = frame.shape[:2]
    cx, cy = int(center[0]), int(center[1])
    mask = np.zeros((h, w), dtype=np.float32)
    cv2.circle(mask, (cx, cy), radius, 1.0, -1, cv2.LINE_AA)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=max(radius * 0.42, 1.0))
    mask = (mask[..., None] * strength).astype(np.float32)
    glow = frame.astype(np.float32).copy()
    cv2.circle(glow, (cx, cy), max(radius - 4, 6), (255, 255, 255), -1, cv2.LINE_AA)
    frame[:] = np.clip(
        frame.astype(np.float32) * (1.0 - mask) + glow * mask,
        0,
        255,
    ).astype(np.uint8)


def draw_carrier_pulse(
    frame: np.ndarray,
    center: tuple[int, int],
    t: float,
    *,
    color_bgr: tuple[int, int, int] = ROBOFLOW_PURPLE_BGR,
) -> None:
    """Animated focus rings on the ball carrier."""
    cx, cy = center
    for i, base_r in enumerate((22, 36, 52)):
        phase = (t + i * 0.22) % 1.0
        wave = 0.5 + 0.5 * np.sin(phase * 2.0 * np.pi)
        radius = int(base_r * (0.92 + 0.12 * wave))
        alpha = 0.22 + 0.18 * wave
        layer = frame.copy()
        cv2.circle(layer, (cx, cy), radius, color_bgr, 2, cv2.LINE_AA)
        frame[:] = cv2.addWeighted(layer, alpha, frame, 1.0 - alpha, 0)
    cv2.circle(frame, (cx, cy), 10, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), 14, color_bgr, 2, cv2.LINE_AA)


def draw_pass_analysis_panel(
    frame: np.ndarray,
    *,
    progress: float,
    revealed: int,
    total: int = 3,
    rank_label: str | None = None,
    rank_color: tuple[int, int, int] | None = None,
) -> np.ndarray:
    """Bottom broadcast-style panel for the freeze / reveal sequence."""
    h, w = frame.shape[:2]
    panel_h = 78
    panel_w = min(460, w - 48)
    px = (w - panel_w) // 2
    py = h - panel_h - 62

    overlay = frame.copy()
    cv2.rectangle(overlay, (px, py), (px + panel_w, py + panel_h), (16, 16, 20), -1)
    cv2.rectangle(overlay, (px, py), (px + panel_w, py + panel_h), (48, 48, 58), 1)
    cv2.rectangle(overlay, (px, py), (px + panel_w, py + 4), ROBOFLOW_PURPLE_BGR, -1)
    frame[:] = cv2.addWeighted(overlay, 0.9, frame, 0.1, 0)

    if revealed == 0:
        title = "PASS ANALYSIS"
        step = int(progress * 9) % 3
        dots = "." * (step + 1)
        subtitle = f"Scanning open lanes{dots}"
    else:
        label = rank_label or f"OPTION {revealed}"
        title = label
        subtitle = f"Route {revealed} of {total}  |  ranked by lane + distance"

    draw_text_shadow(
        frame,
        title,
        (px + 18, py + 30),
        font_scale=0.62,
        color_bgr=rank_color or (255, 255, 255),
        thickness=2,
    )
    draw_text_shadow(
        frame,
        subtitle,
        (px + 18, py + 58),
        font_scale=0.46,
        color_bgr=(175, 175, 185),
        thickness=1,
    )

    bar_w = panel_w - 36
    bar_x = px + 18
    bar_y = py + panel_h - 12
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 4), (40, 40, 48), -1)
    fill = int(bar_w * ease_out_cubic(progress if revealed else (0.35 + 0.65 * progress)))
    cv2.rectangle(
        frame,
        (bar_x, bar_y),
        (bar_x + max(fill, 6), bar_y + 4),
        rank_color or ROBOFLOW_PURPLE_BGR,
        -1,
    )
    return frame


def draw_glow_arrow(
    frame: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    color_bgr: tuple[int, int, int],
    *,
    thickness: int = 4,
    alpha: float = 1.0,
) -> None:
    if alpha <= 0.01:
        return
    layer = frame.copy()
    cv2.arrowedLine(
        layer, start, end, (20, 20, 20), thickness + 3, cv2.LINE_AA, tipLength=0.05
    )
    cv2.arrowedLine(layer, start, end, color_bgr, thickness, cv2.LINE_AA, tipLength=0.05)
    if alpha >= 0.99:
        frame[:] = layer
    else:
        frame[:] = cv2.addWeighted(layer, alpha, frame, 1.0 - alpha, 0)


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
