"""Pitch geometry, homography and radar minimap.

`SoccerPitchConfiguration`, `ViewTransformer`, `draw_pitch` and `draw_points_on_pitch`
are vendored (lightly trimmed) from roboflow/sports
(https://github.com/roboflow/sports, Apache-2.0) so the demos do not depend on the
unpublished `sports` package.

`PitchHomography` wraps a YOLOv8-pose pitch-keypoint model (the
``football-pitch-detection.pt`` weights from roboflow/sports) to estimate a per-frame
homography that maps image points to real pitch coordinates (centimeters). The weights
live on Google Drive; see the README "weights" note if the auto-download is blocked.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np
import numpy.typing as npt
import supervision as sv


# --------------------------------------------------------------------------- #
# Pitch configuration (vendored from sports/configs/soccer.py)
# --------------------------------------------------------------------------- #
@dataclass
class SoccerPitchConfiguration:
    width: int = 7000  # [cm]
    length: int = 12000  # [cm]
    penalty_box_width: int = 4100
    penalty_box_length: int = 2015
    goal_box_width: int = 1832
    goal_box_length: int = 550
    centre_circle_radius: int = 915
    penalty_spot_distance: int = 1100

    @property
    def vertices(self) -> List[Tuple[int, int]]:
        w, length = self.width, self.length
        pbw, pbl = self.penalty_box_width, self.penalty_box_length
        gbw, gbl = self.goal_box_width, self.goal_box_length
        psd, ccr = self.penalty_spot_distance, self.centre_circle_radius
        return [
            (0, 0), (0, (w - pbw) / 2), (0, (w - gbw) / 2), (0, (w + gbw) / 2),
            (0, (w + pbw) / 2), (0, w), (gbl, (w - gbw) / 2), (gbl, (w + gbw) / 2),
            (psd, w / 2), (pbl, (w - pbw) / 2), (pbl, (w - gbw) / 2),
            (pbl, (w + gbw) / 2), (pbl, (w + pbw) / 2), (length / 2, 0),
            (length / 2, w / 2 - ccr), (length / 2, w / 2 + ccr), (length / 2, w),
            (length - pbl, (w - pbw) / 2), (length - pbl, (w - gbw) / 2),
            (length - pbl, (w + gbw) / 2), (length - pbl, (w + pbw) / 2),
            (length - psd, w / 2), (length - gbl, (w - gbw) / 2),
            (length - gbl, (w + gbw) / 2), (length, 0), (length, (w - pbw) / 2),
            (length, (w - gbw) / 2), (length, (w + gbw) / 2), (length, (w + pbw) / 2),
            (length, w), (length / 2 - ccr, w / 2), (length / 2 + ccr, w / 2),
        ]

    edges: List[Tuple[int, int]] = field(default_factory=lambda: [
        (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (7, 8), (10, 11), (11, 12),
        (12, 13), (14, 15), (15, 16), (16, 17), (18, 19), (19, 20), (20, 21),
        (23, 24), (25, 26), (26, 27), (27, 28), (28, 29), (29, 30), (1, 14),
        (2, 10), (3, 7), (4, 8), (5, 13), (6, 17), (14, 25), (18, 26), (23, 27),
        (24, 28), (21, 29), (17, 30),
    ])


# --------------------------------------------------------------------------- #
# Homography (vendored from sports/common/view.py)
# --------------------------------------------------------------------------- #
# RANSAC reprojection threshold in image pixels; rejects mis-detected pitch keypoints.
HOMOGRAPHY_RANSAC_REPROJ_THRESH = 5.0
def _find_homography_matrix(
    source: npt.NDArray,
    target: npt.NDArray,
    *,
    use_ransac: bool = True,
    ransac_thresh: float = HOMOGRAPHY_RANSAC_REPROJ_THRESH,
) -> npt.NDArray:
    src = source.astype(np.float32)
    dst = target.astype(np.float32)
    if use_ransac and len(src) >= 4:
        m, _ = cv2.findHomography(src, dst, cv2.RANSAC, ransacReprojThreshold=ransac_thresh)
        if m is not None:
            return m
    m, _ = cv2.findHomography(src, dst)
    if m is None:
        raise ValueError("Homography matrix could not be calculated.")
    return m


class ViewTransformer:
    def __init__(
        self,
        source: npt.NDArray | None = None,
        target: npt.NDArray | None = None,
        *,
        matrix: npt.NDArray | None = None,
        use_ransac: bool = True,
        ransac_thresh: float = HOMOGRAPHY_RANSAC_REPROJ_THRESH,
    ) -> None:
        if matrix is not None:
            self.m = matrix.astype(np.float64)
            return
        if source is None or target is None:
            raise ValueError("Provide source/target points or a precomputed matrix.")
        if source.shape != target.shape or source.shape[1] != 2:
            raise ValueError("source/target must be matching (N, 2) arrays")
        self.m = _find_homography_matrix(
            source, target, use_ransac=use_ransac, ransac_thresh=ransac_thresh
        )

    def transform_points(self, points: npt.NDArray) -> npt.NDArray:
        if points.size == 0:
            return points
        reshaped = points.reshape(-1, 1, 2).astype(np.float32)
        return cv2.perspectiveTransform(reshaped, self.m).reshape(-1, 2).astype(np.float32)


# --------------------------------------------------------------------------- #
# Radar drawing (vendored from sports/annotators/soccer.py)
# --------------------------------------------------------------------------- #
def draw_pitch(
    config: SoccerPitchConfiguration,
    background_color: sv.Color = sv.Color(34, 139, 34),
    line_color: sv.Color = sv.Color.WHITE,
    padding: int = 50,
    line_thickness: int = 4,
    point_radius: int = 8,
    scale: float = 0.1,
) -> np.ndarray:
    sw, sl = int(config.width * scale), int(config.length * scale)
    scr = int(config.centre_circle_radius * scale)
    spd = int(config.penalty_spot_distance * scale)
    pitch = np.ones((sw + 2 * padding, sl + 2 * padding, 3), np.uint8) * np.array(
        background_color.as_bgr(), np.uint8
    )
    for start, end in config.edges:
        p1 = (int(config.vertices[start - 1][0] * scale) + padding,
              int(config.vertices[start - 1][1] * scale) + padding)
        p2 = (int(config.vertices[end - 1][0] * scale) + padding,
              int(config.vertices[end - 1][1] * scale) + padding)
        cv2.line(pitch, p1, p2, line_color.as_bgr(), line_thickness)
    cv2.circle(pitch, (sl // 2 + padding, sw // 2 + padding), scr,
               line_color.as_bgr(), line_thickness)
    for spot in [(spd + padding, sw // 2 + padding),
                 (sl - spd + padding, sw // 2 + padding)]:
        cv2.circle(pitch, spot, point_radius, line_color.as_bgr(), -1)
    return pitch


def draw_points_on_pitch(
    config: SoccerPitchConfiguration,
    xy: np.ndarray,
    face_color: sv.Color = sv.Color.RED,
    edge_color: sv.Color = sv.Color.BLACK,
    radius: int = 10,
    thickness: int = 2,
    padding: int = 50,
    scale: float = 0.1,
    pitch: Optional[np.ndarray] = None,
) -> np.ndarray:
    if pitch is None:
        pitch = draw_pitch(config=config, padding=padding, scale=scale)
    for point in xy:
        sp = (int(point[0] * scale) + padding, int(point[1] * scale) + padding)
        cv2.circle(pitch, sp, radius, face_color.as_bgr(), -1)
        cv2.circle(pitch, sp, radius, edge_color.as_bgr(), thickness)
    return pitch


# --------------------------------------------------------------------------- #
# Pitch keypoint model -> per-frame homography
# --------------------------------------------------------------------------- #
class PitchHomography:
    """Estimate image->pitch homography from a YOLOv8-pose pitch-keypoint model.

    Args:
        weights_path: path to ``football-pitch-detection.pt`` (roboflow/sports).
        config: pitch configuration whose ``vertices`` are the target points.
        device: torch device string.
        confidence: keypoint confidence threshold.
    """

    def __init__(
        self,
        weights_path: str,
        config: SoccerPitchConfiguration | None = None,
        device: str = "cpu",
        confidence: float = 0.5,
    ) -> None:
        from ultralytics import YOLO  # local import: heavy + optional

        self.model = YOLO(weights_path)
        self.config = config or SoccerPitchConfiguration()
        self.device = device
        self.confidence = confidence
        self._targets = np.array(self.config.vertices, dtype=np.float32)

    def __call__(self, frame_bgr: np.ndarray) -> ViewTransformer | None:
        result = self.model(frame_bgr, device=self.device, verbose=False)[0]
        kps = sv.KeyPoints.from_ultralytics(result)
        if kps.xy.shape[0] == 0:
            return None
        xy = kps.xy[0]
        conf = pitch_keypoint_confidence(kps, n_vertices=len(xy))
        mask = pitch_keypoint_accept_mask(xy, conf, confidence=self.confidence)
        if mask.sum() < 4:
            return None
        return ViewTransformer(
            source=xy[mask],
            target=self._targets[mask],
            ransac_thresh=getattr(self, "ransac_thresh", HOMOGRAPHY_RANSAC_REPROJ_THRESH),
        )


PITCH_CONFIG = SoccerPitchConfiguration()


def infer_goal_defenders(
    pitch_xy_cm: np.ndarray, teams: np.ndarray
) -> tuple[int, int]:
    """Return ``(left_goal_team, right_goal_team)`` from feet on the pitch (cm)."""
    if len(pitch_xy_cm) == 0:
        from world_cup_projects.common.soccernet import TEAM_LEFT, TEAM_RIGHT

        return TEAM_LEFT, TEAM_RIGHT
    medians: list[float] = []
    for team_id in (0, 1):
        mask = teams == team_id
        medians.append(
            float(pitch_xy_cm[mask, 0].mean()) if mask.any() else float("inf")
        )
    if medians[0] <= medians[1]:
        return 0, 1
    return 1, 0


def pitch_layout_reliable(
    pitch_xy_m: np.ndarray,
    teams: np.ndarray | None = None,
    *,
    config: SoccerPitchConfiguration = PITCH_CONFIG,
    min_players: int = 8,
    min_x_spread_m: float = 14.0,
    min_y_spread_m: float = 10.0,
    max_center_colony_frac: float = 0.4,
    center_band_m: float = 9.0,
    min_team_x_sep_m: float = 12.0,
) -> bool:
    """False when homography collapses players onto the halfway line (bad H / early frames)."""
    if pitch_xy_m is None or len(pitch_xy_m) < min_players:
        return False
    xy = np.asarray(pitch_xy_m, dtype=np.float64)
    if not np.isfinite(xy).all():
        return False
    length_m = float(config.length) / 100.0
    center_x = length_m / 2.0
    x_spread = float(np.percentile(xy[:, 0], 90) - np.percentile(xy[:, 0], 10))
    y_spread = float(np.percentile(xy[:, 1], 90) - np.percentile(xy[:, 1], 10))
    if x_spread < min_x_spread_m or y_spread < min_y_spread_m:
        return False
    if (np.abs(xy[:, 0] - center_x) < center_band_m).mean() > max_center_colony_frac:
        return False
    if teams is not None and np.any(teams == 0) and np.any(teams == 1):
        m0 = float(xy[teams == 0, 0].mean())
        m1 = float(xy[teams == 1, 0].mean())
        if abs(m0 - m1) < min_team_x_sep_m:
            return False
    return True


def draw_goals_on_pitch(
    config: SoccerPitchConfiguration,
    *,
    left_defender_team: int,
    right_defender_team: int,
    team_colors: list[sv.Color] | None = None,
    padding: int = 50,
    scale: float = 0.1,
    pitch: np.ndarray | None = None,
    fill_alpha: float = 0.38,
) -> np.ndarray:
    """Highlight each goal mouth in the defending team's color (radar debug)."""
    if team_colors is None:
        team_colors = [
            sv.Color.from_hex("#00BFFF"),
            sv.Color.from_hex("#FF1493"),
        ]
    if pitch is None:
        pitch = draw_pitch(config=config, padding=padding, scale=scale)

    w = config.width
    length = config.length
    gbw, gbl = config.goal_box_width, config.goal_box_length
    y0, y1 = (w - gbw) / 2, (w + gbw) / 2

    def _goal_patch(goal_x_cm: float, defender: int, depth_cm: float) -> None:
        nonlocal pitch
        color = team_colors[defender % len(team_colors)].as_bgr()
        mouth_x = int(goal_x_cm * scale) + padding
        py0 = int(y0 * scale) + padding
        py1 = int(y1 * scale) + padding
        inner_x = int((goal_x_cm + depth_cm) * scale) + padding
        x_lo, x_hi = sorted((mouth_x, inner_x))
        overlay = pitch.copy()
        cv2.rectangle(overlay, (x_lo, py0), (x_hi, py1), color, -1)
        cv2.addWeighted(overlay, fill_alpha, pitch, 1.0 - fill_alpha, 0, pitch)
        cv2.line(pitch, (mouth_x, py0), (mouth_x, py1), color, 5, cv2.LINE_AA)
        cv2.line(pitch, (mouth_x, py0), (mouth_x, py1), (255, 255, 255), 1, cv2.LINE_AA)

    _goal_patch(0.0, left_defender_team, gbl)
    _goal_patch(float(length), right_defender_team, -gbl)
    return pitch


# --------------------------------------------------------------------------- #
# Radar (vendored from sports/examples/soccer/main.py ``render_radar``)
# --------------------------------------------------------------------------- #
_SPORTS_RADAR_COLORS = [
    sv.Color.from_hex("#00BFFF"),
    sv.Color.from_hex("#FF1493"),
]


# Display/radar: same floor as tracker fits; skip only clearly broken frames.
DISPLAY_MIN_KEYPOINTS = 4
DISPLAY_MAX_REPROJ_PX = 10.0


def homography_from_keypoints_simple(
    keypoints: sv.KeyPoints | None,
    *,
    config: SoccerPitchConfiguration = PITCH_CONFIG,
    confidence: float = 0.5,
    min_keypoints: int = DISPLAY_MIN_KEYPOINTS,
    max_reproj_px: float = DISPLAY_MAX_REPROJ_PX,
) -> ViewTransformer | None:
    """Single-frame H from accepted keypoints (no mirror/orientation heuristics).

    Returns ``None`` when too few keypoints pass the mask or reprojection is poor
    (common on clips like SNMOT-189 where area keypoints are misdetected).
    """
    if keypoints is None or keypoints.xy.shape[0] == 0:
        return None
    xy = keypoints.xy[0]
    conf = pitch_keypoint_confidence(keypoints, n_vertices=len(xy))
    mask = pitch_keypoint_accept_mask(xy, conf, confidence=confidence)
    if mask.sum() < min_keypoints:
        return None
    src = xy[mask].astype(np.float32)
    dst = np.array(config.vertices, dtype=np.float32)[mask]
    try:
        t = ViewTransformer(
            source=src,
            target=dst,
            use_ransac=False,
            ransac_thresh=HOMOGRAPHY_RANSAC_REPROJ_THRESH,
        )
    except ValueError:
        return None
    if _mean_reproj_px(t, src, dst) > max_reproj_px:
        return None
    return t


def _keypoint_image_valid(x: float, y: float) -> bool:
    return bool(np.isfinite(x) and np.isfinite(y) and x > 1 and y > 1)


def draw_radar_pitch_keypoints_debug(
    radar: np.ndarray,
    keypoints: sv.KeyPoints,
    transformer: ViewTransformer,
    *,
    config: SoccerPitchConfiguration = PITCH_CONFIG,
    confidence: float = 0.5,
    padding: int = 50,
    scale: float = 0.1,
) -> np.ndarray:
    """Warp all detected pitch keypoints onto the minimap (accepted vs rejected)."""
    if keypoints.xy.shape[0] == 0:
        return radar
    xy = keypoints.xy[0]
    n = len(config.vertices)
    conf = pitch_keypoint_confidence(keypoints, n_vertices=n)
    accept = pitch_keypoint_accept_mask(xy, conf, confidence=confidence)

    def _to_radar_px(cm_xy: np.ndarray) -> tuple[int, int]:
        return (
            int(cm_xy[0] * scale) + padding,
            int(cm_xy[1] * scale) + padding,
        )

    for start, end in config.edges:
        i, j = start - 1, end - 1
        if i >= len(xy) or j >= len(xy):
            continue
        if not (
            _keypoint_image_valid(float(xy[i, 0]), float(xy[i, 1]))
            and _keypoint_image_valid(float(xy[j, 0]), float(xy[j, 1]))
            and accept[i]
            and accept[j]
        ):
            continue
        seg = transformer.transform_points(xy[[i, j]].astype(np.float32))
        cv2.line(
            radar,
            _to_radar_px(seg[0]),
            _to_radar_px(seg[1]),
            (90, 90, 110),
            1,
            cv2.LINE_AA,
        )

    for i in range(min(len(xy), n)):
        x, y = float(xy[i, 0]), float(xy[i, 1])
        if not _keypoint_image_valid(x, y):
            continue
        kp_cm = transformer.transform_points(np.array([[x, y]], dtype=np.float32))
        if accept[i]:
            face = sv.Color.from_hex("#50DC32")
            radius = 14
        else:
            face = sv.Color.from_hex("#5050FF")
            radius = 10
        radar = draw_points_on_pitch(
            config=config,
            xy=kp_cm,
            face_color=face,
            edge_color=sv.Color.WHITE,
            radius=radius,
            pitch=radar,
        )
    return radar


def render_radar_simple(
    detections: sv.Detections,
    keypoints: sv.KeyPoints | None,
    *,
    config: SoccerPitchConfiguration = PITCH_CONFIG,
    confidence: float = 0.5,
    transformer: ViewTransformer | None = None,
    locked_goal_defenders: tuple[int, int] | None = None,
    debug_keypoints: bool = False,
) -> np.ndarray | None:
    """Minimap: H, team-colored goals, keypoints, player feet."""
    from world_cup_projects.common.soccernet import (
        ROLE_GOALKEEPER,
        ROLE_PLAYER,
        TEAM_LEFT,
        TEAM_RIGHT,
    )

    t = homography_from_keypoints_simple(
        keypoints, config=config, confidence=confidence
    )
    if t is None and transformer is not None:
        t = transformer
    if t is None:
        return None

    radar = draw_pitch(config=config)
    outfield_mask = detections.class_id == ROLE_PLAYER
    gk_mask = detections.class_id == ROLE_GOALKEEPER
    feet_cm = None
    teams = None
    gk_feet_cm = None
    layout_ok = False

    if outfield_mask.any():
        outfield = detections[outfield_mask]
        feet = outfield.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
        feet_cm = t.transform_points(feet.astype(np.float32))
        teams = outfield.data.get("team", np.zeros(len(outfield), dtype=int))
        feet_m = feet_cm / 100.0
        layout_ok = pitch_layout_reliable(feet_m, teams, config=config)

    if gk_mask.any():
        gks = detections[gk_mask]
        gk_feet = gks.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
        gk_feet_cm = t.transform_points(gk_feet.astype(np.float32))

    has_players = outfield_mask.any() or gk_mask.any()
    if feet_cm is not None and teams is not None and outfield_mask.any():
        if locked_goal_defenders is not None:
            left_team, right_team = locked_goal_defenders
        elif layout_ok:
            left_team, right_team = infer_goal_defenders(feet_cm, teams)
        else:
            left_team, right_team = TEAM_LEFT, TEAM_RIGHT
        radar = draw_goals_on_pitch(
            config,
            left_defender_team=left_team,
            right_defender_team=right_team,
            team_colors=_SPORTS_RADAR_COLORS,
            pitch=radar,
        )
    elif not has_players:
        left_team, right_team = TEAM_LEFT, TEAM_RIGHT
        radar = draw_goals_on_pitch(
            config,
            left_defender_team=left_team,
            right_defender_team=right_team,
            team_colors=_SPORTS_RADAR_COLORS,
            pitch=radar,
        )

    if keypoints is not None and keypoints.xy.shape[0] > 0:
        if debug_keypoints:
            radar = draw_radar_pitch_keypoints_debug(
                radar, keypoints, t, config=config, confidence=confidence
            )
        else:
            xy = keypoints.xy[0]
            conf = pitch_keypoint_confidence(keypoints, n_vertices=len(config.vertices))
            mask = pitch_keypoint_accept_mask(xy, conf, confidence=confidence)
            if mask.any():
                kp_cm = t.transform_points(xy[mask].astype(np.float32))
                radar = draw_points_on_pitch(
                    config=config,
                    xy=kp_cm,
                    face_color=sv.Color.from_hex("#FFFFFF"),
                    edge_color=sv.Color.from_hex("#333333"),
                    radius=10,
                    pitch=radar,
                )

    if feet_cm is not None and teams is not None and outfield_mask.any():
        from world_cup_projects.common.visual import _valid_pitch_cm

        on_pitch = _valid_pitch_cm(feet_cm, config, margin_cm=80.0)
        for team_id, color in enumerate(_SPORTS_RADAR_COLORS[:2]):
            team_mask = (teams == team_id) & on_pitch
            if not team_mask.any():
                continue
            radar = draw_points_on_pitch(
                config=config,
                xy=feet_cm[team_mask],
                face_color=color,
                edge_color=sv.Color.BLACK,
                radius=20,
                pitch=radar,
            )

    if gk_feet_cm is not None and gk_mask.any():
        from world_cup_projects.common.visual import _valid_pitch_cm

        gk_teams = detections[gk_mask].data.get(
            "team", np.full(int(gk_mask.sum()), -1, dtype=int)
        )
        on_pitch = _valid_pitch_cm(gk_feet_cm, config, margin_cm=80.0)
        for team_id, color in enumerate(_SPORTS_RADAR_COLORS[:2]):
            team_mask = (gk_teams == team_id) & on_pitch
            if not team_mask.any():
                continue
            radar = draw_points_on_pitch(
                config=config,
                xy=gk_feet_cm[team_mask],
                face_color=color,
                edge_color=sv.Color.WHITE,
                radius=16,
                pitch=radar,
            )
        neutral = on_pitch & ~np.isin(gk_teams, (0, 1))
        if neutral.any():
            radar = draw_points_on_pitch(
                config=config,
                xy=gk_feet_cm[neutral],
                face_color=sv.Color.from_hex("#E8E8E8"),
                edge_color=sv.Color.BLACK,
                radius=14,
                pitch=radar,
            )
    return radar


def render_radar_from_transformer(
    detections: sv.Detections,
    transformer: ViewTransformer,
    *,
    config: SoccerPitchConfiguration = PITCH_CONFIG,
    locked_goal_defenders: tuple[int, int] | None = None,
) -> np.ndarray | None:
    """Warp player feet with a precomputed homography."""
    from world_cup_projects.common.soccernet import (
        ROLE_GOALKEEPER,
        ROLE_PLAYER,
        TEAM_LEFT,
        TEAM_RIGHT,
    )

    outfield_mask = detections.class_id == ROLE_PLAYER
    gk_mask = detections.class_id == ROLE_GOALKEEPER
    radar = draw_pitch(config=config)
    transformed_xy = None
    teams = None
    gk_xy = None

    if outfield_mask.any():
        outfield = detections[outfield_mask]
        xy = outfield.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
        transformed_xy = transformer.transform_points(points=xy.astype(np.float32))
        teams = outfield.data.get("team", np.zeros(len(outfield), dtype=int))
        if locked_goal_defenders is not None:
            left_team, right_team = locked_goal_defenders
        else:
            left_team, right_team = infer_goal_defenders(transformed_xy, teams)
    else:
        left_team, right_team = TEAM_LEFT, TEAM_RIGHT

    if gk_mask.any():
        gks = detections[gk_mask]
        gk_feet = gks.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
        gk_xy = transformer.transform_points(points=gk_feet.astype(np.float32))

    radar = draw_goals_on_pitch(
        config,
        left_defender_team=left_team,
        right_defender_team=right_team,
        team_colors=_SPORTS_RADAR_COLORS,
        pitch=radar,
    )
    if transformed_xy is not None and teams is not None:
        for team_id, color in enumerate(_SPORTS_RADAR_COLORS[:2]):
            team_mask = teams == team_id
            if not team_mask.any():
                continue
            radar = draw_points_on_pitch(
                config=config,
                xy=transformed_xy[team_mask],
                face_color=color,
                edge_color=sv.Color.BLACK,
                radius=20,
                pitch=radar,
            )

    if gk_xy is not None and gk_mask.any():
        gk_teams = detections[gk_mask].data.get(
            "team", np.full(len(gk_xy), -1, dtype=int)
        )
        for team_id, color in enumerate(_SPORTS_RADAR_COLORS[:2]):
            team_mask = gk_teams == team_id
            if not team_mask.any():
                continue
            radar = draw_points_on_pitch(
                config=config,
                xy=gk_xy[team_mask],
                face_color=color,
                edge_color=sv.Color.WHITE,
                radius=16,
                pitch=radar,
            )
        neutral = ~np.isin(gk_teams, (0, 1))
        if neutral.any():
            radar = draw_points_on_pitch(
                config=config,
                xy=gk_xy[neutral],
                face_color=sv.Color.from_hex("#E8E8E8"),
                edge_color=sv.Color.BLACK,
                radius=14,
                pitch=radar,
            )
    return radar


def render_radar_sports(
    detections: sv.Detections,
    keypoints: sv.KeyPoints | None,
    *,
    config: SoccerPitchConfiguration = PITCH_CONFIG,
    confidence: float = 0.5,
    use_ransac: bool = False,
    ransac_thresh: float = HOMOGRAPHY_RANSAC_REPROJ_THRESH,
) -> np.ndarray | None:
    """Build minimap from per-frame keypoints (simple homography by default)."""
    del use_ransac, ransac_thresh  # kept for call-site compatibility
    return render_radar_simple(
        detections, keypoints, config=config, confidence=confidence
    )


_MODEL_DIR = __import__("pathlib").Path(__file__).resolve().parent.parent / ".cache" / "models"
_PITCH_MODEL_PATH = _MODEL_DIR / "football-pitch-detection.pt"
_PITCH_MODEL_GDRIVE_ID = "1Ma5Kt86tgpdjCTKfum79YMgNnSjcoOyf"


def ensure_pitch_model() -> "Path":
    """Download the football pitch keypoint model if missing."""
    from pathlib import Path

    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if _PITCH_MODEL_PATH.is_file():
        return _PITCH_MODEL_PATH

    try:
        import gdown
    except ImportError as exc:
        raise ImportError("Install gdown to download pitch model weights.") from exc

    url = f"https://drive.google.com/uc?id={_PITCH_MODEL_GDRIVE_ID}"
    gdown.download(url, str(_PITCH_MODEL_PATH), quiet=False)
    return _PITCH_MODEL_PATH


def load_pitch_model(device: str = "cpu") -> PitchHomography:
    path = ensure_pitch_model()
    return PitchHomography(str(path), config=PITCH_CONFIG, device=device)


def detect_pitch_keypoints(frame: np.ndarray, model: PitchHomography) -> sv.KeyPoints:
    result = model.model(frame, device=model.device, verbose=False)[0]
    return sv.KeyPoints.from_ultralytics(result)


def pitch_keypoint_confidence(
    keypoints: sv.KeyPoints, n_vertices: int | None = None
) -> np.ndarray:
    """Per-vertex confidence; missing entries are 0."""
    n = n_vertices or len(PITCH_CONFIG.vertices)
    if keypoints is None or keypoints.xy.shape[0] == 0:
        return np.zeros(n, dtype=np.float32)
    xy = keypoints.xy[0]
    if keypoints.confidence is None:
        conf = np.ones(len(xy), dtype=np.float32)
    else:
        conf = keypoints.confidence[0].astype(np.float32)
    if len(conf) < n:
        conf = np.pad(conf, (0, n - len(conf)))
    return conf[:n]


def pitch_keypoint_accept_mask(
    xy: np.ndarray,
    conf: np.ndarray,
    *,
    confidence: float = 0.5,
) -> np.ndarray:
    """True where a keypoint is used for homography (same rule as ``PitchHomography``)."""
    n = min(len(xy), len(conf))
    if n == 0:
        return np.zeros(0, dtype=bool)
    return (conf[:n] > confidence) & (xy[:n, 0] > 1) & (xy[:n, 1] > 1)


def view_transformer_from_keypoints(
    keypoints: sv.KeyPoints | None,
    config: SoccerPitchConfiguration = PITCH_CONFIG,
    confidence: float = 0.5,
    *,
    use_ransac: bool = True,
    ransac_thresh: float = HOMOGRAPHY_RANSAC_REPROJ_THRESH,
    orientation_anchor: ViewTransformer | None = None,
) -> ViewTransformer | None:
    """Per-frame H from model keypoints (tries plain + mirrored pitch like the tracker)."""
    if keypoints is None or keypoints.xy.shape[0] == 0:
        return None
    xy = keypoints.xy[0]
    conf = pitch_keypoint_confidence(keypoints, n_vertices=len(xy))
    mask = pitch_keypoint_accept_mask(xy, conf, confidence=confidence)
    if mask.sum() < 4:
        return None
    src = xy[mask].astype(np.float32)
    dst = np.array(config.vertices, dtype=np.float32)[mask]
    length = float(config.length)
    candidates: list[tuple[float, ViewTransformer]] = []
    for target in (dst, _flip_pitch_x_targets(dst, length)):
        try:
            t = ViewTransformer(
                source=src,
                target=target,
                use_ransac=use_ransac,
                ransac_thresh=ransac_thresh,
            )
        except ValueError:
            continue
        candidates.append((_mean_reproj_px(t, src, target), t))
    if not candidates:
        return None
    candidates.sort(key=lambda pair: pair[0])
    for _err, t in candidates:
        if orientation_anchor is None or _orientation_matches_anchor(
            t, orientation_anchor, src
        ):
            return t
    return candidates[0][1]


def _players_on_pitch_score(
    transformer: ViewTransformer,
    detections,
    config: SoccerPitchConfiguration = PITCH_CONFIG,
) -> tuple[int, float]:
    """Count in-bounds players and team separation on the pitch (cm)."""
    from world_cup_projects.common.possession import feet_xy, player_mask
    from world_cup_projects.common.visual import _valid_pitch_cm

    pmask = player_mask(detections)
    if not pmask.any():
        return 0, 0.0
    feet = feet_xy(detections)[pmask].astype(np.float32)
    cm = transformer.transform_points(feet)
    in_bounds = int(_valid_pitch_cm(cm, config, margin_cm=80.0).sum())
    teams = detections.data["team"][pmask]
    separation = 0.0
    if np.any(teams == 0) and np.any(teams == 1):
        separation = abs(
            float(cm[teams == 0, 0].mean()) - float(cm[teams == 1, 0].mean())
        )
    return in_bounds, separation


def resolve_clip_display_anchor(
    sequence,
    keypoints_by_frame: dict[int, sv.KeyPoints | None],
    *,
    confidence: float = 0.5,
    sample_step: int = 20,
    max_reproj_px: float = 8.0,
) -> ViewTransformer | None:
    """Pick one clip-wide H for radar/overlays (avoids per-frame left/right flips).

    Scores plain vs mirrored pitch fits by player feet landing in-bounds and team
    separation on the pitch, not keypoint reprojection alone.
    """
    from world_cup_projects.common.soccernet import iter_gt_detections

    config = PITCH_CONFIG
    length = float(config.length)
    best_score = -1.0
    best_t: ViewTransformer | None = None

    for frame_idx, dets in iter_gt_detections(sequence):
        if frame_idx % sample_step != 0:
            continue
        kps = keypoints_by_frame.get(frame_idx)
        if kps is None or kps.xy.shape[0] == 0:
            continue
        xy = kps.xy[0]
        conf = pitch_keypoint_confidence(kps, n_vertices=len(xy))
        mask = pitch_keypoint_accept_mask(xy, conf, confidence=confidence)
        if mask.sum() < 4:
            continue
        src = xy[mask].astype(np.float32)
        dst = np.array(config.vertices, dtype=np.float32)[mask]
        for target in (dst, _flip_pitch_x_targets(dst, length)):
            try:
                t = ViewTransformer(
                    source=src,
                    target=target,
                    use_ransac=False,
                    ransac_thresh=HOMOGRAPHY_RANSAC_REPROJ_THRESH,
                )
            except ValueError:
                continue
            err = _mean_reproj_px(t, src, target)
            if err > max_reproj_px:
                continue
            in_bounds, separation = _players_on_pitch_score(t, dets, config)
            if in_bounds < 4:
                continue
            score = in_bounds * 12.0 + separation / 50.0 - err
            if score > best_score:
                best_score = score
                best_t = t
    return best_t


def display_homography(
    keypoints: sv.KeyPoints | None,
    locked: ViewTransformer | None,
    *,
    confidence: float = 0.5,
    clip_display_anchor: ViewTransformer | None = None,
) -> ViewTransformer | None:
    """Per-frame display H (simple fit); ``locked`` / anchor only as fallback."""
    del clip_display_anchor
    t = homography_from_keypoints_simple(keypoints, confidence=confidence)
    if t is not None:
        return t
    return locked


def image_to_pitch_cm(
    points_xy: np.ndarray, transformer: ViewTransformer | None
) -> np.ndarray | None:
    if transformer is None or points_xy.size == 0:
        return None
    return transformer.transform_points(points_xy.astype(np.float32))


def pitch_cm_to_image(
    points_cm: np.ndarray, transformer: ViewTransformer | None
) -> np.ndarray | None:
    """Map pitch points (cm) back to image pixels via ``H^{-1}`` (homography sanity check)."""
    if transformer is None or points_cm.size == 0:
        return None
    pts = points_cm.reshape(-1, 1, 2).astype(np.float32)
    try:
        inv = np.linalg.inv(transformer.m)
    except np.linalg.LinAlgError:
        return None
    return cv2.perspectiveTransform(pts, inv).reshape(-1, 2)


def image_to_pitch_m(
    points_xy: np.ndarray, transformer: ViewTransformer | None
) -> np.ndarray | None:
    cm = image_to_pitch_cm(points_xy, transformer)
    if cm is None:
        return None
    return cm / 100.0


def pitch_attack_direction(
    detections: sv.Detections,
    carrier_team: int,
    transformer: ViewTransformer,
    *,
    player_mask_fn,
    feet_fn,
) -> np.ndarray:
    from world_cup_projects.common.geometry import unit
    from world_cup_projects.common.soccernet import TEAM_LEFT

    pmask = player_mask_fn(detections)
    if not pmask.any():
        return np.array([1.0, 0.0])

    feet = feet_fn(detections)[pmask]
    teams = detections.data["team"][pmask]
    pitch_xy = image_to_pitch_m(feet, transformer)
    if pitch_xy is None:
        return np.array([1.0, 0.0])

    own = pitch_xy[teams == carrier_team]
    opp = pitch_xy[teams == (1 - carrier_team)]
    if len(own) == 0 or len(opp) == 0:
        return np.array([1.0, 0.0]) if carrier_team == TEAM_LEFT else np.array([-1.0, 0.0])

    return unit(opp.mean(axis=0) - own.mean(axis=0))


def _flip_pitch_x_targets(dst: np.ndarray, length: float) -> np.ndarray:
    out = dst.copy()
    out[:, 0] = length - out[:, 0]
    return out


def _mean_reproj_px(transformer: ViewTransformer, src: np.ndarray, dst: np.ndarray) -> float:
    reproj = transformer.transform_points(src)
    return float(np.linalg.norm(reproj - dst, axis=1).mean())


def _orientation_matches_anchor(
    candidate: ViewTransformer,
    anchor: ViewTransformer,
    src: np.ndarray,
    *,
    min_corr: float = 0.85,
) -> bool:
    """Reject H that mirrors the pitch left/right vs the first locked estimate."""
    if len(src) < 4:
        return True
    pa = anchor.transform_points(src)
    pc = candidate.transform_points(src)
    corr = np.corrcoef(pa[:, 0], pc[:, 0])[0, 1]
    if not np.isfinite(corr):
        return True
    return float(corr) >= min_corr


class PitchHomographyTracker:
    """Sequence-stable homography without cross-frame point stacking.

    Each frame fits ``H`` from **that frame's** confidence-filtered keypoints only
    (stacking image points across a moving camera breaks the homography model and
    drove speeds to zero). Stability comes from:

    * locking pitch orientation after the first good fit,
    * accepting updates only when reprojection is good and orientation matches,
    * sharing the same gated ``H`` for speed and radar.

    Goal defending teams are voted during warmup and then fixed.
    """

    def __init__(
        self,
        homography: PitchHomography,
        confidence: float = 0.5,
        ransac_thresh: float = HOMOGRAPHY_RANSAC_REPROJ_THRESH,
        max_reproj_px: float = 8.0,
        pool_frames: int = 20,
        min_pool_frames: int = 3,
    ) -> None:
        self.homography = homography
        self.config = homography.config
        self.confidence = confidence
        self.ransac_thresh = ransac_thresh
        self.max_reproj_px = max_reproj_px
        self.goal_warmup_votes = max(min_pool_frames, 5)
        self._targets = np.array(self.config.vertices, dtype=np.float32)
        self._locked: ViewTransformer | None = None
        self._orientation_anchor: ViewTransformer | None = None
        self.locked_goal_defenders: tuple[int, int] | None = None
        self._goal_votes: list[tuple[int, int]] = []

    def _frame_correspondences(
        self, keypoints: sv.KeyPoints
    ) -> tuple[np.ndarray, np.ndarray] | None:
        xy = keypoints.xy[0]
        conf = pitch_keypoint_confidence(keypoints, n_vertices=len(self._targets))
        mask = pitch_keypoint_accept_mask(xy, conf, confidence=self.confidence)
        if mask.sum() < 4:
            return None
        return xy[mask].astype(np.float32), self._targets[mask]

    def _fit_frame(
        self, src: np.ndarray, dst: np.ndarray, *, use_ransac: bool
    ) -> tuple[ViewTransformer, np.ndarray] | None:
        """Pick the best plain or mirrored target for this frame's correspondences."""
        length = float(self.config.length)
        candidates: list[tuple[float, ViewTransformer, np.ndarray]] = []
        for target in (dst, _flip_pitch_x_targets(dst, length)):
            try:
                t = ViewTransformer(
                    source=src,
                    target=target,
                    use_ransac=use_ransac,
                    ransac_thresh=self.ransac_thresh,
                )
            except ValueError:
                continue
            candidates.append((_mean_reproj_px(t, src, target), t, target))
        if not candidates:
            return None

        candidates.sort(key=lambda pair: pair[0])
        for _err, t, target in candidates:
            if self._orientation_anchor is None or _orientation_matches_anchor(
                t, self._orientation_anchor, src
            ):
                return t, target
        _err, t, target = candidates[0]
        return t, target

    def register_goal_vote(self, left_team: int, right_team: int) -> None:
        """Accumulate defending-team votes; call from render when players are visible."""
        if self.locked_goal_defenders is not None:
            return
        self._goal_votes.append((left_team, right_team))
        if len(self._goal_votes) >= self.goal_warmup_votes:
            self._solidify_goal_defenders()

    def _solidify_goal_defenders(self) -> None:
        if self.locked_goal_defenders is not None or not self._goal_votes:
            return
        counts: dict[tuple[int, int], int] = {}
        for pair in self._goal_votes:
            counts[pair] = counts.get(pair, 0) + 1
        self.locked_goal_defenders = max(counts, key=counts.get)

    def finalize_goal_lock(self) -> None:
        """Pick defending teams from warmup votes (call once before rendering)."""
        self._solidify_goal_defenders()

    def register_reliable_goal_vote(
        self, pitch_xy_m: np.ndarray, teams: np.ndarray
    ) -> bool:
        """Vote defending teams only when warped feet look like a real pitch layout."""
        if self.locked_goal_defenders is not None:
            return False
        if not pitch_layout_reliable(pitch_xy_m, teams):
            return False
        left, right = infer_goal_defenders(pitch_xy_m * 100.0, teams)
        self.register_goal_vote(left, right)
        return True

    def update(
        self, frame: np.ndarray, keypoints: sv.KeyPoints | None = None
    ) -> tuple[ViewTransformer | None, ViewTransformer | None]:
        """Return ``(speed_transformer, radar_transformer)`` — same gated per-frame H."""
        if keypoints is None:
            keypoints = detect_pitch_keypoints(frame, self.homography)
        if keypoints.xy.shape[0] == 0:
            return self._locked, self._locked

        pair = self._frame_correspondences(keypoints)
        if pair is None:
            return self._locked, self._locked

        src_now, dst_now = pair
        if len(src_now) < DISPLAY_MIN_KEYPOINTS:
            return self._locked, self._locked
        fitted = self._fit_frame(src_now, dst_now, use_ransac=True)
        if fitted is not None:
            speed_cand, target = fitted
            err = _mean_reproj_px(speed_cand, src_now, target)
            if err <= self.max_reproj_px:
                if self._orientation_anchor is None:
                    self._orientation_anchor = speed_cand
                self._locked = speed_cand

        h = self._locked
        if h is None:
            fallback = self._fit_frame(src_now, dst_now, use_ransac=False)
            if fallback is not None:
                h = fallback[0]
        return h, h


def warmup_goal_defenders(
    pitch_tracker: PitchHomographyTracker | None,
    frames_with_dets,
    frame_transforms: dict[int, ViewTransformer | None],
    *,
    sample_step: int = 8,
) -> tuple[int, int] | None:
    """Sample frames until reliable votes lock left/right defending teams."""
    from world_cup_projects.common.possession import feet_xy, player_mask

    if pitch_tracker is None:
        return None
    for frame_idx, dets in frames_with_dets:
        if int(frame_idx) % sample_step != 0:
            continue
        transformer = frame_transforms.get(int(frame_idx))
        if transformer is None:
            continue
        pmask = player_mask(dets)
        if not pmask.any():
            continue
        pitch_m = image_to_pitch_m(feet_xy(dets)[pmask], transformer)
        if pitch_m is None:
            continue
        teams = dets.data.get("team", np.zeros(len(dets), dtype=int))[pmask]
        pitch_tracker.register_reliable_goal_vote(pitch_m, teams)
        if pitch_tracker.locked_goal_defenders is not None:
            return pitch_tracker.locked_goal_defenders
    pitch_tracker.finalize_goal_lock()
    return pitch_tracker.locked_goal_defenders


def iter_pitch_transformers(
    sequence,
    *,
    device: str = "cpu",
    start: int = 1,
    end: int | None = None,
    ransac_thresh: float = HOMOGRAPHY_RANSAC_REPROJ_THRESH,
    max_reproj_px: float = 8.0,
    confidence: float = 0.5,
    pool_frames: int = 20,
    yield_keypoints: bool = False,
    yield_tracker: bool = False,
):
    """Yield homographies for every frame in a SoccerNet sequence.

    Each item is ``(frame_idx, speed_transformer, radar_transformer)`` from per-frame
    keypoints with orientation-locked, gated updates (shared ``H`` for speed + radar).

    With ``yield_keypoints=True``, append ``keypoints | None``.
    With ``yield_tracker=True``, append the :class:`PitchHomographyTracker` instance.
    """
    import cv2

    homography = load_pitch_model(device=device)
    homography.ransac_thresh = ransac_thresh
    homography.confidence = confidence
    tracker = PitchHomographyTracker(
        homography,
        confidence=confidence,
        ransac_thresh=ransac_thresh,
        max_reproj_px=max_reproj_px,
        pool_frames=pool_frames,
    )
    last = sequence.length if end is None else min(end, sequence.length)

    from world_cup_projects.common.video import read_sequence_frame

    for frame_idx in range(start, last + 1):
        frame = read_sequence_frame(sequence, frame_idx)
        if frame is None:
            if yield_keypoints:
                yield frame_idx, None, None, None
            else:
                yield frame_idx, None, None
            continue
        kps = detect_pitch_keypoints(frame, homography)
        speed_t, radar_t = tracker.update(frame, kps)
        if yield_keypoints and yield_tracker:
            yield frame_idx, speed_t, radar_t, kps, tracker
        elif yield_keypoints:
            yield frame_idx, speed_t, radar_t, kps
        else:
            yield frame_idx, speed_t, radar_t
