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

from collections import deque
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


def render_radar_from_transformer(
    detections: sv.Detections,
    transformer: ViewTransformer,
    *,
    config: SoccerPitchConfiguration = PITCH_CONFIG,
) -> np.ndarray | None:
    """Warp player feet with a precomputed homography (e.g. temporally smoothed)."""
    from world_cup_projects.common.possession import player_mask

    from world_cup_projects.common.soccernet import TEAM_LEFT, TEAM_RIGHT

    pmask = player_mask(detections)
    radar = draw_pitch(config=config)

    if pmask.any():
        players = detections[pmask]
        xy = players.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
        transformed_xy = transformer.transform_points(points=xy.astype(np.float32))
        teams = players.data.get("team", np.zeros(len(players), dtype=int))
        left_team, right_team = infer_goal_defenders(transformed_xy, teams)
    else:
        left_team, right_team = TEAM_LEFT, TEAM_RIGHT
        transformed_xy = None
        teams = None

    radar = draw_goals_on_pitch(
        config,
        left_defender_team=left_team,
        right_defender_team=right_team,
        team_colors=_SPORTS_RADAR_COLORS,
        pitch=radar,
    )
    if transformed_xy is None:
        return radar

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
    return radar


def render_radar_sports(
    detections: sv.Detections,
    keypoints: sv.KeyPoints | None,
    *,
    config: SoccerPitchConfiguration = PITCH_CONFIG,
    use_ransac: bool = False,
    ransac_thresh: float = HOMOGRAPHY_RANSAC_REPROJ_THRESH,
) -> np.ndarray | None:
    """Build minimap like ``roboflow/sports`` ``render_radar`` (plain H, all visible KPs)."""
    if keypoints is None or keypoints.xy.shape[0] == 0:
        return None

    mask = (keypoints.xy[0][:, 0] > 1) & (keypoints.xy[0][:, 1] > 1)
    if int(mask.sum()) < 4:
        return None

    transformer = ViewTransformer(
        source=keypoints.xy[0][mask].astype(np.float32),
        target=np.array(config.vertices, dtype=np.float32)[mask],
        use_ransac=use_ransac,
        ransac_thresh=ransac_thresh,
    )

    return render_radar_from_transformer(detections, transformer, config=config)


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
    keypoints: sv.KeyPoints,
    config: SoccerPitchConfiguration = PITCH_CONFIG,
    confidence: float = 0.5,
) -> ViewTransformer | None:
    if keypoints is None or len(keypoints) == 0:
        return None
    xy = keypoints.xy[0]
    conf = pitch_keypoint_confidence(keypoints, n_vertices=len(xy))
    mask = pitch_keypoint_accept_mask(xy, conf, confidence=confidence)
    if mask.sum() < 4:
        return None
    target = np.array(config.vertices, dtype=np.float32)[mask]
    return ViewTransformer(
        source=xy[mask].astype(np.float32),
        target=target,
        ransac_thresh=HOMOGRAPHY_RANSAC_REPROJ_THRESH,
    )


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


class PitchHomographyTracker:
    """Per-frame homography with temporal keypoint smoothing.

    The pitch-keypoint model is noisy frame-to-frame. We keep rolling buffers of each
    vertex's **image** position (valid under camera motion), then fit homographies.

    * **Speed / metrics** — confidence-filtered buffers, RANSAC, reprojection gate.
    * **Radar** — all visible keypoints (sports ``x,y > 1`` mask), same temporal smooth,
      plain ``findHomography`` on every correspondence (no RANSAC subset jumps).
    """

    def __init__(
        self,
        homography: PitchHomography,
        smooth_window: int = 31,
        confidence: float = 0.5,
        ransac_thresh: float = HOMOGRAPHY_RANSAC_REPROJ_THRESH,
        max_reproj_px: float = 8.0,
    ) -> None:
        self.homography = homography
        self.config = homography.config
        self.smooth_window = smooth_window
        self.confidence = confidence
        self.ransac_thresh = ransac_thresh
        self.max_reproj_px = max_reproj_px
        self._targets = np.array(self.config.vertices, dtype=np.float32)
        n = len(self._targets)
        self._buffers: list[deque] = [deque(maxlen=smooth_window) for _ in range(n)]
        self._radar_buffers: list[deque] = [deque(maxlen=smooth_window) for _ in range(n)]
        self._last: ViewTransformer | None = None
        self._radar_last: ViewTransformer | None = None

    def _append_observations(self, xy: np.ndarray, conf: np.ndarray) -> None:
        for i in range(len(self._targets)):
            if i >= len(xy) or xy[i, 0] <= 1 or xy[i, 1] <= 1:
                continue
            self._radar_buffers[i].append(xy[i])
            if conf[i] > self.confidence:
                self._buffers[i].append(xy[i])

    @staticmethod
    def _correspondences(
        buffers: list[deque], targets: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray] | None:
        src, dst = [], []
        for i, buf in enumerate(buffers):
            if buf:
                src.append(np.mean(buf, axis=0))
                dst.append(targets[i])
        if len(src) < 4:
            return None
        return np.asarray(src, np.float32), np.asarray(dst, np.float32)

    def update(
        self, frame: np.ndarray, keypoints: sv.KeyPoints | None = None
    ) -> tuple[ViewTransformer | None, ViewTransformer | None]:
        """Return ``(speed_transformer, radar_transformer)`` for this frame."""
        if keypoints is None:
            keypoints = detect_pitch_keypoints(frame, self.homography)
        kps = keypoints
        if kps.xy.shape[0] == 0:
            return self._last, self._radar_last

        xy = kps.xy[0]
        conf = pitch_keypoint_confidence(kps, n_vertices=len(self._targets))
        self._append_observations(xy, conf)

        speed_pair = self._correspondences(self._buffers, self._targets)
        if speed_pair is not None:
            src_arr, dst_arr = speed_pair
            try:
                candidate = ViewTransformer(
                    source=src_arr,
                    target=dst_arr,
                    use_ransac=True,
                    ransac_thresh=self.ransac_thresh,
                )
            except ValueError:
                pass
            else:
                if self._last is None:
                    self._last = candidate
                else:
                    reproj = candidate.transform_points(src_arr)
                    err = float(np.linalg.norm(reproj - dst_arr, axis=1).mean())
                    if err <= self.max_reproj_px:
                        self._last = candidate

        radar_pair = self._correspondences(self._radar_buffers, self._targets)
        if radar_pair is not None:
            src_arr, dst_arr = radar_pair
            try:
                self._radar_last = ViewTransformer(
                    source=src_arr, target=dst_arr, use_ransac=False
                )
            except ValueError:
                pass

        return self._last, self._radar_last

    def smoothed_xy(self, *, for_radar: bool = True) -> np.ndarray:
        """Per-vertex rolling-mean positions in image space; NaN where buffer is empty."""
        buffers = self._radar_buffers if for_radar else self._buffers
        out = np.full((len(self._targets), 2), np.nan, dtype=np.float32)
        for i, buf in enumerate(buffers):
            if buf:
                out[i] = np.mean(buf, axis=0)
        return out


def iter_pitch_transformers(
    sequence,
    *,
    device: str = "cpu",
    start: int = 1,
    end: int | None = None,
    smooth_window: int = 31,
    ransac_thresh: float = HOMOGRAPHY_RANSAC_REPROJ_THRESH,
    max_reproj_px: float = 8.0,
    confidence: float = 0.5,
    yield_keypoints: bool = False,
    yield_smoothed_keypoints: bool = False,
):
    """Yield homographies for every frame in a SoccerNet sequence.

    Each item is ``(frame_idx, speed_transformer, radar_transformer)``. Radar uses
    temporally smoothed keypoints and plain homography (no RANSAC). Speed keeps RANSAC
    and a reprojection gate for metric coordinates.

    With ``yield_keypoints=True``, each item ends with ``raw_keypoints | None`` and, if
    ``yield_smoothed_keypoints=True``, ``radar_smooth_xy`` and ``speed_smooth_xy`` arrays
    (shape ``(n_vertices, 2)``, NaN where not buffered). Smoothed points are what H is
    fit from — not display-only.
    """
    import cv2

    homography = load_pitch_model(device=device)
    homography.ransac_thresh = ransac_thresh
    homography.confidence = confidence
    tracker = PitchHomographyTracker(
        homography,
        smooth_window=smooth_window,
        confidence=confidence,
        ransac_thresh=ransac_thresh,
        max_reproj_px=max_reproj_px,
    )
    last = sequence.length if end is None else min(end, sequence.length)

    for frame_idx in range(start, last + 1):
        frame = cv2.imread(str(sequence.frame_path(frame_idx)))
        if frame is None:
            if yield_keypoints and yield_smoothed_keypoints:
                yield frame_idx, None, None, None, None, None
            elif yield_keypoints:
                yield frame_idx, None, None, None
            else:
                yield frame_idx, None, None
            continue
        kps = detect_pitch_keypoints(frame, homography)
        speed_t, radar_t = tracker.update(frame, kps)
        if yield_keypoints and yield_smoothed_keypoints:
            yield (
                frame_idx,
                speed_t,
                radar_t,
                kps,
                tracker.smoothed_xy(for_radar=True),
                tracker.smoothed_xy(for_radar=False),
            )
        elif yield_keypoints:
            yield frame_idx, speed_t, radar_t, kps
        else:
            yield frame_idx, speed_t, radar_t
