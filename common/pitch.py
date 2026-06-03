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
# Reject (or keep previous H) when anchor keypoints jump more than ~half the pitch (cm).
HOMOGRAPHY_ORIENTATION_JUMP_CM = 3500.0


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


def _mirror_homography_matrix(
    matrix: npt.NDArray,
    config: SoccerPitchConfiguration,
    *,
    flip_x: bool,
    flip_y: bool,
) -> npt.NDArray:
    """Reflect pitch coordinates (cm) about touchlines / goallines: ``p' = R @ H @ x``."""
    out = np.array(matrix, dtype=np.float64, copy=True)
    if flip_x:
        rx = np.array(
            [[-1.0, 0.0, float(config.length)], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        out = rx @ out
    if flip_y:
        ry = np.array(
            [[1.0, 0.0, 0.0], [0.0, -1.0, float(config.width)], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        out = ry @ out
    return out


def _canonicalize_homography(
    candidate: ViewTransformer,
    reference: ViewTransformer,
    anchors_image: npt.NDArray,
    config: SoccerPitchConfiguration,
) -> tuple[ViewTransformer, float]:
    """Resolve pitch left/right flips by matching anchor motion to the previous H."""
    ref_pitch = reference.transform_points(anchors_image.astype(np.float32))
    variants = [candidate.m]
    variants.append(_mirror_homography_matrix(candidate.m, config, flip_x=True, flip_y=False))
    variants.append(_mirror_homography_matrix(candidate.m, config, flip_x=False, flip_y=True))
    variants.append(_mirror_homography_matrix(candidate.m, config, flip_x=True, flip_y=True))

    best_m = candidate.m
    best_err = float("inf")
    for m in variants:
        alt = ViewTransformer(matrix=m)
        pitch = alt.transform_points(anchors_image.astype(np.float32))
        err = float(np.median(np.linalg.norm(pitch - ref_pitch, axis=1)))
        if err < best_err:
            best_err = err
            best_m = m
    return ViewTransformer(matrix=best_m), best_err


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

    The pitch-keypoint model is noisy frame-to-frame. We keep a long rolling buffer of
    each vertex's image position, fit H with RANSAC, and **reject** updates whose mean
    reprojection error is worse than the previous H (matrix EMA is avoided — it is not
    a valid homography blend and blows up metric coordinates).
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
        self._last: ViewTransformer | None = None

    def update(
        self, frame: np.ndarray, keypoints: sv.KeyPoints | None = None
    ) -> ViewTransformer | None:
        if keypoints is None:
            keypoints = detect_pitch_keypoints(frame, self.homography)
        kps = keypoints
        if kps.xy.shape[0] == 0:
            return self._last

        xy = kps.xy[0]
        conf = pitch_keypoint_confidence(kps, n_vertices=len(self._targets))
        for i in range(len(self._targets)):
            if i < len(xy) and conf[i] > self.confidence and xy[i, 0] > 1 and xy[i, 1] > 1:
                self._buffers[i].append(xy[i])

        src, dst = [], []
        for i, buf in enumerate(self._buffers):
            if buf:
                src.append(np.mean(buf, axis=0))
                dst.append(self._targets[i])
        if len(src) < 4:
            return self._last

        src_arr = np.asarray(src, np.float32)
        dst_arr = np.asarray(dst, np.float32)
        try:
            candidate = ViewTransformer(
                source=src_arr, target=dst_arr, ransac_thresh=self.ransac_thresh
            )
        except ValueError:
            return self._last

        if self._last is not None:
            candidate, orient_err = _canonicalize_homography(
                candidate, self._last, src_arr, self.config
            )
            if orient_err > HOMOGRAPHY_ORIENTATION_JUMP_CM:
                return self._last
            reproj = candidate.transform_points(src_arr)
            err = float(np.linalg.norm(reproj - dst_arr, axis=1).mean())
            if err > self.max_reproj_px:
                return self._last

        self._last = candidate
        return self._last


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
):
    """Yield ``(frame_idx, transformer)`` for every frame in a SoccerNet sequence.

    With ``yield_keypoints=True``, each item is
    ``(frame_idx, transformer, keypoints | None)`` (one model forward per frame).
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
            if yield_keypoints:
                yield frame_idx, None, None
            else:
                yield frame_idx, None
            continue
        kps = detect_pitch_keypoints(frame, homography)
        transformer = tracker.update(frame, kps)
        if yield_keypoints:
            yield frame_idx, transformer, kps
        else:
            yield frame_idx, transformer
