"""Model-based detection + tracking for non-SoccerNet video (v2 path).

Sources:
- ``football``: YOLO ``football-players-detection`` (DFL-trained; ball/player/gk/referee).
- ``rfdetr``: generic COCO RF-DETR (person + sports ball only; poor team/role split).

Both yield the same ``sv.Detections`` contract as :func:`common.soccernet.iter_gt_detections`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator

import cv2
import numpy as np
import supervision as sv

from trackers import ByteTrackTracker

from world_cup_projects.common.player_tracker import TrackerKind, create_player_tracker
from world_cup_projects.common.tracking_facing import kalman_velocity_arrays
from world_cup_projects.common.soccernet import (
    ROLE_BALL,
    ROLE_GOALKEEPER,
    ROLE_PLAYER,
    ROLE_REFEREE,
    TEAM_NONE,
    SoccerNetSequence,
)
from world_cup_projects.common.teams import (
    JerseyColorTeamClassifier,
    TrackletTeamStabilizer,
    get_crops,
)
from world_cup_projects.common.video import read_sequence_frame

# Class ids in football-players-detection / roboflow/sports football-player-detection.pt
_FP_BALL = 0
_FP_GOALKEEPER = 1
_FP_PLAYER = 2
_FP_REFEREE = 3

_FP_TO_ROLE = {
    _FP_BALL: ROLE_BALL,
    _FP_GOALKEEPER: ROLE_GOALKEEPER,
    _FP_PLAYER: ROLE_PLAYER,
    _FP_REFEREE: ROLE_REFEREE,
}

_MODEL_DIR = Path(__file__).resolve().parent.parent / ".cache" / "models"
_FOOTBALL_PLAYERS_MODEL_PATH = _MODEL_DIR / "football-player-detection.pt"
_FOOTBALL_PLAYERS_MODEL_GDRIVE_ID = "17PXFNlx-jI7VjVo_vQnB1sONjRyvoB-q"
_FOOTBALL_BALL_MODEL_PATH = _MODEL_DIR / "football-ball-detection.pt"
_FOOTBALL_BALL_MODEL_GDRIVE_ID = "1isw4wx-MK9h9LMr36VvIWlJD6ppUvw7V"
DEFAULT_FOOTBALL_PLAYERS_MODEL_ID = "football-players-detection-3zvbc/11"
FOOTBALL_PLAYERS_INFERENCE_V11 = "football-players-detection-3zvbc/11"
# Latest YOLO on Universe (yolo11m, Aug 2025) — use via --detector-backend inference.
FOOTBALL_PLAYERS_INFERENCE_V19 = "football-players-detection-3zvbc/19"
FOOTBALL_PLAYERS_INFERENCE_V20 = "football-players-detection-3zvbc/20"
FOOTBALL_PLAYERS_INFERENCE_RFDETR = "football-players-detection-3zvbc/18"
BEST_FOOTBALL_PLAYERS_YOLO_MODEL_ID = FOOTBALL_PLAYERS_INFERENCE_V19
DEFAULT_BALL_DETECTION_THRESHOLD = 0.20
# Dedicated ball-only YOLOv8x (DFL / Bundesliga ball dataset).
DEFAULT_FOOTBALL_BALL_MODEL_ID = "football-ball-detection-rejhg/4"
KNOWN_FOOTBALL_PLAYER_MODELS = (
    FOOTBALL_PLAYERS_INFERENCE_V11,
    FOOTBALL_PLAYERS_INFERENCE_V19,
    FOOTBALL_PLAYERS_INFERENCE_V20,
    FOOTBALL_PLAYERS_INFERENCE_RFDETR,
)

_FP_CLASS_NAME_TO_ROLE = {
    "ball": ROLE_BALL,
    "goalkeeper": ROLE_GOALKEEPER,
    "player": ROLE_PLAYER,
    "referee": ROLE_REFEREE,
}


class RFDETRDetector:
    """Thin wrapper around an RF-DETR checkpoint returning soccer-relevant detections."""

    def __init__(self, model_name: str = "nano", device: str = "cpu", threshold: float = 0.4):
        import rfdetr
        from rfdetr.assets.coco_classes import COCO_CLASSES

        model_cls = {
            "nano": rfdetr.RFDETRNano,
            "small": rfdetr.RFDETRSmall,
            "medium": rfdetr.RFDETRMedium,
            "base": rfdetr.RFDETRBase,
        }[model_name]
        self.model = model_cls(device=device)
        self.threshold = threshold
        self._classes = COCO_CLASSES

    def detect(self, frame_bgr: np.ndarray) -> sv.Detections:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        det = self.model.predict(rgb, threshold=self.threshold)
        names = np.array([self._classes[c] for c in det.class_id])
        role = np.full(len(det), -1, dtype=int)
        role[names == "person"] = ROLE_PLAYER
        role[names == "sports ball"] = ROLE_BALL
        keep = role >= 0
        det = det[keep]
        det.class_id = role[keep]
        return det


class ColorTeamClassifier:
    """2-cluster KMeans on mean torso color - a fast stand-in for TeamClassifier."""

    def __init__(self) -> None:
        from sklearn.cluster import KMeans

        self.kmeans = KMeans(n_clusters=2, n_init=10)
        self._fitted = False

    @staticmethod
    def _torso_feature(frame: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
        x1, y1, x2, y2 = xyxy.astype(int)
        # upper-middle third of the box = jersey, avoiding head/legs/grass
        h = y2 - y1
        ty1, ty2 = y1 + int(0.15 * h), y1 + int(0.5 * h)
        crop = frame[max(ty1, 0):max(ty2, 1), max(x1, 0):max(x2, 1)]
        if crop.size == 0:
            return np.zeros(3, np.float32)
        return cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).reshape(-1, 3).mean(axis=0)

    def fit(self, features: np.ndarray) -> None:
        if len(features) >= 2:
            self.kmeans.fit(features)
            self._fitted = True

    def predict(self, frame: np.ndarray, players: sv.Detections) -> np.ndarray:
        if not self._fitted or len(players) == 0:
            return np.full(len(players), TEAM_NONE, dtype=int)
        feats = np.stack([self._torso_feature(frame, b) for b in players.xyxy])
        return self.kmeans.predict(feats).astype(int)


def fit_team_classifier(
    sequence: SoccerNetSequence,
    detector: RFDETRDetector,
    *,
    sample_stride: int = 30,
    max_frames: int = 750,
) -> ColorTeamClassifier:
    """Collect torso colors across sampled frames and fit the team classifier."""
    clf = ColorTeamClassifier()
    feats: list[np.ndarray] = []
    last = min(max_frames, sequence.length)
    for frame_idx in range(1, last + 1, sample_stride):
        image = read_sequence_frame(sequence, frame_idx)
        if image is None:
            continue
        det = detector.detect(image)
        players = det[det.class_id == ROLE_PLAYER]
        for box in players.xyxy:
            feats.append(ColorTeamClassifier._torso_feature(image, box))
    if feats:
        clf.fit(np.stack(feats))
    return clf


def iter_rfdetr_detections(
    sequence: SoccerNetSequence,
    detector: RFDETRDetector,
    team_classifier: ColorTeamClassifier,
    *,
    start: int = 1,
    end: int | None = None,
    track_activation_threshold: float = 0.4,
) -> Iterator[tuple[int, sv.Detections]]:
    """Detect + track players across the sequence, yielding GT-compatible detections."""
    tracker = ByteTrackTracker(
        frame_rate=sequence.frame_rate,
        track_activation_threshold=track_activation_threshold,
    )
    last = sequence.length if end is None else min(end, sequence.length)

    for frame_idx in range(start, last + 1):
        image = read_sequence_frame(sequence, frame_idx)
        if image is None:
            yield frame_idx, sv.Detections.empty()
            continue

        det = detector.detect(image)
        players = det[det.class_id == ROLE_PLAYER]
        balls = det[det.class_id == ROLE_BALL]

        tracked = tracker.update(players, frame=image)
        teams = team_classifier.predict(image, tracked)

        # Rebuild clean detections (single, consistent `data` schema) so the optional
        # ball row can be merged without key mismatches.
        n = len(tracked)
        xyxy = tracked.xyxy
        tracker_id = tracked.tracker_id if tracked.tracker_id is not None else np.full(n, -1)
        class_id = np.full(n, ROLE_PLAYER, dtype=int)
        team = teams

        if len(balls):
            box = balls.xyxy[:1]
            xyxy = np.concatenate([xyxy, box], axis=0)
            tracker_id = np.concatenate([tracker_id, [-1]])
            class_id = np.concatenate([class_id, [ROLE_BALL]])
            team = np.concatenate([team, [TEAM_NONE]])

        yield frame_idx, sv.Detections(
            xyxy=xyxy.astype(np.float32),
            tracker_id=tracker_id.astype(int),
            class_id=class_id.astype(int),
            data={
                "team": team.astype(int),
                "jersey": np.asarray([""] * len(xyxy), dtype=object),
            },
        )


def iter_model_detections(
    sequence: SoccerNetSequence,
    *,
    start: int = 1,
    end: int | None = None,
    model_name: str = "nano",
    device: str = "cpu",
    threshold: float = 0.4,
    sample_stride: int = 30,
) -> Iterator[tuple[int, sv.Detections]]:
    """Convenience wrapper: build detector + team classifier, then stream frames."""
    detector = RFDETRDetector(model_name=model_name, device=device, threshold=threshold)
    team_classifier = fit_team_classifier(
        sequence, detector, sample_stride=sample_stride, max_frames=end or sequence.length
    )
    yield from iter_rfdetr_detections(
        sequence, detector, team_classifier, start=start, end=end
    )


def ensure_football_players_model() -> Path:
    """Download football-player-detection.pt (DFL / football-players-detection-3zvbc)."""
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if _FOOTBALL_PLAYERS_MODEL_PATH.is_file():
        return _FOOTBALL_PLAYERS_MODEL_PATH

    try:
        import gdown
    except ImportError as exc:
        raise ImportError("Install gdown to download football player model weights.") from exc

    url = f"https://drive.google.com/uc?id={_FOOTBALL_PLAYERS_MODEL_GDRIVE_ID}"
    gdown.download(url, str(_FOOTBALL_PLAYERS_MODEL_PATH), quiet=False)
    return _FOOTBALL_PLAYERS_MODEL_PATH


def ensure_football_ball_model() -> Path:
    """Download football-ball-detection.pt (Universe football-ball-detection-rejhg)."""
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if _FOOTBALL_BALL_MODEL_PATH.is_file():
        return _FOOTBALL_BALL_MODEL_PATH

    try:
        import gdown
    except ImportError as exc:
        raise ImportError("Install gdown to download football ball model weights.") from exc

    url = f"https://drive.google.com/uc?id={_FOOTBALL_BALL_MODEL_GDRIVE_ID}"
    gdown.download(url, str(_FOOTBALL_BALL_MODEL_PATH), quiet=False)
    return _FOOTBALL_BALL_MODEL_PATH


def ensure_football_players_inference(*, model_id: str = DEFAULT_FOOTBALL_PLAYERS_MODEL_ID) -> None:
    """Verify ``ROBOFLOW_API_KEY`` is set (Inference caches weights on first run)."""
    import os

    if not os.environ.get("ROBOFLOW_API_KEY"):
        raise RuntimeError(
            f"Set ROBOFLOW_API_KEY for player detection inference ({model_id})."
        )


def _map_fp_detections(det: sv.Detections) -> sv.Detections:
    if len(det) == 0:
        return det
    roles = np.array([_FP_TO_ROLE.get(int(c), -1) for c in det.class_id], dtype=int)
    keep = roles >= 0
    det = det[keep]
    det.class_id = roles[keep]
    return det


def _apply_fp_class_thresholds(
    det: sv.Detections,
    *,
    player_threshold: float,
    ball_threshold: float,
) -> sv.Detections:
    """Filter multi-class football detections with a lower threshold for the small ball class."""
    if len(det) == 0:
        return det
    keep = np.zeros(len(det), dtype=bool)
    for role, thr in (
        (ROLE_PLAYER, player_threshold),
        (ROLE_GOALKEEPER, player_threshold),
        (ROLE_REFEREE, player_threshold),
        (ROLE_BALL, ball_threshold),
    ):
        keep |= (det.class_id == role) & (
            det.confidence >= thr if det.confidence is not None else True
        )
    return det[keep]


REFEREE_PLAYER_IOU_THRESHOLD = 0.25
REFEREE_TRACK_IOU_THRESHOLD = 0.4


def _detection_overlap_with_referees(
    subject: sv.Detections,
    referees: sv.Detections,
    *,
    iou_threshold: float = REFEREE_PLAYER_IOU_THRESHOLD,
) -> np.ndarray:
    """Per-row mask: subject detection overlaps any referee box."""
    if len(subject) == 0 or len(referees) == 0:
        return np.zeros(len(subject), dtype=bool)
    ious = sv.box_iou_batch(subject.xyxy, referees.xyxy)
    overlap = ious.max(axis=1) >= iou_threshold
    if overlap.all():
        return overlap
    rcx = (referees.xyxy[:, 0] + referees.xyxy[:, 2]) / 2.0
    rcy = (referees.xyxy[:, 1] + referees.xyxy[:, 3]) / 2.0
    x1, y1, x2, y2 = (
        subject.xyxy[:, 0],
        subject.xyxy[:, 1],
        subject.xyxy[:, 2],
        subject.xyxy[:, 3],
    )
    for i in np.flatnonzero(~overlap):
        inside = (rcx >= x1[i]) & (rcx <= x2[i]) & (rcy >= y1[i]) & (rcy <= y2[i])
        if inside.any():
            overlap[i] = True
    return overlap


def suppress_players_overlapping_referees(
    players: sv.Detections,
    referees: sv.Detections,
    *,
    iou_threshold: float = REFEREE_PLAYER_IOU_THRESHOLD,
) -> sv.Detections:
    """Drop player rows that duplicate a referee detection (same person, two classes)."""
    if len(players) == 0 or len(referees) == 0:
        return players
    drop = _detection_overlap_with_referees(players, referees, iou_threshold=iou_threshold)
    return players[~drop]


def filter_referees_from_detections(
    dets: sv.Detections,
    *,
    blocked_tracker_ids: set[int] | frozenset[int] | None = None,
    iou_threshold: float = REFEREE_PLAYER_IOU_THRESHOLD,
) -> sv.Detections:
    """Remove referee rows and outfield players overlapping or flagged as refs."""
    if len(dets) == 0:
        return dets
    ref_mask = dets.class_id == ROLE_REFEREE
    keep = ~ref_mask
    player_mask = dets.class_id == ROLE_PLAYER
    refs = dets[ref_mask]
    if len(refs) and player_mask.any():
        overlap = _detection_overlap_with_referees(
            dets[player_mask], refs, iou_threshold=iou_threshold
        )
        player_idx = np.flatnonzero(player_mask)
        keep[player_idx[overlap]] = False
    if blocked_tracker_ids and dets.tracker_id is not None:
        for i, tid in enumerate(dets.tracker_id):
            if int(tid) in blocked_tracker_ids:
                keep[i] = False
    return dets[keep]


def collect_referee_tracker_ids(
    frames: Iterable[tuple[int, sv.Detections]],
    *,
    iou_threshold: float = REFEREE_TRACK_IOU_THRESHOLD,
) -> frozenset[int]:
    """Track ids that ever strongly overlap a referee box (misclassified as player)."""
    flagged: set[int] = set()
    for _, dets in frames:
        player_mask = dets.class_id == ROLE_PLAYER
        ref_mask = dets.class_id == ROLE_REFEREE
        if not player_mask.any() or not ref_mask.any() or dets.tracker_id is None:
            continue
        ious = sv.box_iou_batch(dets.xyxy[player_mask], dets.xyxy[ref_mask])
        player_idx = np.flatnonzero(player_mask)
        tids = dets.tracker_id[player_idx]
        for j, mx in enumerate(ious.max(axis=1)):
            if float(mx) >= iou_threshold and int(tids[j]) >= 0:
                flagged.add(int(tids[j]))
    return frozenset(flagged)


def _best_ball_detection(
    balls: sv.Detections,
    *,
    players: sv.Detections | None = None,
) -> sv.Detections:
    """When multiple balls are detected, prefer the one nearest the pitch players."""
    if len(balls) <= 1:
        return balls
    if players is not None and len(players):
        from world_cup_projects.common.possession import feet_xy

        feet = feet_xy(players)
        if len(feet):
            cx = (balls.xyxy[:, 0] + balls.xyxy[:, 2]) / 2
            cy = balls.xyxy[:, 3]
            centers = np.column_stack([cx, cy])
            dists = np.linalg.norm(
                centers[:, None, :] - feet[None, :, :], axis=2
            ).min(axis=1)
            pick = int(np.argmin(dists))
            return balls[pick : pick + 1]
    if balls.confidence is None:
        return balls[:1]
    pick = int(np.argmax(balls.confidence))
    return balls[pick : pick + 1]


def _merge_ball_detections(
    primary: sv.Detections,
    secondary: sv.Detections,
    *,
    players: sv.Detections | None = None,
) -> sv.Detections:
    """Merge two ball detectors, then disambiguate duplicates near the action."""
    if len(primary) == 0:
        return _best_ball_detection(secondary, players=players)
    if len(secondary) == 0:
        return _best_ball_detection(primary, players=players)
    combined = sv.Detections.merge([primary, secondary])
    return _best_ball_detection(combined, players=players)


def _map_inference_fp_detections(inference_result, *, confidence: float) -> sv.Detections:
    """Convert Roboflow Inference object-detection output to role-tagged detections."""
    if hasattr(inference_result, "model_dump"):
        payload = inference_result.model_dump(by_alias=True, exclude_none=True)
    elif hasattr(inference_result, "dict"):
        payload = inference_result.dict(exclude_none=True, by_alias=True)
    else:
        payload = inference_result

    predictions = payload.get("predictions") or []
    if not predictions:
        return sv.Detections.empty()

    xyxy_list: list[list[float]] = []
    class_ids: list[int] = []
    confidences: list[float] = []

    for pred in predictions:
        conf = float(pred.get("confidence", 0.0))
        if conf < confidence:
            continue
        w = float(pred["width"])
        h = float(pred["height"])
        cx = float(pred["x"])
        cy = float(pred["y"])
        x1, y1 = cx - w / 2, cy - h / 2
        x2, y2 = cx + w / 2, cy + h / 2

        role = -1
        if "class_id" in pred:
            role = _FP_TO_ROLE.get(int(pred["class_id"]), -1)
        if role < 0:
            name = str(pred.get("class", "")).strip().lower()
            role = _FP_CLASS_NAME_TO_ROLE.get(name, -1)
        if role < 0:
            continue

        xyxy_list.append([x1, y1, x2, y2])
        class_ids.append(role)
        confidences.append(conf)

    if not xyxy_list:
        return sv.Detections.empty()

    return sv.Detections(
        xyxy=np.asarray(xyxy_list, dtype=np.float32),
        class_id=np.asarray(class_ids, dtype=int),
        confidence=np.asarray(confidences, dtype=np.float32),
    )


class FootballPlayersInferenceDetector:
    """football-players-detection via Roboflow Inference (same API as pitch keypoints)."""

    def __init__(
        self,
        *,
        model_id: str = DEFAULT_FOOTBALL_PLAYERS_MODEL_ID,
        api_key: str | None = None,
        threshold: float = 0.5,
        ball_threshold: float = DEFAULT_BALL_DETECTION_THRESHOLD,
        device: str = "cpu",
    ) -> None:
        import os

        from inference import get_model

        ensure_football_players_inference(model_id=model_id)
        key = api_key or os.environ.get("ROBOFLOW_API_KEY")
        self.model_id = model_id
        self.model = get_model(model_id=model_id, api_key=key)
        self.threshold = threshold
        self.ball_threshold = ball_threshold
        self.device = device  # unused; kept for call-site parity with YOLO path

    def detect(self, frame_bgr: np.ndarray) -> sv.Detections:
        min_conf = min(self.threshold, self.ball_threshold, 0.05)
        result = self.model.infer(frame_bgr, confidence=min_conf)[0]
        det = _map_inference_fp_detections(result, confidence=min_conf)
        return _apply_fp_class_thresholds(
            det,
            player_threshold=self.threshold,
            ball_threshold=self.ball_threshold,
        )


class FootballBallInferenceDetector:
    """Dedicated ball-only model (Roboflow football-ball-detection)."""

    def __init__(
        self,
        *,
        model_id: str = DEFAULT_FOOTBALL_BALL_MODEL_ID,
        api_key: str | None = None,
        threshold: float = DEFAULT_BALL_DETECTION_THRESHOLD,
    ) -> None:
        import os

        from inference import get_model

        ensure_football_players_inference(model_id=model_id)
        key = api_key or os.environ.get("ROBOFLOW_API_KEY")
        self.model_id = model_id
        self.model = get_model(model_id=model_id, api_key=key)
        self.threshold = threshold

    def detect(self, frame_bgr: np.ndarray) -> sv.Detections:
        min_conf = min(self.threshold, 0.05)
        result = self.model.infer(frame_bgr, confidence=min_conf)[0]
        if hasattr(result, "model_dump"):
            payload = result.model_dump(by_alias=True, exclude_none=True)
        elif hasattr(result, "dict"):
            payload = result.dict(exclude_none=True, by_alias=True)
        else:
            payload = result

        predictions = payload.get("predictions") or []
        xyxy_list: list[list[float]] = []
        confidences: list[float] = []
        for pred in predictions:
            conf = float(pred.get("confidence", 0.0))
            if conf < self.threshold:
                continue
            w = float(pred["width"])
            h = float(pred["height"])
            cx = float(pred["x"])
            cy = float(pred["y"])
            xyxy_list.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
            confidences.append(conf)

        if not xyxy_list:
            return sv.Detections.empty()

        return sv.Detections(
            xyxy=np.asarray(xyxy_list, dtype=np.float32),
            class_id=np.full(len(xyxy_list), ROLE_BALL, dtype=int),
            confidence=np.asarray(confidences, dtype=np.float32),
        )


class FootballBallYoloDetector:
    """Dedicated ball-only YOLO (football-ball-detection-rejhg weights)."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        device: str = "cpu",
        threshold: float = DEFAULT_BALL_DETECTION_THRESHOLD,
    ) -> None:
        from ultralytics import YOLO

        path = Path(model_path) if model_path else ensure_football_ball_model()
        self.model = YOLO(str(path))
        self.device = device
        self.threshold = threshold

    def detect(self, frame_bgr: np.ndarray) -> sv.Detections:
        min_conf = min(self.threshold, 0.05)
        results = self.model.predict(
            frame_bgr, conf=min_conf, verbose=False, device=self.device
        )[0]
        det = sv.Detections.from_ultralytics(results)
        if len(det) == 0:
            return sv.Detections.empty()
        if det.confidence is not None:
            det = det[det.confidence >= self.threshold]
        if len(det) == 0:
            return sv.Detections.empty()
        det.class_id = np.full(len(det), ROLE_BALL, dtype=int)
        return det


def create_football_players_detector(
    *,
    backend: str = "yolo",
    model_id: str = DEFAULT_FOOTBALL_PLAYERS_MODEL_ID,
    model_path: str | Path | None = None,
    device: str = "cpu",
    threshold: float = 0.5,
    ball_threshold: float = DEFAULT_BALL_DETECTION_THRESHOLD,
    api_key: str | None = None,
) -> FootballPlayersDetector | FootballPlayersInferenceDetector:
    """Build YOLO (local .pt) or Inference (Universe model id) football player detector."""
    if backend == "inference":
        return FootballPlayersInferenceDetector(
            model_id=model_id,
            api_key=api_key,
            threshold=threshold,
            ball_threshold=ball_threshold,
            device=device,
        )
    if backend == "yolo":
        return FootballPlayersDetector(
            model_path=model_path,
            device=device,
            threshold=threshold,
            ball_threshold=ball_threshold,
        )
    raise ValueError(f"Unknown detector backend: {backend!r} (use 'yolo' or 'inference')")


def create_football_ball_detector(
    *,
    backend: str = "inference",
    model_id: str = DEFAULT_FOOTBALL_BALL_MODEL_ID,
    threshold: float = DEFAULT_BALL_DETECTION_THRESHOLD,
    api_key: str | None = None,
    device: str = "cpu",
) -> FootballBallInferenceDetector | FootballBallYoloDetector | None:
    if backend in (None, "", "none", "off"):
        return None
    if backend == "inference":
        return FootballBallInferenceDetector(
            model_id=model_id,
            api_key=api_key,
            threshold=threshold,
        )
    if backend == "yolo":
        return FootballBallYoloDetector(device=device, threshold=threshold)
    raise ValueError(f"Unknown ball detector backend: {backend!r} (use 'inference', 'yolo', or 'none')")


def wrap_football_detections_cache(args, *, refresh: bool | None = None):
    """Cached football detections with backend + model id in the cache key."""
    from world_cup_projects.common.detection_cache import wrap_detections_cache

    backend = getattr(args, "detector_backend", "yolo")
    model_id = getattr(args, "player_model_id", DEFAULT_FOOTBALL_PLAYERS_MODEL_ID)
    # YOLO keeps legacy ``football`` cache keys; Inference gets its own namespace + model id.
    source_name = "football" if backend == "yolo" else f"football_{backend}"
    cache_params: dict = {
        "device": getattr(args, "device", "cpu"),
        "threshold": getattr(args, "detection_threshold", 0.5),
        "tracker": getattr(args, "tracker", "botsort"),
    }
    if not getattr(args, "legacy_detections_cache", False):
        cache_params["ball_threshold"] = getattr(
            args, "ball_threshold", DEFAULT_BALL_DETECTION_THRESHOLD
        )
    ball_backend = getattr(args, "ball_detector_backend", None)
    if (
        not getattr(args, "legacy_detections_cache", False)
        and ball_backend
        and ball_backend not in ("none", "off", "")
    ):
        cache_params["ball_backend"] = ball_backend
        cache_params["ball_model_id"] = getattr(
            args, "ball_model_id", DEFAULT_FOOTBALL_BALL_MODEL_ID
        )
    if backend == "inference":
        cache_params["backend"] = backend
        cache_params["model_ver"] = model_id.rsplit("/", 1)[-1]

    def _iter_football_cached(sequence, **kwargs):
        params = dict(kwargs)
        params.pop("model_ver", None)
        params["backend"] = backend
        if backend == "inference":
            params["model_id"] = model_id
        return iter_football_model_detections(sequence, **params)

    return wrap_detections_cache(
        _iter_football_cached,
        source_name=source_name,
        refresh=refresh if refresh is not None else getattr(args, "refresh_detections_cache", False),
        **cache_params,
    )


class FootballPlayersDetector:
    """YOLO weights from roboflow/sports (football-players-detection-3zvbc)."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        device: str = "cpu",
        threshold: float = 0.5,
        ball_threshold: float = DEFAULT_BALL_DETECTION_THRESHOLD,
    ) -> None:
        from ultralytics import YOLO

        path = Path(model_path) if model_path else ensure_football_players_model()
        self.model = YOLO(str(path))
        self.device = device
        self.threshold = threshold
        self.ball_threshold = ball_threshold

    def detect(self, frame_bgr: np.ndarray) -> sv.Detections:
        min_conf = min(self.threshold, self.ball_threshold, 0.05)
        results = self.model.predict(
            frame_bgr,
            conf=min_conf,
            verbose=False,
            device=self.device,
        )[0]
        det = _map_fp_detections(sv.Detections.from_ultralytics(results))
        return _apply_fp_class_thresholds(
            det,
            player_threshold=self.threshold,
            ball_threshold=self.ball_threshold,
        )


def fit_football_team_classifier(
    sequence: SoccerNetSequence,
    detector: FootballPlayersDetector,
    *,
    sample_stride: int = 30,
    max_frames: int = 750,
) -> JerseyColorTeamClassifier:
    """Fit jersey-color clusters on outfield players only (excludes gk/referee)."""
    clf = JerseyColorTeamClassifier()
    crops: list[np.ndarray] = []
    last = min(max_frames, sequence.length)
    for frame_idx in range(1, last + 1, sample_stride):
        image = read_sequence_frame(sequence, frame_idx)
        if image is None:
            continue
        det = detector.detect(image)
        players = det[det.class_id == ROLE_PLAYER]
        crops.extend(get_crops(image, players))
    if crops:
        clf.fit(crops)
    return clf


def _empty_detections_data(n: int) -> dict:
    return {
        "team": np.full(n, TEAM_NONE, dtype=int),
        "jersey": np.asarray([""] * n, dtype=object),
        "kf_vx": np.full(n, np.nan, dtype=np.float32),
        "kf_vy": np.full(n, np.nan, dtype=np.float32),
    }


def iter_football_detections(
    sequence: SoccerNetSequence,
    detector: FootballPlayersDetector | FootballPlayersInferenceDetector,
    team_classifier: JerseyColorTeamClassifier,
    *,
    start: int = 1,
    end: int | None = None,
    track_activation_threshold: float = 0.4,
    tracker: TrackerKind = "bytetrack",
    ball_detector: FootballBallInferenceDetector | FootballBallYoloDetector | None = None,
) -> Iterator[tuple[int, sv.Detections]]:
    """Detect + track with football-players-detection; refs excluded from teams."""
    player_tracker = create_player_tracker(
        sequence.frame_rate,
        kind=tracker,
        track_activation_threshold=track_activation_threshold,
    )
    needs_frame = tracker == "botsort"
    last = sequence.length if end is None else min(end, sequence.length)
    team_stabilizer = TrackletTeamStabilizer()

    for frame_idx in range(start, last + 1):
        image = read_sequence_frame(sequence, frame_idx)
        if image is None:
            yield frame_idx, sv.Detections.empty()
            continue

        det = detector.detect(image)
        players = det[det.class_id == ROLE_PLAYER]
        gks = det[det.class_id == ROLE_GOALKEEPER]
        refs = det[det.class_id == ROLE_REFEREE]
        on_pitch = sv.Detections.merge([players, gks]) if len(players) or len(gks) else None
        balls = det[det.class_id == ROLE_BALL]
        if ball_detector is not None:
            extra_balls = ball_detector.detect(image)
            balls = _merge_ball_detections(balls, extra_balls, players=on_pitch)
        balls = _best_ball_detection(balls, players=on_pitch)
        players = suppress_players_overlapping_referees(players, refs)

        trackable = sv.Detections.merge([players, gks])
        tracked = (
            player_tracker.update(
                trackable,
                frame=image if needs_frame else None,
            )
            if len(trackable)
            else sv.Detections.empty()
        )

        parts: list[sv.Detections] = []
        if len(tracked):
            t_players = tracked[tracked.class_id == ROLE_PLAYER]
            t_gks = tracked[tracked.class_id == ROLE_GOALKEEPER]
            if len(t_players):
                player_teams = team_classifier.predict(get_crops(image, t_players))
            else:
                player_teams = np.array([], dtype=int)
            # Goalkeepers keep neutral styling (distinct jersey confuses team clustering).
            if len(t_gks):
                gk_teams = np.full(len(t_gks), TEAM_NONE, dtype=int)
            else:
                gk_teams = np.array([], dtype=int)

            for subset, teams in ((t_players, player_teams), (t_gks, gk_teams)):
                if len(subset) == 0:
                    continue
                tid = subset.tracker_id if subset.tracker_id is not None else np.full(len(subset), -1)
                kf_vx, kf_vy = kalman_velocity_arrays(subset, player_tracker)
                parts.append(
                    sv.Detections(
                        xyxy=subset.xyxy.astype(np.float32),
                        tracker_id=tid.astype(int),
                        class_id=subset.class_id.astype(int),
                        data={
                            "team": teams.astype(int),
                            "jersey": np.asarray([""] * len(subset), dtype=object),
                            "kf_vx": kf_vx,
                            "kf_vy": kf_vy,
                        },
                    )
                )

        if len(refs):
            parts.append(
                sv.Detections(
                    xyxy=refs.xyxy.astype(np.float32),
                    tracker_id=np.full(len(refs), -1, dtype=int),
                    class_id=np.full(len(refs), ROLE_REFEREE, dtype=int),
                    data=_empty_detections_data(len(refs)),
                )
            )

        if len(balls):
            parts.append(
                sv.Detections(
                    xyxy=balls.xyxy.astype(np.float32),
                    tracker_id=np.array([-1], dtype=int),
                    class_id=np.array([ROLE_BALL], dtype=int),
                    data=_empty_detections_data(1),
                )
            )

        if not parts:
            yield frame_idx, sv.Detections.empty()
            continue

        merged = parts[0]
        for p in parts[1:]:
            merged = sv.Detections.merge([merged, p])
        yield frame_idx, team_stabilizer.apply(merged)


def iter_football_model_detections(
    sequence: SoccerNetSequence,
    *,
    start: int = 1,
    end: int | None = None,
    device: str = "cpu",
    threshold: float = 0.5,
    ball_threshold: float = DEFAULT_BALL_DETECTION_THRESHOLD,
    sample_stride: int = 30,
    model_path: str | Path | None = None,
    model_id: str = DEFAULT_FOOTBALL_PLAYERS_MODEL_ID,
    backend: str = "yolo",
    tracker: TrackerKind = "bytetrack",
    ball_backend: str | None = None,
    ball_model_id: str = DEFAULT_FOOTBALL_BALL_MODEL_ID,
) -> Iterator[tuple[int, sv.Detections]]:
    """Convenience wrapper: football-players detect + track + jersey teams."""
    detector = create_football_players_detector(
        backend=backend,
        model_id=model_id,
        model_path=model_path,
        device=device,
        threshold=threshold,
        ball_threshold=ball_threshold,
    )
    ball_detector = create_football_ball_detector(
        backend=ball_backend or "none",
        model_id=ball_model_id,
        threshold=ball_threshold,
        device=device,
    )
    team_classifier = fit_football_team_classifier(
        sequence, detector, sample_stride=sample_stride, max_frames=end or sequence.length
    )
    yield from iter_football_detections(
        sequence,
        detector,
        team_classifier,
        start=start,
        end=end,
        tracker=tracker,
        ball_detector=ball_detector,
    )
