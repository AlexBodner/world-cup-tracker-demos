"""Model-based detection + ByteTrack for non-SoccerNet video (v2 path).

Sources:
- ``football``: YOLO ``football-players-detection`` (DFL-trained; ball/player/gk/referee).
- ``rfdetr``: generic COCO RF-DETR (person + sports ball only; poor team/role split).

Both yield the same ``sv.Detections`` contract as :func:`common.soccernet.iter_gt_detections`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
import supervision as sv

from trackers import ByteTrackTracker

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
    get_crops,
    resolve_goalkeepers_team_id,
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
DEFAULT_FOOTBALL_PLAYERS_MODEL_ID = "football-players-detection-3zvbc/11"


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


def _map_fp_detections(det: sv.Detections) -> sv.Detections:
    if len(det) == 0:
        return det
    roles = np.array([_FP_TO_ROLE.get(int(c), -1) for c in det.class_id], dtype=int)
    keep = roles >= 0
    det = det[keep]
    det.class_id = roles[keep]
    return det


class FootballPlayersDetector:
    """YOLO weights from roboflow/sports (football-players-detection-3zvbc)."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        device: str = "cpu",
        threshold: float = 0.5,
    ) -> None:
        from ultralytics import YOLO

        path = Path(model_path) if model_path else ensure_football_players_model()
        self.model = YOLO(str(path))
        self.device = device
        self.threshold = threshold

    def detect(self, frame_bgr: np.ndarray) -> sv.Detections:
        results = self.model.predict(
            frame_bgr,
            conf=self.threshold,
            verbose=False,
            device=self.device,
        )[0]
        return _map_fp_detections(sv.Detections.from_ultralytics(results))


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
    }


def iter_football_detections(
    sequence: SoccerNetSequence,
    detector: FootballPlayersDetector,
    team_classifier: JerseyColorTeamClassifier,
    *,
    start: int = 1,
    end: int | None = None,
    track_activation_threshold: float = 0.4,
) -> Iterator[tuple[int, sv.Detections]]:
    """Detect + track with football-players-detection; refs excluded from teams."""
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
        balls = det[det.class_id == ROLE_BALL]
        players = det[det.class_id == ROLE_PLAYER]
        gks = det[det.class_id == ROLE_GOALKEEPER]
        refs = det[det.class_id == ROLE_REFEREE]

        trackable = sv.Detections.merge([players, gks])
        tracked = tracker.update(trackable, frame=image) if len(trackable) else sv.Detections.empty()

        parts: list[sv.Detections] = []
        if len(tracked):
            t_players = tracked[tracked.class_id == ROLE_PLAYER]
            t_gks = tracked[tracked.class_id == ROLE_GOALKEEPER]
            if len(t_players):
                player_teams = team_classifier.predict(get_crops(image, t_players))
            else:
                player_teams = np.array([], dtype=int)
            if len(t_gks) and len(t_players):
                gk_teams = resolve_goalkeepers_team_id(t_players, player_teams, t_gks)
            elif len(t_gks):
                gk_teams = np.full(len(t_gks), TEAM_NONE, dtype=int)
            else:
                gk_teams = np.array([], dtype=int)

            for subset, teams in ((t_players, player_teams), (t_gks, gk_teams)):
                if len(subset) == 0:
                    continue
                tid = subset.tracker_id if subset.tracker_id is not None else np.full(len(subset), -1)
                parts.append(
                    sv.Detections(
                        xyxy=subset.xyxy.astype(np.float32),
                        tracker_id=tid.astype(int),
                        class_id=subset.class_id.astype(int),
                        data={
                            "team": teams.astype(int),
                            "jersey": np.asarray([""] * len(subset), dtype=object),
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
                    xyxy=balls.xyxy[:1].astype(np.float32),
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
        yield frame_idx, merged


def iter_football_model_detections(
    sequence: SoccerNetSequence,
    *,
    start: int = 1,
    end: int | None = None,
    device: str = "cpu",
    threshold: float = 0.5,
    sample_stride: int = 30,
    model_path: str | Path | None = None,
) -> Iterator[tuple[int, sv.Detections]]:
    """Convenience wrapper: football-players YOLO + jersey team classifier."""
    detector = FootballPlayersDetector(model_path=model_path, device=device, threshold=threshold)
    team_classifier = fit_football_team_classifier(
        sequence, detector, sample_stride=sample_stride, max_frames=end or sequence.length
    )
    yield from iter_football_detections(
        sequence, detector, team_classifier, start=start, end=end
    )
