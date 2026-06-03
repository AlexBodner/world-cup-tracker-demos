"""RF-DETR detection + trackers.ByteTrackTracker tracking (the v2 "raw pixels" path).

TODO: homography/speed looks miscalibrated on the RF-DETR path — likely a resolution/scale
mismatch between RF-DETR inference resolution and the 1920x1080 frames used for pitch
keypoints; scale boxes back to native frame size before tracking/homography.

This produces the same ``sv.Detections`` contract as
:func:`common.soccernet.iter_gt_detections` - ``tracker_id``, ``class_id`` (role) and a
``data['team']`` array - so the pass / speed demos run unchanged on top of it.

Roles available from COCO RF-DETR: ``person`` -> player, ``sports ball`` -> ball.
Goalkeeper / referee are not separable from players here (documented limitation); teams
come from a lightweight jersey-color classifier instead of the heavier Siglip+UMAP
``TeamClassifier`` in roboflow/sports.
"""

from __future__ import annotations

from typing import Iterator

import cv2
import numpy as np
import supervision as sv

from trackers import ByteTrackTracker

from world_cup_projects.common.soccernet import (
    ROLE_BALL,
    ROLE_PLAYER,
    TEAM_NONE,
    SoccerNetSequence,
)


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
        image = cv2.imread(str(sequence.frame_path(frame_idx)))
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
        image = cv2.imread(str(sequence.frame_path(frame_idx)))
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
