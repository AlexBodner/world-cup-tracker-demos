"""Team assignment for v2 detections (no ground-truth team labels)."""

from __future__ import annotations

from typing import Iterable, List

import numpy as np
import supervision as sv
import torch
from sklearn.cluster import KMeans

from world_cup_projects.common.soccernet import (
    ROLE_GOALKEEPER,
    ROLE_PLAYER,
    TEAM_NONE,
)

SIGLIP_MODEL_PATH = "google/siglip-base-patch16-224"


def get_crops(frame: np.ndarray, detections: sv.Detections) -> List[np.ndarray]:
    return [sv.crop_image(frame, xyxy) for xyxy in detections.xyxy]


class TrackletTeamStabilizer:
    """Per-track jersey team with hysteresis: flip only after N consecutive disagrees."""

    def __init__(self, *, flip_after: int = 16) -> None:
        self.flip_after = max(1, int(flip_after))
        self._stable: dict[int, int] = {}
        self._streak: dict[int, int] = {}

    def apply(self, dets: sv.Detections) -> sv.Detections:
        if dets.tracker_id is None or dets.data is None or len(dets) == 0:
            return dets
        team = np.array(dets.data.get("team", np.full(len(dets), TEAM_NONE)), dtype=int)
        for i, tid in enumerate(dets.tracker_id):
            tid = int(tid)
            if tid < 0 or dets.class_id[i] != ROLE_PLAYER:
                continue
            raw = int(team[i])
            if raw not in (0, 1):
                continue
            if tid not in self._stable:
                self._stable[tid] = raw
                self._streak[tid] = 0
            elif raw == self._stable[tid]:
                self._streak[tid] = 0
            else:
                self._streak[tid] = self._streak.get(tid, 0) + 1
                if self._streak[tid] >= self.flip_after:
                    self._stable[tid] = raw
                    self._streak[tid] = 0
            team[i] = self._stable[tid]
        dets.data["team"] = team
        return dets


def stabilize_teams_by_tracklet(
    frames: Iterable[tuple[int, sv.Detections]],
    *,
    flip_after: int = 16,
) -> list[tuple[int, sv.Detections]]:
    """Run :class:`TrackletTeamStabilizer` across a clip (works on cached detections too)."""
    stabilizer = TrackletTeamStabilizer(flip_after=flip_after)
    return [(int(fi), stabilizer.apply(dets)) for fi, dets in frames]


def resolve_goalkeepers_team_by_goal(
    goalkeepers_pitch_cm: np.ndarray,
    outfield_pitch_cm: np.ndarray,
    outfield_team_id: np.ndarray,
    *,
    pitch_length_cm: float = 12000.0,
    pitch_width_cm: float = 7000.0,
) -> np.ndarray:
    """Assign each GK to the team defending the nearer goal mouth."""
    from world_cup_projects.common.pitch import infer_goal_defenders
    from world_cup_projects.common.soccernet import TEAM_LEFT, TEAM_RIGHT

    if len(goalkeepers_pitch_cm) == 0:
        return np.array([], dtype=int)

    if (
        len(outfield_pitch_cm) >= 4
        and np.any(outfield_team_id == 0)
        and np.any(outfield_team_id == 1)
    ):
        left_def, right_def = infer_goal_defenders(outfield_pitch_cm, outfield_team_id)
    else:
        left_def, right_def = TEAM_LEFT, TEAM_RIGHT

    left_goal = np.array([0.0, pitch_width_cm / 2.0], dtype=np.float32)
    right_goal = np.array([pitch_length_cm, pitch_width_cm / 2.0], dtype=np.float32)
    ids: list[int] = []
    for xy in goalkeepers_pitch_cm:
        d_left = float(np.linalg.norm(xy - left_goal))
        d_right = float(np.linalg.norm(xy - right_goal))
        ids.append(left_def if d_left < d_right else right_def)
    return np.array(ids, dtype=int)


def apply_goalkeeper_teams_by_goal(dets: sv.Detections, transformer) -> sv.Detections:
    """Set ``data['team']`` on goalkeeper rows from pitch distance to each goal."""
    from world_cup_projects.common.pitch import image_to_pitch_cm
    from world_cup_projects.common.possession import feet_xy

    gk_mask = dets.class_id == ROLE_GOALKEEPER
    out_mask = dets.class_id == ROLE_PLAYER
    if not gk_mask.any():
        return dets

    feet = feet_xy(dets)
    gk_cm = image_to_pitch_cm(feet[gk_mask], transformer)
    if gk_cm is None or not np.isfinite(gk_cm).all():
        return dets

    out_cm = None
    out_teams = None
    if out_mask.any():
        out_cm = image_to_pitch_cm(feet[out_mask], transformer)
        out_teams = dets.data.get("team", np.zeros(len(dets), dtype=int))[out_mask]

    gk_teams = resolve_goalkeepers_team_by_goal(
        gk_cm,
        out_cm if out_cm is not None else np.empty((0, 2), dtype=np.float32),
        out_teams if out_teams is not None else np.array([], dtype=int),
    )

    team = np.array(dets.data.get("team", np.full(len(dets), TEAM_NONE)), dtype=int)
    team[gk_mask] = gk_teams
    dets.data["team"] = team
    return dets


def resolve_goalkeepers_team_id(
    players: sv.Detections,
    players_team_id: np.ndarray,
    goalkeepers: sv.Detections,
) -> np.ndarray:
    """Assign each goalkeeper to the nearest team centroid (sports reference)."""
    if len(goalkeepers) == 0:
        return np.array([], dtype=int)

    gk_xy = goalkeepers.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    players_xy = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    team_0 = players_xy[players_team_id == 0].mean(axis=0)
    team_1 = players_xy[players_team_id == 1].mean(axis=0)
    ids: list[int] = []
    for xy in gk_xy:
        d0 = np.linalg.norm(xy - team_0)
        d1 = np.linalg.norm(xy - team_1)
        ids.append(0 if d0 < d1 else 1)
    return np.array(ids, dtype=int)


def _create_batches(sequence: Iterable, batch_size: int):
    batch: list = []
    for element in sequence:
        batch.append(element)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


class TeamClassifier:
    """SigLIP + UMAP + KMeans team classifier (vendored from sports/common/team.py)."""

    def __init__(self, device: str = "cpu", batch_size: int = 32) -> None:
        from transformers import AutoProcessor, SiglipVisionModel
        import umap

        self.device = device
        self.batch_size = batch_size
        self.features_model = SiglipVisionModel.from_pretrained(SIGLIP_MODEL_PATH).to(device)
        self.processor = AutoProcessor.from_pretrained(SIGLIP_MODEL_PATH)
        self.reducer = umap.UMAP(n_components=3)
        self.cluster_model = KMeans(n_clusters=2, random_state=42)
        self._fitted = False

    def extract_features(self, crops: List[np.ndarray]) -> np.ndarray:
        crops_pil = [sv.cv2_to_pillow(crop) for crop in crops]
        data: list[np.ndarray] = []
        with torch.no_grad():
            for batch in _create_batches(crops_pil, self.batch_size):
                inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
                outputs = self.features_model(**inputs)
                embeddings = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()
                data.append(embeddings)
        return np.concatenate(data) if data else np.empty((0, 768))

    def fit(self, crops: List[np.ndarray]) -> None:
        if not crops:
            return
        data = self.extract_features(crops)
        projections = self.reducer.fit_transform(data)
        self.cluster_model.fit(projections)
        self._fitted = True

    def predict(self, crops: List[np.ndarray]) -> np.ndarray:
        if not crops:
            return np.array([], dtype=int)
        if not self._fitted:
            raise RuntimeError("TeamClassifier.fit() must be called before predict().")
        data = self.extract_features(crops)
        projections = self.reducer.transform(data)
        return self.cluster_model.predict(projections)


class JerseyColorTeamClassifier:
    """Fast fallback: cluster mean jersey RGB in player bounding boxes."""

    def __init__(self) -> None:
        self.cluster_model = KMeans(n_clusters=2, random_state=42)
        self._fitted = False

    @staticmethod
    def _jersey_color(crop: np.ndarray) -> np.ndarray:
        import cv2

        h, w = crop.shape[:2]
        torso = crop[int(h * 0.15) : int(h * 0.55), int(w * 0.2) : int(w * 0.8)]
        if torso.size == 0:
            torso = crop
        lab = cv2.cvtColor(torso, cv2.COLOR_BGR2LAB).reshape(-1, 3)
        return lab.mean(axis=0)

    def fit(self, crops: List[np.ndarray]) -> None:
        if not crops:
            return
        colors = np.array([self._jersey_color(c) for c in crops], dtype=np.float32)
        self.cluster_model.fit(colors)
        self._fitted = True

    def predict(self, crops: List[np.ndarray]) -> np.ndarray:
        if not crops:
            return np.array([], dtype=int)
        if not self._fitted:
            raise RuntimeError("JerseyColorTeamClassifier.fit() must be called before predict().")
        colors = np.array([self._jersey_color(c) for c in crops], dtype=np.float32)
        return self.cluster_model.predict(colors)


def stabilize_goalkeeper_teams(
    frames: list[tuple[int, sv.Detections]],
    transforms: dict[int, object | None] | None = None,
    locked_goal_defenders: tuple[int, int] | None = None,
    *,
    keypoints_by_frame: dict[int, object] | None = None,
    pitch_confidence: float = 0.9,
) -> None:
    """Mutate frames so GKs keep a stable defending team (sports-radar pitch space).

    1. Identify tracklets ever detected as ROLE_GOALKEEPER.
    2. Average pitch X from the same per-frame radar H used on the minimap.
    3. Assign the team defending the nearer goal for the whole tracklet.
    """
    from world_cup_projects.common.pitch import (
        homography_from_keypoints_radar,
        image_to_pitch_cm,
    )
    from world_cup_projects.common.possession import feet_xy
    from world_cup_projects.common.soccernet import TEAM_LEFT, TEAM_RIGHT

    if not locked_goal_defenders:
        left_def, right_def = TEAM_LEFT, TEAM_RIGHT
    else:
        left_def, right_def = locked_goal_defenders

    track_positions: dict[int, list[float]] = {}
    gk_track_ids: set[int] = set()

    for frame_idx, dets in frames:
        if dets.tracker_id is None:
            continue
        t = None
        if keypoints_by_frame is not None:
            t = homography_from_keypoints_radar(
                keypoints_by_frame.get(int(frame_idx)),
                confidence=pitch_confidence,
            )
        if t is None and transforms is not None:
            t = transforms.get(frame_idx)
        if t is None:
            continue

        feet = feet_xy(dets)
        feet_cm = image_to_pitch_cm(feet, t)
        if feet_cm is None:
            continue

        for i, tid in enumerate(dets.tracker_id):
            tid = int(tid)
            if tid < 0:
                continue
            if dets.class_id[i] == ROLE_GOALKEEPER:
                gk_track_ids.add(tid)

            if np.isfinite(feet_cm[i]).all():
                track_positions.setdefault(tid, []).append(float(feet_cm[i, 0]))

    stable_assignments: dict[int, int] = {}
    pitch_mid_cm = 12000.0 / 2.0

    for tid in gk_track_ids:
        positions = track_positions.get(tid)
        if not positions:
            continue
        avg_x = sum(positions) / len(positions)
        stable_assignments[tid] = left_def if avg_x < pitch_mid_cm else right_def

    # Patch frames in place
    for _, dets in frames:
        if dets.tracker_id is None or dets.data is None:
            continue
        team_array = dets.data.get("team")
        if team_array is None:
            continue
        for i, tid in enumerate(dets.tracker_id):
            tid = int(tid)
            if tid in stable_assignments:
                dets.class_id[i] = ROLE_GOALKEEPER
                team_array[i] = stable_assignments[tid]
