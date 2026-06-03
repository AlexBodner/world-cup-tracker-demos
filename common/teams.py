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
        h, w = crop.shape[:2]
        torso = crop[int(h * 0.15) : int(h * 0.55), int(w * 0.2) : int(w * 0.8)]
        if torso.size == 0:
            torso = crop
        return torso.reshape(-1, 3).mean(axis=0)

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


def assign_teams(
    frame: np.ndarray,
    detections: sv.Detections,
    classifier: TeamClassifier | JerseyColorTeamClassifier,
) -> np.ndarray:
    """Return a ``team`` array aligned with *detections* rows."""
    n = len(detections)
    team = np.full(n, TEAM_NONE, dtype=int)

    player_rows = np.flatnonzero(
        np.isin(detections.class_id, (ROLE_PLAYER, ROLE_GOALKEEPER))
    )
    if len(player_rows) == 0:
        return team

    players = detections[player_rows]
    player_teams = classifier.predict(get_crops(frame, players))

    outfield_local = players.class_id == ROLE_PLAYER
    gk_local = players.class_id == ROLE_GOALKEEPER

    if outfield_local.any() and gk_local.any():
        outfield = players[outfield_local]
        outfield_teams = player_teams[outfield_local]
        gks = players[gk_local]
        gk_teams = resolve_goalkeepers_team_id(outfield, outfield_teams, gks)
        team[player_rows[outfield_local]] = player_teams[outfield_local]
        team[player_rows[gk_local]] = gk_teams
    else:
        team[player_rows] = player_teams

    return team
