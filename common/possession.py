"""Ball-carrier detection and possession-moment selection.

Works on the ``sv.Detections`` produced by :func:`common.soccernet.iter_gt_detections`
(v1) or any equivalent detector+tracker output that carries ``class_id`` (role) and a
``data['team']`` array (v2).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import supervision as sv

from world_cup_projects.common.soccernet import (
    ROLE_BALL,
    ROLE_GOALKEEPER,
    ROLE_PLAYER,
)

# Nearest-feet-to-ball thresholds for "in possession" (not a pass detector).
CARRIER_MAX_DISTANCE_PX = 80.0
# ~1 m on the pitch: tight control at the feet; GT box + ball annotation add slack vs true 0.5 m.
CARRIER_MAX_DISTANCE_M = 1.0


def feet_xy(detections: sv.Detections) -> np.ndarray:
    """Bottom-center anchor (the players' feet / ground contact point)."""
    return detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)


def bbox_center_xy(detections: sv.Detections) -> np.ndarray:
    """BBox center — better for lane blocking than feet when players lean across a pass."""
    boxes = detections.xyxy
    return np.stack(
        [(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2],
        axis=1,
    )


def player_mask(detections: sv.Detections) -> np.ndarray:
    """Boolean mask of outfield players + goalkeepers."""
    return np.isin(detections.class_id, (ROLE_PLAYER, ROLE_GOALKEEPER))


def ball_xy(detections: sv.Detections) -> np.ndarray | None:
    """Return the ball's ground position, or ``None`` if no ball this frame."""
    mask = detections.class_id == ROLE_BALL
    if not mask.any():
        return None
    boxes = detections.xyxy[mask]
    cx = (boxes[:, 0] + boxes[:, 2]) / 2
    cy = boxes[:, 3]
    return np.stack([cx, cy], axis=1)[0]


@dataclass(frozen=True)
class Carrier:
    index: int          # row index into the detections
    team: int
    distance: float     # pixels (v1) or meters (v2) from ball to carrier feet
    ball: np.ndarray    # (2,) ball ground position


def find_ball_carrier(
    detections: sv.Detections,
    *,
    max_distance_px: float = CARRIER_MAX_DISTANCE_PX,
    transformer=None,
    max_distance_m: float = CARRIER_MAX_DISTANCE_M,
) -> Carrier | None:
    """Nearest player to the ball, if within range (pixels or pitch meters).

    When ``transformer`` is set (metric / homography path), distance is measured on the
    pitch in meters via :func:`common.pitch.image_to_pitch_m`. Otherwise image pixels.
    """
    ball = ball_xy(detections)
    if ball is None:
        return None
    pmask = player_mask(detections)
    if not pmask.any():
        return None

    feet_img = feet_xy(detections)[pmask]
    global_indices = np.flatnonzero(pmask)
    roles = detections.class_id[pmask]

    use_pixels = transformer is None
    dist = None
    
    if transformer is not None:
        from world_cup_projects.common.pitch import image_to_pitch_m

        feet_m = image_to_pitch_m(feet_img, transformer)
        ball_m = image_to_pitch_m(np.array([ball], dtype=np.float32), transformer)
        if feet_m is not None and ball_m is not None:
            dist = np.linalg.norm(feet_m - ball_m[0], axis=1)
            # Relax the limit for Goalkeepers since the ball in hands projects far away
            limit = np.full(len(dist), max_distance_m, dtype=np.float32)
            limit[roles == ROLE_GOALKEEPER] = max_distance_m * 3.5
        else:
            use_pixels = True

    if use_pixels:
        dist = np.hypot(feet_img[:, 0] - ball[0], feet_img[:, 1] - ball[1])
        limit = np.full(len(dist), max_distance_px, dtype=np.float32)
        limit[roles == ROLE_GOALKEEPER] = max_distance_px * 2.5

    if dist is None:
        return None

    local = int(np.argmin(dist))
    if dist[local] > limit[local]:
        return None
    global_idx = int(global_indices[local])
    team = int(detections.data["team"][global_idx])
    return Carrier(global_idx, team, float(dist[local]), ball)
