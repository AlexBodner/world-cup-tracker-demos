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

    When ``transformer`` is set, we check both metric and pixel distance. A player
    is considered a valid carrier if they are within the limit in EITHER space.
    This prevents possession drops when homography is temporarily distorted by camera blur.
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

    # Always calculate pixel distance as a robust fallback
    # To prevent aerial balls from triggering false possession, we heavily penalize 
    # balls that are higher up on the screen (lower Y coordinate) than the player's feet.
    dx = feet_img[:, 0] - ball[0]
    dy = feet_img[:, 1] - ball[1]
    
    # If the ball is "above" the feet (dy > 0), we multiply the vertical distance penalty.
    # This stretches the effective distance for aerial balls, preventing fly-bys.
    dy_penalty = np.where(dy > 10, dy * 2.5, dy)
    
    dist_px = np.hypot(dx, dy_penalty)
    limit_px = np.full(len(dist_px), max_distance_px, dtype=np.float32)
    limit_px[roles == ROLE_GOALKEEPER] = max_distance_px * 2.5

    valid_mask = dist_px <= limit_px
    dist_to_use = dist_px

    if transformer is not None:
        from world_cup_projects.common.pitch import image_to_pitch_m

        feet_m = image_to_pitch_m(feet_img, transformer)
        ball_m = image_to_pitch_m(np.array([ball], dtype=np.float32), transformer)
        if feet_m is not None and ball_m is not None:
            dist_m = np.linalg.norm(feet_m - ball_m[0], axis=1)
            limit_m = np.full(len(dist_m), max_distance_m, dtype=np.float32)
            limit_m[roles == ROLE_GOALKEEPER] = max_distance_m * 3.5
            
            # A player is valid if they pass the metric check OR the pixel check
            valid_mask = valid_mask | (dist_m <= limit_m)
            dist_to_use = dist_m

    if not valid_mask.any():
        return None

    # Find the closest player among the valid candidates
    # We use dist_to_use (metric if available, else px) to pick the 'closest'
    # We set invalid players to infinity so they aren't chosen
    dist_to_use[~valid_mask] = float('inf')
    local = int(np.argmin(dist_to_use))
    
    global_idx = int(global_indices[local])
    team = int(detections.data["team"][global_idx])
    return Carrier(global_idx, team, float(dist_to_use[local]), ball)
