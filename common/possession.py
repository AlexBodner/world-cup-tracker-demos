"""Ball-carrier detection and possession-moment selection.

Works on the ``sv.Detections`` produced by :func:`common.soccernet.iter_gt_detections`
(v1) or any equivalent detector+tracker output that carries ``class_id`` (role) and a
``data['team']`` array (v2).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import supervision as sv

from world_cup_projects.common.possession_config import (
    CARRIER_MAX_DISTANCE_M,
    CARRIER_MAX_DISTANCE_PX,
    CONTROL_MAX_DISTANCE_M,
    CONTROL_MAX_DISTANCE_PX,
    RECEPTION_MAX_DISTANCE_M,
    RECEPTION_MAX_DISTANCE_PX,
)
from world_cup_projects.common.soccernet import (
    ROLE_BALL,
    ROLE_GOALKEEPER,
    ROLE_PLAYER,
)


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
    require_both_spaces: bool = False,
) -> Carrier | None:
    """Nearest player to the ball, if within range (pixels or pitch meters).

    When ``transformer`` is set, we check both metric and pixel distance. By default
    a player is valid if they are within the limit in **either** space (robust to brief
    homography glitches). With ``require_both_spaces=True``, both must pass — used for
    pass-alternative freeze picking so warped metric alone cannot extend possession.
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
    # balls that are significantly above OR below the feet in the 2D image.
    dx = feet_img[:, 0] - ball[0]
    dy = feet_img[:, 1] - ball[1]
    
    # If the ball is significantly displaced on the Y-axis (abs(dy) > 20px), 
    # it is likely an aerial pass (either behind or in front of the player).
    # We use this as a strict veto against the 3D metric projection.
    is_aerial = np.abs(dy) > 20
    
    # We stretch the effective Y distance to prevent pixel-fallback fly-bys.
    dy_penalty = np.where(np.abs(dy) > 10, dy * 2.5, dy)
    
    dist_px = np.hypot(dx, dy_penalty)
    limit_px = np.full(len(dist_px), max_distance_px, dtype=np.float32)
    limit_px[roles == ROLE_GOALKEEPER] = max_distance_px * 2.5

    pixel_valid = dist_px <= limit_px
    valid_mask = pixel_valid
    dist_to_use = dist_px

    if transformer is not None:
        from world_cup_projects.common.pitch import image_to_pitch_m

        feet_m = image_to_pitch_m(feet_img, transformer)
        ball_m = image_to_pitch_m(np.array([ball], dtype=np.float32), transformer)
        if feet_m is not None and ball_m is not None:
            dist_m = np.linalg.norm(feet_m - ball_m[0], axis=1)
            limit_m = np.full(len(dist_m), max_distance_m, dtype=np.float32)
            limit_m[roles == ROLE_GOALKEEPER] = max_distance_m * 3.5

            # Veto metric when the ball is clearly aerial in 2D (flat-ground assumption).
            metric_valid = (dist_m <= limit_m) & ~is_aerial
            if require_both_spaces:
                valid_mask = pixel_valid & metric_valid
            else:
                valid_mask = pixel_valid | metric_valid
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


def carrier_from_tracker_id(
    detections: sv.Detections,
    tracker_id: int,
) -> Carrier | None:
    """Build a :class:`Carrier` for a known player row (e.g. inferred passer at release).

    Does not require the ball to be at their feet — used for pass-alternative freezes on
    the release frame, when control-range carrier lookup would already have failed.
    """
    if tracker_id < 0 or detections.tracker_id is None:
        return None
    pmask = player_mask(detections)
    rows = np.flatnonzero(pmask & (detections.tracker_id == tracker_id))
    if len(rows) == 0:
        return None
    idx = int(rows[0])
    ball = ball_xy(detections)
    feet = feet_xy(detections)[idx]
    if ball is None:
        ball = feet
    dist = float(np.linalg.norm(feet - ball))
    team = int(detections.data["team"][idx])
    return Carrier(idx, team, dist, np.asarray(ball, dtype=np.float64))


def find_control_carrier(
    detections: sv.Detections,
    *,
    transformer=None,
    max_distance_px: float = CONTROL_MAX_DISTANCE_PX,
    max_distance_m: float = CONTROL_MAX_DISTANCE_M,
    require_both_spaces: bool = False,
) -> Carrier | None:
    """Nearest player within tight dribble range (pass passer / lane-scoring gate)."""
    return find_ball_carrier(
        detections,
        max_distance_px=max_distance_px,
        transformer=transformer,
        max_distance_m=max_distance_m,
        require_both_spaces=require_both_spaces,
    )


def find_reception_carrier(
    detections: sv.Detections,
    *,
    transformer=None,
    max_distance_px: float = RECEPTION_MAX_DISTANCE_PX,
    max_distance_m: float = RECEPTION_MAX_DISTANCE_M,
) -> Carrier | None:
    """Nearest player within looser first-touch range (pass detection only)."""
    return find_ball_carrier(
        detections,
        max_distance_px=max_distance_px,
        transformer=transformer,
        max_distance_m=max_distance_m,
    )


def find_active_carrier(
    detections: sv.Detections,
    *,
    transformer=None,
    control_max_distance_px: float = CONTROL_MAX_DISTANCE_PX,
    control_max_distance_m: float = CONTROL_MAX_DISTANCE_M,
    reception_max_distance_px: float = RECEPTION_MAX_DISTANCE_PX,
    reception_max_distance_m: float = RECEPTION_MAX_DISTANCE_M,
) -> tuple[Carrier | None, str | None]:
    """Control carrier if any, else reception; matches pass-detection possession logic."""
    control = find_control_carrier(
        detections,
        transformer=transformer,
        max_distance_px=control_max_distance_px,
        max_distance_m=control_max_distance_m,
    )
    if control is not None:
        return control, "control"
    reception = find_reception_carrier(
        detections,
        transformer=transformer,
        max_distance_px=reception_max_distance_px,
        max_distance_m=reception_max_distance_m,
    )
    if reception is not None:
        return reception, "reception"
    return None, None
