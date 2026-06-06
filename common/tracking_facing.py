"""Kalman velocity facing via tracker replay (for cached detections without KF state)."""

from __future__ import annotations

import numpy as np
import supervision as sv
from trackers.utils.state_representations import XCYCWHStateEstimator, XYXYStateEstimator

from world_cup_projects.common.geometry import unit
from world_cup_projects.common.player_tracker import TrackerKind, create_player_tracker
from world_cup_projects.common.soccernet import ROLE_GOALKEEPER, ROLE_PLAYER

DEFAULT_MIN_SPEED_PX = 0.5


def kalman_velocity_by_tracker_id(tracker, *, min_speed: float = DEFAULT_MIN_SPEED_PX) -> dict[int, np.ndarray]:
    """Map confirmed ``tracker_id`` to feet-referenced Kalman velocity."""
    out: dict[int, np.ndarray] = {}
    for tracklet in tracker.tracks:
        tid = int(tracklet.tracker_id)
        if tid < 0:
            continue
        velocity = kalman_feet_velocity_from_tracklet(tracklet)
        if velocity is None:
            continue
        if float(np.linalg.norm(velocity)) < min_speed:
            continue
        out[tid] = velocity
    return out


def kalman_velocity_arrays(
    detections: sv.Detections,
    tracker,
    *,
    min_speed: float = DEFAULT_MIN_SPEED_PX,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-row ``kf_vx`` / ``kf_vy`` aligned with ``detections``."""
    n = len(detections)
    kf_vx = np.full(n, np.nan, dtype=np.float32)
    kf_vy = np.full(n, np.nan, dtype=np.float32)
    if n == 0 or detections.tracker_id is None:
        return kf_vx, kf_vy
    id_to_vel = kalman_velocity_by_tracker_id(tracker, min_speed=min_speed)
    for i, tid in enumerate(detections.tracker_id):
        velocity = id_to_vel.get(int(tid))
        if velocity is None:
            continue
        kf_vx[i] = float(velocity[0])
        kf_vy[i] = float(velocity[1])
    return kf_vx, kf_vy


def facing_kalman_from_detections(
    detections: sv.Detections,
    *,
    min_speed: float = DEFAULT_MIN_SPEED_PX,
) -> np.ndarray:
    """Unit facing vectors from cached ``kf_vx`` / ``kf_vy`` (NaN rows if missing)."""
    n = len(detections)
    out = np.full((n, 2), np.nan, dtype=np.float64)
    if n == 0 or detections.data is None:
        return out
    kf_vx = detections.data.get("kf_vx")
    kf_vy = detections.data.get("kf_vy")
    if kf_vx is None or kf_vy is None:
        return out
    for i in range(n):
        vx, vy = float(kf_vx[i]), float(kf_vy[i])
        if not np.isfinite(vx) or not np.isfinite(vy):
            continue
        speed = float(np.hypot(vx, vy))
        if speed < min_speed:
            continue
        out[i] = unit(np.array([vx, vy], dtype=np.float64))
    return out


def carrier_kalman_direction(
    detections: sv.Detections,
    carrier_index: int,
    *,
    transformer=None,
    min_speed: float = DEFAULT_MIN_SPEED_PX,
) -> np.ndarray | None:
    """Unit movement direction for the ball carrier from Kalman velocity."""
    if detections.data is None:
        return None
    kf_vx = detections.data.get("kf_vx")
    kf_vy = detections.data.get("kf_vy")
    if kf_vx is None or kf_vy is None:
        return None
    vx, vy = float(kf_vx[carrier_index]), float(kf_vy[carrier_index])
    if not np.isfinite(vx) or not np.isfinite(vy):
        return None
    speed = float(np.hypot(vx, vy))
    if speed < min_speed:
        return None
    vel_img = np.array([vx, vy], dtype=np.float64)
    if transformer is None:
        return unit(vel_img)
    from world_cup_projects.common.pitch import image_to_pitch_m
    from world_cup_projects.common.possession import feet_xy

    feet = feet_xy(detections)[carrier_index]
    p0 = image_to_pitch_m(feet.reshape(1, 2), transformer)
    p1 = image_to_pitch_m((feet + vel_img).reshape(1, 2), transformer)
    if p0 is None or p1 is None:
        return unit(vel_img)
    delta = p1[0] - p0[0]
    if float(np.linalg.norm(delta)) < 1e-6:
        return unit(vel_img)
    return unit(delta)


def detections_have_kalman_velocity(detections: sv.Detections) -> bool:
    return (
        detections.data is not None
        and "kf_vx" in detections.data
        and "kf_vy" in detections.data
    )


def kalman_feet_velocity_from_tracklet(tracklet) -> np.ndarray | None:
    """Feet-referenced velocity from the tracklet Kalman state."""
    est = tracklet.state_estimator
    x = est.kf.x.flatten()
    if len(x) < 7:
        return None
    if isinstance(est, XYXYStateEstimator):
        vx = (float(x[4]) + float(x[6])) / 2.0
        vy = float(x[7])
    elif isinstance(est, XCYCWHStateEstimator):
        vx = float(x[4])
        vy = float(x[5]) + float(x[7]) / 2.0
    else:
        vx, vy = float(x[4]), float(x[5])
    return np.array([vx, vy], dtype=np.float64)


class KalmanFacingReplay:
    """Replay a tracker on trackable detections to recover per-id Kalman velocities."""

    def __init__(
        self,
        frame_rate: float,
        *,
        tracker_kind: TrackerKind = "bytetrack",
        track_activation_threshold: float = 0.4,
        min_speed: float = DEFAULT_MIN_SPEED_PX,
    ) -> None:
        self._min_speed = min_speed
        self._needs_frame = tracker_kind == "botsort"
        self._tracker = create_player_tracker(
            frame_rate,
            kind=tracker_kind,
            track_activation_threshold=track_activation_threshold,
        )

    def advance(
        self,
        frame_dets: sv.Detections,
        frame: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return ``(N, 2)`` unit facing vectors aligned with ``frame_dets`` rows."""
        n = len(frame_dets)
        out = np.full((n, 2), np.nan, dtype=np.float64)
        if n == 0:
            return out

        pmask = np.isin(frame_dets.class_id, (ROLE_PLAYER, ROLE_GOALKEEPER))
        if not pmask.any():
            return out

        trackable = frame_dets[pmask]
        if self._needs_frame and frame is None:
            return out
        self._tracker.update(
            trackable,
            frame=frame if self._needs_frame else None,
        )
        return facing_kalman_from_trackable(
            frame_dets,
            pmask,
            self._tracker,
            min_speed=self._min_speed,
        )


def facing_kalman_from_trackable(
    frame_dets: sv.Detections,
    pmask: np.ndarray,
    tracker,
    *,
    min_speed: float = DEFAULT_MIN_SPEED_PX,
) -> np.ndarray:
    """Align Kalman facing with full-frame detections using cached tracker ids."""
    n = len(frame_dets)
    out = np.full((n, 2), np.nan, dtype=np.float64)
    if frame_dets.tracker_id is None:
        return out
    id_to_vel = kalman_velocity_by_tracker_id(tracker, min_speed=min_speed)
    for i in range(n):
        if not pmask[i]:
            continue
        tid = int(frame_dets.tracker_id[i])
        velocity = id_to_vel.get(tid)
        if velocity is None:
            continue
        out[i] = unit(velocity)
    return out
