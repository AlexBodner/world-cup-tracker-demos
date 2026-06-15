"""Kalman velocity facing via tracker replay (for cached detections without KF state)."""

from __future__ import annotations

import numpy as np
import supervision as sv
from trackers.utils.state_representations import XCYCWHStateEstimator, XYXYStateEstimator

from world_cup_projects.common.geometry import unit
from world_cup_projects.common.player_tracker import TrackerKind, create_player_tracker
from world_cup_projects.common.soccernet import ROLE_GOALKEEPER, ROLE_PLAYER

DEFAULT_MIN_SPEED_PX = 0.5
PLAYER_HEIGHT_M = 1.8
DEFAULT_SPEED_HOMOGRAPHY_WEIGHT = 0.3


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


def kalman_ground_speed_m_s(
    feet_px: np.ndarray,
    vel_px: np.ndarray,
    transformer,
    *,
    fps: float,
    box_height_px: float | None = None,
    min_speed_px: float = DEFAULT_MIN_SPEED_PX,
    homography_weight: float = DEFAULT_SPEED_HOMOGRAPHY_WEIGHT,
) -> float | None:
    """Ground speed (m/s) from Kalman image velocity.

    Blends pitch-homography displacement with a player-height meters-per-pixel
    estimate so broadcast homography scale error does not inflate labels.
    """
    vel = np.asarray(vel_px, dtype=np.float64).reshape(2)
    vx, vy = float(vel[0]), float(vel[1])
    if not np.isfinite(vx) or not np.isfinite(vy) or fps <= 0:
        return None
    speed_px = float(np.hypot(vx, vy))
    if speed_px < min_speed_px:
        return None

    homo_ms: float | None = None
    if transformer is not None:
        from world_cup_projects.common.pitch import image_to_pitch_m

        feet = np.asarray(feet_px, dtype=np.float64).reshape(2)
        p0 = image_to_pitch_m(feet.reshape(1, 2), transformer)
        p1 = image_to_pitch_m((feet + vel).reshape(1, 2), transformer)
        if p0 is not None and p1 is not None:
            delta_m = p1[0] - p0[0]
            speed_m_per_frame = float(np.linalg.norm(delta_m))
            if speed_m_per_frame >= 1e-9:
                homo_ms = speed_m_per_frame * float(fps)

    height_ms: float | None = None
    if box_height_px is not None and box_height_px > 1.0:
        mpp = PLAYER_HEIGHT_M / float(box_height_px)
        height_ms = speed_px * mpp * float(fps)

    if homo_ms is None and height_ms is None:
        return None
    if homo_ms is None:
        return height_ms
    if height_ms is None:
        return homo_ms
    w = float(np.clip(homography_weight, 0.0, 1.0))
    return w * homo_ms + (1.0 - w) * height_ms


class KalmanSpeedDisplaySmoother:
    """EMA on displayed ground speed (m/s) per track."""

    def __init__(self, *, alpha: float = 0.3) -> None:
        self.alpha = float(np.clip(alpha, 0.05, 1.0))
        self._speed: dict[int, float] = {}

    def smooth(self, tracker_id: int, speed_m_s: float) -> float:
        if tracker_id < 0:
            return float(speed_m_s)
        a = self.alpha
        if tracker_id in self._speed:
            speed_m_s = a * float(speed_m_s) + (1.0 - a) * self._speed[tracker_id]
        self._speed[tracker_id] = float(speed_m_s)
        return float(speed_m_s)


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


class KalmanVelocitySmoother:
    """Per-track EMA on ``kf_vx`` / ``kf_vy`` for display (holds last value on NaN)."""

    def __init__(self, *, alpha: float = 0.3) -> None:
        self.alpha = float(np.clip(alpha, 0.05, 1.0))
        self._state: dict[int, tuple[float, float]] = {}

    def smooth_detections(self, dets: sv.Detections) -> sv.Detections:
        if len(dets) == 0 or dets.data is None:
            return dets
        kf_vx = dets.data.get("kf_vx")
        kf_vy = dets.data.get("kf_vy")
        if kf_vx is None or kf_vy is None:
            return dets

        n = len(dets)
        out_vx = np.full(n, np.nan, dtype=np.float32)
        out_vy = np.full(n, np.nan, dtype=np.float32)
        tids = dets.tracker_id if dets.tracker_id is not None else np.full(n, -1, dtype=int)
        a = self.alpha

        for i in range(n):
            tid = int(tids[i])
            raw_x, raw_y = float(kf_vx[i]), float(kf_vy[i])
            has_raw = np.isfinite(raw_x) and np.isfinite(raw_y)

            if tid >= 0 and has_raw:
                if tid in self._state:
                    prev_x, prev_y = self._state[tid]
                    sx = a * raw_x + (1.0 - a) * prev_x
                    sy = a * raw_y + (1.0 - a) * prev_y
                else:
                    sx, sy = raw_x, raw_y
                self._state[tid] = (sx, sy)
                out_vx[i], out_vy[i] = sx, sy
            elif tid >= 0 and tid in self._state:
                sx, sy = self._state[tid]
                out_vx[i], out_vy[i] = sx, sy
            elif has_raw:
                out_vx[i], out_vy[i] = raw_x, raw_y

        data = dict(dets.data)
        data["kf_vx"] = out_vx
        data["kf_vy"] = out_vy
        return sv.Detections(
            xyxy=dets.xyxy,
            class_id=dets.class_id,
            tracker_id=dets.tracker_id,
            confidence=dets.confidence,
            data=data,
        )


class JoystickDotSmoother:
    """EMA on joystick offset from ellipse center (not absolute screen position)."""

    def __init__(self, *, alpha: float = 0.32) -> None:
        self.alpha = float(np.clip(alpha, 0.05, 1.0))
        self._offset: dict[int, tuple[float, float]] = {}

    def smooth(
        self,
        tracker_id: int,
        cx: float,
        cy: float,
        px: float,
        py: float,
    ) -> tuple[int, int]:
        if tracker_id < 0:
            return int(round(px)), int(round(py))
        ox, oy = px - cx, py - cy
        a = self.alpha
        if tracker_id in self._offset:
            pox, poy = self._offset[tracker_id]
            ox = a * ox + (1.0 - a) * pox
            oy = a * oy + (1.0 - a) * poy
        self._offset[tracker_id] = (ox, oy)
        return int(round(cx + ox)), int(round(cy + oy))


class EllipseWidthSmoother:
    """Per-track EMA on bbox width so ground ellipses do not flicker with detector jitter."""

    def __init__(self, *, alpha: float = 0.22) -> None:
        self.alpha = float(np.clip(alpha, 0.05, 1.0))
        self._width: dict[int, float] = {}

    def smooth_detections(self, dets: sv.Detections) -> sv.Detections:
        if len(dets) == 0:
            return dets
        xyxy = dets.xyxy.astype(np.float64).copy()
        tids = dets.tracker_id if dets.tracker_id is not None else np.full(len(dets), -1, dtype=int)
        a = self.alpha
        for i in range(len(dets)):
            x1, y1, x2, y2 = xyxy[i]
            cx = (x1 + x2) * 0.5
            width = float(x2 - x1)
            tid = int(tids[i])
            if tid >= 0 and width > 1.0:
                if tid in self._width:
                    width = a * width + (1.0 - a) * self._width[tid]
                self._width[tid] = width
            xyxy[i, 0] = cx - width * 0.5
            xyxy[i, 2] = cx + width * 0.5
        return sv.Detections(
            xyxy=xyxy.astype(np.float32),
            class_id=dets.class_id,
            tracker_id=dets.tracker_id,
            confidence=dets.confidence,
            data=dets.data,
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
