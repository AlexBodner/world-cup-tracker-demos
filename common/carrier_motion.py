"""Estimate carrier run direction and ball motion from recent GT positions."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import supervision as sv

from world_cup_projects.common.geometry import unit
from world_cup_projects.common.possession import Carrier, player_mask


class TrackPositionHistory:
    """Rolling ``(frame_idx, xy)`` samples per ``tracker_id``."""

    def __init__(self, *, max_samples: int = 10) -> None:
        self._max_samples = max_samples
        self._samples: dict[int, list[tuple[int, np.ndarray]]] = defaultdict(list)

    def record_frame(
        self, frame_idx: int, detections: sv.Detections, positions: np.ndarray
    ) -> None:
        if detections.tracker_id is None:
            return
        pmask = player_mask(detections)
        for i in np.flatnonzero(pmask):
            tid = int(detections.tracker_id[i])
            if tid < 0:
                continue
            buf = self._samples[tid]
            buf.append((int(frame_idx), np.asarray(positions[i], dtype=np.float64)))
            if len(buf) > self._max_samples:
                del buf[: len(buf) - self._max_samples]

    def motion_direction(
        self,
        detections: sv.Detections,
        carrier: Carrier,
        frame_idx: int,
        *,
        lookback_frames: int = 5,
        min_displacement: float = 3.0,
    ) -> np.ndarray | None:
        """Unit displacement over recent frames ending at ``frame_idx`` (exclusive)."""
        if detections.tracker_id is None:
            return None
        tid = int(detections.tracker_id[carrier.index])
        if tid < 0:
            return None
        buf = [
            (f, p)
            for f, p in self._samples.get(tid, [])
            if frame_idx - lookback_frames <= f < frame_idx
        ]
        if len(buf) < 2:
            return None
        f0, p0 = buf[0]
        f1, p1 = buf[-1]
        if f1 <= f0:
            return None
        delta = p1 - p0
        dist = float(np.linalg.norm(delta))
        if dist < min_displacement:
            return None
        return unit(delta)

    def player_facing(
        self,
        detections: sv.Detections,
        frame_idx: int,
        *,
        lookback_frames: int = 4,
        min_displacement: float = 2.0,
    ) -> np.ndarray:
        """Per-row unit facing vectors from recent tracker motion (NaN if unknown)."""
        n = len(detections)
        out = np.full((n, 2), np.nan, dtype=np.float64)
        if detections.tracker_id is None or n == 0:
            return out
        for i in range(n):
            tid = int(detections.tracker_id[i])
            if tid < 0:
                continue
            buf = [
                (f, p)
                for f, p in self._samples.get(tid, [])
                if frame_idx - lookback_frames <= f < frame_idx
            ]
            if len(buf) < 2:
                continue
            delta = buf[-1][1] - buf[0][1]
            if float(np.linalg.norm(delta)) < min_displacement:
                continue
            out[i] = unit(delta)
        return out


class BallPositionHistory:
    """Recent ball ground positions for speed when picking freeze frames."""

    def __init__(self, *, max_samples: int = 12) -> None:
        self._max_samples = max_samples
        self._samples: list[tuple[int, np.ndarray]] = []

    def record(self, frame_idx: int, ball: np.ndarray | None) -> None:
        if ball is None:
            return
        self._samples.append((int(frame_idx), np.asarray(ball, dtype=np.float64)))
        if len(self._samples) > self._max_samples:
            self._samples = self._samples[-self._max_samples :]

    def speed(
        self,
        frame_idx: int,
        *,
        lookback_frames: int,
        fps: float,
        transformer=None,
    ) -> float | None:
        """Ball speed over the lookback window (m/s with homography, else px/s)."""
        if fps <= 0 or lookback_frames < 1:
            return None
        window = [
            (f, p)
            for f, p in self._samples
            if frame_idx - lookback_frames <= f <= frame_idx
        ]
        if len(window) < 2:
            return None
        f0, p0 = window[0]
        f1, p1 = window[-1]
        if f1 <= f0:
            return None
        dt = (f1 - f0) / fps
        if dt <= 0:
            return None

        if transformer is not None:
            from world_cup_projects.common.pitch import image_to_pitch_m

            pts = np.stack([p0, p1], axis=0).astype(np.float32)
            pitch = image_to_pitch_m(pts, transformer)
            if pitch is None:
                return None
            dist = float(np.linalg.norm(pitch[1] - pitch[0]))
        else:
            dist = float(np.linalg.norm(p1 - p0))
        return dist / dt

    def displacement(
        self,
        frame_idx: int,
        *,
        lookback_frames: int,
    ) -> tuple[np.ndarray | None, float | None]:
        """``(delta_xy, speed_px_per_frame)`` over the lookback window ending at ``frame_idx``."""
        window = [
            (f, p)
            for f, p in self._samples
            if frame_idx - lookback_frames <= f <= frame_idx
        ]
        if len(window) < 2:
            return None, None
        f0, p0 = window[0]
        f1, p1 = window[-1]
        if f1 <= f0:
            return None, None
        delta = p1 - p0
        speed_px_per_frame = float(np.linalg.norm(delta)) / (f1 - f0)
        return delta, speed_px_per_frame


# Footballs rarely exceed ~35 m/s even on powerful shots; higher ⇒ bad homography.
MAX_PLAUSIBLE_BALL_SPEED_M_S = 35.0
# Raw ball bbox jitter / re-detect teleports (image px per frame, ~25 fps).
MAX_BALL_PX_PER_FRAME = 22.0


class BallDirectionSmoother:
    """EMA on unit direction only — magnitude comes from the lookback window."""

    def __init__(self, *, alpha: float = 0.25) -> None:
        self.alpha = float(alpha)
        self._direction: np.ndarray | None = None

    def update(self, delta: np.ndarray | None) -> np.ndarray | None:
        from world_cup_projects.common.geometry import unit

        if delta is None:
            return self._direction
        direction = unit(delta)
        if direction is None:
            return self._direction
        if self._direction is None:
            self._direction = direction.copy()
        else:
            blended = self.alpha * direction + (1.0 - self.alpha) * self._direction
            self._direction = unit(blended)
        return self._direction

    def reset(self) -> None:
        self._direction = None


class BallVelocitySmoother:
    """EMA on ball velocity vectors for debug overlays (raw detections are jittery)."""

    def __init__(self, *, alpha: float = 0.22) -> None:
        self.alpha = float(alpha)
        self._velocity: np.ndarray | None = None

    def update(self, velocity: np.ndarray | None) -> np.ndarray | None:
        if velocity is None:
            return self._velocity
        v = np.asarray(velocity, dtype=np.float64)
        if self._velocity is None:
            self._velocity = v.copy()
        else:
            self._velocity = self.alpha * v + (1.0 - self.alpha) * self._velocity
        return self._velocity

    def reset(self) -> None:
        self._velocity = None


def plausible_ball_speed_m_s(speed_m_s: float | None) -> float | None:
    """Drop homography outliers that read as hundreds of m/s."""
    if speed_m_s is None or speed_m_s <= 0:
        return None
    if speed_m_s > MAX_PLAUSIBLE_BALL_SPEED_M_S:
        return None
    return speed_m_s
