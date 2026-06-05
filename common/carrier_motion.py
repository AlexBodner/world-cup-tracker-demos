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
