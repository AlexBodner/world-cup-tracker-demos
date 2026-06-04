"""Per-player speed and distance from tracked detections.

Two calibration modes turn pixel motion into meters:

* **height** - bbox-height local scale (no pitch model).
* **homography** - per-frame pitch keypoints (confidence-filtered) + RANSAC H.

**Homography / warp model (no matrix chain):** we do *not* multiply
``H_j @ H_{j-1}^{-1}`` across frames. Each frame has its own image→pitch map
``H_j`` from pitch keypoints. For a step from frame ``i-1`` to ``i``:

* feet → pitch: ``p_{i-1} = H_{i-1}(xy_{i-1})``, ``p_i = H_i(xy_i)`` (distance)

* instantaneous speed on that step: ``‖p_i - p_{i-1}‖ / Δt`` with the same
  per-step warp (each endpoint in its frame's metric)

Displayed speed at frame ``j`` is the **median** of ``K`` multi-scale velocities:
``v_{j,j-1}, v_{j,j-2}, …, v_{j,j-K}`` where each ``v_{j,j-l} = ‖H_j(xy_j) -
H_{j-l}(xy_{j-l})‖ / Δt_{j-l→j}`` (per-step warps, not an H chain). Longer lags
damp 1-frame homography noise; short lags keep responsiveness. Light image-space
feet smoothing first, then median aggregation. Homography glitch steps are
**excluded** via per-track adaptive thresholds (not a display speed cap).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import supervision as sv

from world_cup_projects.common.possession import feet_xy, player_mask
from world_cup_projects.common.soccernet import TEAM_NONE

PLAYER_HEIGHT_M = 1.8
DEFAULT_SPEED_K_FRAMES = 15
HOMOGRAPHY_XY_SMOOTH = 5  # image feet, speed path only
HOMOGRAPHY_SPEED_SMOOTH = 11
# ~45 km/h — only excludes non-physical homography spikes, not normal sprint peaks.
MAX_PHYSICAL_STEP_MS = 12.5


@dataclass
class PlayerTrack:
    track_id: int
    team: int = TEAM_NONE
    frames: list[int] = field(default_factory=list)
    xy: list[tuple[float, float]] = field(default_factory=list)
    box_h: list[float] = field(default_factory=list)
    speed_ms: np.ndarray | None = None
    distance_m: float = 0.0
    top_speed_ms: float = 0.0


def collect_tracks(detections_iter) -> dict[int, PlayerTrack]:
    """Accumulate per-track feet positions + box heights from a detections iterator."""
    tracks: dict[int, PlayerTrack] = {}
    for frame_idx, dets in detections_iter:
        if dets.tracker_id is None or len(dets) == 0:
            continue
        pmask = player_mask(dets)
        if not pmask.any():
            continue
        feet = feet_xy(dets)
        heights = dets.xyxy[:, 3] - dets.xyxy[:, 1]
        teams = dets.data.get("team", np.full(len(dets), TEAM_NONE))
        for i in np.flatnonzero(pmask):
            tid = int(dets.tracker_id[i])
            if tid < 0:
                continue
            track = tracks.setdefault(tid, PlayerTrack(tid, int(teams[i])))
            track.frames.append(int(frame_idx))
            track.xy.append((float(feet[i, 0]), float(feet[i, 1])))
            track.box_h.append(float(heights[i]))
    return tracks


def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) < 2 or window <= 1:
        return values
    pad = window // 2
    padded = np.pad(values, pad, mode="edge")
    return np.array([np.median(padded[i:i + window]) for i in range(len(values))])


def _smooth_xy(xy: np.ndarray, window: int) -> np.ndarray:
    """Moving-average smooth an (N, 2) trajectory."""
    if len(xy) < 2 or window <= 1:
        return xy
    pad = window // 2
    out = xy.copy()
    for k in range(2):
        col = xy[:, k]
        padded = np.pad(col, pad, mode="edge")
        out[:, k] = np.array(
            [np.mean(padded[i:i + window]) for i in range(len(col))]
        )
    return out


def _to_pitch_m(transformer, xy: np.ndarray) -> np.ndarray | None:
    if transformer is None:
        return None
    pt = transformer.transform_points(xy.reshape(1, 2).astype(np.float32))[0]
    return pt / 100.0


def _transformer_at(
    frames: np.ndarray,
    frame_transforms: dict,
    index: int,
    *,
    transform=None,
):
    transforms = frame_transforms or {}
    return transforms.get(int(frames[index]), transform)


def _instantaneous_speed_homography(
    xy: np.ndarray,
    frames: np.ndarray,
    frame_transforms: dict,
    index: int,
    fps: float,
    *,
    transform=None,
) -> float | None:
    """Speed for one frame step ``index-1 → index`` using each frame's own H."""
    if index < 1:
        return None
    t0 = _transformer_at(frames, frame_transforms, index - 1, transform=transform)
    t1 = _transformer_at(frames, frame_transforms, index, transform=transform)
    if t0 is None or t1 is None:
        return None
    p0 = _to_pitch_m(t0, xy[index - 1])
    p1 = _to_pitch_m(t1, xy[index])
    if p0 is None or p1 is None:
        return None
    dt = (int(frames[index]) - int(frames[index - 1])) / fps
    if dt <= 0:
        return None
    return float(np.linalg.norm(p1 - p0) / dt)


def _lag_speed_homography(
    xy: np.ndarray,
    frames: np.ndarray,
    frame_transforms: dict,
    j: int,
    lag: int,
    fps: float,
    *,
    transform=None,
) -> float | None:
    """Average speed from frame ``j-lag`` to ``j`` using each endpoint's H."""
    j0 = j - lag
    if j0 < 0:
        return None
    t0 = _transformer_at(frames, frame_transforms, j0, transform=transform)
    t1 = _transformer_at(frames, frame_transforms, j, transform=transform)
    if t0 is None or t1 is None:
        return None
    p0 = _to_pitch_m(t0, xy[j0])
    p1 = _to_pitch_m(t1, xy[j])
    if p0 is None or p1 is None:
        return None
    dt = (int(frames[j]) - int(frames[j0])) / fps
    if dt <= 0:
        return None
    return float(np.linalg.norm(p1 - p0) / dt)


def _speed_multi_lag_mean_homography(
    xy: np.ndarray,
    frames: np.ndarray,
    frame_transforms: dict,
    fps: float,
    k: int,
    *,
    transform=None,
    max_step_ms: float | None = None,
) -> np.ndarray:
    """At frame ``j``, median of ``v_{j,j-1} … v_{j,j-K}`` (multi-scale robust estimate)."""
    n = len(frames)
    speed = np.zeros(n, dtype=np.float64)
    if n < 2 or k < 1:
        return speed
    for j in range(1, n):
        lags: list[float] = []
        for lag in range(1, min(k, j) + 1):
            v = _lag_speed_homography(
                xy, frames, frame_transforms, j, lag, fps, transform=transform
            )
            if v is None:
                continue
            if max_step_ms is not None and v > max_step_ms:
                continue
            lags.append(v)
        if lags:
            speed[j] = float(np.median(lags))
    return speed


def _homography_step_lengths_per_frame_h(
    xy: np.ndarray,
    frames: np.ndarray,
    frame_transforms: dict,
    *,
    transform=None,
) -> np.ndarray:
    """Step length for ``j-1 → j`` with ``H_{j-1}`` and ``H_j`` (not a chained warp)."""
    n = len(frames)
    steps = np.zeros(max(n - 1, 0), dtype=np.float64)
    for j in range(1, n):
        t0 = _transformer_at(frames, frame_transforms, j - 1, transform=transform)
        t1 = _transformer_at(frames, frame_transforms, j, transform=transform)
        if t0 is None or t1 is None:
            continue
        p0 = _to_pitch_m(t0, xy[j - 1])
        p1 = _to_pitch_m(t1, xy[j])
        if p0 is None or p1 is None:
            continue
        steps[j - 1] = float(np.linalg.norm(p1 - p0))
    return steps


def _speed_multi_lag_mean(
    positions: np.ndarray,
    frames: np.ndarray,
    fps: float,
    k: int,
    *,
    max_step_ms: float | None = None,
) -> np.ndarray:
    """Median of ``v_{j,j-1} … v_{j,j-K}`` in metric space (height mode)."""
    n = len(frames)
    speed = np.zeros(n, dtype=np.float64)
    if n < 2 or k < 1:
        return speed
    for j in range(1, n):
        lags: list[float] = []
        for lag in range(1, min(k, j) + 1):
            j0 = j - lag
            if np.any(np.isnan(positions[j])) or np.any(np.isnan(positions[j0])):
                continue
            dt = (int(frames[j]) - int(frames[j0])) / fps
            if dt <= 0:
                continue
            v = float(np.linalg.norm(positions[j] - positions[j0]) / dt)
            if max_step_ms is not None and v > max_step_ms:
                continue
            lags.append(v)
        if lags:
            speed[j] = float(np.median(lags))
    return speed


def _credible_speed_threshold_ms(inst_speed: np.ndarray) -> float:
    """Upper bound for plausible step speeds on this track (adaptive + physical ceiling)."""
    v = inst_speed[np.isfinite(inst_speed) & (inst_speed > 0.05)]
    if len(v) < 5:
        return MAX_PHYSICAL_STEP_MS
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med)))
    return max(MAX_PHYSICAL_STEP_MS, med + max(3.5 * mad, 3.0))


def _height_positions_m(xy: np.ndarray, box_h: np.ndarray) -> np.ndarray:
    mpp = PLAYER_HEIGHT_M / np.clip(box_h, 1.0, None)
    return xy * mpp[:, np.newaxis]


def compute_kinematics(
    tracks: dict[int, PlayerTrack],
    fps: float,
    *,
    mode: str = "height",
    transform=None,
    frame_transforms: dict[int, object] | None = None,
    smooth_window: int = 7,
    min_frames: int = 10,
    speed_k_frames: int = DEFAULT_SPEED_K_FRAMES,
) -> dict[int, PlayerTrack]:
    for track in tracks.values():
        if len(track.frames) < min_frames:
            track.speed_ms = np.zeros(len(track.frames))
            continue

        xy = np.asarray(track.xy, dtype=np.float64)
        frames = np.asarray(track.frames)
        box_h = np.asarray(track.box_h, dtype=np.float64)
        transforms = frame_transforms or {}

        if mode == "homography" and (transforms or transform is not None):
            xy_h = _smooth_xy(xy, HOMOGRAPHY_XY_SMOOTH)
            step_m = _homography_step_lengths_per_frame_h(
                xy_h, frames, transforms, transform=transform
            )
            dt = np.clip(np.diff(frames), 1, None) / fps
            inst_speed = np.divide(
                step_m, dt, out=np.zeros_like(step_m), where=dt > 0
            )
            step_thresh = _credible_speed_threshold_ms(inst_speed)
            credible = inst_speed <= step_thresh
            step_m = np.where(credible, step_m, 0.0)
            speed = _speed_multi_lag_mean_homography(
                xy_h,
                frames,
                transforms,
                fps,
                speed_k_frames,
                transform=transform,
                max_step_ms=step_thresh,
            )
            speed_smooth = max(smooth_window, HOMOGRAPHY_SPEED_SMOOTH)
        else:
            mpp = PLAYER_HEIGHT_M / np.clip(box_h, 1.0, None)
            step_px = np.linalg.norm(np.diff(xy, axis=0), axis=1)
            step_mpp = (mpp[:-1] + mpp[1:]) / 2.0
            step_m = step_px * step_mpp
            dt = np.clip(np.diff(frames), 1, None) / fps
            inst_speed = np.divide(
                step_m, dt, out=np.zeros_like(step_m), where=dt > 0
            )
            step_thresh = _credible_speed_threshold_ms(inst_speed)
            credible = inst_speed <= step_thresh
            step_m = np.where(credible, step_m, 0.0)
            speed_pos = _height_positions_m(xy, box_h)
            speed = _speed_multi_lag_mean(
                speed_pos, frames, fps, speed_k_frames, max_step_ms=step_thresh
            )
            speed_smooth = smooth_window

        speed = np.clip(speed, 0, None)
        speed = _smooth(speed, speed_smooth)

        track.speed_ms = speed
        track.distance_m = float(np.sum(step_m))
        positive = speed[speed > 0.3]
        track.top_speed_ms = (
            float(np.percentile(positive, 90)) if len(positive) >= 5 else float(np.max(speed))
        )
    return tracks


def speed_at_frame(track: PlayerTrack, frame_idx: int) -> float | None:
    if track.speed_ms is None or frame_idx not in track.frames:
        return None
    return float(track.speed_ms[track.frames.index(frame_idx)])


MS_TO_KMH = 3.6


def format_speed_kmh(ms: float, *, decimals: int = 1) -> str:
    """User-facing speed label (``ms`` is meters per second; displays km/h)."""
    return f"{ms * MS_TO_KMH:.{decimals}f} km/h"


def format_speed_ms(ms: float, *, decimals: int = 1) -> str:
    """Alias for :func:`format_speed_kmh` (internal kinematics stay in m/s)."""
    return format_speed_kmh(ms, decimals=decimals)
