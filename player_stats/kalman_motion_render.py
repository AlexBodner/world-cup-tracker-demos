"""Render team ellipses with Kalman joystick direction dots (no radar or pass overlays)."""

from __future__ import annotations

from typing import Callable, Iterator, Literal

import cv2
import numpy as np
import supervision as sv

from world_cup_projects.common.detect import (
    collect_referee_tracker_ids,
    filter_referees_from_detections,
)
from world_cup_projects.common.player_tracker import TrackerKind, create_player_tracker
from world_cup_projects.common.soccernet import ROLE_GOALKEEPER, ROLE_PLAYER
from world_cup_projects.common.tracking_facing import (
    EllipseWidthSmoother,
    JoystickDotSmoother,
    KalmanSpeedDisplaySmoother,
    KalmanVelocitySmoother,
    detections_have_kalman_velocity,
    kalman_ground_speed_m_s,
    kalman_velocity_arrays,
)
from world_cup_projects.common.video import (
    H264StreamWriter,
    SequentialVideoReader,
    read_sequence_frame,
)
from world_cup_projects.common.pitch import (
    ensure_pitch_homography_maps,
    load_pitch_homography_cache,
    warmup_goal_defenders_radar,
)
from world_cup_projects.common.teams import (
    enforce_one_goalkeeper_per_team_frames,
    lock_teams_by_tracklet_majority,
    stabilize_goalkeeper_teams,
)
from world_cup_projects.common.visual import annotate_kalman_motion_players

DetectionSource = Callable[..., Iterator[tuple[int, sv.Detections]]]
SpeedSource = Literal["kalman", "multilag"]


def _iter_frames(sequence, detections_source: DetectionSource, *, max_frames: int | None):
    end = max_frames if max_frames is not None else sequence.length
    yield from detections_source(sequence, start=1, end=end)


def _load_pitch_keypoints(
    sequence,
    frame_list: list[tuple[int, sv.Detections]],
    *,
    pitch_device: str,
) -> dict[int, object] | None:
    if not frame_list:
        return None
    clip_end = int(frame_list[-1][0])
    maps = load_pitch_homography_cache(sequence.name, end=clip_end, device=pitch_device)
    if maps is None:
        maps = ensure_pitch_homography_maps(
            sequence,
            device=pitch_device,
            detections_by_frame={fi: dets for fi, dets in frame_list},
        )
    return maps.keypoints


def _stabilize_goalkeeper_teams(
    sequence,
    frame_list: list[tuple[int, sv.Detections]],
    *,
    pitch_device: str,
    keypoints: dict[int, object] | None,
) -> None:
    """Assign stable defending-team colors to goalkeeper tracklets (mutates ``frame_list``)."""
    if not frame_list or keypoints is None:
        return
    locked = warmup_goal_defenders_radar(frame_list, keypoints)
    stabilize_goalkeeper_teams(
        frame_list,
        keypoints_by_frame=keypoints,
        locked_goal_defenders=locked,
    )


def _patch_goalkeeper_teams_if_needed(
    frame_idx: int,
    dets: sv.Detections,
    keypoints: dict[int, object] | None,
    *,
    pitch_confidence: float = 0.9,
) -> sv.Detections:
    """Per-frame fallback for untracked GKs (``tracker_id < 0``) missed by clip stabilization."""
    from world_cup_projects.common.pitch import homography_from_keypoints_radar
    from world_cup_projects.common.teams import apply_goalkeeper_teams_by_goal

    gk_mask = dets.class_id == ROLE_GOALKEEPER
    if not gk_mask.any() or dets.data is None or keypoints is None:
        return dets
    teams = np.array(dets.data.get("team", np.full(len(dets), -1)), dtype=int)
    if np.all(np.isin(teams[gk_mask], (0, 1))):
        return dets
    transformer = homography_from_keypoints_radar(
        keypoints.get(int(frame_idx)),
        confidence=pitch_confidence,
    )
    if transformer is None:
        return dets
    return apply_goalkeeper_teams_by_goal(dets, transformer)


def _build_frame_transformers(
    keypoints: dict[int, object] | None,
    *,
    pitch_confidence: float = 0.9,
) -> dict[int, object]:
    """Per-frame pitch homographies for metric Kalman speed."""
    from world_cup_projects.common.pitch import homography_from_keypoints_radar

    if not keypoints:
        return {}
    transformers: dict[int, object] = {}
    for frame_idx, kp in keypoints.items():
        transformer = homography_from_keypoints_radar(
            kp,
            confidence=pitch_confidence,
        )
        if transformer is not None:
            transformers[int(frame_idx)] = transformer
    return transformers


def _prepare_kalman_motion_frames(
    sequence,
    detections_source: DetectionSource,
    *,
    max_frames: int | None,
    team_flip_after: int,
    pitch_device: str,
    pitch_confidence: float,
) -> tuple[
    list[tuple[int, sv.Detections]],
    dict[int, object] | None,
    dict[int, object],
    float,
    bool,
]:
    """Load detections, stabilize teams, and build per-frame homographies."""
    frame_list = list(_iter_frames(sequence, detections_source, max_frames=max_frames))
    blocked_ref_tids = collect_referee_tracker_ids(frame_list)
    frame_list = [
        (
            int(fi),
            filter_referees_from_detections(
                dets, blocked_tracker_ids=blocked_ref_tids
            ),
        )
        for fi, dets in frame_list
    ]
    frame_list = lock_teams_by_tracklet_majority(frame_list)
    keypoints = _load_pitch_keypoints(
        sequence, frame_list, pitch_device=pitch_device
    )
    _stabilize_goalkeeper_teams(
        sequence, frame_list, pitch_device=pitch_device, keypoints=keypoints
    )
    frame_list = enforce_one_goalkeeper_per_team_frames(
        frame_list, frame_width=float(sequence.width)
    )
    transformers = _build_frame_transformers(
        keypoints, pitch_confidence=pitch_confidence
    )
    use_cached_kalman = bool(
        frame_list and detections_have_kalman_velocity(frame_list[0][1])
    )
    return frame_list, keypoints, transformers, float(sequence.frame_rate), use_cached_kalman


def _compute_multilag_speed_ms_lookup(
    frame_list: list[tuple[int, sv.Detections]],
    transformers: dict[int, object],
    fps: float,
    *,
    speed_k_frames: int = 5,
    min_frames: int = 2,
) -> dict[tuple[int, int], float]:
    """Multi-lag homography speed (m/s) per (tracker_id, frame_idx)."""
    from world_cup_projects.player_stats.speed_distance import (
        collect_tracks,
        compute_kinematics,
        speed_at_frame,
    )

    tracks = collect_tracks(frame_list)
    compute_kinematics(
        tracks,
        fps,
        mode="homography",
        frame_transforms=transformers,
        speed_k_frames=speed_k_frames,
        min_frames=min_frames,
        smooth_window=1,
    )
    lookup: dict[tuple[int, int], float] = {}
    for tid, track in tracks.items():
        for fi in track.frames:
            speed_m_s = speed_at_frame(track, fi)
            lookup[(int(tid), int(fi))] = float(speed_m_s) if speed_m_s is not None else 0.0
    return lookup


def _attach_multilag_speed_ms(
    dets: sv.Detections,
    frame_idx: int,
    lookup: dict[tuple[int, int], float],
) -> sv.Detections:
    n = len(dets)
    speed_ms = np.zeros(n, dtype=np.float32)
    tids = dets.tracker_id
    if tids is not None:
        for i in range(n):
            tid = int(tids[i])
            if tid >= 0:
                speed_ms[i] = lookup.get((tid, int(frame_idx)), 0.0)
    data = dict(dets.data) if dets.data else {}
    data["speed_ms"] = speed_ms
    return sv.Detections(
        xyxy=dets.xyxy,
        class_id=dets.class_id,
        tracker_id=dets.tracker_id,
        confidence=dets.confidence,
        data=data,
    )


def _attach_kalman_velocity(
    dets: sv.Detections,
    player_tracker,
    *,
    needs_frame: bool,
    image: np.ndarray | None,
) -> sv.Detections:
    """Replay one tracker step and merge ``kf_vx`` / ``kf_vy`` onto full-frame detections."""
    pmask = np.isin(dets.class_id, (ROLE_PLAYER, ROLE_GOALKEEPER))
    n = len(dets)
    kf_vx = np.full(n, np.nan, dtype=np.float32)
    kf_vy = np.full(n, np.nan, dtype=np.float32)
    if pmask.any():
        trackable = dets[pmask]
        player_tracker.update(
            trackable,
            frame=image if needs_frame else None,
        )
        vx_sub, vy_sub = kalman_velocity_arrays(trackable, player_tracker)
        kf_vx[pmask] = vx_sub
        kf_vy[pmask] = vy_sub
    data = dict(dets.data) if dets.data else {}
    data["kf_vx"] = kf_vx
    data["kf_vy"] = kf_vy
    return sv.Detections(
        xyxy=dets.xyxy,
        class_id=dets.class_id,
        tracker_id=dets.tracker_id,
        confidence=dets.confidence,
        data=data,
    )


def render_kalman_motion_video(
    sequence,
    out_path: str,
    *,
    detections_source: DetectionSource,
    max_frames: int | None = None,
    tracker_kind: TrackerKind = "botsort",
    min_speed_px: float = 0.5,
    max_speed_px: float = 4.0,
    pitch_device: str = "cpu",
    smooth_alpha: float = 0.28,
    dot_smooth_alpha: float = 0.32,
    width_smooth_alpha: float = 0.22,
    team_flip_after: int = 16,
    show_speed: bool = True,
    min_speed_ms: float = 0.0,
    speed_smooth_alpha: float = 0.22,
    max_speed_labels: int = 0,
    pitch_confidence: float = 0.9,
    speed_source: SpeedSource = "kalman",
    speed_k_frames: int = 5,
    prepared_frames: tuple[
        list[tuple[int, sv.Detections]],
        dict[int, object] | None,
        dict[int, object],
        float,
        bool,
    ]
    | None = None,
) -> dict:
    """Write an MP4 with team ellipses and Kalman joystick dots on each frame."""
    if prepared_frames is None:
        prepared_frames = _prepare_kalman_motion_frames(
            sequence,
            detections_source,
            max_frames=max_frames,
            team_flip_after=team_flip_after,
            pitch_device=pitch_device,
            pitch_confidence=pitch_confidence,
        )
    frame_list, keypoints, transformers, fps, use_cached_kalman = prepared_frames

    multilag_lookup: dict[tuple[int, int], float] | None = None
    if show_speed and speed_source == "multilag":
        multilag_lookup = _compute_multilag_speed_ms_lookup(
            frame_list,
            transformers,
            fps,
            speed_k_frames=speed_k_frames,
        )

    player_tracker = None
    needs_frame = tracker_kind == "botsort"
    if not use_cached_kalman:
        player_tracker = create_player_tracker(sequence.frame_rate, kind=tracker_kind)

    video_reader: SequentialVideoReader | None = None
    if getattr(sequence, "video_path", None) is not None:
        video_reader = SequentialVideoReader(sequence.video_path)

    def _load_frame_image(frame_idx: int) -> np.ndarray:
        if video_reader is not None:
            image = video_reader.read(frame_idx)
        else:
            image = read_sequence_frame(sequence, frame_idx)
        if image is None:
            image = np.full((sequence.height, sequence.width, 3), 30, np.uint8)
        return image

    try:
        writer: H264StreamWriter | cv2.VideoWriter = H264StreamWriter(
            out_path,
            width=sequence.width,
            height=sequence.height,
            fps=sequence.frame_rate,
        )
        use_h264_stream = True
    except RuntimeError:
        use_h264_stream = False
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            out_path,
            fourcc,
            sequence.frame_rate,
            (sequence.width, sequence.height),
        )

    velocity_smoother = KalmanVelocitySmoother(alpha=smooth_alpha)
    dot_smoother = JoystickDotSmoother(alpha=dot_smooth_alpha)
    speed_smoother = (
        KalmanSpeedDisplaySmoother(alpha=speed_smooth_alpha)
        if show_speed
        else None
    )
    width_smoother = EllipseWidthSmoother(alpha=width_smooth_alpha)
    n_frames = 0
    n_dots = 0
    for frame_idx, dets in frame_list:
        image = _load_frame_image(frame_idx)
        dets = _patch_goalkeeper_teams_if_needed(frame_idx, dets, keypoints)
        if player_tracker is not None:
            dets = _attach_kalman_velocity(
                dets,
                player_tracker,
                needs_frame=needs_frame,
                image=image,
            )
        dets = velocity_smoother.smooth_detections(dets)
        dets = width_smoother.smooth_detections(dets)
        if multilag_lookup is not None:
            dets = _attach_multilag_speed_ms(dets, frame_idx, multilag_lookup)

        out = annotate_kalman_motion_players(
            image,
            dets,
            min_speed_px=min_speed_px,
            max_speed_px=max_speed_px,
            dot_smoother=dot_smoother,
            speed_smoother=speed_smoother,
            transformer=transformers.get(int(frame_idx)),
            fps=fps,
            show_speed=show_speed,
            min_speed_ms=min_speed_ms,
            max_speed_labels=max_speed_labels,
            speed_source=speed_source,
        )
        if dets.data is not None:
            teams = dets.data.get("team", np.full(len(dets), -1))
            pmask = np.isin(dets.class_id, (ROLE_PLAYER, ROLE_GOALKEEPER))
            n_dots += int(np.sum(pmask & np.isin(teams, (0, 1))))

        writer.write(out)
        n_frames += 1

    if isinstance(writer, H264StreamWriter):
        writer.close()
    else:
        writer.release()

    if video_reader is not None:
        video_reader.close()

    return {
        "sequence": sequence.name,
        "frames": n_frames,
        "dots_drawn": n_dots,
        "min_speed_px": min_speed_px,
        "max_speed_px": max_speed_px,
        "kalman_source": "cached" if use_cached_kalman else "replay",
        "smooth_alpha": smooth_alpha,
        "dot_smooth_alpha": dot_smooth_alpha,
        "width_smooth_alpha": width_smooth_alpha,
        "team_flip_after": team_flip_after,
        "show_speed": show_speed,
        "min_speed_ms": min_speed_ms,
        "max_speed_labels": max_speed_labels,
        "speed_smooth_alpha": speed_smooth_alpha,
        "team_lock": "majority",
        "speed_source": speed_source,
        "speed_k_frames": speed_k_frames if speed_source == "multilag" else None,
        "encoder": "h264" if use_h264_stream else "mp4v",
    }


def _summarize_speed_pairs(
    kalman_ms: list[float],
    multilag_ms: list[float],
) -> dict:
    k = np.asarray(kalman_ms, dtype=np.float64)
    m = np.asarray(multilag_ms, dtype=np.float64)
    diff = k - m
    return {
        "samples": int(len(k)),
        "kalman_ms": {
            "median": float(np.median(k)),
            "mean": float(np.mean(k)),
            "p90": float(np.percentile(k, 90)),
        },
        "multilag_ms": {
            "median": float(np.median(m)),
            "mean": float(np.mean(m)),
            "p90": float(np.percentile(m, 90)),
        },
        "kalman_minus_multilag_ms": {
            "median": float(np.median(diff)),
            "mean": float(np.mean(diff)),
            "p90": float(np.percentile(diff, 90)),
        },
    }


def _collect_speed_comparison_pairs(
    frame_list: list[tuple[int, sv.Detections]],
    transformers: dict[int, object],
    fps: float,
    *,
    tracker_kind: TrackerKind,
    use_cached_kalman: bool,
    smooth_alpha: float,
    speed_k_frames: int,
    min_moving_ms: float = 0.3,
) -> dict:
    """Pair Kalman vs multi-lag m/s on the same player-frames (after velocity EMA)."""
    from world_cup_projects.common.possession import feet_xy

    multilag_lookup = _compute_multilag_speed_ms_lookup(
        frame_list,
        transformers,
        fps,
        speed_k_frames=speed_k_frames,
    )
    player_tracker = None
    needs_frame = tracker_kind == "botsort"
    if not use_cached_kalman:
        player_tracker = create_player_tracker(fps, kind=tracker_kind)

    velocity_smoother = KalmanVelocitySmoother(alpha=smooth_alpha)
    kalman_ms: list[float] = []
    multilag_ms: list[float] = []

    for frame_idx, dets in frame_list:
        if player_tracker is not None:
            dets = _attach_kalman_velocity(
                dets,
                player_tracker,
                needs_frame=needs_frame,
                image=None,
            )
        dets = velocity_smoother.smooth_detections(dets)
        if dets.data is None:
            continue
        kf_vx = dets.data.get("kf_vx")
        kf_vy = dets.data.get("kf_vy")
        if kf_vx is None or kf_vy is None:
            continue
        transformer = transformers.get(int(frame_idx))
        feet = feet_xy(dets)
        teams = dets.data.get("team", np.full(len(dets), -1))
        pmask = np.isin(dets.class_id, (ROLE_PLAYER, ROLE_GOALKEEPER))
        tids = dets.tracker_id if dets.tracker_id is not None else np.full(len(dets), -1)
        for i in np.flatnonzero(pmask):
            if int(teams[i]) not in (0, 1):
                continue
            tid = int(tids[i])
            if tid < 0:
                continue
            vx, vy = float(kf_vx[i]), float(kf_vy[i])
            if not np.isfinite(vx) or not np.isfinite(vy):
                continue
            speed_m_s = kalman_ground_speed_m_s(
                feet[i],
                np.array([vx, vy], dtype=np.float64),
                transformer,
                fps=fps,
            )
            if speed_m_s is None:
                continue
            k_ms = float(speed_m_s)
            m_ms = multilag_lookup.get((tid, int(frame_idx)), 0.0)
            if k_ms < min_moving_ms and m_ms < min_moving_ms:
                continue
            kalman_ms.append(k_ms)
            multilag_ms.append(m_ms)

    return _summarize_speed_pairs(kalman_ms, multilag_ms)


def compare_kalman_motion_speeds(
    sequence,
    out_dir: str,
    *,
    detections_source: DetectionSource,
    max_frames: int | None = None,
    tracker_kind: TrackerKind = "botsort",
    pitch_device: str = "cpu",
    team_flip_after: int = 16,
    pitch_confidence: float = 0.9,
    speed_k_frames: int = 5,
    **render_kwargs,
) -> dict:
    """Render Kalman and multi-lag label videos plus a paired speed summary."""
    from pathlib import Path

    prepared = _prepare_kalman_motion_frames(
        sequence,
        detections_source,
        max_frames=max_frames,
        team_flip_after=team_flip_after,
        pitch_device=pitch_device,
        pitch_confidence=pitch_confidence,
    )
    frame_list, _, transformers, fps, use_cached_kalman = prepared
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    stem = f"kalman_motion_{sequence.name}"
    out_kalman = str(out_root / f"{stem}_kalman.mp4")
    out_multilag = str(out_root / f"{stem}_multilag.mp4")
    stats_path = str(out_root / f"{stem}_speed_compare.json")

    shared = {
        "detections_source": detections_source,
        "max_frames": max_frames,
        "tracker_kind": tracker_kind,
        "pitch_device": pitch_device,
        "team_flip_after": team_flip_after,
        "pitch_confidence": pitch_confidence,
        "prepared_frames": prepared,
        **render_kwargs,
    }
    manifest_kalman = render_kalman_motion_video(
        sequence,
        out_kalman,
        speed_source="kalman",
        **shared,
    )
    manifest_multilag = render_kalman_motion_video(
        sequence,
        out_multilag,
        speed_source="multilag",
        speed_k_frames=speed_k_frames,
        **shared,
    )
    comparison = _collect_speed_comparison_pairs(
        frame_list,
        transformers,
        fps,
        tracker_kind=tracker_kind,
        use_cached_kalman=use_cached_kalman,
        smooth_alpha=render_kwargs.get("smooth_alpha", 0.28),
        speed_k_frames=speed_k_frames,
    )
    result = {
        "sequence": sequence.name,
        "speed_k_frames": speed_k_frames,
        "outputs": {
            "kalman": out_kalman,
            "multilag": out_multilag,
            "stats": stats_path,
        },
        "comparison": comparison,
        "kalman_manifest": manifest_kalman,
        "multilag_manifest": manifest_multilag,
    }
    Path(stats_path).write_text(
        __import__("json").dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    return result
