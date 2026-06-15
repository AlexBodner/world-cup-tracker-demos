"""Render team ellipses with Kalman joystick direction dots (no radar or pass overlays)."""

from __future__ import annotations

from typing import Callable, Iterator

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
    stabilize_goalkeeper_teams,
    stabilize_teams_by_tracklet,
)
from world_cup_projects.common.visual import annotate_kalman_motion_players

DetectionSource = Callable[..., Iterator[tuple[int, sv.Detections]]]


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
    tracker_kind: TrackerKind = "bytetrack",
    min_speed_px: float = 0.5,
    max_speed_px: float = 4.0,
    pitch_device: str = "cpu",
    smooth_alpha: float = 0.28,
    dot_smooth_alpha: float = 0.32,
    width_smooth_alpha: float = 0.22,
    team_flip_after: int = 16,
    show_speed: bool = True,
    min_speed_kmh: float = 4.0,
    speed_homography_weight: float = 0.3,
    speed_smooth_alpha: float = 0.22,
    pitch_confidence: float = 0.9,
) -> dict:
    """Write an MP4 with team ellipses and Kalman joystick dots on each frame."""
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
    frame_list = stabilize_teams_by_tracklet(frame_list, flip_after=team_flip_after)
    keypoints = _load_pitch_keypoints(
        sequence, frame_list, pitch_device=pitch_device
    )
    _stabilize_goalkeeper_teams(
        sequence, frame_list, pitch_device=pitch_device, keypoints=keypoints
    )
    transformers = _build_frame_transformers(
        keypoints, pitch_confidence=pitch_confidence
    )
    fps = float(sequence.frame_rate)

    use_cached_kalman = bool(
        frame_list and detections_have_kalman_velocity(frame_list[0][1])
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
            min_speed_kmh=min_speed_kmh,
            speed_homography_weight=speed_homography_weight,
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
        "min_speed_kmh": min_speed_kmh,
        "speed_homography_weight": speed_homography_weight,
        "speed_smooth_alpha": speed_smooth_alpha,
        "encoder": "h264" if use_h264_stream else "mp4v",
    }
