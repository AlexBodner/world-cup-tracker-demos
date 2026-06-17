"""Render tracked players only — team ellipses, tracker IDs, ball (no motion overlay)."""

from __future__ import annotations

from typing import Callable, Iterator

import cv2
import numpy as np
import supervision as sv

from world_cup_projects.common.video import (
    H264StreamWriter,
    SequentialVideoReader,
    read_sequence_frame,
)
from world_cup_projects.common.visual import annotate_ball, annotate_tracking_players
from world_cup_projects.player_stats.kalman_motion_render import (
    _patch_goalkeeper_teams_if_needed,
    _prepare_kalman_motion_frames,
)

DetectionSource = Callable[..., Iterator[tuple[int, sv.Detections]]]


def render_tracking_video(
    sequence,
    out_path: str,
    *,
    detections_source: DetectionSource,
    max_frames: int | None = None,
    pitch_device: str = "cpu",
    pitch_confidence: float = 0.9,
    team_flip_after: int = 16,
) -> dict:
    """Write an MP4 with per-track colors and tracker IDs (no Kalman / radar)."""
    frame_list, keypoints, _, _, _ = _prepare_kalman_motion_frames(
        sequence,
        detections_source,
        max_frames=max_frames,
        team_flip_after=team_flip_after,
        pitch_device=pitch_device,
        pitch_confidence=pitch_confidence,
    )

    video_reader: SequentialVideoReader | None = None
    if getattr(sequence, "video_path", None) is not None:
        video_reader = SequentialVideoReader(sequence.video_path)

    def _load_frame_image(frame_idx: int):
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

    n_frames = 0
    for frame_idx, dets in frame_list:
        image = _load_frame_image(frame_idx)
        dets = _patch_goalkeeper_teams_if_needed(
            frame_idx, dets, keypoints, pitch_confidence=pitch_confidence
        )
        out = annotate_tracking_players(image, dets)
        out = annotate_ball(out, dets)
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
        "team_lock": "majority",
        "gk_dedup": "one_per_team",
        "color_mode": "tracker_id",
        "encoder": "h264" if use_h264_stream else "mp4v",
    }
