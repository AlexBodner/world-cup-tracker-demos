"""Debug overlay: smoothed ball velocity arrow + speed from raw detection positions."""

from __future__ import annotations

from typing import Callable, Iterator

import numpy as np
import supervision as sv

from world_cup_projects.common.carrier_motion import (
    MAX_BALL_PX_PER_FRAME,
    BallDirectionSmoother,
    BallPositionHistory,
    plausible_ball_speed_m_s,
)
from world_cup_projects.common.geometry import unit
from world_cup_projects.common.pitch import load_pitch_homography_cache
from world_cup_projects.common.possession import ball_xy
from world_cup_projects.common.teams import stabilize_teams_by_tracklet
from world_cup_projects.common.video import H264StreamWriter, SequentialVideoReader, read_sequence_frame
from world_cup_projects.common.visual import annotate_ball, annotate_players, draw_ball_speed_arrow, draw_hud_bar
from world_cup_projects.player_stats.pass_events import PassDetectionConfig, _active_carrier

DetectionSource = Callable[..., Iterator[tuple[int, sv.Detections]]]

IN_FLIGHT_LOOKBACK = 6
POSSESSION_LOOKBACK = 2
POSSESSION_BRIDGE_FRAMES = 4
MIN_POSSESSION_FRAMES = 2


def _speed_lookback(
    *,
    frame_idx: int,
    at_possession: bool,
    possession_start: int | None,
    default_lookback: int,
) -> int | None:
    """In-flight uses a long window; at-feet uses only frames since control began."""
    if not at_possession:
        return default_lookback
    if possession_start is None:
        return None
    span = frame_idx - possession_start + 1
    if span < MIN_POSSESSION_FRAMES:
        return None
    return min(POSSESSION_LOOKBACK, span)


def _load_pitch_transformers(
    sequence_name: str,
    *,
    pitch_device: str,
    pitch_confidence: float,
) -> dict[int, object]:
    """Full-clip pitch cache (partial ``end`` breaks cache lookup for 750-frame caches)."""
    maps = load_pitch_homography_cache(
        sequence_name,
        end=None,
        pitch_confidence=pitch_confidence,
        device=pitch_device,
    )
    if maps is None:
        return {}
    return maps.transforms


def render_ball_speed_debug_video(
    sequence,
    out_path: str,
    *,
    detections_source: DetectionSource,
    start_frame: int = 1,
    end_frame: int | None = None,
    lookback_frames: int = 6,
    smooth_alpha: float = 0.25,
    pitch_device: str = "cpu",
    pitch_confidence: float = 0.9,
    team_flip_after: int = 16,
) -> dict:
    """Write MP4 with ball speed arrow from raw ``ball_xy`` (no Kalman on ball)."""
    end = end_frame if end_frame is not None else sequence.length
    frame_list = list(detections_source(sequence, start=start_frame, end=end))
    frame_list = stabilize_teams_by_tracklet(frame_list, flip_after=team_flip_after)

    transformers = _load_pitch_transformers(
        sequence.name,
        pitch_device=pitch_device,
        pitch_confidence=pitch_confidence,
    )

    fps = float(sequence.frame_rate)
    touch_cfg = PassDetectionConfig().for_frame_rate(fps)
    ball_history = BallPositionHistory()
    direction_smoother = BallDirectionSmoother(alpha=smooth_alpha)
    possession_tid: int | None = None
    possession_start: int | None = None
    missing_ball_streak = 0
    reader = SequentialVideoReader(sequence.video_path)
    writer = H264StreamWriter(
        out_path,
        width=sequence.width,
        height=sequence.height,
        fps=fps,
    )

    try:
        for frame_idx, dets in frame_list:
            image = reader.read(frame_idx)
            if image is None:
                image = read_sequence_frame(sequence, frame_idx)
            if image is None:
                continue

            ball = ball_xy(dets)
            transformer = transformers.get(int(frame_idx))
            carrier, touch_kind = _active_carrier(
                dets, transformer=transformer, config=touch_cfg
            )
            carrier_tid = (
                int(dets.tracker_id[carrier.index])
                if carrier is not None and dets.tracker_id is not None
                else None
            )

            if ball is None:
                missing_ball_streak += 1
                if missing_ball_streak > POSSESSION_BRIDGE_FRAMES:
                    possession_tid = None
                    possession_start = None
                    direction_smoother.reset()
            else:
                missing_ball_streak = 0
                ball_history.record(frame_idx, ball)
                if carrier_tid is not None and touch_kind == "control":
                    if carrier_tid != possession_tid:
                        possession_tid = carrier_tid
                        possession_start = frame_idx
                        direction_smoother.reset()

            at_possession = (
                possession_tid is not None and carrier_tid == possession_tid
            )
            in_flight = not at_possession
            speed_lb = _speed_lookback(
                frame_idx=frame_idx,
                at_possession=at_possession,
                possession_start=possession_start,
                default_lookback=lookback_frames,
            )

            out = annotate_players(image.copy(), dets, show_tracker_ids=True)
            out = annotate_ball(out, dets)

            if ball is not None and speed_lb is not None:
                raw_delta, px_per_frame = ball_history.displacement(
                    frame_idx, lookback_frames=speed_lb
                )
                if in_flight:
                    direction_smoother.reset()
                if (
                    raw_delta is not None
                    and px_per_frame is not None
                    and px_per_frame <= MAX_BALL_PX_PER_FRAME
                ):
                    direction = unit(raw_delta) if in_flight else direction_smoother.update(raw_delta)
                    if direction is None:
                        direction = unit(raw_delta)
                    if direction is not None:
                        arrow_delta = direction * px_per_frame
                        speed_m_s = plausible_ball_speed_m_s(
                            ball_history.speed(
                                frame_idx,
                                lookback_frames=speed_lb,
                                fps=fps,
                                transformer=transformer,
                            )
                        )
                        speed_px_s = px_per_frame * fps
                        phase = "flight" if in_flight else "poss"

                        if speed_m_s is not None:
                            label = f"{speed_m_s:.1f} m/s ({phase})"
                        else:
                            label = f"{speed_px_s:.0f} px/s ({phase})"

                        out = draw_ball_speed_arrow(
                            out,
                            ball,
                            arrow_delta,
                            speed_label=label,
                            min_speed_px=0.8,
                        )

            homography_note = "pitch ok" if transformers else "no pitch cache"
            phase_note = "flight" if in_flight else f"poss #{possession_tid}"
            out = draw_hud_bar(
                out,
                f"Ball speed  f{frame_idx}  {phase_note}  {homography_note}",
            )
            writer.write(out)
    finally:
        reader.close()
        writer.close()

    return {
        "sequence": sequence.name,
        "frames": f"{start_frame}-{end}",
        "speed_source": "ball_xy lookback displacement + direction EMA",
        "ball_tracked": False,
        "lookback_frames": lookback_frames,
        "smooth_alpha": smooth_alpha,
        "pitch_transforms": len(transformers),
    }
