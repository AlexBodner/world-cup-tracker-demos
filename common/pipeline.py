"""Shared analysis pipeline for pass-network and explain scripts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import supervision as sv

from world_cup_projects.common.pitch import (
    ensure_pitch_homography_maps,
    warmup_goal_defenders_radar,
)
from world_cup_projects.common.soccernet import iter_gt_detections


@dataclass
class MetricContext:
    transforms: dict[int, Any]
    radar_transforms: dict[int, Any]
    keypoints: dict[int, Any]
    locked_goals: tuple[int, int] | None


def load_detections_source(args, sequence):
    if args.source == "football":
        from world_cup_projects.common.detect import wrap_football_detections_cache

        return wrap_football_detections_cache(args)
    if args.source == "rfdetr":
        from world_cup_projects.common.detect import iter_model_detections
        from world_cup_projects.common.detection_cache import wrap_detections_cache

        return wrap_detections_cache(
            iter_model_detections,
            source_name="rfdetr",
            refresh=args.refresh_detections_cache,
            device=args.device,
        )
    return iter_gt_detections


def prepare_model_frames(
    frames: list[tuple[int, sv.Detections]],
    *,
    frame_width: float = 1920,
) -> list[tuple[int, sv.Detections]]:
    from world_cup_projects.common.teams import (
        enforce_one_goalkeeper_per_team_frames,
        stabilize_teams_by_tracklet,
    )

    frames = stabilize_teams_by_tracklet(frames)
    return enforce_one_goalkeeper_per_team_frames(
        frames, frame_width=float(frame_width)
    )


def load_metric_context(
    sequence,
    frames: list[tuple[int, sv.Detections]],
    *,
    device: str,
    pitch_confidence: float,
    end: int | None = None,
    source: str | None = None,
    frame_width: float | None = None,
    refresh: bool = False,
) -> MetricContext:
    """Pitch homography maps plus optional goalkeeper team stabilization."""
    clip_end = end if end is not None else sequence.length
    detections_by_frame = {int(fi): d for fi, d in frames}
    maps = ensure_pitch_homography_maps(
        sequence,
        device=device,
        end=clip_end,
        pitch_confidence=pitch_confidence,
        refresh=refresh,
        detections_by_frame=detections_by_frame,
    )
    locked_goals = None
    if source in ("football", "rfdetr") and maps.keypoints:
        locked_goals = warmup_goal_defenders_radar(
            frames,
            maps.keypoints,
            confidence=pitch_confidence,
        )
        from world_cup_projects.common.teams import stabilize_goalkeeper_teams

        gk_kwargs: dict[str, Any] = {
            "locked_goal_defenders": locked_goals,
            "keypoints_by_frame": maps.keypoints,
            "pitch_confidence": pitch_confidence,
        }
        if frame_width is not None:
            gk_kwargs["frame_width"] = float(frame_width)
        stabilize_goalkeeper_teams(frames, **gk_kwargs)
    return MetricContext(
        transforms=maps.transforms,
        radar_transforms=maps.radar_transforms,
        keypoints=maps.keypoints,
        locked_goals=locked_goals,
    )
