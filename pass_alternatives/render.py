"""Render the pass-alternatives demo video."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass

import cv2
import numpy as np
import supervision as sv

from world_cup_projects.common.pitch import (
    ViewTransformer,
    image_to_pitch_m,
    iter_pitch_transformers,
    pitch_attack_direction,
)
from world_cup_projects.common.possession import (
    CARRIER_MAX_DISTANCE_M,
    CARRIER_MAX_DISTANCE_PX,
    Carrier,
    feet_xy,
    find_ball_carrier,
    player_mask,
)
from world_cup_projects.common.soccernet import (
    SoccerNetSequence,
    iter_gt_detections,
)
from world_cup_projects.common.visual import (
    ROBOFLOW_PURPLE_BGR,
    annotate_ball,
    annotate_players,
    draw_branding_tag,
    draw_glow_arrow,
    draw_hud_bar,
    draw_pitch_keypoints_debug,
    draw_radar_minimap,
    draw_score_chip,
    draw_text_shadow,
)
from world_cup_projects.pass_alternatives.pass_options import (
    PassOption,
    PassWeights,
    top_pass_options,
)

RANK_COLORS_BGR = [(80, 220, 60), (40, 220, 240), (40, 140, 255)]
RANK_LABELS = ["BEST", "2ND", "3RD"]

DetectionSource = Callable[..., Iterator[tuple[int, sv.Detections]]]


@dataclass(frozen=True)
class PassEvent:
    frame_idx: int
    carrier: Carrier
    options: list[PassOption]
    top_score: float


def _iter_frames(
    sequence: SoccerNetSequence,
    detections_source: DetectionSource,
    *,
    max_frames: int | None,
) -> Iterator[tuple[int, sv.Detections]]:
    end = max_frames if max_frames is not None else sequence.length
    yield from detections_source(sequence, start=1, end=end)


def _score_options(
    dets: sv.Detections,
    carrier: Carrier,
    *,
    weights: PassWeights,
    transformer: ViewTransformer | None,
    metric: bool,
) -> list[PassOption]:
    if metric and transformer is not None:
        pitch_feet = image_to_pitch_m(feet_xy(dets), transformer)
        if pitch_feet is not None:
            attack_dir = pitch_attack_direction(
                dets,
                carrier.team,
                transformer,
                player_mask_fn=player_mask,
                feet_fn=feet_xy,
            )
            return top_pass_options(
                dets,
                carrier,
                k=3,
                weights=weights,
                attack_dir=attack_dir,
                positions=pitch_feet,
            )
    return top_pass_options(dets, carrier, k=3, weights=weights)


def _pitch_transform_map(
    sequence: SoccerNetSequence,
    *,
    max_frames: int | None,
    pitch_device: str,
    pitch_confidence: float = 0.5,
) -> dict[int, ViewTransformer | None]:
    return {
        frame_idx: speed_t
        for frame_idx, speed_t, _radar_t in iter_pitch_transformers(
            sequence,
            device=pitch_device,
            end=max_frames,
            confidence=pitch_confidence,
        )
    }


def _load_pitch_maps(
    sequence: SoccerNetSequence,
    *,
    max_frames: int | None,
    pitch_device: str,
    pitch_confidence: float,
    need_transforms: bool,
    need_keypoints: bool,
) -> tuple[
    dict[int, ViewTransformer | None],
    dict[int, sv.KeyPoints | None] | None,
    dict[int, ViewTransformer | None],
]:
    """One model pass per frame when transforms and/or debug keypoints are needed."""
    transforms: dict[int, ViewTransformer | None] = {}
    radar_transforms: dict[int, ViewTransformer | None] = {}
    keypoints: dict[int, sv.KeyPoints | None] | None = (
        {} if (need_keypoints or need_transforms) else None
    )
    for frame_idx, speed_t, radar_t, kps in iter_pitch_transformers(
        sequence,
        device=pitch_device,
        end=max_frames,
        confidence=pitch_confidence,
        yield_keypoints=True,
    ):
        if keypoints is not None:
            keypoints[frame_idx] = kps
        radar_transforms[frame_idx] = radar_t
        if need_transforms:
            transforms[frame_idx] = speed_t
    return transforms, keypoints, radar_transforms


def plan_events(
    sequence: SoccerNetSequence,
    *,
    max_events: int = 4,
    min_gap_frames: int = 90,
    carrier_max_distance_px: float = CARRIER_MAX_DISTANCE_PX,
    carrier_max_distance_m: float = CARRIER_MAX_DISTANCE_M,
    weights: PassWeights = PassWeights(),
    detections_source: DetectionSource = iter_gt_detections,
    max_frames: int | None = None,
    metric: bool = False,
    pitch_device: str = "cpu",
    frame_transforms: dict[int, ViewTransformer | None] | None = None,
) -> list[PassEvent]:
    """Pick the most compelling freeze moments across the clip."""
    transformers: dict[int, ViewTransformer | None] = frame_transforms or {}
    if metric and not transformers:
        transformers = _pitch_transform_map(
            sequence, max_frames=max_frames, pitch_device=pitch_device
        )

    candidates: list[PassEvent] = []
    for frame_idx, dets in _iter_frames(
        sequence, detections_source, max_frames=max_frames
    ):
        transformer = transformers.get(frame_idx) if metric else None
        carrier = find_ball_carrier(
            dets,
            max_distance_px=carrier_max_distance_px,
            transformer=transformer,
            max_distance_m=carrier_max_distance_m,
        )
        if carrier is None:
            continue
        options = _score_options(
            dets, carrier, weights=weights, transformer=transformer, metric=metric
        )
        if len(options) < 3:
            continue
        candidates.append(PassEvent(frame_idx, carrier, options, options[0].score))

    candidates.sort(key=lambda e: e.top_score, reverse=True)
    chosen: list[PassEvent] = []
    for event in candidates:
        if all(abs(event.frame_idx - c.frame_idx) >= min_gap_frames for c in chosen):
            chosen.append(event)
        if len(chosen) >= max_events:
            break
    chosen.sort(key=lambda e: e.frame_idx)
    return chosen


def _annotate_live(
    frame: np.ndarray,
    dets: sv.Detections,
    *,
    transformer=None,
    radar_transformer: ViewTransformer | None = None,
    keypoints: sv.KeyPoints | None = None,
    pitch_confidence: float = 0.5,
) -> np.ndarray:
    frame = annotate_players(frame, dets)
    frame = annotate_ball(frame, dets)
    if radar_transformer is not None or keypoints is not None:
        frame = draw_radar_minimap(
            frame, dets, keypoints, transformer=radar_transformer
        )
    if keypoints is not None:
        frame = draw_pitch_keypoints_debug(
            frame, keypoints, confidence_threshold=pitch_confidence
        )
    frame = draw_hud_bar(frame, "PASS ALTERNATIVES")
    return draw_branding_tag(frame)


def _draw_pass_overlay(
    frame: np.ndarray,
    dets: sv.Detections,
    event: PassEvent,
    *,
    keypoints: sv.KeyPoints | None = None,
    pitch_confidence: float = 0.5,
) -> np.ndarray:
    """Dim the frame and draw the ranked pass arrows from the carrier."""
    dim = (frame.astype(np.float32) * 0.35).astype(np.uint8)
    if keypoints is not None:
        dim = draw_pitch_keypoints_debug(
            dim, keypoints, confidence_threshold=pitch_confidence
        )
    dim = annotate_players(dim, dets)
    dim = annotate_ball(dim, dets)

    feet = feet_xy(dets)
    carrier_xy = feet[event.carrier.index]
    cx, cy = int(carrier_xy[0]), int(carrier_xy[1])

    cv2.circle(dim, (cx, cy), 18, (255, 255, 255), 2)
    draw_text_shadow(dim, "BALL CARRIER", (cx - 58, cy + 40), font_scale=0.5, color_bgr=(255, 255, 255))

    for rank, option in enumerate(event.options):
        color = RANK_COLORS_BGR[rank]
        recv_xy = feet[option.receiver_index]
        rx, ry = int(recv_xy[0]), int(recv_xy[1])
        draw_glow_arrow(dim, (cx, cy), (rx, ry), color, thickness=5)
        ring = 20 if rank == 0 else 14
        cv2.circle(dim, (rx, ry), ring, color, 3 if rank == 0 else 2)
        if rank == 0:
            cv2.circle(dim, (rx, ry), ring + 6, (255, 255, 255), 1)
        midx, midy = (cx + rx) // 2, (cy + ry) // 2
        draw_score_chip(dim, f"{RANK_LABELS[rank]}  {option.score:.2f}", (midx, midy), bg_bgr=color)

    dim = draw_hud_bar(dim, "PASS ALTERNATIVES  -  top 3 open lanes")
    return draw_branding_tag(dim)


def render_demo(
    sequence: SoccerNetSequence,
    out_path: str,
    *,
    max_frames: int | None = None,
    freeze_seconds: float = 1.5,
    max_events: int = 4,
    weights: PassWeights = PassWeights(),
    detections_source: DetectionSource = iter_gt_detections,
    metric: bool = False,
    pitch_device: str = "cpu",
    version_tag: str = "v1",
    frame_transforms: dict | None = None,
    carrier_max_distance_px: float = CARRIER_MAX_DISTANCE_PX,
    carrier_max_distance_m: float = CARRIER_MAX_DISTANCE_M,
    debug_pitch_keypoints: bool = False,
    pitch_confidence: float = 0.5,
) -> dict:
    """Render the full demo MP4. Returns a small manifest dict."""
    transforms: dict = frame_transforms or {}
    frame_radar_transforms: dict[int, ViewTransformer | None] = {}
    frame_keypoints: dict[int, sv.KeyPoints | None] | None = None
    need_transforms = metric and not transforms
    if need_transforms or debug_pitch_keypoints or metric:
        transforms, frame_keypoints, frame_radar_transforms = _load_pitch_maps(
            sequence,
            max_frames=max_frames,
            pitch_device=pitch_device,
            pitch_confidence=pitch_confidence,
            need_transforms=need_transforms,
            need_keypoints=debug_pitch_keypoints or metric,
        )
    events = plan_events(
        sequence,
        max_events=max_events,
        weights=weights,
        detections_source=detections_source,
        max_frames=max_frames,
        metric=metric,
        pitch_device=pitch_device,
        frame_transforms=transforms,
        carrier_max_distance_m=carrier_max_distance_m,
        carrier_max_distance_px=carrier_max_distance_px,
    )

    events_by_frame = {e.frame_idx: e for e in events}
    freeze_frames = int(round(freeze_seconds * sequence.frame_rate))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        out_path, fourcc, sequence.frame_rate, (sequence.width, sequence.height)
    )
    for frame_idx, dets in _iter_frames(
        sequence, detections_source, max_frames=max_frames
    ):
        image = cv2.imread(str(sequence.frame_path(frame_idx)))
        if image is None:
            image = np.full((sequence.height, sequence.width, 3), 30, np.uint8)
        transformer = transforms.get(frame_idx) if transforms else None
        kps = frame_keypoints.get(frame_idx) if frame_keypoints is not None else None
        radar_t = frame_radar_transforms.get(frame_idx)
        live = _annotate_live(
            image,
            dets,
            transformer=transformer,
            radar_transformer=radar_t,
            keypoints=kps,
            pitch_confidence=pitch_confidence,
        )
        writer.write(live)

        if frame_idx in events_by_frame:
            overlay = _draw_pass_overlay(
                image,
                dets,
                events_by_frame[frame_idx],
                keypoints=kps,
                pitch_confidence=pitch_confidence,
            )
            for _ in range(freeze_frames):
                writer.write(overlay)

    writer.release()
    return {
        "sequence": sequence.name,
        "output": out_path,
        "version": version_tag,
        "metric": metric,
        "events": [
            {
                "frame": e.frame_idx,
                "carrier_team": e.carrier.team,
                "top_score": round(e.top_score, 3),
                "options": [round(o.score, 3) for o in e.options],
            }
            for e in events
        ],
    }
