"""Render the pass-alternatives demo video."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import supervision as sv

from world_cup_projects.common.carrier_motion import BallPositionHistory, TrackPositionHistory
from world_cup_projects.common.player_tracker import TrackerKind
from world_cup_projects.common.tracking_facing import (
    KalmanFacingReplay,
    carrier_kalman_direction,
    detections_have_kalman_velocity,
    facing_kalman_from_detections,
)
from world_cup_projects.common.clips import pitch_keypoints_unreliable
from world_cup_projects.common.pitch import (
    PitchHomographyTracker,
    ViewTransformer,
    homography_from_keypoints_radar,
    image_to_pitch_cm,
    image_to_pitch_m,
    iter_pitch_transformers,
    load_pitch_homography_cache,
    pitch_attack_direction,
    render_radar_simple,
    warmup_goal_defenders,
)
from world_cup_projects.pass_alternatives.lane_visual import (
    PassEvent,
    apply_pass_lane_geometry,
    draw_pass_lane_legend,
    draw_pass_lanes_on_radar,
    draw_pass_overlay,
    draw_receiver_highlight,
    pass_line_label_xy,
)
from world_cup_projects.pass_alternatives.pass_options import PassWeights, top_pass_options
from world_cup_projects.common.possession import (
    Carrier,
    ball_xy,
    bbox_center_xy,
    feet_xy,
    find_control_carrier,
    player_mask,
)
from world_cup_projects.common.soccernet import (
    SoccerNetSequence,
    iter_gt_detections,
)
from world_cup_projects.common.video import (
    H264StreamWriter,
    SequentialVideoReader,
    finalize_video_for_playback,
    read_sequence_frame,
)
from world_cup_projects.common.visual import (
    ROBOFLOW_PURPLE_BGR,
    annotate_ball,
    annotate_players,
    draw_branding_tag,
    draw_carrier_halo,
    draw_carrier_pulse,
    draw_carrier_spotlight,
    draw_glow_arrow,
    draw_hud_bar,
    draw_pass_analysis_panel,
    draw_pitch_keypoints_debug,
    draw_radar_minimap,
    draw_score_chip,
    draw_text_shadow,
    ease_out_cubic,
)
from world_cup_projects.pass_alternatives.pass_options import (
    PassOption,
    PassWeights,
    remap_lane_debug_to_pitch_cm,
    top_pass_options,
)

RANK_COLORS_BGR = [(80, 220, 60), (0, 215, 255), (40, 140, 255)]
RANK_LABELS = ["BEST", "2ND", "3RD"]

# Cinematic slowdown before each freeze, then staggered pass-line reveals.
MAX_RAMP_HOLD = 6
DEFAULT_SLOWDOWN_RAMP_SECONDS = 0.72
DEFAULT_OPTION_REVEAL_SECONDS = 0.6
DEFAULT_FREEZE_SECONDS = 2.5
DEFAULT_FINAL_OPTION_EXTRA_SECONDS = 1.0

DetectionSource = Callable[..., Iterator[tuple[int, sv.Detections]]]

# Backward-compatible re-exports for notebooks and legacy imports.
_draw_pass_overlay = draw_pass_overlay
__all__ = ["PassEvent", "draw_pass_overlay", "_draw_pass_overlay"]


def _iter_frames(
    sequence: SoccerNetSequence,
    detections_source: DetectionSource,
    *,
    max_frames: int | None,
) -> Iterator[tuple[int, sv.Detections]]:
    end = max_frames if max_frames is not None else sequence.length
    yield from detections_source(sequence, start=1, end=end)


def _patch_goalkeeper_teams(
    detections_source: DetectionSource,
    transforms: dict[int, ViewTransformer | None],
) -> DetectionSource:
    """Infer goalkeeper team from pitch distance to each goal mouth."""
    from world_cup_projects.common.teams import apply_goalkeeper_teams_by_goal

    def _wrapped(
        seq: SoccerNetSequence,
        *,
        start: int = 1,
        end: int | None = None,
        **kwargs: object,
    ) -> Iterator[tuple[int, sv.Detections]]:
        del kwargs
        last = seq.length if end is None else min(end, seq.length)
        for frame_idx, dets in detections_source(seq, start=start, end=last):
            t = transforms.get(frame_idx)
            if t is not None:
                dets = apply_goalkeeper_teams_by_goal(dets, t)
            yield frame_idx, dets

    return _wrapped


def _cache_detections(
    sequence: SoccerNetSequence,
    detections_source: DetectionSource,
    *,
    max_frames: int | None,
) -> DetectionSource:
    """Run the detector/tracker once; render passes reuse cached frames."""
    end = max_frames if max_frames is not None else sequence.length
    cached = list(detections_source(sequence, start=1, end=end))

    def _replay(
        seq: SoccerNetSequence,
        *,
        start: int = 1,
        end: int | None = None,
        **kwargs: object,
    ) -> Iterator[tuple[int, sv.Detections]]:
        del seq, kwargs
        last = sequence.length if end is None else min(end, sequence.length)
        for frame_idx, dets in cached:
            if start <= frame_idx <= last:
                yield frame_idx, dets

    return _replay


def _score_options(
    dets: sv.Detections,
    carrier: Carrier,
    *,
    weights: PassWeights,
    transformer: ViewTransformer | None,
    metric: bool,
) -> list[PassOption]:
    motion_dir = None
    if weights.use_carrier_motion:
        motion_dir = carrier_kalman_direction(
            dets,
            carrier.index,
            transformer=transformer if metric else None,
        )

    if metric and transformer is not None:
        feet_img = feet_xy(dets)
        pitch_feet = image_to_pitch_m(feet_img, transformer)
        pitch_cm = image_to_pitch_cm(feet_img, transformer)
        body_pitch_m = image_to_pitch_m(bbox_center_xy(dets), transformer)
        if pitch_feet is not None and pitch_cm is not None:
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
                carrier_motion_dir=motion_dir,
                pitch_cm=pitch_cm,
                body_pitch_m=body_pitch_m,
            )
    return top_pass_options(
        dets,
        carrier,
        k=3,
        weights=weights,
        carrier_motion_dir=motion_dir,
    )


def _pitch_transform_map(
    sequence: SoccerNetSequence,
    *,
    max_frames: int | None,
    pitch_device: str,
    pitch_confidence: float = 0.9,
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
    PitchHomographyTracker | None,
]:
    """One model pass per frame when transforms and/or debug keypoints are needed."""
    clip_end = max_frames if max_frames is not None else sequence.length
    cached = load_pitch_homography_cache(
        sequence.name,
        end=clip_end,
        pitch_confidence=pitch_confidence,
        device=pitch_device,
    )
    if cached is not None:
        transforms = cached.transforms if need_transforms else {}
        radar_transforms = cached.radar_transforms
        keypoints = (
            cached.keypoints if (need_keypoints or need_transforms) else None
        )
        return transforms, keypoints, radar_transforms, None

    transforms: dict[int, ViewTransformer | None] = {}
    radar_transforms: dict[int, ViewTransformer | None] = {}
    keypoints: dict[int, sv.KeyPoints | None] | None = (
        {} if (need_keypoints or need_transforms) else None
    )
    tracker: PitchHomographyTracker | None = None
    for frame_idx, speed_t, radar_t, kps, pitch_tracker in iter_pitch_transformers(
        sequence,
        device=pitch_device,
        end=max_frames,
        confidence=pitch_confidence,
        yield_keypoints=True,
        yield_tracker=True,
    ):
        tracker = pitch_tracker
        if keypoints is not None:
            keypoints[frame_idx] = kps
        radar_transforms[frame_idx] = radar_t
        if need_transforms:
            transforms[frame_idx] = speed_t
    if tracker is not None:
        tracker.finalize_goal_lock()
    return transforms, keypoints, radar_transforms, tracker


def _freeze_moment_score(
    pass_top: float,
    carrier: Carrier,
    *,
    ball_speed: float | None,
    weights: PassWeights,
    metric: bool,
) -> float | None:
    """Rank freeze candidates: pass quality + tight feet + slow ball."""
    if weights.use_ball_control_gate and ball_speed is not None:
        if metric:
            if ball_speed >= weights.ball_speed_skip_m:
                return None
            ref, cap = weights.ball_speed_ref_m, weights.ball_speed_max_m
            tight_ref = weights.carrier_tight_ref_m
        else:
            if ball_speed >= weights.ball_speed_skip_px_s:
                return None
            ref, cap = weights.ball_speed_ref_px_s, weights.ball_speed_max_px_s
            tight_ref = weights.carrier_tight_ref_px

        score = pass_top
        if ball_speed <= ref:
            score += 0.05
        elif ball_speed < cap:
            t = (ball_speed - ref) / (cap - ref)
            score -= weights.ball_speed_penalty * t
        else:
            score -= weights.ball_speed_penalty

        if carrier.distance < tight_ref:
            score += weights.carrier_tight_bonus * (
                1.0 - carrier.distance / tight_ref
            )
        return score

    score = pass_top
    if metric:
        tight_ref = weights.carrier_tight_ref_m
    else:
        tight_ref = weights.carrier_tight_ref_px
    if carrier.distance < tight_ref:
        score += weights.carrier_tight_bonus * (1.0 - carrier.distance / tight_ref)
    return score


def _resolve_freeze_frame_earlier(
    event: PassEvent,
    by_frame: dict[int, PassEvent],
    *,
    weights: PassWeights,
    metric: bool,
    instant_speed_by_frame: dict[int, float],
) -> PassEvent:
    """Keep the pass moment but freeze earlier — before release or score peak."""
    if not weights.freeze_nudge_earlier:
        return event
    slack = weights.freeze_nudge_score_slack
    eps = weights.freeze_separation_eps_m if metric else weights.freeze_separation_eps_px
    best = event

    # Ball leaving feet: walk back while separating or ball just sped up.
    while True:
        prev = by_frame.get(best.frame_idx - 1)
        if prev is None or prev.top_score < best.top_score - slack:
            break
        separating = best.carrier.distance > prev.carrier.distance + eps
        instant = instant_speed_by_frame.get(best.frame_idx)
        fast_ball = instant is not None and (
            (metric and instant >= weights.freeze_release_ball_speed_skip_m)
            or (not metric and instant >= weights.freeze_release_ball_speed_skip_px_s)
        )
        if not (separating or fast_ball):
            break
        best = prev

    # Lane score peaked on this frame vs the previous — prefer one frame earlier
    # (still in control) when the prior frame is nearly as good.
    prev = by_frame.get(best.frame_idx - 1)
    if (
        prev is not None
        and best.top_score > prev.top_score
        and prev.top_score >= best.top_score - slack
    ):
        best = prev

    return best


def _apply_freeze_frame_nudges(
    candidates: list[PassEvent],
    *,
    weights: PassWeights,
    metric: bool,
    instant_speed_by_frame: dict[int, float],
) -> list[PassEvent]:
    by_frame = {e.frame_idx: e for e in candidates}
    resolved: dict[int, PassEvent] = {}
    for event in candidates:
        nudged = _resolve_freeze_frame_earlier(
            event,
            by_frame,
            weights=weights,
            metric=metric,
            instant_speed_by_frame=instant_speed_by_frame,
        )
        prev = resolved.get(nudged.frame_idx)
        if prev is None or nudged.top_score > prev.top_score:
            resolved[nudged.frame_idx] = nudged
    return sorted(resolved.values(), key=lambda e: e.frame_idx)


def _select_pass_moments(
    candidates: list[PassEvent],
    *,
    weights: PassWeights,
    min_gap_frames: int,
    max_events: int | None,
) -> list[PassEvent]:
    """Pick freeze frames where pass context is strong (score threshold + local peak)."""
    if not candidates:
        return []

    by_frame = sorted(candidates, key=lambda e: e.frame_idx)
    score_at = {e.frame_idx: e.top_score for e in by_frame}
    half = weights.freeze_local_peak_half_window

    peaks: list[PassEvent] = []
    for event in by_frame:
        if event.top_score < weights.freeze_min_pick_score:
            continue
        if event.options[0].score < weights.freeze_min_pass_score:
            continue
        if weights.freeze_detect_local_peaks:
            f = event.frame_idx
            neighbor_scores = [
                score_at.get(f + d, -1.0)
                for d in range(-half, half + 1)
                if d != 0 and (f + d) in score_at
            ]
            if neighbor_scores and event.top_score <= max(neighbor_scores):
                continue
        peaks.append(event)

    peaks.sort(key=lambda e: e.top_score, reverse=True)
    chosen: list[PassEvent] = []
    for event in peaks:
        if all(abs(event.frame_idx - c.frame_idx) >= min_gap_frames for c in chosen):
            chosen.append(event)
        if max_events is not None and len(chosen) >= max_events:
            break
    chosen.sort(key=lambda e: e.frame_idx)
    return chosen


def plan_events(
    sequence: SoccerNetSequence,
    *,
    max_events: int | None = None,
    min_gap_frames: int = 90,
    carrier_max_distance_px: float | None = None,
    carrier_max_distance_m: float | None = None,
    weights: PassWeights = PassWeights(),
    detections_source: DetectionSource = iter_gt_detections,
    max_frames: int | None = None,
    metric: bool = False,
    pitch_device: str = "cpu",
    frame_transforms: dict[int, ViewTransformer | None] | None = None,
) -> list[PassEvent]:
    """Detect good pass moments (threshold + local peaks), optional ``max_events`` cap."""
    transformers: dict[int, ViewTransformer | None] = frame_transforms or {}
    if metric and not transformers:
        transformers = _pitch_transform_map(
            sequence, max_frames=max_frames, pitch_device=pitch_device
        )

    max_px = (
        carrier_max_distance_px
        if carrier_max_distance_px is not None
        else weights.freeze_carrier_max_distance_px
    )
    max_m = (
        carrier_max_distance_m
        if carrier_max_distance_m is not None
        else weights.freeze_carrier_max_distance_m
    )
    require_both = weights.freeze_require_both_spaces and metric

    history = TrackPositionHistory()
    ball_history = BallPositionHistory()
    fps = float(sequence.frame_rate)
    candidates: list[PassEvent] = []
    instant_speed_by_frame: dict[int, float] = {}
    for frame_idx, dets in _iter_frames(
        sequence, detections_source, max_frames=max_frames
    ):
        transformer = transformers.get(frame_idx) if metric else None
        ball_history.record(frame_idx, ball_xy(dets))
        feet_img = feet_xy(dets)
        hist_xy = feet_img
        if metric and transformer is not None:
            pitch_feet = image_to_pitch_m(feet_img, transformer)
            if pitch_feet is not None:
                hist_xy = pitch_feet
        history.record_frame(frame_idx, dets, hist_xy)

        carrier = find_control_carrier(
            dets,
            max_distance_px=max_px,
            transformer=transformer,
            max_distance_m=max_m,
            require_both_spaces=require_both and transformer is not None,
        )
        if carrier is None:
            continue
        if metric and frame_idx < 30:
            continue
        options = _score_options(
            dets,
            carrier,
            weights=weights,
            transformer=transformer,
            metric=metric,
        )
        if len(options) < 2:
            continue
        top = options[0]
        if top.length < weights.min_length or top.length > weights.max_length:
            continue
        ball_speed = ball_history.speed(
            frame_idx,
            lookback_frames=weights.ball_speed_lookback_frames,
            fps=fps,
            transformer=transformer if metric else None,
        )
        pick_score = _freeze_moment_score(
            top.score,
            carrier,
            ball_speed=ball_speed,
            weights=weights,
            metric=metric,
        )
        if pick_score is None:
            continue
        instant = ball_history.speed(
            frame_idx,
            lookback_frames=1,
            fps=fps,
            transformer=transformer if metric else None,
        )
        if instant is not None:
            instant_speed_by_frame[frame_idx] = instant
        candidates.append(PassEvent(frame_idx, carrier, options, pick_score))

    candidates = _apply_freeze_frame_nudges(
        candidates,
        weights=weights,
        metric=metric,
        instant_speed_by_frame=instant_speed_by_frame,
    )
    return _select_pass_moments(
        candidates,
        weights=weights,
        min_gap_frames=min_gap_frames,
        max_events=max_events,
    )


def _annotate_live(
    frame: np.ndarray,
    dets: sv.Detections,
    *,
    keypoints: sv.KeyPoints | None = None,
    pitch_confidence: float = 0.9,
    metric: bool = False,
    show_radar: bool = True,
    radar_transformer: ViewTransformer | None = None,
    locked_goal_defenders: tuple[int, int] | None = None,
    debug_pitch_keypoints: bool = False,
    facing: np.ndarray | None = None,
    facing_motion: np.ndarray | None = None,
    facing_kalman: np.ndarray | None = None,
) -> np.ndarray:
    frame = annotate_players(
        frame,
        dets,
        facing=facing,
        facing_motion=facing_motion,
        facing_kalman=facing_kalman,
        show_tracker_ids=True,
    )
    frame = annotate_ball(frame, dets)
    carrier = find_control_carrier(
        dets,
        transformer=radar_transformer if metric else None,
    )
    if carrier is not None:
        feet = feet_xy(dets)[carrier.index]
        draw_carrier_halo(frame, (int(feet[0]), int(feet[1])))
    if show_radar and metric and keypoints is not None:
        frame = draw_radar_minimap(
            frame,
            dets,
            keypoints,
            pitch_confidence=pitch_confidence,
            locked_goal_defenders=locked_goal_defenders,
            debug_keypoints=True,
        )
    if debug_pitch_keypoints and keypoints is not None:
        frame = draw_pitch_keypoints_debug(
            frame, keypoints, confidence_threshold=pitch_confidence
        )
    frame = draw_hud_bar(frame, "PASS ALTERNATIVES")
    return draw_branding_tag(frame)


def _slowdown_hold_count(
    frames_until_event: int,
    *,
    ramp_frames: int,
    max_extra_holds: int = MAX_RAMP_HOLD,
) -> int:
    """Repeat count for live frames as playback eases into a freeze."""
    if frames_until_event <= 0 or frames_until_event > ramp_frames:
        return 1
    t = 1.0 - frames_until_event / ramp_frames
    return 1 + int((t * t) * max(0, max_extra_holds - 1))


def render_demo(
    sequence: SoccerNetSequence,
    out_path: str,
    *,
    max_frames: int | None = None,
    freeze_seconds: float = DEFAULT_FREEZE_SECONDS,
    slowdown_ramp_seconds: float = DEFAULT_SLOWDOWN_RAMP_SECONDS,
    option_reveal_seconds: float = DEFAULT_OPTION_REVEAL_SECONDS,
    final_option_extra_seconds: float = DEFAULT_FINAL_OPTION_EXTRA_SECONDS,
    max_events: int | None = None,
    weights: PassWeights = PassWeights(),
    detections_source: DetectionSource = iter_gt_detections,
    metric: bool = False,
    pitch_device: str = "cpu",
    version_tag: str = "v1",
    frame_transforms: dict | None = None,
    carrier_max_distance_px: float | None = None,
    carrier_max_distance_m: float | None = None,
    debug_pitch_keypoints: bool = False,
    pitch_confidence: float = 0.9,
    show_radar: bool | None = None,
    facing_mode: Literal["motion", "kalman", "both"] = "kalman",
    tracker_kind: TrackerKind = "bytetrack",
) -> dict:
    """Render the full demo MP4. Returns a small manifest dict."""
    detections_source = _cache_detections(
        sequence, detections_source, max_frames=max_frames
    )
    if show_radar is None:
        show_radar = not pitch_keypoints_unreliable(sequence.name)
    transforms: dict = frame_transforms or {}
    frame_radar_transforms: dict[int, ViewTransformer | None] = {}
    frame_keypoints: dict[int, sv.KeyPoints | None] | None = None
    pitch_tracker: PitchHomographyTracker | None = None
    need_transforms = metric and not transforms
    if need_transforms or debug_pitch_keypoints or metric:
        transforms, frame_keypoints, frame_radar_transforms, pitch_tracker = (
            _load_pitch_maps(
                sequence,
                max_frames=max_frames,
                pitch_device=pitch_device,
                pitch_confidence=pitch_confidence,
                need_transforms=need_transforms,
                need_keypoints=debug_pitch_keypoints or metric,
            )
        )
    frame_list = list(_iter_frames(sequence, detections_source, max_frames=max_frames))
    locked_goals = warmup_goal_defenders(
        pitch_tracker,
        frame_list,
        transforms,
        keypoints_by_frame=frame_keypoints,
        confidence=pitch_confidence,
    )
    if metric and frame_keypoints is not None:
        from world_cup_projects.common.teams import stabilize_goalkeeper_teams

        stabilize_goalkeeper_teams(
            frame_list,
            transforms=transforms,
            locked_goal_defenders=locked_goals,
            keypoints_by_frame=frame_keypoints,
            pitch_confidence=pitch_confidence,
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
    event_frames = sorted(events_by_frame)
    ramp_frames = max(1, int(round(slowdown_ramp_seconds * sequence.frame_rate)))
    reveal_frames = max(4, int(round(option_reveal_seconds * sequence.frame_rate)))

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

    use_h264_stream = True
    try:
        writer: H264StreamWriter | cv2.VideoWriter = H264StreamWriter(
            out_path,
            width=sequence.width,
            height=sequence.height,
            fps=sequence.frame_rate,
        )
    except RuntimeError:
        use_h264_stream = False
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            out_path, fourcc, sequence.frame_rate, (sequence.width, sequence.height)
        )
    track_history = TrackPositionHistory()
    use_cached_kalman = False
    kalman_replay = None
    if facing_mode in ("kalman", "both"):
        sample = next(
            _iter_frames(sequence, detections_source, max_frames=max_frames),
            None,
        )
        if sample is not None and detections_have_kalman_velocity(sample[1]):
            use_cached_kalman = True
        else:
            kalman_replay = KalmanFacingReplay(
                sequence.frame_rate, tracker_kind=tracker_kind
            )
    for frame_idx, dets in _iter_frames(
        sequence, detections_source, max_frames=max_frames
    ):
        image = _load_frame_image(frame_idx)
        transformer = transforms.get(frame_idx) if transforms else None
        kps = frame_keypoints.get(frame_idx) if frame_keypoints is not None else None
        if pitch_tracker is not None and transformer is not None and locked_goals is None:
            from world_cup_projects.common.soccernet import ROLE_PLAYER

            omask = dets.class_id == ROLE_PLAYER
            if omask.any():
                pitch_m = image_to_pitch_m(feet_xy(dets)[omask], transformer)
                if pitch_m is not None:
                    teams = dets.data.get("team", np.zeros(len(dets), dtype=int))[
                        omask
                    ]
                    if pitch_tracker.register_reliable_goal_vote(pitch_m, teams):
                        locked_goals = pitch_tracker.locked_goal_defenders
        track_history.record_frame(frame_idx, dets, feet_xy(dets))
        facing_motion = (
            track_history.player_facing(dets, frame_idx)
            if facing_mode in ("motion", "both")
            else None
        )
        if facing_mode in ("kalman", "both"):
            if use_cached_kalman:
                facing_kalman = facing_kalman_from_detections(dets)
            else:
                facing_kalman = kalman_replay.advance(dets, image) if kalman_replay else None
        else:
            facing_kalman = None
        live = _annotate_live(
            image,
            dets,
            keypoints=kps,
            pitch_confidence=pitch_confidence,
            metric=metric,
            show_radar=show_radar,
            radar_transformer=transformer,
            locked_goal_defenders=locked_goals,
            debug_pitch_keypoints=debug_pitch_keypoints,
            facing_motion=facing_motion,
            facing_kalman=facing_kalman,
        )
        frames_until = next(
            (ef - frame_idx for ef in event_frames if ef >= frame_idx),
            None,
        )
        hold = (
            _slowdown_hold_count(frames_until, ramp_frames=ramp_frames)
            if frames_until is not None
            else 1
        )
        for _ in range(hold):
            writer.write(live)

        if frame_idx in events_by_frame:
            event = events_by_frame[frame_idx]
            n_options = min(3, len(event.options))
            overlay_kwargs = dict(
                weights=weights,
                metric=metric,
                keypoints=kps,
                pitch_confidence=pitch_confidence,
                transformer=transformer,
                show_lane_debug=metric and kps is not None,
                show_radar=show_radar,
                locked_goal_defenders=locked_goals,
                debug_pitch_keypoints=debug_pitch_keypoints,
            )
            phases: list[tuple[int, int]] = [(0, reveal_frames)]
            phases.extend((i, reveal_frames) for i in range(1, n_options + 1))
            min_freeze = sum(h for _, h in phases)
            extra_hold = max(0, int(round(freeze_seconds * sequence.frame_rate)) - min_freeze)
            final_extra = max(4, int(round(final_option_extra_seconds * sequence.frame_rate)))
            if phases:
                phases[-1] = (phases[-1][0], phases[-1][1] + extra_hold + final_extra)

            for revealed, phase_hold in phases:
                for step in range(phase_hold):
                    progress = (step + 1) / max(phase_hold, 1)
                    overlay = draw_pass_overlay(
                        image,
                        dets,
                        event,
                        revealed_options=revealed,
                        reveal_progress=progress,
                        facing_motion=facing_motion,
                        facing_kalman=facing_kalman,
                        **overlay_kwargs,
                    )
                    writer.write(overlay)

    if use_h264_stream:
        writer.close()
    else:
        writer.release()
        finalize_video_for_playback(Path(out_path))
    if video_reader is not None:
        video_reader.close()
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
                "options": [
                    {"score": round(o.score, 3), "length_m": round(o.length, 2)}
                    if metric
                    else round(o.score, 3)
                    for o in e.options
                ],
            }
            for e in events
        ],
    }
