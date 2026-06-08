"""Infer pass events from per-frame ball-carrier handoffs.

A pass is detected when possession moves from teammate A to teammate B on the
same team within a short frame window. Each event is scored with the same lane
openness model used by the pass-alternatives demo.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import asdict, dataclass

import numpy as np
import supervision as sv

from world_cup_projects.common.pitch import (
    image_to_pitch_cm,
    image_to_pitch_m,
    pitch_attack_direction,
)
from world_cup_projects.common.possession import (
    CARRIER_MAX_DISTANCE_M,
    CARRIER_MAX_DISTANCE_PX,
    Carrier,
    ball_xy,
    bbox_center_xy,
    feet_xy,
    find_ball_carrier,
    player_mask,
)
from world_cup_projects.common.tracking_facing import carrier_kalman_direction
from world_cup_projects.pass_alternatives.pass_options import (
    PassOption,
    PassWeights,
    score_pass_options,
)

DetectionIterator = Iterator[tuple[int, sv.Detections]]


@dataclass(frozen=True)
class PassDetectionConfig:
    """Heuristic gates for carrier-to-carrier pass inference."""

    min_carrier_gap_frames: int = 3
    max_pass_gap_frames: int = 120
    min_ball_travel_m: float = 2.0
    min_ball_travel_px: float = 40.0
    carrier_max_distance_m: float = CARRIER_MAX_DISTANCE_M
    carrier_max_distance_px: float = CARRIER_MAX_DISTANCE_PX


@dataclass(frozen=True)
class InferredPass:
    """One directed pass A -> B at the release frame."""

    frame_idx: int
    passer_tid: int
    receiver_tid: int
    team: int
    gap_frames: int
    pass_length_m: float | None
    quality_score: float
    openness: float
    forward_gain: float
    rivals_in_lane: int
    motion_alignment: float
    receiver_space: float

    def to_dict(self) -> dict:
        return asdict(self)


class PassQualityScorer:
    """Score an actual passer-to-receiver lane on a single frame."""

    def __init__(
        self,
        *,
        weights: PassWeights = PassWeights.metric(),
        metric: bool = True,
        transformers: dict[int, object] | None = None,
    ) -> None:
        self._weights = weights
        self._metric = metric
        self._transformers = transformers or {}

    def option_for_receiver(
        self,
        frame_idx: int,
        dets: sv.Detections,
        carrier: Carrier,
        receiver_tid: int,
    ) -> PassOption | None:
        """Return the scored lane to ``receiver_tid``, or None if not a teammate option."""
        if receiver_tid < 0:
            return None
        pmask = player_mask(dets)
        receiver_rows = np.flatnonzero(
            pmask & (dets.tracker_id == receiver_tid)
        )
        if len(receiver_rows) == 0:
            return None

        motion_dir = None
        if self._weights.use_carrier_motion:
            transformer = self._transformers.get(frame_idx)
            motion_dir = carrier_kalman_direction(
                dets,
                carrier.index,
                transformer=transformer if self._metric else None,
            )

        transformer = self._transformers.get(frame_idx)
        if self._metric and transformer is not None:
            feet_img = feet_xy(dets)
            pitch_feet = image_to_pitch_m(feet_img, transformer)
            pitch_cm = image_to_pitch_cm(feet_img, transformer)
            body_pitch_m = image_to_pitch_m(bbox_center_xy(dets), transformer)
            if pitch_feet is None or pitch_cm is None:
                return None
            attack_dir = pitch_attack_direction(
                dets,
                carrier.team,
                transformer,
                player_mask_fn=player_mask,
                feet_fn=feet_xy,
            )
            options = score_pass_options(
                dets,
                carrier,
                weights=self._weights,
                attack_dir=attack_dir,
                positions=pitch_feet,
                carrier_motion_dir=motion_dir,
                pitch_cm=pitch_cm,
                body_pitch_m=body_pitch_m,
            )
        else:
            options = score_pass_options(
                dets,
                carrier,
                weights=self._weights,
                carrier_motion_dir=motion_dir,
            )

        receiver_index = int(receiver_rows[0])
        for option in options:
            if option.receiver_index == receiver_index:
                return option
        return None


def _ball_travel(
    ball_from: np.ndarray,
    ball_to: np.ndarray,
    *,
    metric: bool,
    transformer_from,
    transformer_to,
    min_travel_m: float,
    min_travel_px: float,
) -> float | None:
    """Distance the ball moved between two carrier frames (m or px)."""
    if metric and transformer_from is not None and transformer_to is not None:
        p0 = image_to_pitch_m(np.array([ball_from], dtype=np.float32), transformer_from)
        p1 = image_to_pitch_m(np.array([ball_to], dtype=np.float32), transformer_to)
        if p0 is not None and p1 is not None:
            return float(np.linalg.norm(p1[0] - p0[0]))
        return None
    return float(np.linalg.norm(ball_to - ball_from))


def _pass_length_m(option: PassOption | None, metric: bool) -> float | None:
    if option is None:
        return None
    if metric:
        return float(option.length)
    return None


def detect_pass_events(
    detections_iter: DetectionIterator,
    *,
    scorer: PassQualityScorer,
    config: PassDetectionConfig = PassDetectionConfig(),
    metric: bool = True,
    transformers: dict[int, object] | None = None,
) -> list[InferredPass]:
    """Scan tracked detections and infer directed pass events."""
    transformers = transformers or {}
    last: tuple[int, sv.Detections, Carrier] | None = None
    events: list[InferredPass] = []

    for frame_idx, dets in detections_iter:
        transformer = transformers.get(frame_idx) if metric else None
        carrier = find_ball_carrier(
            dets,
            max_distance_px=config.carrier_max_distance_px,
            transformer=transformer,
            max_distance_m=config.carrier_max_distance_m,
        )
        if carrier is None:
            continue

        tid = int(dets.tracker_id[carrier.index]) if dets.tracker_id is not None else -1
        if tid < 0 or carrier.team not in (0, 1):
            continue

        if last is not None:
            last_frame, last_dets, last_carrier = last
            last_tid = (
                int(last_dets.tracker_id[last_carrier.index])
                if last_dets.tracker_id is not None
                else -1
            )
            gap = frame_idx - last_frame
            if (
                last_tid >= 0
                and last_tid != tid
                and last_carrier.team == carrier.team
                and config.min_carrier_gap_frames <= gap <= config.max_pass_gap_frames
            ):
                travel = _ball_travel(
                    last_carrier.ball,
                    carrier.ball,
                    metric=metric,
                    transformer_from=transformers.get(last_frame),
                    transformer_to=transformer,
                    min_travel_m=config.min_ball_travel_m,
                    min_travel_px=config.min_ball_travel_px,
                )
                min_travel = config.min_ball_travel_m if metric else config.min_ball_travel_px
                if travel is not None and travel >= min_travel:
                    option = scorer.option_for_receiver(
                        last_frame, last_dets, last_carrier, tid
                    )
                    events.append(
                        InferredPass(
                            frame_idx=last_frame,
                            passer_tid=last_tid,
                            receiver_tid=tid,
                            team=int(last_carrier.team),
                            gap_frames=gap,
                            pass_length_m=_pass_length_m(option, metric),
                            quality_score=float(option.score) if option else 0.0,
                            openness=float(option.openness) if option else 0.0,
                            forward_gain=float(option.forward_gain) if option else 0.0,
                            rivals_in_lane=int(option.rivals_in_lane) if option else 0,
                            motion_alignment=float(option.motion_alignment) if option else 0.0,
                            receiver_space=float(option.receiver_space) if option else 0.0,
                        )
                    )

        last = (frame_idx, dets, carrier)

    return events
