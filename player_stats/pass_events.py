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

    min_carrier_gap_frames: int = 2
    max_pass_gap_frames: int = 120
    min_ball_travel_m: float = 3.0
    min_ball_travel_px: float = 40.0
    # Tightened distance to require the ball to be closer to outfield feet
    carrier_max_distance_m: float = 0.8
    carrier_max_distance_px: float = 60.0
    # Allow 2-touch passes minimum, relying on dual-distance to catch them
    min_consecutive_possession_frames: int = 2


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
    
    events: list[InferredPass] = []
    
    current_candidate_tid = -1
    candidate_frames = 0
    candidate_first_frame = -1
    candidate_first_carrier = None
    candidate_first_dets = None
    missing_ball_tolerance = 0  # Allow a few frames of occlusion without dropping possession

    # Stores the latest frame where a confirmed carrier had the ball (release frame)
    confirmed_passer: tuple[int, sv.Detections, Carrier, int] | None = None

    for frame_idx, dets in detections_iter:
        transformer = transformers.get(frame_idx) if metric else None
        carrier = find_ball_carrier(
            dets,
            max_distance_px=config.carrier_max_distance_px,
            transformer=transformer,
            max_distance_m=config.carrier_max_distance_m,
        )
        
        # 1. Occlusion Tolerance: Ball or Carrier missing
        if carrier is None:
            if current_candidate_tid >= 0:
                missing_ball_tolerance += 1
                if missing_ball_tolerance > 3:  # Drop candidate after ~100ms of no ball
                    current_candidate_tid = -1
                    candidate_frames = 0
            continue

        tid = int(dets.tracker_id[carrier.index]) if dets.tracker_id is not None else -1
        if tid < 0 or carrier.team not in (0, 1):
            continue

        # 2. Update Candidate State
        if tid == current_candidate_tid:
            # Same player touches it again (or still has it)
            candidate_frames += 1
            missing_ball_tolerance = 0  # Reset tolerance
        else:
            # New player touched the ball
            current_candidate_tid = tid
            candidate_frames = 1
            missing_ball_tolerance = 0
            candidate_first_frame = frame_idx
            candidate_first_carrier = carrier
            candidate_first_dets = dets

        # 3. Check for Confirmation
        from world_cup_projects.common.soccernet import ROLE_GOALKEEPER
        role = dets.class_id[carrier.index]
        required_frames = 1 if role == ROLE_GOALKEEPER else config.min_consecutive_possession_frames

        if candidate_frames == required_frames:
            # We confirmed possession for this player!
            if current_candidate_tid == 20 or carrier.team == 1:
                print(f"[DEBUG] Frame {frame_idx}: Confirmed possession for TID {current_candidate_tid} (Team {carrier.team})")
                
            if confirmed_passer is not None:
                p_frame, p_dets, p_carrier, p_tid = confirmed_passer

                # Gap is from the passer's LAST touch to the receiver's FIRST touch
                gap = candidate_first_frame - p_frame

                if p_tid != current_candidate_tid and p_carrier.team == carrier.team:
                    if config.min_carrier_gap_frames <= gap <= config.max_pass_gap_frames:
                        travel = _ball_travel(
                            p_carrier.ball,
                            candidate_first_carrier.ball,
                            metric=metric,
                            transformer_from=transformers.get(p_frame),
                            transformer_to=transformers.get(candidate_first_frame),
                            min_travel_m=config.min_ball_travel_m,
                            min_travel_px=config.min_ball_travel_px,
                        )
                        min_travel = config.min_ball_travel_m if metric else config.min_ball_travel_px
                        if travel is not None and travel >= min_travel:
                            if p_tid == 20 or current_candidate_tid == 20 or carrier.team == 1:
                                print(f"[DEBUG] ACCEPTED pass {p_tid}->{current_candidate_tid} (gap: {gap}, travel: {travel:.2f}m)")
                                
                            option = scorer.option_for_receiver(
                                p_frame, p_dets, p_carrier, current_candidate_tid
                            )
                            events.append(
                                InferredPass(
                                    frame_idx=p_frame,
                                    passer_tid=p_tid,
                                    receiver_tid=current_candidate_tid,
                                    team=int(p_carrier.team),
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
                        else:
                            if p_tid == 20 or current_candidate_tid == 20 or carrier.team == 1:
                                print(f"[DEBUG] Rejected pass {p_tid}->{current_candidate_tid} due to short travel: {travel}")
                    else:
                        if p_tid == 20 or current_candidate_tid == 20 or carrier.team == 1:
                            print(f"[DEBUG] Rejected pass {p_tid}->{current_candidate_tid} due to gap: {gap} frames")

            # Now that they are confirmed, they become the passer for the NEXT pass
            confirmed_passer = (frame_idx, dets, carrier, current_candidate_tid)

        elif candidate_frames > required_frames:
            # They are still holding it. Update their "last touch" frame for accurate release timing.
            confirmed_passer = (frame_idx, dets, carrier, current_candidate_tid)

    return events

