"""Per-frame ball-carrier signals for pass detection and debug overlays."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import asdict, dataclass

import numpy as np
import supervision as sv

from world_cup_projects.common.possession import (
    Carrier,
    ball_xy,
    feet_xy,
    find_active_carrier,
    find_ball_carrier,
    player_mask,
)
from world_cup_projects.common.possession_config import (
    AERIAL_DY_THRESHOLD_PX,
    CONTROL_MAX_DISTANCE_M,
    CONTROL_MAX_DISTANCE_PX,
    RECEPTION_MAX_DISTANCE_M,
    RECEPTION_MAX_DISTANCE_PX,
)
from world_cup_projects.common.possession_touch import (
    TouchValidationConfig,
    is_aerial_flyby_below_feet,
    is_aerial_touch,
    is_valid_possession_touch,
)
from world_cup_projects.common.soccernet import ROLE_GOALKEEPER


DetectionIterator = Iterator[tuple[int, sv.Detections]]


@dataclass(frozen=True)
class CarrierTrackingConfig:
    """Distance gates for control vs one-touch reception."""

    control_max_distance_m: float = CONTROL_MAX_DISTANCE_M
    control_max_distance_px: float = CONTROL_MAX_DISTANCE_PX
    reception_max_distance_m: float = RECEPTION_MAX_DISTANCE_M
    reception_max_distance_px: float = RECEPTION_MAX_DISTANCE_PX
    min_pass_gap_frames: int = 1
    max_pass_gap_frames: int = 75
    min_arrival_frames: int = 3
    min_reception_arrival_frames: int = 2
    min_arrival_control_frames: int = 1
    min_control_frames: int = 2
    min_gk_control_frames: int = 1
    pre_flight_release_window: int = 10
    adjacent_pass_max_gap_frames: int = 15
    aerial_dy_threshold_px: float = AERIAL_DY_THRESHOLD_PX
    missing_ball_tolerance: int = 10


@dataclass(frozen=True)
class CarrierFrameState:
    """Snapshot of how possession is interpreted on one frame."""

    frame_idx: int
    ball_present: bool
    nearest_tid: int | None
    nearest_dist_px: float | None
    control_tid: int | None
    control_dist: float | None
    reception_tid: int | None
    reception_dist: float | None
    active_tid: int | None
    active_kind: str | None
    last_release_tid: int | None
    possession_anchor_tid: int | None = None
    in_flight: bool = False
    arrival_candidate_tid: int | None = None
    arrival_streak: int = 0
    control_streak: int = 0
    nearest_teammate_tid: int | None = None
    pass_emitted: bool = False
    pass_from_tid: int | None = None
    pass_to_tid: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _nearest_player(
    dets: sv.Detections,
    ball: np.ndarray,
) -> tuple[int, float] | None:
    pmask = player_mask(dets)
    if not pmask.any() or dets.tracker_id is None:
        return None
    feet = feet_xy(dets)[pmask]
    tids = dets.tracker_id[pmask]
    dist = np.hypot(feet[:, 0] - ball[0], feet[:, 1] - ball[1])
    local = int(np.argmin(dist))
    tid = int(tids[local])
    if tid < 0:
        return None
    return tid, float(dist[local])


def _carrier_at_threshold(
    dets: sv.Detections,
    *,
    transformer,
    max_distance_m: float,
    max_distance_px: float,
) -> Carrier | None:
    return find_ball_carrier(
        dets,
        max_distance_px=max_distance_px,
        transformer=transformer,
        max_distance_m=max_distance_m,
    )


def _active_carrier_kind(
    dets: sv.Detections,
    *,
    transformer,
    config: CarrierTrackingConfig,
) -> tuple[Carrier | None, str | None]:
    return find_active_carrier(
        dets,
        transformer=transformer,
        control_max_distance_px=config.control_max_distance_px,
        control_max_distance_m=config.control_max_distance_m,
        reception_max_distance_px=config.reception_max_distance_px,
        reception_max_distance_m=config.reception_max_distance_m,
    )


def _touch_validation_config(config: CarrierTrackingConfig) -> TouchValidationConfig:
    return TouchValidationConfig(aerial_dy_threshold_px=config.aerial_dy_threshold_px)


@dataclass
class _DebugTeamState:
    release: tuple[int, Carrier, int] | None = None
    last_touch: tuple[int, Carrier, int] | None = None
    possession_tid: int = -1
    control_streak: int = 0
    in_flight: bool = False
    arrival_candidate_tid: int = -1
    arrival_streak: int = 0
    arrival_control_streak: int = 0


def _nearest_teammate(
    dets: sv.Detections,
    ball: np.ndarray,
    *,
    reference_team: int | None,
) -> tuple[int, float] | None:
    pmask = player_mask(dets)
    if not pmask.any() or dets.tracker_id is None or reference_team is None:
        return None
    feet = feet_xy(dets)[pmask]
    tids = dets.tracker_id[pmask]
    teams = dets.data["team"][pmask]
    teammate_mask = teams == reference_team
    if not teammate_mask.any():
        return None
    feet = feet[teammate_mask]
    tids = tids[teammate_mask]
    dist = np.hypot(feet[:, 0] - ball[0], feet[:, 1] - ball[1])
    local = int(np.argmin(dist))
    tid = int(tids[local])
    if tid < 0:
        return None
    return tid, float(dist[local])


def build_carrier_timeline(
    detections_iter: DetectionIterator,
    *,
    config: CarrierTrackingConfig = CarrierTrackingConfig(),
    metric: bool = True,
    transformers: dict[int, object] | None = None,
) -> list[CarrierFrameState]:
    """Walk the clip and record control/reception carrier signals per frame."""
    transformers = transformers or {}
    timeline: list[CarrierFrameState] = []

    team_states: dict[int, _DebugTeamState] = {
        0: _DebugTeamState(),
        1: _DebugTeamState(),
    }

    def _expire_stale_releases(frame_idx: int) -> None:
        for state in team_states.values():
            if state.release is None or state.in_flight:
                continue
            if frame_idx - state.release[0] > config.max_pass_gap_frames:
                state.release = None

    missing_ball_streak = 0
    for frame_idx, dets in detections_iter:
        _expire_stale_releases(frame_idx)
        transformer = transformers.get(frame_idx) if metric else None
        ball = ball_xy(dets)
        if ball is None:
            missing_ball_streak += 1
        else:
            missing_ball_streak = 0
        nearest = _nearest_player(dets, ball) if ball is not None else None

        control = (
            _carrier_at_threshold(
                dets,
                transformer=transformer,
                max_distance_m=config.control_max_distance_m,
                max_distance_px=config.control_max_distance_px,
            )
            if ball is not None
            else None
        )
        reception = (
            _carrier_at_threshold(
                dets,
                transformer=transformer,
                max_distance_m=config.reception_max_distance_m,
                max_distance_px=config.reception_max_distance_px,
            )
            if ball is not None
            else None
        )

        control_tid = None
        control_dist = None
        if control is not None and dets.tracker_id is not None:
            control_tid = int(dets.tracker_id[control.index])
            control_dist = float(control.distance)

        reception_tid = None
        reception_dist = None
        if reception is not None and dets.tracker_id is not None:
            reception_tid = int(dets.tracker_id[reception.index])
            reception_dist = float(reception.distance)

        active_tid = None
        active_kind = None
        if control_tid is not None and control_tid >= 0:
            active_tid = control_tid
            active_kind = "control"
        elif reception_tid is not None and reception_tid >= 0:
            active_tid = reception_tid
            active_kind = "reception"

        pass_emitted = False
        pass_from_tid = None
        pass_to_tid = None
        debug_team: int | None = None
        debug_anchor_tid: int | None = None
        debug_in_flight = False
        debug_arrival_candidate: int | None = None
        debug_arrival_streak = 0
        debug_control_streak = 0

        carrier, touch_kind = _active_carrier_kind(
            dets, transformer=transformer, config=config
        )
        touch_kind = touch_kind or "reception"
        if carrier is None or not is_valid_possession_touch(
            dets,
            carrier,
            touch_kind=touch_kind,
            config=_touch_validation_config(config),
        ):
            if ball is not None:
                for state in team_states.values():
                    if (
                        not state.in_flight
                        and state.release is None
                        and state.last_touch is not None
                        and frame_idx - state.last_touch[0]
                        <= config.pre_flight_release_window
                    ):
                        # last_touch was recorded on a valid frame; no re-check needed.
                        state.release = state.last_touch
                    state.in_flight = True
                    state.arrival_candidate_tid = -1
                    state.arrival_streak = 0
                    state.arrival_control_streak = 0
            elif missing_ball_streak <= config.missing_ball_tolerance:
                for state in team_states.values():
                    if state.release is None and state.last_touch is None:
                        continue
                    if (
                        not state.in_flight
                        and state.release is None
                        and state.last_touch is not None
                        and frame_idx - state.last_touch[0]
                        <= config.pre_flight_release_window
                    ):
                        state.release = state.last_touch
                    state.in_flight = True
            else:
                for state in team_states.values():
                    state.in_flight = False
                    state.arrival_candidate_tid = -1
                    state.arrival_streak = 0
                    state.arrival_control_streak = 0
        elif dets.tracker_id is not None:
            tid = int(dets.tracker_id[carrier.index])
            team = int(carrier.team)
            if tid >= 0 and team in (0, 1):
                debug_team = team
                state = team_states[team]
                was_in_flight = state.in_flight
                state.in_flight = False
                state.last_touch = (frame_idx, carrier, tid)
                min_control = (
                    config.min_gk_control_frames
                    if int(dets.class_id[carrier.index]) == ROLE_GOALKEEPER
                    else config.min_control_frames
                )

                if touch_kind == "control":
                    if tid == state.possession_tid:
                        state.control_streak += 1
                    else:
                        state.possession_tid = tid
                        state.control_streak = 1
                    if state.control_streak >= min_control:
                        if state.release is None or state.release[2] == tid:
                            state.release = (frame_idx, carrier, tid)
                else:
                    state.possession_tid = tid
                    state.control_streak = 0
                    if int(dets.class_id[carrier.index]) == ROLE_GOALKEEPER:
                        state.release = (frame_idx, carrier, tid)

                release = state.release
                if release is not None:
                    release_frame, _, release_tid = release
                    if tid == release_tid:
                        state.release = (frame_idx, carrier, tid)
                        state.arrival_candidate_tid = -1
                        state.arrival_streak = 0
                    else:
                        if tid == state.arrival_candidate_tid:
                            state.arrival_streak += 1
                            if touch_kind == "control":
                                state.arrival_control_streak += 1
                            else:
                                state.arrival_control_streak = 0
                        else:
                            state.arrival_candidate_tid = tid
                            state.arrival_streak = 1
                            state.arrival_control_streak = (
                                1 if touch_kind == "control" else 0
                            )
                        gap_frames = frame_idx - release_frame
                        min_arrival = (
                            config.min_reception_arrival_frames
                            if touch_kind == "reception"
                            and gap_frames >= config.adjacent_pass_max_gap_frames
                            else config.min_arrival_frames
                        )
                        arrival_ready = state.arrival_streak >= min_arrival
                        adjacent = gap_frames < config.adjacent_pass_max_gap_frames
                        if arrival_ready:
                            if adjacent:
                                arrival_ready = (
                                    state.arrival_control_streak
                                    >= config.min_control_frames
                                )
                            else:
                                arrival_ready = (
                                    state.arrival_control_streak
                                    >= config.min_arrival_control_frames
                                )
                        if arrival_ready:
                            gap = frame_idx - release_frame
                            if (
                                config.min_pass_gap_frames
                                <= gap
                                <= config.max_pass_gap_frames
                            ):
                                pass_emitted = True
                                pass_from_tid = release_tid
                                pass_to_tid = tid
                            state.release = (frame_idx, carrier, tid)
                            state.arrival_candidate_tid = -1
                            state.arrival_streak = 0

                debug_anchor_tid = state.release[2] if state.release else None
                debug_in_flight = state.in_flight
                debug_arrival_candidate = (
                    state.arrival_candidate_tid
                    if state.arrival_candidate_tid >= 0
                    else None
                )
                debug_arrival_streak = state.arrival_streak
                debug_control_streak = state.control_streak

        anchor_tid = debug_anchor_tid
        anchor_team = debug_team
        nearest_teammate = (
            _nearest_teammate(dets, ball, reference_team=anchor_team)
            if ball is not None
            else None
        )
        in_flight = debug_in_flight
        arrival_candidate_tid = debug_arrival_candidate
        arrival_streak = debug_arrival_streak

        timeline.append(
            CarrierFrameState(
                frame_idx=frame_idx,
                ball_present=ball is not None,
                nearest_tid=nearest[0] if nearest else None,
                nearest_dist_px=nearest[1] if nearest else None,
                control_tid=control_tid,
                control_dist=control_dist,
                reception_tid=reception_tid,
                reception_dist=reception_dist,
                active_tid=active_tid,
                active_kind=active_kind,
                last_release_tid=anchor_tid,
                possession_anchor_tid=anchor_tid,
                in_flight=in_flight,
                arrival_candidate_tid=arrival_candidate_tid,
                arrival_streak=arrival_streak,
                control_streak=debug_control_streak,
                nearest_teammate_tid=(
                    nearest_teammate[0] if nearest_teammate else None
                ),
                pass_emitted=pass_emitted,
                pass_from_tid=pass_from_tid,
                pass_to_tid=pass_to_tid,
            )
        )

    return timeline
