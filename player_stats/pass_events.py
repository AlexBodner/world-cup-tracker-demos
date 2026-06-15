"""Infer pass events from per-frame ball-carrier handoffs.

Simple rule set (per team, frame by frame)
------------------------------------------
1. **Valid touch** — ball is nearest this player's feet; reject aerial *control*
   and reception fly-bys below the feet (chest-height receptions are OK).
2. **Passer** — player who *released* the ball:
   - outfield: ``min_control_frames`` consecutive control frames, OR
   - goalkeeper: one control/reception frame, OR
   - any role: last valid touch within ``pre_flight_release_window`` when the
     ball goes in-flight (covers punts and one-touch releases).
3. **Receiver** — teammate who gets the ball after a gap:
   - ``min_arrival_frames`` consecutive valid touches (filters deflections).
   - if gap passer→receiver < ``adjacent_pass_max_gap_frames``, receiver must
     also show brief control (filters fly-by false receptions on quick plays).
   - longer gaps still need ``min_arrival_control_frames`` with the ball at
     the feet (filters early credit while the ball is still travelling in).
4. **Emit pass** — same team, frame gap in range, min ball travel, no opponent
   *touch* (control or reception) between passer and receiver, dedupe nearby
   duplicates.
5. **Emit turnover** — team A releases the ball; an opponent touches it before
   any teammate of A arrives (snapshot release on first opponent touch). When
   that opponent later completes a pass, attribute the turnover to their
   first touch frame. Teammate arrivals are ignored if an opponent controlled
   the ball in between.

Distance gates: tight *control* at the feet; looser *reception* for first
contact / one-touch. Anchors survive in-flight stretches (ball visible but no
player in range) and brief ball-detection dropouts (``missing_ball_tolerance``).
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
    Carrier,
    ball_xy,
    bbox_center_xy,
    feet_xy,
    find_active_carrier,
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
    is_valid_possession_touch,
    reception_aerial_veto_threshold,
)
from world_cup_projects.common.soccernet import ROLE_GOALKEEPER
from world_cup_projects.common.tracking_facing import carrier_kalman_direction
from world_cup_projects.pass_alternatives.pass_options import (
    PassOption,
    PassWeights,
    score_pass_options,
    top_pass_options,
)
from world_cup_projects.player_stats.carrier_tracking import (
    CarrierFrameState,
    CarrierTrackingConfig,
    build_carrier_timeline,
)

DetectionIterator = Iterator[tuple[int, sv.Detections]]


@dataclass(frozen=True)
class PassDetectionConfig:
    """Heuristic gates for carrier-to-carrier pass inference.

    Defaults assume ~25 fps. Frame counts can be scaled via
    :meth:`for_frame_rate` when working at a different rate.
    """

    min_carrier_gap_frames: int = 1
    max_pass_gap_frames: int = 110  # ~4.4 s at 25 fps; long aerials + ball dropouts
    min_ball_travel_m: float = 1.0
    min_ball_travel_px: float = 25.0
    control_max_distance_m: float = CONTROL_MAX_DISTANCE_M
    control_max_distance_px: float = CONTROL_MAX_DISTANCE_PX
    reception_max_distance_m: float = RECEPTION_MAX_DISTANCE_M
    reception_max_distance_px: float = RECEPTION_MAX_DISTANCE_PX
    missing_ball_tolerance: int = 10  # ~0.4 s at 25 fps; bridge intermittent ball loss
    dedupe_window_frames: int = 12
    min_arrival_frames: int = 3
    min_reception_arrival_frames: int = 2
    min_arrival_control_frames: int = 1
    min_control_frames: int = 3
    min_gk_control_frames: int = 1
    pre_flight_release_window: int = 10  # ~0.4 s before ball leaves range
    adjacent_pass_max_gap_frames: int = 15  # quick plays need control at receiver
    aerial_dy_threshold_px: float = AERIAL_DY_THRESHOLD_PX
    max_plausible_travel_m: float = 40.0
    min_long_gap_opponent_control_streak: int = 2
    transit_min_speed_m_s: float = 9.0
    transit_min_feet_px: float = 30.0
    transit_min_speed_px_per_frame: float = 10.0
    max_plausible_transit_speed_m_s: float = 35.0
    ball_speed_lookback_frames: int = 10
    ball_speed_min_lookback_frames: int = 3

    def for_frame_rate(self, fps: float, *, base_fps: float = 25.0) -> PassDetectionConfig:
        """Scale time-based frame counts for a different video frame rate."""
        if fps <= 0:
            return self
        scale = fps / base_fps
        return PassDetectionConfig(
            min_carrier_gap_frames=self.min_carrier_gap_frames,
            max_pass_gap_frames=max(1, round(self.max_pass_gap_frames * scale)),
            min_ball_travel_m=self.min_ball_travel_m,
            min_ball_travel_px=self.min_ball_travel_px,
            control_max_distance_m=self.control_max_distance_m,
            control_max_distance_px=self.control_max_distance_px,
            reception_max_distance_m=self.reception_max_distance_m,
            reception_max_distance_px=self.reception_max_distance_px,
            missing_ball_tolerance=max(1, round(self.missing_ball_tolerance * scale)),
            dedupe_window_frames=max(1, round(self.dedupe_window_frames * scale)),
            min_arrival_frames=max(1, round(self.min_arrival_frames * scale)),
            min_reception_arrival_frames=max(
                1, round(self.min_reception_arrival_frames * scale)
            ),
            min_arrival_control_frames=max(
                1, round(self.min_arrival_control_frames * scale)
            ),
            min_control_frames=max(1, round(self.min_control_frames * scale)),
            min_gk_control_frames=max(1, round(self.min_gk_control_frames * scale)),
            pre_flight_release_window=max(
                1, round(self.pre_flight_release_window * scale)
            ),
            adjacent_pass_max_gap_frames=max(
                1, round(self.adjacent_pass_max_gap_frames * scale)
            ),
            aerial_dy_threshold_px=self.aerial_dy_threshold_px,
            max_plausible_travel_m=self.max_plausible_travel_m,
            min_long_gap_opponent_control_streak=self.min_long_gap_opponent_control_streak,
            transit_min_speed_m_s=self.transit_min_speed_m_s,
            transit_min_feet_px=self.transit_min_feet_px,
            transit_min_speed_px_per_frame=self.transit_min_speed_px_per_frame,
            max_plausible_transit_speed_m_s=self.max_plausible_transit_speed_m_s,
            ball_speed_lookback_frames=max(
                1, round(self.ball_speed_lookback_frames * scale)
            ),
            ball_speed_min_lookback_frames=max(
                1, round(self.ball_speed_min_lookback_frames * scale)
            ),
        )

    def touch_validation_config(self) -> TouchValidationConfig:
        return TouchValidationConfig(
            aerial_dy_threshold_px=self.aerial_dy_threshold_px,
            transit_min_speed_m_s=self.transit_min_speed_m_s,
            transit_min_feet_px=self.transit_min_feet_px,
            transit_min_speed_px_per_frame=self.transit_min_speed_px_per_frame,
            max_plausible_transit_speed_m_s=self.max_plausible_transit_speed_m_s,
            ball_speed_lookback_frames=self.ball_speed_lookback_frames,
            ball_speed_min_lookback_frames=self.ball_speed_min_lookback_frames,
        )

    def tracking_config(self) -> CarrierTrackingConfig:
        return CarrierTrackingConfig(
            control_max_distance_m=self.control_max_distance_m,
            control_max_distance_px=self.control_max_distance_px,
            reception_max_distance_m=self.reception_max_distance_m,
            reception_max_distance_px=self.reception_max_distance_px,
            min_pass_gap_frames=self.min_carrier_gap_frames,
            max_pass_gap_frames=self.max_pass_gap_frames,
            min_arrival_frames=self.min_arrival_frames,
            min_reception_arrival_frames=self.min_reception_arrival_frames,
            min_arrival_control_frames=self.min_arrival_control_frames,
            min_control_frames=self.min_control_frames,
            min_gk_control_frames=self.min_gk_control_frames,
            pre_flight_release_window=self.pre_flight_release_window,
            adjacent_pass_max_gap_frames=self.adjacent_pass_max_gap_frames,
            aerial_dy_threshold_px=self.aerial_dy_threshold_px,
            missing_ball_tolerance=self.missing_ball_tolerance,
        )


@dataclass(frozen=True)
class InferredPass:
    """One directed pass A -> B at the release frame."""

    frame_idx: int
    passer_tid: int
    receiver_tid: int
    team: int
    gap_frames: int
    pass_length_m: float | None
    quality_score: float | None
    openness: float | None
    forward_gain: float | None
    rivals_in_lane: int | None
    motion_alignment: float | None
    receiver_space: float | None
    touch_kind: str = "control"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class InferredTurnover:
    """Possession lost: passer released the ball, opponent received it in-flight."""

    release_frame: int
    interception_frame: int
    passer_tid: int
    passer_team: int
    interceptor_tid: int
    interceptor_team: int
    gap_frames: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class PossessionScanResult:
    """Completed passes and interceptions from one scan of the clip."""

    passes: tuple[InferredPass, ...]
    turnovers: tuple[InferredTurnover, ...]


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

    def top_options(
        self,
        frame_idx: int,
        dets: sv.Detections,
        carrier: Carrier,
        k: int = 3,
    ) -> list[PassOption]:
        """Return the top K pass options for the current carrier."""
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
                return []
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
                k=k,
                weights=self._weights,
                attack_dir=attack_dir,
                positions=pitch_feet,
                carrier_motion_dir=motion_dir,
                pitch_cm=pitch_cm,
                body_pitch_m=body_pitch_m,
            )
        return top_pass_options(
            dets,
            carrier,
            k=k,
            weights=self._weights,
            carrier_motion_dir=motion_dir,
        )

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
) -> float | None:
    """Distance the ball moved between two carrier frames (m or px)."""
    travel_px = float(np.linalg.norm(ball_to - ball_from))
    if metric and transformer_from is not None and transformer_to is not None:
        p0 = image_to_pitch_m(np.array([ball_from], dtype=np.float32), transformer_from)
        p1 = image_to_pitch_m(np.array([ball_to], dtype=np.float32), transformer_to)
        if p0 is not None and p1 is not None:
            travel_m = float(np.linalg.norm(p1[0] - p0[0]))
            return travel_m
        return travel_px
    return travel_px


def _effective_ball_travel(
    ball_from: np.ndarray,
    ball_to: np.ndarray,
    *,
    metric: bool,
    transformer_from,
    transformer_to,
    config: PassDetectionConfig,
) -> float | None:
    """Ball travel with pixel fallback when pitch projection is unreliable."""
    travel = _ball_travel(
        ball_from,
        ball_to,
        metric=metric,
        transformer_from=transformer_from,
        transformer_to=transformer_to,
    )
    if travel is None:
        return None
    if metric and travel > config.max_plausible_travel_m:
        travel_px = float(np.linalg.norm(ball_to - ball_from))
        return travel_px
    return travel


def _ball_speed_reference(
    frames_by_idx: dict[int, sv.Detections],
    frame_idx: int,
    *,
    max_lookback: int,
    min_lookback: int = 3,
    transformers: dict[int, object] | None = None,
    metric: bool = False,
) -> tuple[np.ndarray | None, int, object | None]:
    """Nearest prior ball position when the previous frame has no detection (up to ``max_lookback``)."""
    for gap in range(max(min_lookback, 2), max_lookback + 1):
        older = frames_by_idx.get(frame_idx - gap)
        if older is None:
            continue
        older_ball = ball_xy(older)
        if older_ball is None:
            continue
        prev_t = transformers.get(frame_idx - gap) if metric and transformers else None
        return np.asarray(older_ball, dtype=np.float64), gap, prev_t
    return None, 1, None


def _touch_validation_kwargs(
    frames_by_idx: dict[int, sv.Detections],
    frame_idx: int,
    *,
    transformers: dict[int, object],
    metric: bool,
    fps: float,
    max_lookback: int = 10,
    min_lookback: int = 3,
) -> dict:
    transformer = transformers.get(frame_idx) if metric else None
    prev = frames_by_idx.get(frame_idx - 1)
    prev_ball = ball_xy(prev) if prev is not None else None
    prev_transformer = (
        transformers.get(frame_idx - 1) if metric and prev_ball is not None else None
    )
    speed_prev_ball = (
        np.asarray(prev_ball, dtype=np.float64) if prev_ball is not None else None
    )
    speed_frame_gap = 1
    speed_prev_transformer = prev_transformer
    if speed_prev_ball is None and transformer is None:
        speed_prev_ball, speed_frame_gap, speed_prev_transformer = _ball_speed_reference(
            frames_by_idx,
            frame_idx,
            max_lookback=max_lookback,
            min_lookback=min_lookback,
            transformers=transformers,
            metric=metric,
        )
    return {
        "prev_ball": np.asarray(prev_ball, dtype=np.float64) if prev_ball is not None else None,
        "speed_prev_ball": speed_prev_ball,
        "speed_frame_gap": speed_frame_gap,
        "speed_prev_transformer": speed_prev_transformer,
        "fps": fps,
        "transformer": transformer,
    }


def _first_opponent_touch_in_window(
    frames_by_idx: dict[int, sv.Detections],
    *,
    start_frame: int,
    end_frame: int,
    passer_team: int,
    config: PassDetectionConfig,
    transformers: dict[int, object],
    metric: bool,
    require_control: bool = False,
    player_tid: int | None = None,
    fps: float = 25.0,
) -> tuple[int, int, str] | None:
    """First ``(frame, opponent_tid, touch_kind)`` matching the demo opponent-touch gate."""
    touch_cfg = config.touch_validation_config()
    for frame_idx in range(start_frame + 1, end_frame):
        dets = frames_by_idx.get(frame_idx)
        if dets is None:
            continue
        transformer = transformers.get(frame_idx) if metric else None
        carrier, touch_kind = _active_carrier(
            dets, transformer=transformer, config=config
        )
        touch_kind = touch_kind or "reception"
        if carrier is None or int(carrier.team) == passer_team:
            continue
        if require_control and touch_kind != "control":
            continue
        if dets.tracker_id is None:
            continue
        tid = int(dets.tracker_id[carrier.index])
        if player_tid is not None and tid != player_tid:
            continue
        if not is_valid_possession_touch(
            dets,
            carrier,
            touch_kind=touch_kind,
            config=touch_cfg,
            **_touch_validation_kwargs(
                frames_by_idx,
                frame_idx,
                transformers=transformers,
                metric=metric,
                fps=fps,
                max_lookback=config.ball_speed_lookback_frames,
                min_lookback=config.ball_speed_min_lookback_frames,
            ),
        ):
            continue
        return frame_idx, tid, touch_kind
    return None


def _opponent_control_streak_between(
    frames_by_idx: dict[int, sv.Detections],
    *,
    start_frame: int,
    end_frame: int,
    passer_team: int,
    config: PassDetectionConfig,
    transformers: dict[int, object],
    metric: bool,
    min_streak: int,
    fps: float = 25.0,
) -> bool:
    """True if an opponent had ``min_streak`` consecutive control frames in window."""
    touch_cfg = config.touch_validation_config()
    streak = 0
    for frame_idx in range(start_frame + 1, end_frame):
        dets = frames_by_idx.get(frame_idx)
        if dets is None:
            streak = 0
            continue
        transformer = transformers.get(frame_idx) if metric else None
        carrier, touch_kind = _active_carrier(
            dets, transformer=transformer, config=config
        )
        touch_kind = touch_kind or "reception"
        if carrier is None or int(carrier.team) == passer_team:
            streak = 0
            continue
        if touch_kind != "control":
            streak = 0
            continue
        if not is_valid_possession_touch(
            dets,
            carrier,
            touch_kind=touch_kind,
            config=touch_cfg,
            **_touch_validation_kwargs(
                frames_by_idx,
                frame_idx,
                transformers=transformers,
                metric=metric,
                fps=fps,
                max_lookback=config.ball_speed_lookback_frames,
                min_lookback=config.ball_speed_min_lookback_frames,
            ),
        ):
            streak = 0
            continue
        streak += 1
        if streak >= min_streak:
            return True
    return False


def _opponent_blocks_between(
    frames_by_idx: dict[int, sv.Detections],
    *,
    start_frame: int,
    end_frame: int,
    passer_team: int,
    gap_frames: int,
    config: PassDetectionConfig,
    transformers: dict[int, object],
    metric: bool,
    fps: float = 25.0,
) -> bool:
    """Short gaps: any opponent touch; long gaps: sustained opponent control."""
    if gap_frames >= config.adjacent_pass_max_gap_frames:
        return _opponent_control_streak_between(
            frames_by_idx,
            start_frame=start_frame,
            end_frame=end_frame,
            passer_team=passer_team,
            config=config,
            transformers=transformers,
            metric=metric,
            min_streak=config.min_long_gap_opponent_control_streak,
            fps=fps,
        )
    return _opponent_touch_between(
        frames_by_idx,
        start_frame=start_frame,
        end_frame=end_frame,
        passer_team=passer_team,
        config=config,
        transformers=transformers,
        metric=metric,
        require_control=False,
        fps=fps,
    )


def _opponent_touch_between(
    frames_by_idx: dict[int, sv.Detections],
    *,
    start_frame: int,
    end_frame: int,
    passer_team: int,
    config: PassDetectionConfig,
    transformers: dict[int, object],
    metric: bool,
    require_control: bool = False,
    fps: float = 25.0,
) -> bool:
    """True if an opponent had a valid touch between two teammate beats."""
    return (
        _first_opponent_touch_in_window(
            frames_by_idx,
            start_frame=start_frame,
            end_frame=end_frame,
            passer_team=passer_team,
            config=config,
            transformers=transformers,
            metric=metric,
            require_control=require_control,
            fps=fps,
        )
        is not None
    )


def _opponent_control_between(
    frames_by_idx: dict[int, sv.Detections],
    *,
    start_frame: int,
    end_frame: int,
    passer_team: int,
    config: PassDetectionConfig,
    transformers: dict[int, object],
    metric: bool,
    fps: float = 25.0,
) -> bool:
    """True if an opponent had sustained control between two teammate touches."""
    return _opponent_touch_between(
        frames_by_idx,
        start_frame=start_frame,
        end_frame=end_frame,
        passer_team=passer_team,
        config=config,
        transformers=transformers,
        metric=metric,
        require_control=True,
        fps=fps,
    )


def _min_travel_threshold(config: PassDetectionConfig, *, metric: bool) -> float:
    return config.min_ball_travel_m if metric else config.min_ball_travel_px


def _try_emit_pass(
    events: list[InferredPass],
    *,
    release_frame: int,
    release_dets: sv.Detections,
    release_carrier: Carrier,
    passer_tid: int,
    receiver_tid: int,
    arrival_frame: int,
    arrival_carrier: Carrier,
    touch_kind: str,
    scorer: PassQualityScorer,
    config: PassDetectionConfig,
    metric: bool,
    transformers: dict[int, object],
) -> bool:
    """Append one inferred pass if all gates pass."""
    gap = arrival_frame - release_frame
    if gap < config.min_carrier_gap_frames or gap > config.max_pass_gap_frames:
        return False

    travel = _effective_ball_travel(
        release_carrier.ball,
        arrival_carrier.ball,
        metric=metric,
        transformer_from=transformers.get(release_frame),
        transformer_to=transformers.get(arrival_frame),
        config=config,
    )
    min_travel = _min_travel_threshold(config, metric=metric)
    if travel is None or travel < min_travel:
        return False

    if _recent_duplicate(
        events,
        passer_tid,
        receiver_tid,
        release_frame,
        window=config.dedupe_window_frames,
    ):
        return False

    option = scorer.option_for_receiver(
        release_frame, release_dets, release_carrier, receiver_tid
    )
    events.append(
        InferredPass(
            frame_idx=release_frame,
            passer_tid=passer_tid,
            receiver_tid=receiver_tid,
            team=int(release_carrier.team),
            gap_frames=gap,
            pass_length_m=_pass_length_m(option, metric),
            quality_score=float(option.score) if option else None,
            openness=float(option.openness) if option else None,
            forward_gain=float(option.forward_gain) if option else None,
            rivals_in_lane=int(option.rivals_in_lane) if option else None,
            motion_alignment=float(option.motion_alignment) if option else None,
            receiver_space=float(option.receiver_space) if option else None,
            touch_kind=touch_kind,
        )
    )
    return True


def _pass_length_m(option: PassOption | None, metric: bool) -> float | None:
    if option is None:
        return None
    if metric:
        return float(option.length)
    return None


def _active_carrier(
    dets: sv.Detections,
    *,
    transformer,
    config: PassDetectionConfig,
) -> tuple[Carrier | None, str | None]:
    return find_active_carrier(
        dets,
        transformer=transformer,
        control_max_distance_px=config.control_max_distance_px,
        control_max_distance_m=config.control_max_distance_m,
        reception_max_distance_px=config.reception_max_distance_px,
        reception_max_distance_m=config.reception_max_distance_m,
    )


@dataclass
class _TeamPossessionState:
    release: tuple[int, sv.Detections, Carrier, int] | None = None
    last_touch: tuple[int, sv.Detections, Carrier, int] | None = None
    possession_tid: int = -1
    control_streak: int = 0
    in_flight: bool = False
    arrival_candidate_tid: int = -1
    arrival_streak: int = 0
    arrival_control_streak: int = 0
    turnover_snapshot: tuple[int, int] | None = None


def _is_goalkeeper(dets: sv.Detections, carrier_index: int) -> bool:
    return int(dets.class_id[carrier_index]) == ROLE_GOALKEEPER


def _min_control_frames_for(
    dets: sv.Detections,
    carrier: Carrier,
    *,
    config: PassDetectionConfig,
) -> int:
    if _is_goalkeeper(dets, carrier.index):
        return config.min_gk_control_frames
    return config.min_control_frames


def _bridge_missing_ball(
    team_states: dict[int, _TeamPossessionState],
    *,
    frame_idx: int,
    config: PassDetectionConfig,
    frames_by_idx: dict[int, sv.Detections],
    transformers: dict[int, object],
    metric: bool,
    fps: float = 25.0,
) -> None:
    """Brief ball dropout: keep release anchor and arrival streak (like in-flight)."""
    for team_id, state in team_states.items():
        if state.release is None and state.last_touch is None:
            continue
        if not state.in_flight:
            _promote_pre_flight_release(
                state,
                frame_idx,
                config=config,
                team=team_id,
                other_state=team_states[1 - team_id],
                frames_by_idx=frames_by_idx,
                transformers=transformers,
                metric=metric,
                fps=fps,
            )
        state.in_flight = True


def _end_missing_ball_bridge(team_states: dict[int, _TeamPossessionState]) -> None:
    """Ball gone too long — stop bridging and let stale-release expiry run."""
    for state in team_states.values():
        state.in_flight = False
        if state.release is not None:
            continue
        state.arrival_candidate_tid = -1
        state.arrival_streak = 0
        state.arrival_control_streak = 0


def _promote_pre_flight_release(
    state: _TeamPossessionState,
    frame_idx: int,
    *,
    config: PassDetectionConfig,
    team: int | None = None,
    other_state: _TeamPossessionState | None = None,
    frames_by_idx: dict[int, sv.Detections] | None = None,
    transformers: dict[int, object] | None = None,
    metric: bool = False,
    fps: float = 25.0,
) -> None:
    """Credit a brief touch as the passer when the ball immediately goes in-flight."""
    if state.last_touch is None:
        return
    touch_frame, touch_dets, touch_carrier, touch_tid = state.last_touch
    if frame_idx - touch_frame > config.pre_flight_release_window:
        return
    touch_cfg = config.touch_validation_config()
    if is_aerial_flyby_below_feet(
        touch_dets,
        touch_carrier,
        threshold_px=reception_aerial_veto_threshold(touch_cfg),
    ):
        return
    if state.release is not None and state.release[3] == touch_tid:
        return
    if state.release is not None and state.release[3] != touch_tid:
        release_frame = state.release[0]
        opponent_between = False
        if (
            team is not None
            and frames_by_idx is not None
            and transformers is not None
        ):
            opponent_between = _opponent_blocks_between(
                frames_by_idx,
                start_frame=release_frame,
                end_frame=touch_frame,
                passer_team=team,
                gap_frames=touch_frame - release_frame,
                config=config,
                transformers=transformers,
                metric=metric,
                fps=fps,
            )
        other_in_play = (
            other_state is not None
            and other_state.release is not None
            and touch_frame > other_state.release[0]
        )
        if opponent_between or other_in_play:
            # Opponent has the ball or a teammate deflected en route — drop the
            # stale passer anchor; a fly-by near another teammate is not a new pass.
            state.release = None
            state.arrival_candidate_tid = -1
            state.arrival_streak = 0
            state.arrival_control_streak = 0
        return
    if state.release is not None:
        return
    state.release = (touch_frame, touch_dets, touch_carrier, touch_tid)


def _recent_duplicate(
    events: list[InferredPass],
    passer_tid: int,
    receiver_tid: int,
    frame_idx: int,
    *,
    window: int,
) -> bool:
    for event in reversed(events):
        if frame_idx - event.frame_idx > window:
            break
        if event.passer_tid == passer_tid and event.receiver_tid == receiver_tid:
            return True
    return False


def _confirm_release(
    state: _TeamPossessionState,
    frame_idx: int,
    dets: sv.Detections,
    carrier: Carrier,
    tid: int,
) -> None:
    state.release = (frame_idx, dets, carrier, tid)
    if state.turnover_snapshot is not None:
        state.turnover_snapshot = (frame_idx, tid)


def _recent_turnover_duplicate(
    events: list[InferredTurnover],
    passer_tid: int,
    interceptor_tid: int,
    release_frame: int,
    *,
    window: int,
) -> bool:
    for event in reversed(events):
        if release_frame - event.release_frame > window:
            break
        if (
            event.passer_tid == passer_tid
            and event.interceptor_tid == interceptor_tid
        ):
            return True
    return False


def _try_emit_turnover(
    losing_state: _TeamPossessionState,
    *,
    interceptor_tid: int,
    interceptor_team: int,
    interception_frame: int,
    turnovers: list[InferredTurnover],
    config: PassDetectionConfig,
    release_frame: int | None = None,
    passer_tid: int | None = None,
) -> bool:
    """Rule 5: opponent takes the ball while our release is still pending."""
    if release_frame is None or passer_tid is None:
        release = losing_state.release
        if release is None:
            return False
        release_frame, _, _, passer_tid = release
    gap = interception_frame - release_frame
    if gap < config.min_carrier_gap_frames or gap > config.max_pass_gap_frames:
        return False

    if _recent_turnover_duplicate(
        turnovers,
        passer_tid,
        interceptor_tid,
        release_frame,
        window=config.dedupe_window_frames,
    ):
        return False

    turnovers.append(
        InferredTurnover(
            release_frame=release_frame,
            interception_frame=interception_frame,
            passer_tid=passer_tid,
            passer_team=1 - interceptor_team,
            interceptor_tid=interceptor_tid,
            interceptor_team=interceptor_team,
            gap_frames=gap,
        )
    )
    losing_state.release = None
    losing_state.in_flight = False
    losing_state.arrival_candidate_tid = -1
    losing_state.arrival_streak = 0
    losing_state.arrival_control_streak = 0
    losing_state.turnover_snapshot = None
    return True


def _min_arrival_frames_for(
    *,
    touch_kind: str,
    gap_frames: int,
    config: PassDetectionConfig,
) -> int:
    """Reception-only paths on longer gaps need fewer consecutive touches."""
    if (
        touch_kind == "reception"
        and gap_frames >= config.adjacent_pass_max_gap_frames
    ):
        return config.min_reception_arrival_frames
    return config.min_arrival_frames


def _arrival_ready_for_pass(
    *,
    arrival_streak: int,
    arrival_control_streak: int,
    touch_kind: str,
    gap_frames: int,
    config: PassDetectionConfig,
) -> bool:
    """Streak length + control-at-feet gate (stricter on quick adjacent plays)."""
    if arrival_streak < _min_arrival_frames_for(
        touch_kind=touch_kind, gap_frames=gap_frames, config=config
    ):
        return False
    adjacent = gap_frames < config.adjacent_pass_max_gap_frames
    if adjacent:
        return arrival_control_streak >= config.min_control_frames
    if touch_kind == "reception":
        return True
    return arrival_control_streak >= config.min_arrival_control_frames


def _on_confirmed_possession(
    state: _TeamPossessionState,
    frame_idx: int,
    dets: sv.Detections,
    carrier: Carrier,
    tid: int,
) -> None:
    """Credit sustained possession on this team."""
    if state.release is not None and state.release[3] != tid:
        return
    _confirm_release(state, frame_idx, dets, carrier, tid)


def _first_valid_touch_frame(
    frames_by_idx: dict[int, sv.Detections],
    *,
    player_tid: int,
    start_frame: int,
    end_frame: int,
    config: PassDetectionConfig,
    transformers: dict[int, object],
    metric: bool,
    fps: float = 25.0,
) -> int | None:
    """First frame in ``(start_frame, end_frame)`` where ``player_tid`` has a valid touch."""
    touch_cfg = config.touch_validation_config()
    for frame_idx in range(start_frame + 1, end_frame):
        dets = frames_by_idx.get(frame_idx)
        if dets is None:
            continue
        transformer = transformers.get(frame_idx) if metric else None
        carrier, touch_kind = _active_carrier(
            dets, transformer=transformer, config=config
        )
        touch_kind = touch_kind or "reception"
        if carrier is None or not is_valid_possession_touch(
            dets,
            carrier,
            touch_kind=touch_kind,
            config=touch_cfg,
            **_touch_validation_kwargs(
                frames_by_idx,
                frame_idx,
                transformers=transformers,
                metric=metric,
                fps=fps,
                max_lookback=config.ball_speed_lookback_frames,
                min_lookback=config.ball_speed_min_lookback_frames,
            ),
        ):
            continue
        if dets.tracker_id is None:
            continue
        if int(dets.tracker_id[carrier.index]) == player_tid:
            return frame_idx
    return None


def scan_possession_events(
    detections_iter: DetectionIterator,
    *,
    scorer: PassQualityScorer,
    config: PassDetectionConfig = PassDetectionConfig(),
    metric: bool = True,
    transformers: dict[int, object] | None = None,
    fps: float = 25.0,
) -> PossessionScanResult:
    """Scan tracked detections for passes (rule 4) and turnovers (rule 5)."""
    transformers = transformers or {}
    passes: list[InferredPass] = []
    turnovers: list[InferredTurnover] = []
    frames_by_idx: dict[int, sv.Detections] = {}
    team_states: dict[int, _TeamPossessionState] = {
        0: _TeamPossessionState(),
        1: _TeamPossessionState(),
    }
    missing_ball_streak = 0

    def _expire_stale_releases(frame_idx: int) -> None:
        for state in team_states.values():
            if state.release is None or state.in_flight:
                continue
            release_frame = state.release[0]
            if frame_idx - release_frame > config.max_pass_gap_frames:
                state.release = None
                state.arrival_candidate_tid = -1
                state.arrival_streak = 0
                state.arrival_control_streak = 0

    def _update_control_streak(state: _TeamPossessionState, tid: int) -> None:
        if tid == state.possession_tid:
            state.control_streak += 1
        else:
            state.possession_tid = tid
            state.control_streak = 1

    for frame_idx, dets in detections_iter:
        frames_by_idx[frame_idx] = dets
        _expire_stale_releases(frame_idx)
        ball = ball_xy(dets)
        if ball is None:
            missing_ball_streak += 1
        else:
            missing_ball_streak = 0

        transformer = transformers.get(frame_idx) if metric else None
        carrier, touch_kind = _active_carrier(
            dets, transformer=transformer, config=config
        )

        touch_kind = touch_kind or "reception"
        if carrier is None or not is_valid_possession_touch(
            dets,
            carrier,
            touch_kind=touch_kind,
            config=config.touch_validation_config(),
            **_touch_validation_kwargs(
                frames_by_idx,
                frame_idx,
                transformers=transformers,
                metric=metric,
                fps=fps,
                max_lookback=config.ball_speed_lookback_frames,
                min_lookback=config.ball_speed_min_lookback_frames,
            ),
        ):
            if ball is not None:
                for team_id, state in team_states.items():
                    if not state.in_flight:
                        _promote_pre_flight_release(
                            state,
                            frame_idx,
                            config=config,
                            team=team_id,
                            other_state=team_states[1 - team_id],
                            frames_by_idx=frames_by_idx,
                            transformers=transformers,
                            metric=metric,
                            fps=fps,
                        )
                    state.in_flight = True
                    if state.release is None:
                        state.arrival_candidate_tid = -1
                        state.arrival_streak = 0
                        state.arrival_control_streak = 0
            elif missing_ball_streak <= config.missing_ball_tolerance:
                _bridge_missing_ball(
                    team_states,
                    frame_idx=frame_idx,
                    config=config,
                    frames_by_idx=frames_by_idx,
                    transformers=transformers,
                    metric=metric,
                    fps=fps,
                )
            else:
                _end_missing_ball_bridge(team_states)
            continue

        tid = int(dets.tracker_id[carrier.index]) if dets.tracker_id is not None else -1
        team = int(carrier.team)
        if tid < 0 or team not in (0, 1):
            continue

        state = team_states[team]
        other_state = team_states[1 - team]
        if (
            other_state.release is not None
            and frame_idx > other_state.release[0]
        ):
            other_state.turnover_snapshot = (
                other_state.release[0],
                other_state.release[3],
            )

        state.in_flight = False
        state.last_touch = (frame_idx, dets, carrier, tid)
        min_control = _min_control_frames_for(dets, carrier, config=config)

        if touch_kind == "control":
            _update_control_streak(state, tid)
            if state.control_streak >= min_control:
                _on_confirmed_possession(state, frame_idx, dets, carrier, tid)
        else:
            state.possession_tid = tid
            state.control_streak = 0
            if _is_goalkeeper(dets, carrier.index):
                _on_confirmed_possession(state, frame_idx, dets, carrier, tid)

        release = state.release
        if release is None:
            continue

        release_frame, release_dets, release_carrier, release_tid = release

        if tid == release_tid:
            state.release = (frame_idx, dets, carrier, tid)
            state.arrival_candidate_tid = -1
            state.arrival_streak = 0
            state.arrival_control_streak = 0
            continue

        if tid == state.arrival_candidate_tid:
            state.arrival_streak += 1
            if touch_kind == "control":
                state.arrival_control_streak += 1
            else:
                state.arrival_control_streak = 0
        else:
            state.arrival_candidate_tid = tid
            state.arrival_streak = 1
            state.arrival_control_streak = 1 if touch_kind == "control" else 0

        gap_frames = frame_idx - release_frame
        arrival_ready = _arrival_ready_for_pass(
            arrival_streak=state.arrival_streak,
            arrival_control_streak=state.arrival_control_streak,
            touch_kind=touch_kind,
            gap_frames=gap_frames,
            config=config,
        )

        if not arrival_ready:
            continue

        opponent_blocked = _opponent_blocks_between(
            frames_by_idx,
            start_frame=release_frame,
            end_frame=frame_idx,
            passer_team=team,
            gap_frames=gap_frames,
            config=config,
            transformers=transformers,
            metric=metric,
            fps=fps,
        )
        if opponent_blocked:
            # Stale anchors survive opponent possession; credit the arriving teammate
            # without emitting a pass through the press.
            _confirm_release(state, frame_idx, dets, carrier, tid)
            state.arrival_candidate_tid = -1
            state.arrival_streak = 0
            state.arrival_control_streak = 0
            continue

        if _try_emit_pass(
            passes,
            release_frame=release_frame,
            release_dets=release_dets,
            release_carrier=release_carrier,
            passer_tid=release_tid,
            receiver_tid=tid,
            arrival_frame=frame_idx,
            arrival_carrier=carrier,
            touch_kind=touch_kind,
            scorer=scorer,
            config=config,
            metric=metric,
            transformers=transformers,
        ):
            losing = other_state
            snapshot = losing.turnover_snapshot
            if snapshot is not None:
                losing_release_frame, losing_passer_tid = snapshot
                intercept_frame = _first_valid_touch_frame(
                    frames_by_idx,
                    player_tid=release_tid,
                    start_frame=losing_release_frame,
                    end_frame=frame_idx,
                    config=config,
                    transformers=transformers,
                    metric=metric,
                    fps=fps,
                )
                if intercept_frame is None:
                    intercept_frame = release_frame
                if intercept_frame is not None and _opponent_control_between(
                    frames_by_idx,
                    start_frame=losing_release_frame,
                    end_frame=intercept_frame,
                    passer_team=1 - team,
                    config=config,
                    transformers=transformers,
                    metric=metric,
                    fps=fps,
                ):
                    _try_emit_turnover(
                        losing,
                        interceptor_tid=release_tid,
                        interceptor_team=team,
                        interception_frame=intercept_frame,
                        turnovers=turnovers,
                        config=config,
                        release_frame=losing_release_frame,
                        passer_tid=losing_passer_tid,
                    )
        _confirm_release(state, frame_idx, dets, carrier, tid)
        if not _opponent_control_between(
            frames_by_idx,
            start_frame=release_frame,
            end_frame=frame_idx,
            passer_team=team,
            config=config,
            transformers=transformers,
            metric=metric,
        ):
            state.turnover_snapshot = None
        state.arrival_candidate_tid = -1
        state.arrival_streak = 0
        state.arrival_control_streak = 0

    return PossessionScanResult(
        passes=tuple(passes),
        turnovers=tuple(turnovers),
    )


def detect_pass_events(
    detections_iter: DetectionIterator,
    *,
    scorer: PassQualityScorer,
    config: PassDetectionConfig = PassDetectionConfig(),
    metric: bool = True,
    transformers: dict[int, object] | None = None,
) -> list[InferredPass]:
    """Return completed passes only (see :func:`scan_possession_events` for turnovers)."""
    return list(
        scan_possession_events(
            detections_iter,
            scorer=scorer,
            config=config,
            metric=metric,
            transformers=transformers,
        ).passes
    )


def build_pass_carrier_timeline(
    detections_iter: DetectionIterator,
    *,
    config: PassDetectionConfig = PassDetectionConfig(),
    metric: bool = True,
    transformers: dict[int, object] | None = None,
) -> list[CarrierFrameState]:
    """Per-frame carrier signals for debug overlays."""
    return build_carrier_timeline(
        detections_iter,
        config=config.tracking_config(),
        metric=metric,
        transformers=transformers,
    )
