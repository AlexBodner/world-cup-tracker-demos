"""Filmstrip turnover explain — pass attempt tracked until opponent intercepts.

Uses the same touch/control gates as ``pass_events.scan_possession_events`` (demo
logic). Explain-only choices are limited to frame picking for panels, visual
lock timing, and transit fly-by filtering for inflight panels — never alternate
possession rules or shared detection helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import supervision as sv

from world_cup_projects.common.visual import (
    annotate_ball,
    draw_branding_tag,
    draw_carrier_pulse,
    draw_carrier_spotlight,
    draw_hud_bar,
    draw_text_shadow,
)
from world_cup_projects.player_stats.carrier_tracking import CarrierFrameState
from world_cup_projects.player_stats.pass_events import (
    InferredPass,
    InferredTurnover,
    PassDetectionConfig,
    _first_opponent_touch_in_window,
    _first_valid_touch_frame,
    _opponent_control_between,
)
from world_cup_projects.player_stats.pass_network_render import (
    TURNOVER_ACCENT_BGR,
    _get_player_box,
    _team_color,
)
from world_cup_projects.common.possession import ball_xy, feet_xy
from world_cup_projects.player_stats.pass_explain_visual import (
    AERIAL_DY_THRESHOLD_PX,
    PassExplainContext,
    PassExplainVideoTiming,
    PassStripPlan,
    ROBOFLOW_PURPLE_BGR,
    _BRANDING,
    _FOCUS_DIM,
    _MIN_ARRIVAL,
    _MIN_CONTROL,
    _PANEL_BG_BGR,
    _RECEIVER_EXPLAIN_TIGHT_PX,
    _SPOTLIGHT_STRENGTH,
    _action_focus_points,
    _apply_focus_spotlights,
    _build_flight_frame,
    _build_passer_control_frame,
    _compose_panel_row,
    _compose_strip,
    _crop_rect_from_points,
    _dim_frame,
    _draw_anchor_player,
    _draw_dashed_line,
    _draw_focus_player,
    _draw_pass_ball_marker,
    _draw_strip_title,
    _explain_streak_sublabel,
    _feet_for_tid,
    _find_passer_control_run,
    _find_passer_touch_run,
    _find_receiver_control_run,
    _find_receiver_tight_feet_frame,
    _fit_social_square,
    _letterbox_frame,
    _pass_anchor_points,
    _pass_ball_for_ctx,
    _pass_ball_point_for_frame,
    _passer_panel_badge,
    _pick_flight_frame,
    _receiver_arrival_panels,
    _receiver_confirm_search_end,
    _strip_crop_rect,
    _streak_emphasis,
    _ui_scale,
    build_pass_explain_context,
    explain_video_frame_range,
    frames_needed_for_explain,
    render_pass_detect_timeline,
    write_pass_explain_gif,
    write_pass_explain_video,
)

# Intercepts are credited on reception (~120px) before the ball visibly settles at feet.
_INTERCEPTOR_EXPLAIN_CONTACT_PX = 35.0
_INTERCEPTOR_POST_RECEIPT_LOOKAHEAD = 24

_TURNOVER_BRANDING = "Roboflow | pass detection"
_PANEL_W = 960
_PANEL_H = 540
_GUTTER_W = 152


@dataclass(frozen=True)
class TurnoverExplainLogic:
    """Detection gates surfaced in turnover explain copy."""

    release_kind: Literal["control_streak", "pre_flight", "short_touch"]
    release_strip_subtitle: str
    intercept_strip_subtitle: str
    first_opponent_touch_frame: int | None
    opponent_control_between: bool
    gap_frames: int
    min_gap_met: bool
    pre_flight_release_window: int
    inflight_opponent_control_frame: int | None
    inflight_opponent_control_tid: int | None
    demo_inflight_opponent_control_frame: int | None
    demo_inflight_opponent_control_tid: int | None
    inflight_flyby_skipped: bool
    interceptor_control_frame: int | None


@dataclass(frozen=True)
class TurnoverExplainContext:
    """Turnover event plus a pass-shaped explain context for shared strip logic."""

    turnover: InferredTurnover
    pass_ctx: PassExplainContext
    logic: TurnoverExplainLogic


def turnover_to_pass_adapter(turnover: InferredTurnover) -> InferredPass:
    """Map turnover onto pass-explain frame helpers (interceptor = pseudo-receiver)."""
    return InferredPass(
        frame_idx=turnover.release_frame,
        passer_tid=turnover.passer_tid,
        receiver_tid=turnover.interceptor_tid,
        team=turnover.passer_team,
        gap_frames=turnover.gap_frames,
        pass_length_m=None,
        quality_score=None,
        openness=None,
        forward_gain=None,
        rivals_in_lane=None,
        motion_alignment=None,
        receiver_space=None,
    )


def _player_team(dets: sv.Detections, tid: int) -> int | None:
    if dets.tracker_id is None or dets.data is None:
        return None
    team_arr = dets.data.get("team")
    if team_arr is None:
        return None
    mask = dets.tracker_id == tid
    if not mask.any():
        return None
    return int(team_arr[mask][0])


# Explain-only: fast ball through the control radius without settling at feet.
_EXPLAIN_TRANSIT_MIN_BALL_SPEED_PX = 10.0
_EXPLAIN_TRANSIT_MIN_FEET_DISTANCE_PX = 30.0

_FLYBY_CHIP_BGR = (22, 24, 28)
_FLYBY_CHIP_ACCENT_BGR = (85, 140, 200)
_FLYBY_SLATE_BGR = (95, 100, 108)


def _turnover_interceptor_search_end(
    turnover: InferredTurnover,
    *,
    lookahead: int = _INTERCEPTOR_POST_RECEIPT_LOOKAHEAD,
) -> int:
    """Cap interceptor follow-up near the credited reception — not the whole clip."""
    return turnover.interception_frame + lookahead


def _find_demo_opponent_control_in_flight(
    turnover: InferredTurnover,
    dets_by_frame: dict[int, sv.Detections],
    *,
    config: PassDetectionConfig,
    transformers: dict[int, object],
    metric: bool,
) -> tuple[int | None, int | None]:
    """First opponent control in flight — same gate as ``_opponent_control_between``."""
    hit = _first_opponent_touch_in_window(
        dets_by_frame,
        start_frame=turnover.release_frame,
        end_frame=turnover.interception_frame,
        passer_team=turnover.passer_team,
        config=config,
        transformers=transformers,
        metric=metric,
        require_control=True,
    )
    if hit is None:
        return None, None
    frame_idx, tid, _kind = hit
    return frame_idx, tid


def _is_explain_transit_flyby_control(
    dets_by_frame: dict[int, sv.Detections],
    frame_idx: int,
    opponent_tid: int,
) -> bool:
    """Explain-only veto — ball still moving quickly and not tight at opponent feet."""
    dets = dets_by_frame.get(frame_idx)
    prev = dets_by_frame.get(frame_idx - 1)
    if dets is None or prev is None or dets.tracker_id is None:
        return False
    ball = ball_xy(dets)
    prev_ball = ball_xy(prev)
    if ball is None or prev_ball is None:
        return False
    speed = float(np.hypot(ball[0] - prev_ball[0], ball[1] - prev_ball[1]))
    if speed < _EXPLAIN_TRANSIT_MIN_BALL_SPEED_PX:
        return False
    mask = dets.tracker_id == opponent_tid
    if not mask.any():
        return False
    feet = feet_xy(dets)[int(np.where(mask)[0][0])]
    dist = float(np.hypot(ball[0] - feet[0], ball[1] - feet[1]))
    return dist >= _EXPLAIN_TRANSIT_MIN_FEET_DISTANCE_PX


def _resolve_inflight_opponent_control_for_explain(
    turnover: InferredTurnover,
    dets_by_frame: dict[int, sv.Detections],
    *,
    config: PassDetectionConfig,
    transformers: dict[int, object],
    metric: bool,
) -> tuple[int | None, int | None, int | None, int | None]:
    """``(show_frame, show_tid, demo_frame, demo_tid)`` for inflight panels/copy."""
    demo_fi, demo_tid = _find_demo_opponent_control_in_flight(
        turnover,
        dets_by_frame,
        config=config,
        transformers=transformers,
        metric=metric,
    )
    if demo_fi is None or demo_tid is None:
        return None, None, None, None
    # Explain never freezes non-interceptor inflight touches as possession.
    if demo_tid != turnover.interceptor_tid:
        return None, None, demo_fi, demo_tid
    if _is_explain_transit_flyby_control(dets_by_frame, demo_fi, demo_tid):
        return None, None, demo_fi, demo_tid
    return demo_fi, demo_tid, demo_fi, demo_tid


def _inflight_flyby_skipped(
    *,
    demo_frame: int | None,
    show_frame: int | None,
) -> bool:
    return demo_frame is not None and show_frame is None


def _flight_flyby_skip_copy(
    *,
    opponent_tid: int | None,
    frame: int | None,
) -> str:
    return (
        f"Demo: #{opponent_tid} control credited at f{frame} (rule 5)  ·  "
        "Explain: transit contact — not possession"
    )


def _flight_flyby_chip_lines(
    logic: TurnoverExplainLogic,
) -> tuple[str, str]:
    tid = logic.demo_inflight_opponent_control_tid
    fi = logic.demo_inflight_opponent_control_frame
    return (
        f"Rule 5 credits #{tid} at f{fi}",
        "Filtered — ball in transit, not possession",
    )


def _flight_flyby_skip_copy_from_logic(logic: TurnoverExplainLogic) -> str:
    return _flight_flyby_skip_copy(
        opponent_tid=logic.demo_inflight_opponent_control_tid,
        frame=logic.demo_inflight_opponent_control_frame,
    )


def _find_demo_interceptor_control_frame(
    turnover: InferredTurnover,
    dets_by_frame: dict[int, sv.Detections],
    *,
    config: PassDetectionConfig,
    transformers: dict[int, object],
    metric: bool,
    search_end: int,
) -> int | None:
    """First interceptor control after reception — demo ``require_control`` gate."""
    hit = _first_opponent_touch_in_window(
        dets_by_frame,
        start_frame=turnover.interception_frame,
        end_frame=search_end + 1,
        passer_team=turnover.passer_team,
        config=config,
        transformers=transformers,
        metric=metric,
        require_control=True,
        player_tid=turnover.interceptor_tid,
    )
    if hit is None:
        return None
    return hit[0]


def _find_turnover_intercept_panels(
    turnover: InferredTurnover,
    timeline_by_frame: dict[int, CarrierFrameState],
    dets_by_frame: dict[int, sv.Detections],
    *,
    config: PassDetectionConfig,
    transformers: dict[int, object],
    metric: bool,
    confirm_frame: int,
    min_arrival_frames: int = _MIN_ARRIVAL,
) -> tuple[int, ...]:
    """Reception → interceptor control → visual lock (mirrors passer control strip)."""
    event = turnover.interception_frame
    search_end = _turnover_interceptor_search_end(turnover)
    panels: list[int] = [event]
    control = _find_demo_interceptor_control_frame(
        turnover,
        dets_by_frame,
        config=config,
        transformers=transformers,
        metric=metric,
        search_end=search_end,
    )
    if control is not None and control != event:
        panels.append(control)
    if confirm_frame not in panels:
        panels.append(confirm_frame)
    panels = sorted(set(panels))
    while len(panels) < min_arrival_frames:
        prepend = panels[0] - 1
        if prepend <= turnover.release_frame:
            break
        panels.insert(0, prepend)
    if event not in panels:
        panels.append(event)
        panels = sorted(set(panels))
    return tuple(panels[-min_arrival_frames:])


def _find_turnover_release_panels(
    turnover: InferredTurnover,
    timeline_by_frame: dict[int, CarrierFrameState],
    *,
    config: PassDetectionConfig,
    min_control_frames: int = _MIN_CONTROL,
) -> tuple[tuple[int, ...], Literal["control_streak", "pre_flight", "short_touch"]]:
    """Panels that end on the credited release frame — mirrors pass detection rules."""
    release = turnover.release_frame
    passer_tid = turnover.passer_tid

    streak: list[int] = []
    fi = release
    while fi >= max(1, release - 24):
        st = timeline_by_frame.get(fi)
        if (
            st is not None
            and not st.in_flight
            and st.active_tid == passer_tid
            and st.active_kind in ("control", "reception")
        ):
            streak.insert(0, fi)
            fi -= 1
        else:
            break

    if len(streak) >= min_control_frames:
        return tuple(streak[-min_control_frames:]), "control_streak"

    control_run = _find_passer_touch_run(
        release, passer_tid, timeline_by_frame, touch_kind="control"
    )
    if control_run:
        last_control = control_run[-1]
        gap_to_release = release - last_control
        if 1 < gap_to_release <= config.pre_flight_release_window:
            in_flight_frame = last_control + gap_to_release // 2
            for candidate in range(last_control + 1, release):
                st = timeline_by_frame.get(candidate)
                if st is not None and (st.in_flight or st.active_tid is None):
                    in_flight_frame = candidate
                    break
            return (last_control, in_flight_frame, release), "pre_flight"

    if streak:
        panels = list(streak)
        while len(panels) < min_control_frames:
            prev = panels[0] - 1
            st = timeline_by_frame.get(prev)
            if (
                st is not None
                and not st.in_flight
                and st.active_tid == passer_tid
                and st.active_kind in ("control", "reception")
            ):
                panels.insert(0, prev)
            else:
                break
        while len(panels) < min_control_frames:
            panels.insert(0, max(1, panels[0] - 1))
        return tuple(panels[-min_control_frames:]), "short_touch"

    return (
        tuple(max(1, release - offset) for offset in range(min_control_frames - 1, -1, -1)),
        "short_touch",
    )


def build_turnover_explain_logic(
    turnover: InferredTurnover,
    *,
    timeline_by_frame: dict[int, CarrierFrameState],
    dets_by_frame: dict[int, sv.Detections],
    config: PassDetectionConfig,
    transformers: dict[int, object] | None = None,
    metric: bool = True,
) -> TurnoverExplainLogic:
    """Summarise rule 4 release + rule 5 interception gates for on-screen copy."""
    transformers = transformers or {}
    _, release_kind = _find_turnover_release_panels(
        turnover,
        timeline_by_frame,
        config=config,
        min_control_frames=config.min_control_frames,
    )
    min_c = config.min_control_frames
    if release_kind == "control_streak":
        release_subtitle = (
            f"{min_c} consecutive control touches at the feet → release credited"
        )
    elif release_kind == "pre_flight":
        release_subtitle = (
            f"pre-flight release — last touch within {config.pre_flight_release_window}f "
            "before ball leaves feet"
        )
    else:
        release_subtitle = "release credited — pending pass attempt stays open"

    first_touch = _first_valid_touch_frame(
        dets_by_frame,
        player_tid=turnover.interceptor_tid,
        start_frame=turnover.release_frame,
        end_frame=turnover.interception_frame + config.max_pass_gap_frames,
        config=config,
        transformers=transformers,
        metric=metric,
    )
    opp_control = False
    if first_touch is not None:
        opp_control = _opponent_control_between(
            dets_by_frame,
            start_frame=turnover.release_frame,
            end_frame=first_touch,
            passer_team=turnover.passer_team,
            config=config,
            transformers=transformers,
            metric=metric,
        )

    gap = turnover.gap_frames
    min_gap_met = config.min_carrier_gap_frames <= gap <= config.max_pass_gap_frames
    inflight_fi, inflight_tid, demo_inflight_fi, demo_inflight_tid = (
        _resolve_inflight_opponent_control_for_explain(
            turnover,
            dets_by_frame,
            config=config,
            transformers=transformers,
            metric=metric,
        )
    )
    search_end = first_touch or turnover.interception_frame
    interceptor_control = _find_demo_interceptor_control_frame(
        turnover,
        dets_by_frame,
        config=config,
        transformers=transformers,
        metric=metric,
        search_end=search_end + 12,
    )
    flyby_skipped = _inflight_flyby_skipped(
        demo_frame=demo_inflight_fi,
        show_frame=inflight_fi,
    )
    if flyby_skipped and demo_inflight_tid is not None:
        intercept_subtitle = (
            f"Demo flags #{demo_inflight_tid} in-flight control (f{demo_inflight_fi}) · "
            f"explain filters transit · #{turnover.interceptor_tid} reception at f{turnover.interception_frame}"
        )
    elif opp_control and inflight_fi is not None and inflight_tid != turnover.interceptor_tid:
        intercept_subtitle = (
            f"rule 5 — demo credits opponent #{inflight_tid} control in flight "
            f"(f{inflight_fi}); turnover to #{turnover.interceptor_tid} at f{first_touch}"
        )
    elif interceptor_control is not None:
        intercept_subtitle = (
            f"rule 5 — interceptor #{turnover.interceptor_tid} control "
            f"(f{interceptor_control}) after reception"
        )
    elif first_touch is not None:
        intercept_subtitle = (
            f"rule 5 — interceptor #{turnover.interceptor_tid} reception "
            f"(f{first_touch}) while release pending"
        )
    else:
        intercept_subtitle = (
            "rule 5 — opponent takes the ball during a pending release"
        )

    return TurnoverExplainLogic(
        release_kind=release_kind,
        release_strip_subtitle=release_subtitle,
        intercept_strip_subtitle=intercept_subtitle,
        first_opponent_touch_frame=first_touch,
        opponent_control_between=opp_control,
        gap_frames=gap,
        min_gap_met=min_gap_met,
        pre_flight_release_window=config.pre_flight_release_window,
        inflight_opponent_control_frame=inflight_fi,
        inflight_opponent_control_tid=inflight_tid,
        demo_inflight_opponent_control_frame=demo_inflight_fi,
        demo_inflight_opponent_control_tid=demo_inflight_tid,
        inflight_flyby_skipped=flyby_skipped,
        interceptor_control_frame=interceptor_control,
    )


def _turnover_release_label(
    frame_idx: int,
    panels: tuple[int, ...],
    release_frame: int,
    *,
    min_control_frames: int,
) -> str:
    if frame_idx == release_frame:
        return "RELEASE CREDITED"
    if frame_idx not in panels:
        return "CONTROL"
    panel_idx = panels.index(frame_idx) + 1
    return f"CONTROL {panel_idx}/{min_control_frames}"


def _turnover_release_sublabel(
    ctx: TurnoverExplainContext,
    frame_idx: int,
    panels: tuple[int, ...],
    *,
    panel_idx: int,
    locked: bool,
) -> str:
    logic = ctx.logic
    release = ctx.turnover.release_frame
    if locked and frame_idx == release:
        return "pass attempt stays open until teammate or opponent wins it"
    if logic.release_kind == "pre_flight":
        if frame_idx == panels[0]:
            return "last control touch before ball leaves feet"
        if frame_idx != release and frame_idx != panels[0]:
            return (
                f"ball in flight — pre-flight window "
                f"(≤{logic.pre_flight_release_window}f after last touch)"
            )
    if frame_idx == release:
        return "release frame — turnover snapshot can start on opponent touch"
    return _explain_streak_sublabel(
        panel_idx, len(panels), locked=False, kind="control"
    )


def _turnover_intercept_label(
    ctx: TurnoverExplainContext,
    frame_idx: int,
    panels: tuple[int, ...],
    *,
    locked: bool,
) -> str:
    logic = ctx.logic
    t = ctx.turnover
    if locked:
        return "INTERCEPT LOCKED"
    if frame_idx == t.interception_frame:
        return "OPPONENT RECEPTION"
    if logic.interceptor_control_frame is not None and frame_idx == logic.interceptor_control_frame:
        return "OPPONENT CONTROL"
    if frame_idx not in panels:
        return "CLOSING"
    panel_idx = panels.index(frame_idx) + 1
    return f"CLOSING {panel_idx}/{len(panels)}"


def _turnover_intercept_sublabel(
    ctx: TurnoverExplainContext,
    frame_idx: int,
    *,
    panel_idx: int,
    locked: bool,
) -> str:
    logic = ctx.logic
    t = ctx.turnover
    if locked:
        return "ball at interceptor feet — possession flips"
    if frame_idx == t.interception_frame:
        return "demo credits interceptor reception — turnover emitted"
    if logic.interceptor_control_frame is not None and frame_idx == logic.interceptor_control_frame:
        return "demo credits interceptor control touch"
    return _explain_streak_sublabel(
        panel_idx, len(ctx.pass_ctx.strip_plan.receiver_frames),
        locked=False,
        kind="arrival",
    )


def _ball_feet_distance_px(
    dets: sv.Detections,
    player_tid: int,
) -> float | None:
    ball = ball_xy(dets)
    feet = _feet_for_tid(dets, player_tid)
    if ball is None or feet is None:
        return None
    if abs(float(ball[1] - feet[1])) > AERIAL_DY_THRESHOLD_PX:
        return None
    return float(np.hypot(ball[0] - feet[0], ball[1] - feet[1]))


def _find_interceptor_closest_contact_frame(
    turnover: InferredTurnover,
    dets_by_frame: dict[int, sv.Detections],
    *,
    start_frame: int,
    search_end: int,
    max_contact_px: float = _INTERCEPTOR_EXPLAIN_CONTACT_PX,
) -> int | None:
    """Frame with the smallest ball-to-interceptor-feet distance near reception."""
    event = turnover.interception_frame
    search_end = min(search_end, _turnover_interceptor_search_end(turnover))
    start_frame = max(start_frame, event - 2)
    best_fi: int | None = None
    best_dist = float("inf")
    for fi in range(start_frame, search_end + 1):
        dets = dets_by_frame.get(fi)
        if dets is None:
            continue
        dist = _ball_feet_distance_px(dets, turnover.interceptor_tid)
        if dist is None:
            continue
        if dist < best_dist:
            best_dist = dist
            best_fi = fi
    if best_fi is None or best_dist > max_contact_px:
        return None
    return best_fi


def _find_interceptor_visual_confirm_frame(
    turnover: InferredTurnover,
    dets_by_frame: dict[int, sv.Detections],
    *,
    timeline_by_frame: dict[int, CarrierFrameState] | None = None,
    min_control_frames: int = 1,
) -> int:
    """Explain-only intercept lock — ball at interceptor feet, not reception credit."""
    pass_event = turnover_to_pass_adapter(turnover)
    release = turnover.release_frame
    event_frame = turnover.interception_frame
    search_end = _turnover_interceptor_search_end(turnover)
    start = max(release + 1, event_frame - 2)

    tight = _find_receiver_tight_feet_frame(
        pass_event,
        dets_by_frame,
        search_end=search_end,
        start_frame=start,
        max_distance_px=_RECEIVER_EXPLAIN_TIGHT_PX,
    )
    if tight is not None:
        return tight

    if timeline_by_frame:
        control_run = _find_receiver_control_run(
            release,
            turnover.interceptor_tid,
            timeline_by_frame,
            search_end=search_end,
        )
        control_after = tuple(fi for fi in control_run if fi >= event_frame)
        if len(control_after) >= min_control_frames:
            return control_after[min_control_frames - 1]

    closest = _find_interceptor_closest_contact_frame(
        turnover,
        dets_by_frame,
        start_frame=start,
        search_end=search_end,
    )
    if closest is not None:
        return closest

    return event_frame


def build_turnover_strip_plan(
    turnover: InferredTurnover,
    timeline_by_frame: dict[int, CarrierFrameState],
    *,
    dets_by_frame: dict[int, sv.Detections] | None = None,
    config: PassDetectionConfig = PassDetectionConfig(),
    transformers: dict[int, object] | None = None,
    metric: bool = True,
    min_control_frames: int = _MIN_CONTROL,
    min_arrival_frames: int = _MIN_ARRIVAL,
) -> PassStripPlan:
    """Strip plan anchored on credited release + visual intercept contact."""
    pass_event = turnover_to_pass_adapter(turnover)
    release = turnover.release_frame
    arrival = release + turnover.gap_frames
    dets_map = dets_by_frame or {}
    transformers = transformers or {}

    passer_frames, _ = _find_turnover_release_panels(
        turnover,
        timeline_by_frame,
        config=config,
        min_control_frames=min_control_frames,
    )
    flight_frame = _pick_flight_frame(
        pass_event,
        release,
        arrival,
        turnover.passer_tid,
        timeline_by_frame,
        dets_map,
    )
    inflight_fi, inflight_tid, demo_fi, demo_tid = _resolve_inflight_opponent_control_for_explain(
        turnover,
        dets_map,
        config=config,
        transformers=transformers,
        metric=metric,
    )
    flight_frames: tuple[int, ...]
    if inflight_fi is not None and inflight_fi != flight_frame:
        flight_frames = (flight_frame, inflight_fi)
    elif (
        demo_fi is not None
        and inflight_fi is None
        and demo_fi != flight_frame
    ):
        flight_frames = (flight_frame, demo_fi)
    else:
        flight_frames = (flight_frame,)

    confirm = _find_interceptor_visual_confirm_frame(
        turnover,
        dets_map,
        timeline_by_frame=timeline_by_frame,
        min_control_frames=1,
    )
    receiver_frames = _find_turnover_intercept_panels(
        turnover,
        timeline_by_frame,
        dets_map,
        config=config,
        transformers=transformers,
        metric=metric,
        confirm_frame=confirm,
        min_arrival_frames=min_arrival_frames,
    )
    return PassStripPlan(
        passer_frames=passer_frames,
        flight_frames=flight_frames,
        receiver_frames=receiver_frames,
        summary_frame=confirm,
        passer_confirm_frame=release,
        receiver_confirm_frame=confirm,
    )


def pick_explain_turnover(
    turnovers: list[InferredTurnover],
    *,
    turnover_index: int | None = None,
    timeline_by_frame: dict[int, CarrierFrameState] | None = None,
    dets_by_frame: dict[int, sv.Detections] | None = None,
    min_control_frames: int = _MIN_CONTROL,
) -> InferredTurnover:
    if not turnovers:
        raise ValueError("No turnovers found in scan")
    if turnover_index is not None:
        if turnover_index < 0 or turnover_index >= len(turnovers):
            raise ValueError(f"turnover_index {turnover_index} out of range")
        return turnovers[turnover_index]
    if timeline_by_frame is None or dets_by_frame is None:
        return turnovers[0]
    scored: list[tuple[int, InferredTurnover]] = []
    for t in turnovers:
        control = _find_passer_control_run(
            t.release_frame,
            t.passer_tid,
            timeline_by_frame,
            min_control_frames=min_control_frames,
        )
        if len(control) >= min_control_frames and t.gap_frames >= 8:
            scored.append((t.gap_frames, t))
    if scored:
        return max(scored, key=lambda x: x[0])[1]
    return max(turnovers, key=lambda t: t.gap_frames)


def build_turnover_explain_context(
    turnover: InferredTurnover,
    *,
    frame_rate: float,
    frames_by_idx: dict[int, np.ndarray],
    dets_by_frame: dict[int, sv.Detections],
    timeline_by_frame: dict[int, CarrierFrameState],
    keypoints_by_frame: dict | None = None,
    radar_transformers: dict | None = None,
    metric: bool = True,
    config: PassDetectionConfig | None = None,
    min_control_frames: int = _MIN_CONTROL,
    min_arrival_frames: int = _MIN_ARRIVAL,
    strip_plan: PassStripPlan | None = None,
) -> TurnoverExplainContext:
    detection_config = config or PassDetectionConfig().for_frame_rate(frame_rate)
    if strip_plan is None:
        strip_plan = build_turnover_strip_plan(
            turnover,
            timeline_by_frame,
            dets_by_frame=dets_by_frame,
            config=detection_config,
            transformers=radar_transformers,
            metric=metric,
            min_control_frames=min_control_frames,
            min_arrival_frames=min_arrival_frames,
        )
    pass_event = turnover_to_pass_adapter(turnover)
    pass_ctx = build_pass_explain_context(
        pass_event,
        frame_rate=frame_rate,
        frames_by_idx=frames_by_idx,
        dets_by_frame=dets_by_frame,
        timeline_by_frame=timeline_by_frame,
        keypoints_by_frame=keypoints_by_frame,
        radar_transformers=radar_transformers,
        metric=metric,
        min_control_frames=min_control_frames,
        min_arrival_frames=min_arrival_frames,
        strip_plan=strip_plan,
    )
    logic = build_turnover_explain_logic(
        turnover,
        timeline_by_frame=timeline_by_frame,
        dets_by_frame=dets_by_frame,
        config=detection_config,
        transformers=radar_transformers,
        metric=metric,
    )
    return TurnoverExplainContext(turnover=turnover, pass_ctx=pass_ctx, logic=logic)


def _render_turnover_passer_panel(
    ctx: TurnoverExplainContext,
    frame_idx: int,
    *,
    layout: Literal["talk", "social"],
    crop: tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    pctx = ctx.pass_ctx
    t = ctx.turnover
    dets = pctx.dets_by_frame[frame_idx]
    color = _team_color(t.passer_team)
    panels = pctx.strip_plan.passer_frames
    panel_idx = panels.index(frame_idx) + 1 if frame_idx in panels else 1
    level = _streak_emphasis(panel_idx, len(panels))
    release = t.release_frame
    locked = frame_idx >= release
    frame_img = _build_passer_control_frame(
        pctx.frames[frame_idx],
        dets,
        t.passer_tid,
        color,
        emphasis=level,
        locked=locked,
        ball_pt=_pass_ball_for_ctx(pctx, frame_idx),
    )
    label = _turnover_release_label(
        frame_idx,
        panels,
        release,
        min_control_frames=pctx.min_control_frames,
    )
    sublabel = _turnover_release_sublabel(
        ctx, frame_idx, panels, panel_idx=panel_idx, locked=locked
    )
    badge = "RELEASE LOCKED" if locked else _passer_panel_badge(
        panel_idx, len(panels), locked=False
    )
    return _compose_panel_row(
        frame_img,
        frame_idx,
        label,
        color,
        layout=layout,
        step=panel_idx,
        step_total=len(panels),
        sublabel=sublabel,
        badge=badge,
        emphasis=level,
        locked=locked,
        accent_bgr=ROBOFLOW_PURPLE_BGR if locked else color,
        crop=crop,
    )


def render_strip_turnover_passer(
    ctx: TurnoverExplainContext,
    *,
    layout: Literal["talk", "social"] = "talk",
) -> np.ndarray:
    pctx = ctx.pass_ctx
    p = pctx.pass_event
    crop = _strip_crop_rect(pctx, pctx.strip_plan.passer_frames, p.passer_tid)
    panels = [
        _render_turnover_passer_panel(ctx, fi, layout=layout, crop=crop)
        for fi in pctx.strip_plan.passer_frames
    ]
    out = _compose_strip(
        panels,
        title="CREDIT THE RELEASE",
        subtitle=ctx.logic.release_strip_subtitle,
        layout=layout,
        accent_bgr=ROBOFLOW_PURPLE_BGR,
    )
    return draw_branding_tag(out, _TURNOVER_BRANDING)


def _build_turnover_flight_frame(
    ctx: TurnoverExplainContext,
    frame_idx: int,
) -> np.ndarray:
    """Ball in flight with pass arrow toward the intercepting opponent."""
    p = ctx.pass_ctx.pass_event
    t = ctx.turnover
    full = ctx.pass_ctx.frames[frame_idx]
    dets = ctx.pass_ctx.dets_by_frame[frame_idx]
    passer_color = _team_color(t.passer_team)
    opp_color = _team_color(t.interceptor_team)
    ball_pt = _pass_ball_for_ctx(ctx.pass_ctx, frame_idx)
    out = _build_flight_frame(full, dets, p.passer_tid, passer_color, ball_pt=ball_pt)

    passer_feet = _feet_for_tid(dets, p.passer_tid)
    p0, p1 = _pass_anchor_points(p, ctx.pass_ctx.dets_by_frame)
    if passer_feet:
        _draw_dashed_line(out, passer_feet, p1, opp_color, alpha=0.35)
    intercept_feet = _feet_for_tid(dets, t.interceptor_tid)
    if intercept_feet:
        _draw_focus_player(out, dets, t.interceptor_tid, opp_color, prominent=False, emphasis=0.55)
    return out


def _flyby_trajectory_points(
    pctx: PassExplainContext,
    frame_idx: int,
    *,
    lookback: int = 3,
) -> list[tuple[int, int]]:
    pts: list[tuple[int, int]] = []
    for fi in range(max(1, frame_idx - lookback), frame_idx + 1):
        pt = _pass_ball_for_ctx(pctx, fi)
        if pt is not None:
            pts.append(pt)
    return pts


def _draw_ball_trajectory_arrow(
    image: np.ndarray,
    points: list[tuple[int, int]],
    *,
    color_bgr: tuple[int, int, int] = (210, 215, 225),
) -> None:
    if len(points) < 2:
        return
    layer = image.copy()
    for i in range(len(points) - 1):
        cv2.line(layer, points[i], points[i + 1], color_bgr, 2, cv2.LINE_AA)
    cv2.circle(layer, points[0], 4, color_bgr, -1, cv2.LINE_AA)
    cv2.arrowedLine(
        layer,
        points[-2],
        points[-1],
        color_bgr,
        2,
        tipLength=0.28,
        line_type=cv2.LINE_AA,
    )
    image[:] = cv2.addWeighted(layer, 0.72, image, 0.28, 0)


def _draw_transit_zone_veto(
    image: np.ndarray,
    feet: tuple[int, int],
    *,
    radius: int = 38,
) -> None:
    """Muted control-radius ring with slash — contact zone, not possession."""
    layer = image.copy()
    cv2.circle(layer, feet, radius, (88, 92, 100), 1, cv2.LINE_AA)
    cv2.line(
        layer,
        (feet[0] - radius + 6, feet[1] + radius // 2),
        (feet[0] + radius - 6, feet[1] - radius // 2),
        (78, 82, 92),
        2,
        cv2.LINE_AA,
    )
    image[:] = cv2.addWeighted(layer, 0.65, image, 0.35, 0)


def _draw_flyby_status_chip(
    image: np.ndarray,
    *,
    line1: str,
    line2: str,
) -> None:
    h, w = image.shape[:2]
    bar_h = 50
    y0 = h - bar_h - 12
    overlay = image.copy()
    cv2.rectangle(overlay, (14, y0), (w - 14, h - 12), _FLYBY_CHIP_BGR, -1)
    image[:] = cv2.addWeighted(overlay, 0.80, image, 0.20, 0)
    cv2.rectangle(image, (14, y0), (w - 14, h - 12), (58, 62, 72), 1, cv2.LINE_AA)
    cv2.line(image, (14, y0), (14, y0 + bar_h), _FLYBY_CHIP_ACCENT_BGR, 3)
    draw_text_shadow(
        image,
        line1,
        (26, y0 + 20),
        font_scale=0.50,
        color_bgr=(228, 230, 236),
        thickness=1,
    )
    draw_text_shadow(
        image,
        line2,
        (26, y0 + 38),
        font_scale=0.42,
        color_bgr=(148, 152, 162),
        thickness=1,
    )


def _build_flyby_skip_frame(
    ctx: TurnoverExplainContext,
    frame_idx: int,
    *,
    flyby_tid: int,
) -> np.ndarray:
    """Ball transiting an opponent — demo credits control; explain filters."""
    pctx = ctx.pass_ctx
    t = ctx.turnover
    full = pctx.frames[frame_idx]
    dets = pctx.dets_by_frame[frame_idx]
    passer_color = _team_color(t.passer_team)
    flyby_team = _player_team(dets, flyby_tid)
    flyby_color = _team_color(flyby_team if flyby_team is not None else t.interceptor_team)
    ball_pt = _pass_ball_for_ctx(pctx, frame_idx)

    out = _dim_frame(full, 0.30)
    _draw_anchor_player(out, dets, t.passer_tid, passer_color)
    trajectory = _flyby_trajectory_points(pctx, frame_idx)
    if trajectory:
        _draw_ball_trajectory_arrow(out, trajectory)
    if ball_pt is not None:
        out = draw_carrier_spotlight(out, full, ball_pt, radius=160, strength=0.75)
        out = _draw_pass_ball_marker(out, dets, ball_pt)

    flyby_feet = _feet_for_tid(dets, flyby_tid)
    if flyby_feet is not None:
        _draw_transit_zone_veto(out, flyby_feet)
        _draw_focus_player(
            out, dets, flyby_tid, flyby_color, prominent=True, emphasis=0.32, locked=False
        )

    intercept_feet = _feet_for_tid(dets, t.interceptor_tid)
    if intercept_feet:
        _draw_focus_player(
            out,
            dets,
            t.interceptor_tid,
            _team_color(t.interceptor_team),
            prominent=False,
            emphasis=0.38,
        )

    chip1, chip2 = _flight_flyby_chip_lines(ctx.logic)
    _draw_flyby_status_chip(out, line1=chip1, line2=chip2)
    return out


def _render_intercept_panel(
    ctx: TurnoverExplainContext,
    frame_idx: int,
    *,
    layout: Literal["talk", "social"],
    crop: tuple[int, int, int, int] | None,
) -> np.ndarray:
    t = ctx.turnover
    logic = ctx.logic
    pctx = ctx.pass_ctx
    p = pctx.pass_event
    dets = pctx.dets_by_frame[frame_idx]
    passer_color = _team_color(t.passer_team)
    opp_color = _team_color(t.interceptor_team)
    panels = pctx.strip_plan.receiver_frames
    panel_idx = panels.index(frame_idx) + 1 if frame_idx in panels else 1
    confirm = pctx.strip_plan.receiver_confirm_frame
    locked = frame_idx >= confirm
    is_control = (
        logic.interceptor_control_frame is not None
        and frame_idx == logic.interceptor_control_frame
    )
    level = _streak_emphasis(panel_idx, len(panels))
    if is_control:
        level = max(level, 0.85)

    dim_level = 0.30 - 0.14 * level
    dimmed = _dim_frame(pctx.frames[frame_idx], dim_level)
    interceptor_feet = _feet_for_tid(dets, t.interceptor_tid)
    ball = _pass_ball_for_ctx(pctx, frame_idx)

    if level < 0.45:
        out = dimmed
        _draw_anchor_player(out, dets, p.passer_tid, passer_color)
        if interceptor_feet is not None:
            out = draw_carrier_spotlight(
                out, pctx.frames[frame_idx], interceptor_feet, radius=110, strength=0.42
            )
        _draw_focus_player(out, dets, t.interceptor_tid, opp_color, emphasis=0.22)
    else:
        centers: list[tuple[int, int]] = []
        if interceptor_feet is not None:
            centers.append(interceptor_feet)
        if level >= 0.55 and ball is not None:
            centers.append(ball)
        out = _apply_focus_spotlights(
            dimmed,
            pctx.frames[frame_idx],
            centers,
            strength=0.40 + 0.52 * level,
        )
        _draw_anchor_player(out, dets, p.passer_tid, passer_color)
        _draw_focus_player(
            out,
            dets,
            t.interceptor_tid,
            opp_color,
            emphasis=level,
            locked=locked or is_control,
        )
        if level >= 0.55 and ball is not None and interceptor_feet is not None:
            out = _draw_pass_ball_marker(out, dets, ball)
            if locked or is_control:
                cv2.line(out, interceptor_feet, ball, opp_color, 3, cv2.LINE_AA)

    is_reception = frame_idx == t.interception_frame
    badge = "INTERCEPT LOCKED" if locked else (
        "OPPONENT CONTROL" if is_control
        else "RECEPTION CREDITED" if is_reception
        else f"CLOSING {panel_idx}/{len(panels)}"
    )
    sublabel = _turnover_intercept_sublabel(
        ctx, frame_idx, panel_idx=panel_idx, locked=locked
    )
    label = _turnover_intercept_label(ctx, frame_idx, panels, locked=locked)
    return _compose_panel_row(
        out,
        frame_idx,
        label,
        opp_color,
        layout=layout,
        step=panel_idx,
        step_total=len(panels),
        sublabel=sublabel,
        badge=badge,
        emphasis=level,
        locked=locked or is_control,
        accent_bgr=TURNOVER_ACCENT_BGR if (locked or is_control) else opp_color,
        crop=crop,
    )


def _render_inflight_opponent_control_panel(
    ctx: TurnoverExplainContext,
    frame_idx: int,
    *,
    layout: Literal["talk", "social"],
    crop: tuple[int, int, int, int] | None,
) -> np.ndarray:
    logic = ctx.logic
    t = ctx.turnover
    pctx = ctx.pass_ctx
    opp_tid = logic.inflight_opponent_control_tid
    if opp_tid is None:
        opp_tid = t.interceptor_tid
    dets = pctx.dets_by_frame[frame_idx]
    opp_team = _player_team(dets, opp_tid)
    opp_color = _team_color(opp_team if opp_team is not None else t.interceptor_team)
    passer_color = _team_color(t.passer_team)
    out = _build_passer_control_frame(
        pctx.frames[frame_idx],
        dets,
        opp_tid,
        opp_color,
        emphasis=0.9,
        locked=True,
        ball_pt=_pass_ball_for_ctx(pctx, frame_idx),
    )
    _draw_anchor_player(out, dets, t.passer_tid, passer_color)
    return _compose_panel_row(
        out,
        frame_idx,
        "OPPONENT CONTROL",
        opp_color,
        layout=layout,
        step=2,
        step_total=2,
        sublabel=(
            "demo _opponent_control_between gate — valid opponent control "
            f"while release pending (f{frame_idx})"
        ),
        badge="DEMO: OPP CONTROL",
        emphasis=0.9,
        locked=True,
        accent_bgr=TURNOVER_ACCENT_BGR,
        crop=crop,
    )


def _render_inflight_flyby_skip_panel(
    ctx: TurnoverExplainContext,
    frame_idx: int,
    *,
    layout: Literal["talk", "social"],
    crop: tuple[int, int, int, int] | None,
) -> np.ndarray:
    logic = ctx.logic
    flyby_tid = logic.demo_inflight_opponent_control_tid
    if flyby_tid is None:
        flyby_tid = ctx.turnover.interceptor_tid
    dets = ctx.pass_ctx.dets_by_frame[frame_idx]
    flyby_team = _player_team(dets, flyby_tid)
    flyby_color = _team_color(
        flyby_team if flyby_team is not None else ctx.turnover.interceptor_team
    )
    out = _build_flyby_skip_frame(ctx, frame_idx, flyby_tid=flyby_tid)
    _draw_anchor_player(out, dets, ctx.turnover.passer_tid, _team_color(ctx.turnover.passer_team))
    return _compose_panel_row(
        out,
        frame_idx,
        "IN-FLIGHT PRESSURE",
        flyby_color,
        layout=layout,
        step=2,
        step_total=2,
        sublabel=_flight_flyby_skip_copy_from_logic(logic),
        badge="NOT POSSESSION",
        emphasis=0.50,
        locked=False,
        accent_bgr=_FLYBY_SLATE_BGR,
        crop=crop,
    )


def render_strip_intercept(
    ctx: TurnoverExplainContext,
    *,
    layout: Literal["talk", "social"] = "talk",
) -> np.ndarray:
    t = ctx.turnover
    pctx = ctx.pass_ctx
    crop = _crop_rect_from_points(
        pctx.frames[pctx.strip_plan.receiver_frames[0]].shape,
        _action_focus_points(
            pctx.dets_by_frame[pctx.strip_plan.receiver_frames[0]],
            t.passer_tid,
            t.interceptor_tid,
        ),
        min_frac=0.40,
    )
    panels = [
        _render_intercept_panel(ctx, fi, layout=layout, crop=crop)
        for fi in pctx.strip_plan.receiver_frames
    ]
    out = _compose_strip(
        panels,
        title="OPPONENT INTERCEPTS",
        subtitle=ctx.logic.intercept_strip_subtitle,
        layout=layout,
        accent_bgr=TURNOVER_ACCENT_BGR,
    )
    return draw_branding_tag(out, _TURNOVER_BRANDING)


def _render_flight_travel_panel(
    ctx: TurnoverExplainContext,
    frame_idx: int,
    *,
    layout: Literal["talk", "social"],
    crop: tuple[int, int, int, int] | None,
    step: int,
    step_total: int,
) -> np.ndarray:
    t = ctx.turnover
    logic = ctx.logic
    frame_img = _build_turnover_flight_frame(ctx, frame_idx)
    frame_img = _fit_panel_frame_turnover(frame_img, crop=crop)
    if logic.inflight_flyby_skipped:
        sublabel = (
            "pass in flight — next panel: in-flight pressure (filtered)"
        )
    else:
        sublabel = "pass attempt in flight — release still pending"
    return _compose_panel_row(
        frame_img,
        frame_idx,
        "BALL TRAVELLING",
        _team_color(t.passer_team),
        layout=layout,
        step=step,
        step_total=step_total,
        sublabel=sublabel,
        badge="IN FLIGHT",
        emphasis=1.0,
        locked=False,
        accent_bgr=TURNOVER_ACCENT_BGR,
        crop=None,
    )


def render_strip_turnover_flight(
    ctx: TurnoverExplainContext,
    *,
    layout: Literal["talk", "social"] = "talk",
) -> np.ndarray:
    t = ctx.turnover
    pctx = ctx.pass_ctx
    logic = ctx.logic
    flight_frames = pctx.strip_plan.flight_frames
    fi0 = flight_frames[0]
    dets0 = pctx.dets_by_frame[fi0]
    ball_pt = _pass_ball_for_ctx(pctx, fi0)
    focus_tids = [t.passer_tid, t.interceptor_tid]
    if logic.inflight_opponent_control_tid is not None:
        focus_tids.append(logic.inflight_opponent_control_tid)
    elif logic.inflight_flyby_skipped and logic.demo_inflight_opponent_control_tid is not None:
        focus_tids.append(logic.demo_inflight_opponent_control_tid)
    points = _action_focus_points(dets0, *focus_tids, ball_pt=ball_pt)
    crop = _crop_rect_from_points(pctx.frames[fi0].shape, points)
    panels: list[np.ndarray] = []
    step_total = len(flight_frames)
    panels.append(
        _render_flight_travel_panel(
            ctx, fi0, layout=layout, crop=crop, step=1, step_total=step_total
        )
    )
    if len(flight_frames) > 1:
        opp_fi = flight_frames[1]
        panels.append(
            _render_inflight_flyby_skip_panel(
                ctx, opp_fi, layout=layout, crop=crop
            )
        )
    if logic.inflight_flyby_skipped:
        flight_subtitle = _flight_flyby_skip_copy_from_logic(logic)
    elif logic.inflight_opponent_control_frame is not None:
        flight_subtitle = (
            f"demo credits opponent #{logic.inflight_opponent_control_tid} control "
            f"in flight (f{logic.inflight_opponent_control_frame}) — rule 5 press gate"
        )
    else:
        flight_subtitle = "release stays pending — opponent can still win the ball in flight"
    out = _compose_strip(
        panels,
        title="BALL TRAVELLING",
        subtitle=flight_subtitle,
        layout=layout,
        accent_bgr=TURNOVER_ACCENT_BGR,
    )
    return draw_branding_tag(out, _TURNOVER_BRANDING)


def _fit_panel_frame_turnover(
    frame_img: np.ndarray,
    *,
    crop: tuple[int, int, int, int] | None,
) -> np.ndarray:
    from world_cup_projects.player_stats.pass_explain_visual import _apply_crop, _fit_panel_frame

    if crop is not None:
        frame_img = _apply_crop(frame_img, crop)
    return _fit_panel_frame(frame_img, crop=None)


def render_turnover_summary(
    ctx: TurnoverExplainContext,
    *,
    layout: Literal["talk", "social"] = "talk",
) -> np.ndarray:
    t = ctx.turnover
    pctx = ctx.pass_ctx
    fi = pctx.strip_plan.summary_frame
    frame = pctx.frames[fi]
    dets = pctx.dets_by_frame[fi]
    passer_color = _team_color(t.passer_team)
    opp_color = _team_color(t.interceptor_team)
    dimmed = _dim_frame(frame, 0.22)
    centers = [
        f
        for tid in (t.passer_tid, t.interceptor_tid)
        if (f := _feet_for_tid(dets, tid)) is not None
    ]
    out = _apply_focus_spotlights(dimmed, frame, centers, strength=0.78)
    _draw_focus_player(out, dets, t.passer_tid, passer_color, prominent=True, locked=True)
    _draw_focus_player(out, dets, t.interceptor_tid, opp_color, prominent=True, locked=True)
    out = annotate_ball(out, dets)

    passer_feet = _feet_for_tid(dets, t.passer_tid)
    intercept_feet = _feet_for_tid(dets, t.interceptor_tid)
    if passer_feet and intercept_feet:
        cv2.line(out, passer_feet, intercept_feet, TURNOVER_ACCENT_BGR, 2, cv2.LINE_AA)
        ball_pt = _pass_ball_for_ctx(pctx, fi)
        if ball_pt:
            cv2.circle(out, ball_pt, 10, TURNOVER_ACCENT_BGR, 2, cv2.LINE_AA)

    ibox = _get_player_box(dets, t.interceptor_tid)
    if ibox is not None:
        pcx = int((ibox[0] + ibox[2]) / 2)
        pcy = int(ibox[3])
        draw_carrier_pulse(out, (pcx, pcy), 1.0, color_bgr=TURNOVER_ACCENT_BGR)

    points = _action_focus_points(dets, t.passer_tid, t.interceptor_tid)
    crop = _crop_rect_from_points(frame.shape, points, min_frac=0.42)
    out = _letterbox_frame(
        _fit_panel_frame_turnover(out, crop=crop),
        target_w=_PANEL_W + _GUTTER_W,
        target_h=_PANEL_H,
    )
    _draw_strip_title(
        out,
        "POSSESSION LOST",
        ctx.logic.intercept_strip_subtitle,
        layout=layout,
    )
    out = draw_hud_bar(out, "TURNOVER")
    return draw_branding_tag(out, _TURNOVER_BRANDING)


def render_turnover_explain_strips(
    ctx: TurnoverExplainContext,
    *,
    layout: Literal["talk", "social"] = "talk",
) -> dict[str, np.ndarray]:
    pctx = ctx.pass_ctx
    strips = {
        "strip_passer": render_strip_turnover_passer(ctx, layout=layout),
        "strip_flight": render_strip_turnover_flight(ctx, layout=layout),
        "strip_intercept": render_strip_intercept(ctx, layout=layout),
        "summary": render_turnover_summary(ctx, layout=layout),
    }
    if layout == "social":
        strips = {k: _fit_social_square(v) for k, v in strips.items()}
    return strips


def render_turnover_detect_timeline(strips: dict[str, np.ndarray], *, gap: int = 8) -> np.ndarray:
    order = ["strip_passer", "strip_flight", "strip_intercept", "summary"]
    renamed = {k: strips[k] for k in order if k in strips}
    return render_pass_detect_timeline(renamed, gap=gap)


def write_turnover_explain_frames(
    out_dir: str | Path,
    strips: dict[str, np.ndarray],
    *,
    timeline: bool = True,
) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for stem, image in strips.items():
        path = out_dir / f"pass_turnover_{stem}.png"
        cv2.imwrite(str(path), image)
        written.append(path)
    if timeline:
        path = out_dir / "pass_turnover_timeline.png"
        cv2.imwrite(str(path), render_turnover_detect_timeline(strips))
        written.append(path)
    return written


def render_turnover_video_summary(ctx: TurnoverExplainContext) -> np.ndarray:
    t = ctx.turnover
    pctx = ctx.pass_ctx
    fi = pctx.strip_plan.summary_frame
    frame = pctx.frames[fi]
    dets = pctx.dets_by_frame[fi]
    passer_color = _team_color(t.passer_team)
    opp_color = _team_color(t.interceptor_team)
    dimmed = _dim_frame(frame, 0.22)
    centers = [
        f
        for tid in (t.passer_tid, t.interceptor_tid)
        if (f := _feet_for_tid(dets, tid)) is not None
    ]
    out = _apply_focus_spotlights(dimmed, frame, centers, strength=0.78)
    _draw_focus_player(out, dets, t.passer_tid, passer_color, prominent=True, locked=True)
    _draw_focus_player(out, dets, t.interceptor_tid, opp_color, prominent=True, locked=True)
    out = annotate_ball(out, dets)
    passer_feet = _feet_for_tid(dets, t.passer_tid)
    intercept_feet = _feet_for_tid(dets, t.interceptor_tid)
    if passer_feet and intercept_feet:
        cv2.line(out, passer_feet, intercept_feet, TURNOVER_ACCENT_BGR, 2, cv2.LINE_AA)
    ibox = _get_player_box(dets, t.interceptor_tid)
    if ibox is not None:
        pcx = int((ibox[0] + ibox[2]) / 2)
        pcy = int(ibox[3])
        draw_carrier_pulse(out, (pcx, pcy), 1.0, color_bgr=TURNOVER_ACCENT_BGR)
    out = draw_hud_bar(out, "POSSESSION LOST")
    draw_text_shadow(
        out,
        f"#{t.passer_tid} intercepted by #{t.interceptor_tid}",
        (18, out.shape[0] - 18),
        font_scale=0.58,
        color_bgr=TURNOVER_ACCENT_BGR,
        thickness=1,
    )
    return draw_branding_tag(out, _TURNOVER_BRANDING)


def build_turnover_explain_video_sequence(
    ctx: TurnoverExplainContext,
    *,
    timing: PassExplainVideoTiming = PassExplainVideoTiming(),
) -> list[np.ndarray]:
    """Slow-mo walkthrough: release credit → flight → intercept lock → turnover."""
    pctx = ctx.pass_ctx
    t = ctx.turnover
    plan = pctx.strip_plan
    start, end = explain_video_frame_range(plan)
    passer_lock = t.release_frame
    intercept_event = t.interception_frame
    intercept_lock = plan.receiver_confirm_frame
    hold_n = max(1, int(round(timing.hold_locked_seconds * timing.output_fps)))
    summary_n = max(1, int(round(timing.summary_hold_seconds * timing.output_fps)))
    out: list[np.ndarray] = []

    logic = ctx.logic
    flyby_skip_fi = (
        logic.demo_inflight_opponent_control_frame
        if logic.inflight_flyby_skipped
        else None
    )
    interceptor_control = logic.interceptor_control_frame
    hold_frames = {passer_lock, intercept_event, intercept_lock}
    if flyby_skip_fi is not None:
        hold_frames.add(flyby_skip_fi)
    if interceptor_control is not None:
        hold_frames.add(interceptor_control)

    for fi in range(start, end + 1):
        if fi <= passer_lock:
            phase = "passer"
        elif fi < intercept_event:
            phase = "flight"
        else:
            phase = "intercept"

        if phase == "passer":
            locked = fi >= passer_lock
            frame = _build_passer_control_frame(
                pctx.frames[fi],
                pctx.dets_by_frame[fi],
                ctx.turnover.passer_tid,
                _team_color(ctx.turnover.passer_team),
                emphasis=1.0 if locked else 0.7,
                locked=locked,
                ball_pt=_pass_ball_for_ctx(pctx, fi),
            )
            frame = draw_hud_bar(frame, "CREDIT THE RELEASE")
        elif phase == "flight":
            if flyby_skip_fi is not None and fi == flyby_skip_fi:
                flyby_tid = logic.demo_inflight_opponent_control_tid or t.interceptor_tid
                frame = _build_flyby_skip_frame(ctx, fi, flyby_tid=flyby_tid)
                frame = draw_hud_bar(frame, "IN-FLIGHT PRESSURE")
            else:
                frame = _build_turnover_flight_frame(ctx, fi)
                frame = draw_hud_bar(frame, "BALL TRAVELLING")
        else:
            dets = pctx.dets_by_frame[fi]
            locked = fi >= intercept_lock
            is_reception = fi == intercept_event
            is_control = interceptor_control is not None and fi == interceptor_control
            opp = _team_color(ctx.turnover.interceptor_team)
            passer = _team_color(ctx.turnover.passer_team)
            if is_control or locked:
                frame = _build_passer_control_frame(
                    pctx.frames[fi],
                    dets,
                    t.interceptor_tid,
                    opp,
                    emphasis=1.0,
                    locked=True,
                    ball_pt=_pass_ball_for_ctx(pctx, fi),
                )
                _draw_anchor_player(frame, dets, t.passer_tid, passer)
            else:
                frame = pctx.frames[fi].copy()
                dimmed = _dim_frame(frame, _FOCUS_DIM)
                centers = [
                    f
                    for tid in (ctx.turnover.interceptor_tid,)
                    if (f := _feet_for_tid(dets, tid)) is not None
                ]
                frame = _apply_focus_spotlights(
                    dimmed, frame, centers, strength=_SPOTLIGHT_STRENGTH
                )
                _draw_anchor_player(frame, dets, ctx.turnover.passer_tid, passer)
                _draw_focus_player(
                    frame, dets, ctx.turnover.interceptor_tid, opp, prominent=True, locked=False
                )
                frame = annotate_ball(frame, dets)
            title = (
                "OPPONENT CONTROL"
                if is_control
                else "OPPONENT RECEPTION"
                if is_reception
                else "OPPONENT INTERCEPTS"
            )
            frame = draw_hud_bar(frame, title)

        frame = draw_branding_tag(frame, _TURNOVER_BRANDING)
        repeats = hold_n if fi in hold_frames else 1
        out.extend([frame] * repeats)

    out.extend([render_turnover_video_summary(ctx)] * summary_n)
    return out


write_turnover_explain_video = write_pass_explain_video
write_turnover_explain_gif = write_pass_explain_gif
