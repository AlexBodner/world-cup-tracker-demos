"""Conference pass-lane explain visuals — radar + rank colors aligned with pass network demo.

Imports shared scoring/rendering from ``explain_visual`` and ``render`` but fixes:

* Minimap uses ``homography_from_keypoints_radar`` (sports orientation, not tracker lock).
* Goal colours locked via ``warmup_goal_defenders_radar`` like the network clip.
* No pitch keypoint dots on the minimap (``debug_keypoints=False``).
* Per-lane rank colours when scoring lanes one-by-one in the explain video.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

from world_cup_projects.common.pitch import (
    homography_from_keypoints_radar,
    image_to_pitch_cm,
    render_radar_simple,
    warmup_goal_defenders_radar,
)
from world_cup_projects.common.possession import feet_xy, player_mask
from world_cup_projects.common.video import write_gif_from_mp4, write_h264_video
from world_cup_projects.common.visual import (
    TEAM_COLORS,
    annotate_ball,
    annotate_players,
    draw_branding_tag,
    draw_carrier_spotlight,
    draw_radar_minimap,
    draw_score_chip,
)
from world_cup_projects.pass_alternatives.explain_visual import (
    ExplainContext,
    _CANDIDATE_BGR,
    _draw_scoring_panel,
    _draw_step_badge,
    _dim_frame,
    _ensure_lane_debug,
    _fit_social_square,
    build_explain_context,
)
from world_cup_projects.pass_alternatives.lane_visual import (
    apply_pass_lane_geometry,
    draw_candidate_lanes_on_radar,
    draw_pass_arrows_on_frame,
    draw_pass_lanes_on_radar,
)
from world_cup_projects.pass_alternatives.explain_conference_metrics import (
    LaneMetricGeometry,
    PenaltyGeometry,
    build_attack_axis_radar_draw,
    build_penalty_radar_draw,
    build_radar_metric_draw,
    compute_attack_axis_geometry,
    compute_lane_metric_geometry,
    compute_penalty_geometry,
    draw_attack_axis_on_frame,
    draw_attack_axis_on_radar,
    draw_forward_on_frame,
    draw_forward_on_radar,
    draw_openness_on_frame,
    draw_openness_on_radar,
    draw_penalty_back_on_frame,
    draw_penalty_back_on_radar,
    draw_penalty_run_on_frame,
    draw_penalty_run_on_radar,
    draw_penalty_teammate_on_frame,
    draw_penalty_teammate_on_radar,
    draw_space_distance_label,
    draw_space_on_frame,
    draw_space_on_radar,
    _draw_attack_axis_panel,
    _draw_penalty_panel,
    _draw_single_metric_panel,
    _fmt_lane_clearance,
    _m_per_px,
)
from world_cup_projects.pass_alternatives.lane_visual import _RANK_BGR as LANE_RANK_BGR
from world_cup_projects.pass_alternatives.pass_options import (
    PassOption,
    PassWeights,
    decompose_lane_score,
    remap_lane_debug_to_pitch_cm,
)
from world_cup_projects.common.visual import ROBOFLOW_PURPLE_BGR

PITCH_CONFIDENCE = 0.9


def _sports_radar_homography(ctx: ExplainContext):
    """Same minimap H as ``_draw_pass_overlay`` / pass-network freeze (no mirror lock)."""
    if ctx.keypoints is None:
        return None
    return homography_from_keypoints_radar(
        ctx.keypoints, confidence=ctx.pitch_confidence
    )


def _lane_transformer(ctx: ExplainContext):
    """Main-camera corridor H — matches pass-network freeze overlay."""
    if not ctx.metric:
        return None
    if ctx.radar_transformer is not None:
        return ctx.radar_transformer
    return _sports_radar_homography(ctx)


def build_conference_radar(
    ctx: ExplainContext,
    options: list[PassOption],
    *,
    candidate_indices: list[int] | None = None,
    display_ranks: list[int] | None = None,
    radar_overlay=None,
) -> np.ndarray | None:
    """Minimap with optional candidate lines / ranked corridors (no KP debug overlay)."""
    if not ctx.metric or ctx.keypoints is None:
        return None
    radar_h = _sports_radar_homography(ctx)
    if radar_h is None:
        return None
    radar = render_radar_simple(
        ctx.dets,
        ctx.keypoints,
        confidence=ctx.pitch_confidence,
        transformer=radar_h,
        locked_goal_defenders=ctx.locked_goal_defenders,
        debug_keypoints=False,
    )
    if radar is None:
        return None
    feet_img = feet_xy(ctx.dets)
    pitch_cm = image_to_pitch_cm(feet_img, radar_h)
    if pitch_cm is None:
        return radar
    if candidate_indices:
        radar = draw_candidate_lanes_on_radar(
            radar,
            pitch_cm,
            ctx.carrier.index,
            candidate_indices,
        )
    if options:
        remapped = remap_lane_debug_to_pitch_cm(
            options,
            ctx.carrier,
            pitch_cm,
            feet_img,
            weights=ctx.weights,
        )
        radar = draw_pass_lanes_on_radar(
            radar,
            remapped,
            pitch_cm,
            display_ranks=display_ranks,
        )
    if radar_overlay is not None:
        radar = radar_overlay(radar, pitch_cm)
    return radar


def attach_conference_radar(
    canvas: np.ndarray,
    ctx: ExplainContext,
    options: list[PassOption],
    *,
    candidate_indices: list[int] | None = None,
    display_ranks: list[int] | None = None,
    radar_overlay=None,
) -> np.ndarray:
    radar = build_conference_radar(
        ctx,
        options,
        candidate_indices=candidate_indices,
        display_ranks=display_ranks,
        radar_overlay=radar_overlay,
    )
    if radar is None:
        return canvas
    return draw_radar_minimap(
        canvas,
        ctx.dets,
        ctx.keypoints,
        pitch_confidence=ctx.pitch_confidence,
        locked_goal_defenders=ctx.locked_goal_defenders,
        prebuilt_radar=radar,
        debug_keypoints=False,
        scale_frac=0.33,
    )


def enrich_conference_context(
    ctx: ExplainContext,
    *,
    keypoints_by_frame: dict[int, object] | None = None,
    warmup_frames: list[tuple[int, object]] | None = None,
) -> ExplainContext:
    """Attach goal-lock + pitch confidence used by the network demo."""
    locked = ctx.locked_goal_defenders
    if locked is None and keypoints_by_frame is not None:
        frames = warmup_frames if warmup_frames else [(ctx.frame_idx, ctx.dets)]
        locked = warmup_goal_defenders_radar(
            frames,
            keypoints_by_frame,
            confidence=PITCH_CONFIDENCE,
        )
    return replace(
        ctx,
        locked_goal_defenders=locked,
        pitch_confidence=PITCH_CONFIDENCE,
    )


def render_step1_candidates(
    ctx: ExplainContext,
    *,
    layout: Literal["talk", "social"] = "talk",
) -> np.ndarray:
    out = _dim_frame(ctx.frame)
    feet = feet_xy(ctx.dets)
    cx, cy = int(feet[ctx.carrier.index, 0]), int(feet[ctx.carrier.index, 1])
    pmask = player_mask(ctx.dets)
    teams = ctx.dets.data["team"]
    teammates = pmask & (teams == ctx.carrier.team)
    teammates[ctx.carrier.index] = False
    teammate_idxs = [int(i) for i in np.flatnonzero(teammates)]

    for idx in teammate_idxs:
        rx, ry = int(feet[idx, 0]), int(feet[idx, 1])
        cv2.line(out, (cx, cy), (rx, ry), _CANDIDATE_BGR, 3, cv2.LINE_AA)
        cv2.circle(out, (rx, ry), 14, _CANDIDATE_BGR, 3)

    out = annotate_players(out, ctx.dets, show_tracker_ids=True)
    out = annotate_ball(out, ctx.dets)
    out = draw_carrier_spotlight(out, ctx.frame, (cx, cy))
    draw_score_chip(out, "ON BALL", (cx, cy - 36), bg_bgr=ROBOFLOW_PURPLE_BGR)
    _draw_step_badge(
        out, step=1, total=4, title="WHO CAN RECEIVE?", subtitle="", layout=layout
    )
    return draw_branding_tag(
        attach_conference_radar(out, ctx, [], candidate_indices=teammate_idxs),
        "Roboflow · pass lane AI",
    )


def render_step2_corridors(
    ctx: ExplainContext,
    *,
    layout: Literal["talk", "social"] = "talk",
) -> np.ndarray:
    out = _dim_frame(ctx.frame, 0.32)
    options = _ensure_lane_debug(
        ctx, list(ctx.top3) if ctx.top3 else list(ctx.options[:3])
    )
    feet = feet_xy(ctx.dets)
    lane_t = _lane_transformer(ctx)
    if lane_t is not None and options:
        out = apply_pass_lane_geometry(out, options, lane_t, feet)
    out = annotate_players(out, ctx.dets, show_tracker_ids=True)
    out = annotate_ball(out, ctx.dets)
    out = draw_carrier_spotlight(
        out, ctx.frame, (int(feet[ctx.carrier.index, 0]), int(feet[ctx.carrier.index, 1]))
    )
    if options:
        out = draw_pass_arrows_on_frame(
            out,
            feet,
            ctx.carrier.index,
            options,
            metric=ctx.metric,
            show_chips=False,
            show_length=ctx.metric,
            arrow_thickness=2,
        )
    _draw_step_badge(
        out, step=2, total=4, title="CHECK EACH LANE", subtitle="", layout=layout
    )
    return draw_branding_tag(
        attach_conference_radar(out, ctx, options), "Roboflow · pass lane AI"
    )


def _metric_lane_bundle(ctx: ExplainContext, rank: int):
    if not ctx.top3:
        return None
    rank = max(0, min(rank, len(ctx.top3) - 1))
    lane_opts = _ensure_lane_debug(ctx, [ctx.top3[rank]])
    option = lane_opts[0]
    feet = feet_xy(ctx.dets)
    geom = compute_lane_metric_geometry(ctx, option)
    breakdown = decompose_lane_score(
        option,
        ctx.weights,
        carrier_feet=feet[ctx.carrier.index],
        attack_dir=ctx.attack_dir,
        carrier_motion_dir=ctx.carrier_motion_dir,
    )
    penalty = (
        compute_penalty_geometry(ctx, option, breakdown) if geom is not None else None
    )
    return {
        "rank": rank,
        "lane_opts": lane_opts,
        "option": option,
        "feet": feet,
        "geom": geom,
        "penalty": penalty,
        "lane_t": _lane_transformer(ctx),
        "mpp": _m_per_px(ctx, option),
        "rank_color": LANE_RANK_BGR[min(rank, len(LANE_RANK_BGR) - 1)],
        "breakdown": breakdown,
        "ranks": [rank],
    }


def _metric_radar_overlay(ctx: ExplainContext, option, geom: LaneMetricGeometry, draw_fn):
    """Bind metric draw helpers to sports-radar pitch cm (same H as lane corridors)."""

    def overlay(radar: np.ndarray, pitch_cm: np.ndarray) -> np.ndarray:
        draw = build_radar_metric_draw(ctx, option, geom, pitch_cm)
        return draw_fn(radar, draw)

    return overlay


def _penalty_radar_overlay(ctx: ExplainContext, option, penalty: PenaltyGeometry, draw_fn):
    def overlay(radar: np.ndarray, pitch_cm: np.ndarray) -> np.ndarray:
        draw = build_penalty_radar_draw(ctx, option, penalty, pitch_cm)
        return draw_fn(radar, draw)

    return overlay


def _penalty_kinds(breakdown) -> list[str]:
    kinds: list[str] = []
    if breakdown.teammate_penalty > 0.005:
        kinds.append("tm")
    if breakdown.backward_run_penalty > 0.005:
        kinds.append("run")
    if breakdown.backward_attack_penalty > 0.005:
        kinds.append("back")
    return kinds


def _conference_step_order(ctx: ExplainContext, *, rank: int = 0) -> list[str]:
    order = [
        "01_candidates",
        "02_corridors",
        "03_attack_axis",
        "03_open",
        "03_forward",
        "03_space",
    ]
    bundle = _metric_lane_bundle(ctx, rank)
    if bundle is not None:
        for kind in _penalty_kinds(bundle["breakdown"]):
            order.append(f"03_pen_{kind}")
    order.append("04_ranking")
    return order


def _metric_title(rank: int, metric: str) -> str:
    if len(metric) > 0 and rank >= 0:
        return f"{metric}  ·  LANE #{rank + 1}"
    return metric


def _attack_axis_radar_overlay(ctx: ExplainContext, axis):
    def overlay(radar: np.ndarray, pitch_cm: np.ndarray) -> np.ndarray:
        draw = build_attack_axis_radar_draw(
            axis, pitch_cm, ctx.dets, ctx.carrier.team
        )
        own_color = TEAM_COLORS[ctx.carrier.team].as_bgr()
        opp_color = TEAM_COLORS[1 - ctx.carrier.team].as_bgr()
        return draw_attack_axis_on_radar(
            radar, draw, own_color=own_color, opp_color=opp_color
        )

    return overlay


def render_attack_axis(
    ctx: ExplainContext,
    *,
    layout: Literal["talk", "social"] = "talk",
) -> np.ndarray:
    lane_t = _lane_transformer(ctx)
    transformer = lane_t if lane_t is not None else ctx.transformer
    axis = compute_attack_axis_geometry(ctx, transformer)
    if axis is None:
        return _dim_frame(ctx.frame, 0.32)

    dim = _dim_frame(ctx.frame, 0.35)
    own_color = TEAM_COLORS[ctx.carrier.team].as_bgr()
    opp_color = TEAM_COLORS[1 - ctx.carrier.team].as_bgr()
    out = draw_attack_axis_on_frame(
        dim,
        ctx.frame,
        axis,
        transformer,
        own_color=own_color,
        opp_color=opp_color,
    )
    out = annotate_players(out, ctx.dets, show_tracker_ids=True)
    out = annotate_ball(out, ctx.dets)
    _draw_attack_axis_panel(out, axis, layout=layout)
    _draw_step_badge(
        out, step=3, total=4, title="ATTACK AXIS", subtitle="", layout=layout
    )
    return draw_branding_tag(
        attach_conference_radar(
            out,
            ctx,
            [],
            radar_overlay=_attack_axis_radar_overlay(ctx, axis),
        ),
        "Roboflow · pass lane AI",
    )


def render_metric_open(
    ctx: ExplainContext,
    *,
    layout: Literal["talk", "social"] = "talk",
    rank: int = 0,
) -> np.ndarray:
    bundle = _metric_lane_bundle(ctx, rank)
    if bundle is None or bundle["geom"] is None:
        return _dim_frame(ctx.frame, 0.32)

    dim = _dim_frame(ctx.frame, 0.28)
    original = ctx.frame
    lane_opts = bundle["lane_opts"]
    feet = bundle["feet"]
    geom = bundle["geom"]
    lane_t = bundle["lane_t"]
    rank_i = bundle["rank"]

    if lane_t is not None:
        dim = apply_pass_lane_geometry(
            dim, lane_opts, lane_t, feet, display_ranks=bundle["ranks"]
        )
    lane_debug = bundle["option"].lane_debug
    blocking_tm = (
        lane_debug.blocking_teammate_indices if lane_debug is not None else ()
    )
    out = draw_openness_on_frame(
        dim,
        original,
        geom,
        lane_t,
        m_per_px=bundle["mpp"],
        rank_color=bundle["rank_color"],
        feet_xy=feet,
        blocking_teammate_indices=blocking_tm,
    )
    out = annotate_players(out, ctx.dets, show_tracker_ids=True)
    out = annotate_ball(out, ctx.dets)
    _draw_single_metric_panel(
        out,
        label="OPENNESS",
        raw=geom.openness_m,
        ref=ctx.weights.open_ref,
        term=bundle["breakdown"].openness_term,
        weight=ctx.weights.openness,
        color=(80, 220, 60),
        layout=layout,
        detail=(
            f"opp corridor {geom.lane_width_m or ctx.weights.lane_width:.1f} m  |  "
            f"tm corridor {geom.teammate_lane_width_m or 0.0:.1f} m"
        ),
    )
    _draw_step_badge(
        out, step=3, total=4, title=_metric_title(rank_i, "OPEN"), subtitle="", layout=layout
    )
    return draw_branding_tag(
        attach_conference_radar(
            out,
            ctx,
            lane_opts,
            display_ranks=bundle["ranks"],
            radar_overlay=_metric_radar_overlay(
                ctx, bundle["option"], geom, draw_openness_on_radar
            ),
        ),
        "Roboflow · pass lane AI",
    )


def render_metric_forward(
    ctx: ExplainContext,
    *,
    layout: Literal["talk", "social"] = "talk",
    rank: int = 0,
) -> np.ndarray:
    bundle = _metric_lane_bundle(ctx, rank)
    if bundle is None or bundle["geom"] is None:
        return _dim_frame(ctx.frame, 0.32)

    dim = _dim_frame(ctx.frame, 0.28)
    original = ctx.frame
    geom = bundle["geom"]
    lane_t = bundle["lane_t"]
    rank_i = bundle["rank"]
    lane_opts = bundle["lane_opts"]
    out = draw_forward_on_frame(
        dim,
        original,
        geom,
        lane_t,
        m_per_px=bundle["mpp"],
        rank_color=bundle["rank_color"],
    )
    out = annotate_players(out, ctx.dets, show_tracker_ids=True)
    out = annotate_ball(out, ctx.dets)
    _draw_single_metric_panel(
        out,
        label="FORWARD",
        raw=geom.forward_gain_m,
        ref=ctx.weights.forward_ref,
        term=bundle["breakdown"].forward_term,
        weight=ctx.weights.forward,
        color=(0, 200, 255),
        layout=layout,
    )
    _draw_step_badge(
        out, step=3, total=4, title=_metric_title(rank_i, "FORWARD"), subtitle="", layout=layout
    )
    return draw_branding_tag(
        attach_conference_radar(
            out,
            ctx,
            lane_opts,
            display_ranks=bundle["ranks"],
            radar_overlay=_metric_radar_overlay(
                ctx, bundle["option"], geom, draw_forward_on_radar
            ),
        ),
        "Roboflow · pass lane AI",
    )


def render_metric_space(
    ctx: ExplainContext,
    *,
    layout: Literal["talk", "social"] = "talk",
    rank: int = 0,
) -> np.ndarray:
    bundle = _metric_lane_bundle(ctx, rank)
    if bundle is None or bundle["geom"] is None:
        return _dim_frame(ctx.frame, 0.32)

    dim = _dim_frame(ctx.frame, 0.28)
    original = ctx.frame
    geom = bundle["geom"]
    lane_t = bundle["lane_t"]
    feet = bundle["feet"]
    lane_opts = bundle["lane_opts"]
    rank_i = bundle["rank"]

    out = draw_space_on_frame(
        dim, original, geom, lane_t, m_per_px=bundle["mpp"]
    )
    out = annotate_players(out, ctx.dets, show_tracker_ids=True)
    avoid_pts = [(int(feet[i, 0]), int(feet[i, 1])) for i in range(len(feet))]
    draw_space_distance_label(
        out, geom, lane_t, m_per_px=bundle["mpp"], avoid_pts=avoid_pts
    )
    out = annotate_ball(out, ctx.dets)
    _draw_single_metric_panel(
        out,
        label="SPACE",
        raw=geom.space_m,
        ref=ctx.weights.space_ref,
        term=bundle["breakdown"].space_term,
        weight=ctx.weights.space,
        color=(200, 140, 255),
        layout=layout,
    )
    _draw_step_badge(
        out, step=3, total=4, title=_metric_title(rank_i, "SPACE"), subtitle="", layout=layout
    )
    return draw_branding_tag(
        attach_conference_radar(
            out,
            ctx,
            lane_opts,
            display_ranks=bundle["ranks"],
            radar_overlay=_metric_radar_overlay(
                ctx, bundle["option"], geom, draw_space_on_radar
            ),
        ),
        "Roboflow · pass lane AI",
    )


def render_penalty_teammate(
    ctx: ExplainContext,
    *,
    layout: Literal["talk", "social"] = "talk",
    rank: int = 0,
) -> np.ndarray:
    bundle = _metric_lane_bundle(ctx, rank)
    if bundle is None or bundle["geom"] is None or bundle["penalty"] is None:
        return _dim_frame(ctx.frame, 0.32)
    dim = _dim_frame(ctx.frame, 0.28)
    geom, penalty = bundle["geom"], bundle["penalty"]
    out = draw_penalty_teammate_on_frame(
        dim, ctx.frame, penalty, geom, bundle["lane_t"],
        m_per_px=bundle["mpp"], rank_color=bundle["rank_color"],
    )
    out = annotate_players(out, ctx.dets, show_tracker_ids=True)
    out = annotate_ball(out, ctx.dets)
    _draw_penalty_panel(
        out, label="TEAMMATE", penalty=bundle["breakdown"].teammate_penalty,
        detail=f"clearance {penalty.teammate_openness_m:.1f} m", layout=layout,
    )
    _draw_step_badge(
        out, step=3, total=4,
        title=_metric_title(bundle["rank"], "PEN · TM"), subtitle="", layout=layout,
    )
    return draw_branding_tag(
        attach_conference_radar(
            out, ctx, bundle["lane_opts"], display_ranks=bundle["ranks"],
            radar_overlay=_penalty_radar_overlay(
                ctx, bundle["option"], penalty, draw_penalty_teammate_on_radar
            ),
        ),
        "Roboflow · pass lane AI",
    )


def render_penalty_run(
    ctx: ExplainContext,
    *,
    layout: Literal["talk", "social"] = "talk",
    rank: int = 0,
) -> np.ndarray:
    bundle = _metric_lane_bundle(ctx, rank)
    if bundle is None or bundle["geom"] is None or bundle["penalty"] is None:
        return _dim_frame(ctx.frame, 0.32)
    dim = _dim_frame(ctx.frame, 0.28)
    geom, penalty = bundle["geom"], bundle["penalty"]
    out = draw_penalty_run_on_frame(
        dim, ctx.frame, penalty, geom, bundle["lane_t"],
        m_per_px=bundle["mpp"], rank_color=bundle["rank_color"],
    )
    out = annotate_players(out, ctx.dets, show_tracker_ids=True)
    out = annotate_ball(out, ctx.dets)
    _draw_penalty_panel(
        out, label="RUN", penalty=bundle["breakdown"].backward_run_penalty,
        detail=f"align cos {penalty.pass_align:.2f}", layout=layout,
    )
    _draw_step_badge(
        out, step=3, total=4,
        title=_metric_title(bundle["rank"], "PEN · RUN"), subtitle="", layout=layout,
    )
    return draw_branding_tag(
        attach_conference_radar(
            out, ctx, bundle["lane_opts"], display_ranks=bundle["ranks"],
            radar_overlay=_penalty_radar_overlay(
                ctx, bundle["option"], penalty, draw_penalty_run_on_radar
            ),
        ),
        "Roboflow · pass lane AI",
    )


def render_penalty_back(
    ctx: ExplainContext,
    *,
    layout: Literal["talk", "social"] = "talk",
    rank: int = 0,
) -> np.ndarray:
    bundle = _metric_lane_bundle(ctx, rank)
    if bundle is None or bundle["geom"] is None or bundle["penalty"] is None:
        return _dim_frame(ctx.frame, 0.32)
    dim = _dim_frame(ctx.frame, 0.28)
    geom, penalty = bundle["geom"], bundle["penalty"]
    out = draw_penalty_back_on_frame(
        dim, ctx.frame, penalty, geom, bundle["lane_t"],
        m_per_px=bundle["mpp"], rank_color=bundle["rank_color"],
    )
    out = annotate_players(out, ctx.dets, show_tracker_ids=True)
    out = annotate_ball(out, ctx.dets)
    _draw_penalty_panel(
        out, label="BACK", penalty=bundle["breakdown"].backward_attack_penalty,
        detail=f"fwd {geom.forward_gain_m:.1f} m", layout=layout,
    )
    _draw_step_badge(
        out, step=3, total=4,
        title=_metric_title(bundle["rank"], "PEN · BACK"), subtitle="", layout=layout,
    )
    return draw_branding_tag(
        attach_conference_radar(
            out, ctx, bundle["lane_opts"], display_ranks=bundle["ranks"],
            radar_overlay=_penalty_radar_overlay(
                ctx, bundle["option"], penalty, draw_penalty_back_on_radar
            ),
        ),
        "Roboflow · pass lane AI",
    )


def _rank_scoring_segments(
    ctx: ExplainContext,
    rank: int,
    *,
    layout: Literal["talk", "social"],
) -> list[np.ndarray]:
    bundle = _metric_lane_bundle(ctx, rank)
    if bundle is None:
        return []
    segments = [
        render_metric_open(ctx, layout=layout, rank=rank),
        render_metric_forward(ctx, layout=layout, rank=rank),
        render_metric_space(ctx, layout=layout, rank=rank),
    ]
    for kind in _penalty_kinds(bundle["breakdown"]):
        if kind == "tm":
            segments.append(render_penalty_teammate(ctx, layout=layout, rank=rank))
        elif kind == "run":
            segments.append(render_penalty_run(ctx, layout=layout, rank=rank))
        elif kind == "back":
            segments.append(render_penalty_back(ctx, layout=layout, rank=rank))
    return segments


def render_step3_scoring(
    ctx: ExplainContext,
    *,
    layout: Literal["talk", "social"] = "talk",
    rank: int = 0,
) -> np.ndarray:
    out = _dim_frame(ctx.frame, 0.32)
    if not ctx.top3:
        return out

    rank = max(0, min(rank, len(ctx.top3) - 1))
    lane_opts = _ensure_lane_debug(ctx, [ctx.top3[rank]])
    option = lane_opts[0]
    feet = feet_xy(ctx.dets)
    breakdown = decompose_lane_score(
        option,
        ctx.weights,
        carrier_feet=feet[ctx.carrier.index],
        attack_dir=ctx.attack_dir,
        carrier_motion_dir=ctx.carrier_motion_dir,
    )
    lane_t = _lane_transformer(ctx)
    ranks = [rank]
    if lane_t is not None:
        out = apply_pass_lane_geometry(
            out, lane_opts, lane_t, feet, display_ranks=ranks
        )
    out = annotate_players(out, ctx.dets, show_tracker_ids=True)
    out = annotate_ball(out, ctx.dets)
    out = draw_carrier_spotlight(
        out, ctx.frame, (int(feet[ctx.carrier.index, 0]), int(feet[ctx.carrier.index, 1]))
    )
    out = draw_pass_arrows_on_frame(
        out,
        feet,
        ctx.carrier.index,
        lane_opts,
        metric=ctx.metric,
        show_chips=True,
        show_length=ctx.metric,
        display_ranks=ranks,
        arrow_thickness=2,
    )
    _draw_scoring_panel(out, breakdown, ctx.weights, layout=layout)
    rank_title = f"SCORE LANE #{rank + 1}" if len(ctx.top3) > 1 else "SCORE THE LANE"
    _draw_step_badge(
        out, step=3, total=4, title=rank_title, subtitle="", layout=layout
    )
    return draw_branding_tag(
        attach_conference_radar(out, ctx, lane_opts, display_ranks=ranks),
        "Roboflow · pass lane AI",
    )


def render_step4_ranking(
    ctx: ExplainContext,
    *,
    layout: Literal["talk", "social"] = "talk",
) -> np.ndarray:
    out = _dim_frame(ctx.frame)
    feet = feet_xy(ctx.dets)
    cx, cy = int(feet[ctx.carrier.index, 0]), int(feet[ctx.carrier.index, 1])
    top3 = _ensure_lane_debug(ctx, list(ctx.top3))
    lane_t = _lane_transformer(ctx)
    if lane_t is not None and top3:
        out = apply_pass_lane_geometry(out, top3, lane_t, feet)
    out = annotate_players(out, ctx.dets, show_tracker_ids=True)
    out = annotate_ball(out, ctx.dets)
    out = draw_carrier_spotlight(out, ctx.frame, (cx, cy))
    if top3:
        out = draw_pass_arrows_on_frame(
            out,
            feet,
            ctx.carrier.index,
            top3,
            metric=ctx.metric,
            show_chips=True,
            show_length=ctx.metric,
            arrow_thickness=2,
        )
    _draw_step_badge(
        out, step=4, total=4, title="PICK TOP 3", subtitle="", layout=layout
    )
    return draw_branding_tag(
        attach_conference_radar(out, ctx, top3), "Roboflow · pass lane AI"
    )


def render_conference_timeline(
    steps: dict[str, np.ndarray],
    *,
    gap: int = 14,
    bg: tuple[int, int, int] = (12, 12, 18),
) -> np.ndarray:
    order = [k for k in steps if k.startswith("01_") or k.startswith("02_")]
    order += sorted(k for k in steps if k.startswith("03_"))
    order += [k for k in steps if k.startswith("04_")]
    panels = [steps[k] for k in order if k in steps]
    if not panels:
        return np.zeros((100, 100, 3), dtype=np.uint8)
    w = max(p.shape[1] for p in panels)
    h = sum(p.shape[0] for p in panels) + gap * (len(panels) - 1)
    canvas = np.full((h, w, 3), bg, dtype=np.uint8)
    y = 0
    for panel in panels:
        ph, pw = panel.shape[:2]
        x = (w - pw) // 2
        canvas[y : y + ph, x : x + pw] = panel
        y += ph + gap
    return canvas


def render_conference_grid(
    steps: dict[str, np.ndarray],
    *,
    cell_size: tuple[int, int] = (480, 270),
    cols: int = 3,
) -> np.ndarray:
    order = [k for k in steps if k.startswith("01_") or k.startswith("02_")]
    order += sorted(k for k in steps if k.startswith("03_"))
    order += [k for k in steps if k.startswith("04_")]
    cells = [
        cv2.resize(steps[k], cell_size, interpolation=cv2.INTER_AREA)
        for k in order
        if k in steps
    ]
    if not cells:
        raise ValueError("No conference step images for grid")
    rows = [
        np.hstack(cells[i : i + cols])
        for i in range(0, len(cells), cols)
        if len(cells[i : i + cols]) == cols
    ]
    if len(cells) % cols:
        last = cells[len(cells) - len(cells) % cols :]
        pad = cols - len(last)
        blank = np.zeros_like(last[0])
        last = last + [blank] * pad
        rows.append(np.hstack(last))
    return np.vstack(rows)


def write_conference_frames(
    out_dir: str | Path,
    steps: dict[str, np.ndarray],
    *,
    grid: bool = True,
    timeline: bool = True,
) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for stem, image in steps.items():
        path = out_dir / f"pass_lane_{stem}.png"
        cv2.imwrite(str(path), image)
        written.append(path)
    if grid:
        grid_path = out_dir / "pass_lane_grid.png"
        cv2.imwrite(str(grid_path), render_conference_grid(steps))
        written.append(grid_path)
    if timeline:
        timeline_path = out_dir / "pass_lane_timeline.png"
        cv2.imwrite(str(timeline_path), render_conference_timeline(steps))
        written.append(timeline_path)
    return written


def render_conference_steps(
    ctx: ExplainContext,
    *,
    layout: Literal["talk", "social"] = "talk",
) -> dict[str, np.ndarray]:
    steps = {
        "01_candidates": render_step1_candidates(ctx, layout=layout),
        "02_corridors": render_step2_corridors(ctx, layout=layout),
        "03_attack_axis": render_attack_axis(ctx, layout=layout),
        "03_open": render_metric_open(ctx, layout=layout, rank=0),
        "03_forward": render_metric_forward(ctx, layout=layout, rank=0),
        "03_space": render_metric_space(ctx, layout=layout, rank=0),
    }
    bundle = _metric_lane_bundle(ctx, 0)
    if bundle is not None:
        for kind in _penalty_kinds(bundle["breakdown"]):
            if kind == "tm":
                steps["03_pen_tm"] = render_penalty_teammate(ctx, layout=layout, rank=0)
            elif kind == "run":
                steps["03_pen_run"] = render_penalty_run(ctx, layout=layout, rank=0)
            elif kind == "back":
                steps["03_pen_back"] = render_penalty_back(ctx, layout=layout, rank=0)
    steps["04_ranking"] = render_step4_ranking(ctx, layout=layout)
    if layout == "social":
        steps = {k: _fit_social_square(v) for k, v in steps.items()}
    return steps


@dataclass(frozen=True)
class ConferenceVideoTiming:
    output_fps: float = 6.0
    step_hold_seconds: float = 2.8
    metric_hold_seconds: float = 2.4
    summary_hold_seconds: float = 3.2
    crossfade_frames: int = 6
    crf: int = 16


def _blend_frames(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
    return cv2.addWeighted(a, 1.0 - alpha, b, alpha, 0)


def build_conference_video_sequence(
    ctx: ExplainContext,
    *,
    layout: Literal["talk", "social"] = "talk",
    timing: ConferenceVideoTiming = ConferenceVideoTiming(),
) -> list[np.ndarray]:
    fps = timing.output_fps
    step_hold = max(1, int(round(timing.step_hold_seconds * fps)))
    metric_hold = max(1, int(round(timing.metric_hold_seconds * fps)))
    summary_hold = max(1, int(round(timing.summary_hold_seconds * fps)))
    fade_n = max(0, timing.crossfade_frames)

    segments: list[np.ndarray] = [
        render_step1_candidates(ctx, layout=layout),
        render_step2_corridors(ctx, layout=layout),
        render_attack_axis(ctx, layout=layout),
    ]
    rank_segments: list[np.ndarray] = []
    for rank in range(min(3, len(ctx.top3))):
        rank_segments.extend(_rank_scoring_segments(ctx, rank, layout=layout))
    segments.extend(rank_segments)
    segments.append(render_step4_ranking(ctx, layout=layout))

    holds = [step_hold, step_hold, step_hold] + [metric_hold] * len(rank_segments) + [
        summary_hold
    ]
    out: list[np.ndarray] = []
    for i, (seg, hold) in enumerate(zip(segments, holds)):
        if i > 0 and fade_n > 0:
            prev = segments[i - 1]
            for f in range(1, fade_n + 1):
                out.append(_blend_frames(prev, seg, f / (fade_n + 1)))
        out.extend([seg] * hold)
    return out


def write_conference_video(
    path: str | Path,
    frames_bgr: list[np.ndarray],
    *,
    fps: float = 6.0,
    crf: int = 16,
) -> Path:
    path = Path(path)
    try:
        return write_h264_video(path, frames_bgr, fps=fps, crf=crf)
    except RuntimeError:
        h, w = frames_bgr[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
        if not writer.isOpened():
            raise RuntimeError(f"Could not open video writer: {path}")
        for frame in frames_bgr:
            writer.write(frame)
        writer.release()
        return path


def write_conference_gif(
    path: str | Path,
    mp4_path: str | Path,
    *,
    fps: float = 6.0,
    width: int | None = 1280,
) -> Path:
    return write_gif_from_mp4(path, mp4_path, fps=fps, width=width)


def build_conference_context(
    sequence,
    dets,
    frame_idx: int,
    frame: np.ndarray,
    *,
    weights: PassWeights | None = None,
    metric: bool = True,
    pitch_device: str = "cpu",
    transformer=None,
    radar_transformer=None,
    keypoints=None,
    keypoints_by_frame: dict | None = None,
    warmup_frames: list[tuple[int, object]] | None = None,
) -> ExplainContext | None:
    ctx = build_explain_context(
        sequence,
        dets,
        frame_idx,
        frame,
        weights=weights,
        metric=metric,
        pitch_device=pitch_device,
        transformer=transformer,
        radar_transformer=radar_transformer,
        keypoints=keypoints,
    )
    if ctx is None:
        return None
    kp_map = keypoints_by_frame
    if kp_map is None and keypoints is not None:
        kp_map = {frame_idx: keypoints}
    return enrich_conference_context(
        ctx, keypoints_by_frame=kp_map, warmup_frames=warmup_frames
    )


__all__ = [
    "PITCH_CONFIDENCE",
    "ConferenceVideoTiming",
    "attach_conference_radar",
    "build_conference_context",
    "build_conference_radar",
    "build_conference_video_sequence",
    "enrich_conference_context",
    "render_conference_steps",
    "render_attack_axis",
    "render_metric_forward",
    "render_metric_open",
    "render_metric_space",
    "write_conference_frames",
    "write_conference_gif",
    "write_conference_video",
]
