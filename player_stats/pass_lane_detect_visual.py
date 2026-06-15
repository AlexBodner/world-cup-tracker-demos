"""Pass-corridor explain frames — actual pass line + interception threat.

Same filmstrip language as ``pass_explain_visual`` (gutter panels, spotlights,
Roboflow branding) but focused on the scored lane of a detected pass: corridor
geometry at release and the nearest rival who can intercept it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import supervision as sv

from world_cup_projects.common.pitch import (
    ViewTransformer,
    homography_from_keypoints_radar,
    image_to_pitch_cm,
    image_to_pitch_m,
    pitch_attack_direction,
    render_radar_simple,
    warmup_goal_defenders_radar,
)
from world_cup_projects.common.possession import (
    Carrier,
    ball_xy,
    bbox_center_xy,
    feet_xy,
    player_mask,
)
from world_cup_projects.common.tracking_facing import carrier_kalman_direction
from world_cup_projects.common.video import write_gif_from_mp4, write_h264_video
from world_cup_projects.common.visual import (
    ROBOFLOW_PURPLE_BGR,
    annotate_ball,
    annotate_players,
    draw_branding_tag,
    draw_hud_bar,
    draw_radar_minimap,
    draw_text_shadow,
)
from world_cup_projects.pass_alternatives.explain_conference_metrics import (
    LaneMetricGeometry,
    _m_per_px,
    build_radar_metric_draw,
    compute_lane_metric_geometry,
    draw_openness_on_frame,
    draw_openness_on_radar,
)
from world_cup_projects.pass_alternatives.explain_visual import (
    ExplainContext,
    _dim_frame,
    _ensure_lane_debug,
    _fit_social_square,
    _ui_scale,
)
from world_cup_projects.pass_alternatives.lane_visual import (
    apply_pass_lane_geometry,
    draw_pass_arrows_on_frame,
    pass_line_label_xy,
)
from world_cup_projects.pass_alternatives.pass_options import (
    PassOption,
    PassWeights,
    ScoreBreakdown,
    decompose_lane_score,
)
from world_cup_projects.player_stats.pass_events import PassQualityScorer
from world_cup_projects.player_stats.pass_explain_visual import (
    PassExplainContext,
    PassExplainVideoTiming,
    _FOCUS_DIM,
    _GUTTER_W,
    _PANEL_BG_BGR,
    _PANEL_H,
    _PANEL_W,
    _SPOTLIGHT_STRENGTH,
    _action_focus_points,
    _apply_focus_spotlights,
    _compose_panel_row,
    _compose_strip,
    _crop_rect_from_points,
    _draw_focus_player,
    _draw_strip_title,
    _feet_for_tid,
    _fit_panel_frame,
    _letterbox_frame,
    _team_color,
    write_pass_explain_video,
)

_BRANDING = "Roboflow | pass corridor"
_PASS_LINE_BGR = (40, 220, 255)
_RIVAL_BGR = (60, 80, 255)
_OPEN_DASH_BGR = (80, 220, 60)
_PITCH_CONFIDENCE = 0.9


@dataclass(frozen=True)
class PassLaneDetectBundle:
    """Scored lane geometry for one inferred pass at release."""

    release_frame: int
    frame: np.ndarray
    dets: sv.Detections
    explain_ctx: ExplainContext
    option: PassOption
    geom: LaneMetricGeometry | None
    lane_opts: tuple[PassOption, ...]
    lane_t: ViewTransformer | None
    mpp: float
    rank_color: tuple[int, int, int]
    breakdown: ScoreBreakdown


@dataclass(frozen=True)
class PassLaneDetectContext:
    pass_ctx: PassExplainContext
    bundle: PassLaneDetectBundle


def _carrier_from_tid(dets: sv.Detections, tid: int) -> Carrier | None:
    ball = ball_xy(dets)
    if ball is None or dets.tracker_id is None:
        return None
    mask = dets.tracker_id == tid
    if not mask.any():
        return None
    idx = int(np.flatnonzero(mask)[0])
    team_arr = dets.data.get("team") if dets.data else None
    team = int(team_arr[idx]) if team_arr is not None else 0
    feet = feet_xy(dets)[idx]
    dist = float(np.hypot(feet[0] - ball[0], feet[1] - ball[1]))
    return Carrier(idx, team, dist, np.asarray(ball, dtype=np.float64))


def _build_explain_ctx_at_release(
    pass_ctx: PassExplainContext,
    carrier: Carrier,
    *,
    weights: PassWeights,
) -> ExplainContext | None:
    fi = pass_ctx.pass_event.frame_idx
    frame = pass_ctx.frames.get(fi)
    dets = pass_ctx.dets_by_frame.get(fi)
    if frame is None or dets is None:
        return None

    transformer = pass_ctx.radar_transformers.get(fi)
    keypoints = pass_ctx.keypoints_by_frame.get(fi)
    if pass_ctx.metric and keypoints is not None and transformer is None:
        transformer = homography_from_keypoints_radar(
            keypoints, confidence=_PITCH_CONFIDENCE
        )
    radar_h = transformer
    if keypoints is not None and radar_h is None:
        radar_h = homography_from_keypoints_radar(
            keypoints, confidence=_PITCH_CONFIDENCE
        )

    motion_dir = None
    if weights.use_carrier_motion and pass_ctx.metric and transformer is not None:
        motion_dir = carrier_kalman_direction(
            dets, carrier.index, transformer=transformer
        )

    attack_dir = None
    options: list[PassOption] = []
    if pass_ctx.metric and transformer is not None:
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
            from world_cup_projects.pass_alternatives.pass_options import score_pass_options

            options = score_pass_options(
                dets,
                carrier,
                weights=weights,
                attack_dir=attack_dir,
                positions=pitch_feet,
                carrier_motion_dir=motion_dir,
                pitch_cm=pitch_cm,
                body_pitch_m=body_pitch_m,
            )

    pmask = player_mask(dets)
    teams = dets.data["team"]
    teammates = pmask & (teams == carrier.team)
    teammates[carrier.index] = False

    locked = None
    warmup = [(fi, dets)]
    if pass_ctx.keypoints_by_frame:
        locked = warmup_goal_defenders_radar(
            warmup,
            pass_ctx.keypoints_by_frame,
            confidence=_PITCH_CONFIDENCE,
        )

    return ExplainContext(
        frame_idx=fi,
        frame=frame,
        dets=dets,
        carrier=carrier,
        options=tuple(options),
        top3=tuple(options[:3]),
        weights=weights,
        transformer=transformer,
        radar_transformer=radar_h,
        keypoints=keypoints,
        metric=pass_ctx.metric,
        teammate_count=int(teammates.sum()),
        attack_dir=attack_dir,
        carrier_motion_dir=motion_dir,
        locked_goal_defenders=locked,
        pitch_confidence=_PITCH_CONFIDENCE,
    )


def build_pass_lane_detect_bundle(
    pass_ctx: PassExplainContext,
    scorer: PassQualityScorer,
    *,
    weights: PassWeights | None = None,
) -> PassLaneDetectBundle | None:
    """Score the actual pass lane at release and rebuild intercept geometry."""
    w = weights or PassWeights.metric() if pass_ctx.metric else PassWeights()
    fi = pass_ctx.pass_event.frame_idx
    frame = pass_ctx.frames.get(fi)
    dets = pass_ctx.dets_by_frame.get(fi)
    if frame is None or dets is None:
        return None

    carrier = _carrier_from_tid(dets, pass_ctx.pass_event.passer_tid)
    if carrier is None:
        return None

    explain_ctx = _build_explain_ctx_at_release(pass_ctx, carrier, weights=w)
    if explain_ctx is None:
        return None

    option = scorer.option_for_receiver(
        fi, dets, carrier, pass_ctx.pass_event.receiver_tid
    )
    if option is None:
        return None

    lane_opts = tuple(_ensure_lane_debug(explain_ctx, [option]))
    option = lane_opts[0]
    geom = compute_lane_metric_geometry(explain_ctx, option)
    feet_img = feet_xy(dets)
    breakdown = decompose_lane_score(
        option,
        explain_ctx.weights,
        carrier_feet=feet_img[carrier.index],
        attack_dir=explain_ctx.attack_dir,
        carrier_motion_dir=explain_ctx.carrier_motion_dir,
    )
    lane_t = explain_ctx.transformer
    mpp = _m_per_px(explain_ctx, option)
    rank_color = _PASS_LINE_BGR

    return PassLaneDetectBundle(
        release_frame=fi,
        frame=frame,
        dets=dets,
        explain_ctx=explain_ctx,
        option=option,
        geom=geom,
        lane_opts=lane_opts,
        lane_t=lane_t,
        mpp=mpp,
        rank_color=rank_color,
        breakdown=breakdown,
    )


def build_pass_lane_detect_context(
    pass_ctx: PassExplainContext,
    scorer: PassQualityScorer,
    *,
    weights: PassWeights | None = None,
) -> PassLaneDetectContext | None:
    bundle = build_pass_lane_detect_bundle(pass_ctx, scorer, weights=weights)
    if bundle is None:
        return None
    return PassLaneDetectContext(pass_ctx=pass_ctx, bundle=bundle)


def _lane_crop_rect(
    lane_ctx: PassLaneDetectContext,
) -> tuple[int, int, int, int] | None:
    p = lane_ctx.pass_ctx.pass_event
    b = lane_ctx.bundle
    points = _action_focus_points(b.dets, p.passer_tid, p.receiver_tid)
    if b.geom is not None and b.geom.openness_rival_idx is not None:
        rival_feet = _feet_for_tid(b.dets, int(b.dets.tracker_id[b.geom.openness_rival_idx]))
        if rival_feet is not None:
            points.append(rival_feet)
    return _crop_rect_from_points(b.frame.shape, points, min_frac=0.40)


def _draw_simple_pass_line(
    frame: np.ndarray,
    feet: np.ndarray,
    carrier_index: int,
    receiver_index: int,
    color_bgr: tuple[int, int, int],
) -> None:
    c = (int(feet[carrier_index, 0]), int(feet[carrier_index, 1]))
    r = (int(feet[receiver_index, 0]), int(feet[receiver_index, 1]))
    cv2.line(frame, c, r, color_bgr, 3, cv2.LINE_AA)
    lx, ly = pass_line_label_xy(c, r)
    draw_text_shadow(
        frame,
        "PASS LINE",
        (lx - 28, ly - 8),
        font_scale=0.48,
        color_bgr=color_bgr,
        thickness=1,
    )


def _build_pass_line_frame(lane_ctx: PassLaneDetectContext) -> np.ndarray:
    """Corridor + pass segment at release — passer and receiver lit."""
    p = lane_ctx.pass_ctx.pass_event
    b = lane_ctx.bundle
    color = _team_color(p.team)
    dimmed = _dim_frame(b.frame, _FOCUS_DIM)
    feet = feet_xy(b.dets)
    centers = [
        f
        for tid in (p.passer_tid, p.receiver_tid)
        if (f := _feet_for_tid(b.dets, tid)) is not None
    ]
    out = _apply_focus_spotlights(dimmed, b.frame, centers, strength=_SPOTLIGHT_STRENGTH)
    for tid in (p.passer_tid, p.receiver_tid):
        _draw_focus_player(out, b.dets, tid, color, prominent=True, locked=True)

    if b.lane_t is not None and b.lane_opts:
        out = apply_pass_lane_geometry(
            out, list(b.lane_opts), b.lane_t, feet, display_ranks=[0]
        )
        out = draw_pass_arrows_on_frame(
            out,
            feet,
            b.explain_ctx.carrier.index,
            list(b.lane_opts),
            metric=lane_ctx.pass_ctx.metric,
            show_chips=False,
            show_length=lane_ctx.pass_ctx.metric,
            arrow_thickness=3,
            display_ranks=[0],
        )
    else:
        _draw_simple_pass_line(
            out, feet, b.explain_ctx.carrier.index, b.option.receiver_index, _PASS_LINE_BGR
        )

    out = annotate_players(out, b.dets, show_tracker_ids=True)
    out = annotate_ball(out, b.dets)
    return out


def _build_intercept_frame(lane_ctx: PassLaneDetectContext) -> np.ndarray:
    """Pass line plus nearest rival inside the corridor (openness geometry)."""
    p = lane_ctx.pass_ctx.pass_event
    b = lane_ctx.bundle
    color = _team_color(p.team)
    dimmed = _dim_frame(b.frame, _FOCUS_DIM)
    feet = feet_xy(b.dets)

    if b.geom is not None and b.lane_t is not None:
        if b.lane_opts:
            dimmed = apply_pass_lane_geometry(
                dimmed, list(b.lane_opts), b.lane_t, feet, display_ranks=[0]
            )
        lane_debug = b.option.lane_debug
        blocking_tm = (
            lane_debug.blocking_teammate_indices if lane_debug is not None else ()
        )
        out = draw_openness_on_frame(
            dimmed,
            b.frame,
            b.geom,
            b.lane_t,
            m_per_px=b.mpp,
            rank_color=b.rank_color,
            feet_xy=feet,
            blocking_teammate_indices=blocking_tm,
        )
    else:
        out = _build_pass_line_frame(lane_ctx)

    for tid in (p.passer_tid, p.receiver_tid):
        _draw_focus_player(out, b.dets, tid, color, prominent=True, locked=True)

    if b.geom is not None and b.geom.openness_rival_idx is not None:
        rival_tid = int(b.dets.tracker_id[b.geom.openness_rival_idx])
        rival_color = _team_color(1 - p.team)
        _draw_focus_player(out, b.dets, rival_tid, rival_color, prominent=True, locked=True)
        rival_feet = _feet_for_tid(b.dets, rival_tid)
        if rival_feet is not None:
            draw_text_shadow(
                out,
                "INTERCEPTOR",
                (rival_feet[0] - 36, rival_feet[1] - 44),
                font_scale=0.52,
                color_bgr=_RIVAL_BGR,
                thickness=1,
            )

    out = annotate_players(out, b.dets, show_tracker_ids=True)
    out = annotate_ball(out, b.dets)
    return out


def _openness_label(lane_ctx: PassLaneDetectContext) -> str:
    b = lane_ctx.bundle
    p = lane_ctx.pass_ctx.pass_event
    if b.geom is None:
        return "lane geometry unavailable"
    opp = b.geom.opponent_openness_m
    if b.geom.openness_rival_idx is None or opp >= b.explain_ctx.weights.open_ref * 0.99:
        return "lane clear — no rival in corridor"
    return f"{opp:.1f} m to nearest blocker"


def _attach_lane_radar(
    canvas: np.ndarray,
    lane_ctx: PassLaneDetectContext,
) -> np.ndarray:
    b = lane_ctx.bundle
    if not lane_ctx.pass_ctx.metric or b.explain_ctx.keypoints is None:
        return canvas
    radar_h = b.explain_ctx.radar_transformer
    if radar_h is None:
        radar_h = homography_from_keypoints_radar(
            b.explain_ctx.keypoints, confidence=b.explain_ctx.pitch_confidence
        )
    if radar_h is None:
        return canvas
    radar = render_radar_simple(
        b.dets,
        b.explain_ctx.keypoints,
        confidence=b.explain_ctx.pitch_confidence,
        transformer=radar_h,
        locked_goal_defenders=b.explain_ctx.locked_goal_defenders,
        debug_keypoints=False,
    )
    if radar is None:
        return canvas
    pitch_cm = image_to_pitch_cm(feet_xy(b.dets), radar_h)
    if pitch_cm is not None and b.geom is not None:
        draw = build_radar_metric_draw(
            b.explain_ctx, b.option, b.geom, pitch_cm, rank_color=b.rank_color
        )
        radar = draw_openness_on_radar(radar, draw)
    return draw_radar_minimap(canvas, b.dets, b.explain_ctx.keypoints, prebuilt_radar=radar, scale_frac=0.28)


def _render_pass_line_panel(
    lane_ctx: PassLaneDetectContext,
    *,
    layout: Literal["talk", "social"],
    crop: tuple[int, int, int, int] | None,
) -> np.ndarray:
    p = lane_ctx.pass_ctx.pass_event
    b = lane_ctx.bundle
    color = _team_color(p.team)
    frame_img = _build_pass_line_frame(lane_ctx)
    length = ""
    if lane_ctx.pass_ctx.metric and b.option.length > 0:
        length = f" · {b.option.length:.1f} m"
    return _compose_panel_row(
        frame_img,
        b.release_frame,
        "PASS LINE",
        color,
        layout=layout,
        step=1,
        step_total=2,
        sublabel=f"#{p.passer_tid} → #{p.receiver_tid}{length}",
        badge="CORRIDOR",
        emphasis=1.0,
        locked=True,
        accent_bgr=ROBOFLOW_PURPLE_BGR,
        crop=crop,
    )


def _render_intercept_panel(
    lane_ctx: PassLaneDetectContext,
    *,
    layout: Literal["talk", "social"],
    crop: tuple[int, int, int, int] | None,
) -> np.ndarray:
    p = lane_ctx.pass_ctx.pass_event
    b = lane_ctx.bundle
    color = _team_color(p.team)
    frame_img = _build_intercept_frame(lane_ctx)
    return _compose_panel_row(
        frame_img,
        b.release_frame,
        "INTERCEPTION",
        color,
        layout=layout,
        step=2,
        step_total=2,
        sublabel=_openness_label(lane_ctx),
        badge="THREAT",
        emphasis=1.0,
        locked=True,
        accent_bgr=_RIVAL_BGR,
        crop=crop,
    )


def render_strip_pass_line(
    lane_ctx: PassLaneDetectContext,
    *,
    layout: Literal["talk", "social"] = "talk",
) -> np.ndarray:
    crop = _lane_crop_rect(lane_ctx)
    panel = _render_pass_line_panel(lane_ctx, layout=layout, crop=crop)
    header_h = int(62 * _ui_scale(layout))
    header = np.full((header_h, panel.shape[1], 3), _PANEL_BG_BGR, dtype=np.uint8)
    cv2.rectangle(header, (0, 0), (3, header_h), _PASS_LINE_BGR, -1)
    _draw_strip_title(
        header,
        "THE PASS LINE",
        "shaded corridor from passer to receiver at release",
        layout=layout,
    )
    out = np.vstack([header, panel])
    return draw_branding_tag(out, _BRANDING)


def render_strip_intercept(
    lane_ctx: PassLaneDetectContext,
    *,
    layout: Literal["talk", "social"] = "talk",
) -> np.ndarray:
    crop = _lane_crop_rect(lane_ctx)
    panel = _render_intercept_panel(lane_ctx, layout=layout, crop=crop)
    header_h = int(62 * _ui_scale(layout))
    header = np.full((header_h, panel.shape[1], 3), _PANEL_BG_BGR, dtype=np.uint8)
    cv2.rectangle(header, (0, 0), (3, header_h), _RIVAL_BGR, -1)
    sub = _openness_label(lane_ctx)
    _draw_strip_title(
        header,
        "INTERCEPTION THREAT",
        sub,
        layout=layout,
    )
    out = np.vstack([header, panel])
    return draw_branding_tag(out, _BRANDING)


def render_lane_detect_video_summary(lane_ctx: PassLaneDetectContext) -> np.ndarray:
    out = _build_intercept_frame(lane_ctx)
    p = lane_ctx.pass_ctx.pass_event
    out = draw_hud_bar(out, "LANE SCORED")
    qs = p.quality_score
    sub = (
        f"openness {p.openness:.1f} m · score {qs:.2f}"
        if p.openness is not None and qs is not None
        else _openness_label(lane_ctx)
    )
    draw_text_shadow(
        out,
        f"#{p.passer_tid} → #{p.receiver_tid}  ·  {sub}",
        (18, out.shape[0] - 18),
        font_scale=0.52,
        color_bgr=(200, 200, 210),
        thickness=1,
    )
    out = _attach_lane_radar(out, lane_ctx)
    return draw_branding_tag(out, _BRANDING)


def render_lane_detect_summary(
    lane_ctx: PassLaneDetectContext,
    *,
    layout: Literal["talk", "social"] = "talk",
) -> np.ndarray:
    p = lane_ctx.pass_ctx.pass_event
    b = lane_ctx.bundle
    color = _team_color(p.team)
    out = _build_intercept_frame(lane_ctx)
    crop = _lane_crop_rect(lane_ctx)
    out = _letterbox_frame(
        _fit_panel_frame(out, crop=crop),
        target_w=_PANEL_W + _GUTTER_W,
        target_h=_PANEL_H,
    )
    qs = p.quality_score
    qs_label = f" · score {qs:.2f}" if qs is not None else ""
    _draw_strip_title(
        out,
        "LANE SCORED",
        f"openness {p.openness:.1f} m{qs_label}" if p.openness is not None else _openness_label(lane_ctx),
        layout=layout,
    )
    out = _attach_lane_radar(out, lane_ctx)
    out = draw_hud_bar(out, f"#{p.passer_tid} → #{p.receiver_tid}")
    return draw_branding_tag(out, _BRANDING)


def render_pass_lane_detect_strips(
    lane_ctx: PassLaneDetectContext,
    *,
    layout: Literal["talk", "social"] = "talk",
) -> dict[str, np.ndarray]:
    strips = {
        "strip_pass_line": render_strip_pass_line(lane_ctx, layout=layout),
        "strip_intercept": render_strip_intercept(lane_ctx, layout=layout),
        "summary": render_lane_detect_summary(lane_ctx, layout=layout),
    }
    if layout == "social":
        strips = {k: _fit_social_square(v) for k, v in strips.items()}
    return strips


def render_pass_lane_detect_timeline(
    strips: dict[str, np.ndarray],
    *,
    gap: int = 8,
) -> np.ndarray:
    order = ["strip_pass_line", "strip_intercept", "summary"]
    parts = [strips[k] for k in order if k in strips]
    if not parts:
        raise ValueError("No lane strips to compose")
    target_w = max(p.shape[1] for p in parts)
    aligned = []
    for part in parts:
        if part.shape[1] < target_w:
            pad = target_w - part.shape[1]
            part = cv2.copyMakeBorder(
                part, 0, 0, 0, pad, cv2.BORDER_CONSTANT, value=(8, 8, 10)
            )
        aligned.append(part)
    sep = np.full((gap, target_w, 3), (8, 8, 10), dtype=np.uint8)
    out = aligned[0]
    for part in aligned[1:]:
        out = np.vstack([out, sep, part])
    return out


def write_pass_lane_detect_frames(
    out_dir: str | Path,
    strips: dict[str, np.ndarray],
    *,
    timeline: bool = True,
) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for stem, image in strips.items():
        path = out_dir / f"pass_lane_detect_{stem}.png"
        cv2.imwrite(str(path), image)
        written.append(path)
    if timeline:
        path = out_dir / "pass_lane_detect_timeline.png"
        cv2.imwrite(str(path), render_pass_lane_detect_timeline(strips))
        written.append(path)
    return written


def render_lane_detect_video_pass_line(lane_ctx: PassLaneDetectContext) -> np.ndarray:
    out = _build_pass_line_frame(lane_ctx)
    out = draw_hud_bar(out, "THE PASS LINE")
    return draw_branding_tag(out, _BRANDING)


def render_lane_detect_video_intercept(lane_ctx: PassLaneDetectContext) -> np.ndarray:
    out = _build_intercept_frame(lane_ctx)
    out = draw_hud_bar(out, "INTERCEPTION THREAT")
    draw_text_shadow(
        out,
        _openness_label(lane_ctx),
        (18, out.shape[0] - 18),
        font_scale=0.52,
        color_bgr=(200, 200, 210),
        thickness=1,
    )
    return draw_branding_tag(out, _BRANDING)


def build_pass_lane_detect_video_sequence(
    lane_ctx: PassLaneDetectContext,
    *,
    timing: PassExplainVideoTiming = PassExplainVideoTiming(),
) -> list[np.ndarray]:
    hold = max(1, int(round(timing.hold_locked_seconds * timing.output_fps)))
    summary_n = max(1, int(round(timing.summary_hold_seconds * timing.output_fps)))
    out: list[np.ndarray] = []
    out.extend([render_lane_detect_video_pass_line(lane_ctx)] * hold)
    out.extend([render_lane_detect_video_intercept(lane_ctx)] * hold)
    out.extend([render_lane_detect_video_summary(lane_ctx)] * summary_n)
    return out


def write_pass_lane_detect_video(
    path: str | Path,
    frames_bgr: list[np.ndarray],
    *,
    fps: float = 8.0,
    crf: int = 16,
) -> Path:
    return write_pass_explain_video(path, frames_bgr, fps=fps, crf=crf)


def write_pass_lane_detect_gif(
    path: str | Path,
    mp4_path: str | Path,
    *,
    fps: float = 8.0,
    width: int | None = 1280,
) -> Path:
    return write_gif_from_mp4(path, mp4_path, fps=fps, width=width)
