"""Step-by-step passing-lane explanation frames for talks and social posts.

Produces four static images that walk through how we identify lanes, score them,
rank the top three, and why some options are rejected. Designed for conference
slides and Twitter/LinkedIn carousels.
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
    detect_pitch_keypoints,
    homography_from_keypoints_radar,
    image_to_pitch_cm,
    image_to_pitch_m,
    load_pitch_model,
    pitch_attack_direction,
    render_radar_simple,
)
from world_cup_projects.common.possession import (
    Carrier,
    bbox_center_xy,
    feet_xy,
    find_control_carrier,
    player_mask,
)
from world_cup_projects.common.tracking_facing import carrier_kalman_direction
from world_cup_projects.common.video import (
    read_sequence_frame,
    write_gif_from_mp4,
    write_h264_video,
)
from world_cup_projects.common.visual import (
    ROBOFLOW_PURPLE_BGR,
    annotate_ball,
    annotate_players,
    draw_branding_tag,
    draw_carrier_spotlight,
    draw_radar_minimap,
    draw_score_chip,
    draw_text_shadow,
)
from world_cup_projects.pass_alternatives.lane_visual import (
    apply_pass_lane_geometry,
    draw_candidate_lanes_on_radar,
    draw_pass_arrows_on_frame,
    draw_pass_lanes_on_radar,
)
from world_cup_projects.pass_alternatives.pass_options import (
    PassOption,
    PassWeights,
    ScoreBreakdown,
    decompose_lane_score,
    remap_lane_debug_to_pitch_cm,
    score_pass_options,
    top_pass_options,
)

_RANK_BGR = [(80, 220, 60), (0, 215, 255), (40, 140, 255)]
_CANDIDATE_BGR = (120, 120, 130)
_REJECT_BGR = (70, 70, 220)
_STEP_BADGE_BGR = (22, 22, 28)
_PANEL_BG = (14, 14, 18)
_PRESENTATION = True  # minimal on-image copy — captions live in the deck/post


@dataclass(frozen=True)
class ExplainContext:
    """Everything needed to render the four explanation steps for one freeze frame."""

    frame_idx: int
    frame: np.ndarray
    dets: sv.Detections
    carrier: Carrier
    options: tuple[PassOption, ...]
    top3: tuple[PassOption, ...]
    weights: PassWeights
    transformer: ViewTransformer | None
    radar_transformer: ViewTransformer | None
    keypoints: sv.KeyPoints | None
    metric: bool
    teammate_count: int
    attack_dir: np.ndarray | None = None
    carrier_motion_dir: np.ndarray | None = None
    locked_goal_defenders: tuple[int, int] | None = None
    pitch_confidence: float = 0.9


def _ui_scale(layout: Literal["talk", "social"]) -> float:
    """Modest bump over demo defaults — keep well below collision threshold."""
    return 1.22 if layout == "social" else 1.08


def _badge_box(
    canvas: np.ndarray,
    layout: Literal["talk", "social"],
) -> tuple[int, int, int, int, float]:
    """Return pad, badge_w, badge_h, scale for the step title strip."""
    _h, w = canvas.shape[:2]
    scale = _ui_scale(layout)
    pad = 14
    badge_h = int(46 + 6 * scale)
    badge_w = int(min(w * 0.40, 440)) if layout == "talk" else int(w * 0.90)
    return pad, badge_w, badge_h, scale, w


def rejection_reason(option: PassOption, *, rank: int, weights: PassWeights) -> str:
    """Human-readable reason a lane was not promoted to the top three."""
    if option.length < weights.min_length:
        return f"Too short ({option.length:.1f} m < {weights.min_length:.0f} m)"
    if option.length > weights.max_length:
        return f"Too long ({option.length:.1f} m > {weights.max_length:.0f} m)"
    if option.rivals_in_lane > 0 and option.opponent_openness < weights.open_ref * 0.35:
        return f"Blocked — {option.rivals_in_lane} rival(s) in corridor"
    if option.forward_gain < 0:
        return "Backward — toward own goal"
    if option.motion_alignment < weights.backward_cos_threshold:
        return "Against run direction"
    if option.teammates_in_lane > 0:
        return "Teammate obstructing lane"
    if rank >= 3:
        return f"Rank #{rank + 1} — lower score than top 3"
    if option.score < weights.freeze_min_pass_score:
        return "Score below freeze threshold"
    return "Low openness or tight receiver space"


def _dim_frame(frame: np.ndarray, factor: float = 0.38) -> np.ndarray:
    return (frame.astype(np.float32) * factor).astype(np.uint8)


def _draw_step_badge(
    canvas: np.ndarray,
    *,
    step: int,
    total: int,
    title: str,
    subtitle: str,
    layout: Literal["talk", "social"],
) -> None:
    pad, badge_w, badge_h, scale, _w = _badge_box(canvas, layout)
    if subtitle and not _PRESENTATION:
        badge_h = int(66 + 6 * scale)
    x1 = pad + badge_w
    y1 = pad + badge_h
    cv2.rectangle(canvas, (pad, pad), (x1, y1), _STEP_BADGE_BGR, -1)
    cv2.rectangle(canvas, (pad, pad), (x1, y1), (55, 55, 65), 1)
    cv2.rectangle(canvas, (pad, pad), (pad + 5, y1), ROBOFLOW_PURPLE_BGR, -1)
    from world_cup_projects.common.visual import cv2_safe_text

    draw_text_shadow(
        canvas,
        f"STEP {step}/{total}",
        (pad + 14, pad + int(18 * scale)),
        font_scale=0.36 * scale,
        color_bgr=(150, 150, 160),
        thickness=1,
    )
    draw_text_shadow(
        canvas,
        cv2_safe_text(title),
        (pad + 14, pad + int(38 * scale)),
        font_scale=0.62 * scale,
        color_bgr=(255, 255, 255),
        thickness=2,
    )
    if subtitle and not _PRESENTATION:
        draw_text_shadow(
            canvas,
            cv2_safe_text(subtitle),
            (pad + 14, pad + int(58 * scale)),
            font_scale=0.44 * scale,
            color_bgr=(175, 175, 185),
            thickness=1,
        )


def _draw_side_panel(
    canvas: np.ndarray,
    lines: list[str],
    *,
    layout: Literal["talk", "social"],
    panel_width: int | None = None,
) -> None:
    if _PRESENTATION or not lines:
        return
    h, w = canvas.shape[:2]
    pw = panel_width or (int(w * 0.30) if layout == "talk" else int(w * 0.92))
    px = w - pw - 16
    py = 120 if layout == "talk" else h - int(h * 0.28)
    ph = h - py - 24 if layout == "talk" else int(h * 0.24)
    if layout == "social":
        px = int(w * 0.04)
        pw = int(w * 0.92)

    overlay = canvas.copy()
    cv2.rectangle(overlay, (px, py), (px + pw, py + ph), _PANEL_BG, -1)
    cv2.rectangle(overlay, (px, py), (px + pw, py + ph), (48, 48, 58), 1)
    cv2.rectangle(overlay, (px, py), (px + pw, py + 4), ROBOFLOW_PURPLE_BGR, -1)
    canvas[:] = cv2.addWeighted(overlay, 0.92, canvas, 0.08, 0)

    y = py + 28
    fs = 0.44 if layout == "talk" else 0.50
    for line in lines:
        draw_text_shadow(
            canvas,
            line,
            (px + 14, y),
            font_scale=fs,
            color_bgr=(220, 220, 228),
            thickness=1,
        )
        y += int(22 * (1.15 if layout == "social" else 1.0))


def _draw_score_bar(
    canvas: np.ndarray,
    x: int,
    y: int,
    label: str,
    value: float,
    max_value: float,
    *,
    color: tuple[int, int, int],
    label_w: int,
    bar_w: int,
    scale: float,
    negative: bool = False,
    panel_right: int | None = None,
) -> int:
    """One compact row: label | bar | value."""
    bar_h = 11
    bar_x = x + label_w
    bar_y = y - bar_h + 2
    cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (40, 40, 48), -1)
    denom = max(max_value, 1e-6)
    fill = int(bar_w * min(max(abs(value) / denom, 0.0), 1.0))
    bar_color = _REJECT_BGR if negative else color
    cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + max(fill, 2), bar_y + bar_h), bar_color, -1)
    draw_text_shadow(
        canvas,
        label,
        (x, y),
        font_scale=0.40 * scale,
        color_bgr=(200, 200, 210),
        thickness=1,
    )
    sign = "-" if negative and value > 0.005 else ""
    value_text = f"{sign}{abs(value):.2f}"
    font_scale = 0.38 * scale
    thickness = 1
    (tw, _), _ = cv2.getTextSize(value_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    value_x = bar_x + bar_w + 8
    if panel_right is not None:
        value_x = min(value_x, panel_right - 8 - tw)
    draw_text_shadow(
        canvas,
        value_text,
        (value_x, y),
        font_scale=font_scale,
        color_bgr=bar_color,
        thickness=thickness,
    )
    return y + int(26 * scale)


def _draw_scoring_panel(
    canvas: np.ndarray,
    breakdown: ScoreBreakdown,
    weights: PassWeights,
    *,
    layout: Literal["talk", "social"],
) -> None:
    h, w = canvas.shape[:2]
    pad, _badge_w, badge_h, scale, _ = _badge_box(canvas, layout)
    # Upper-left panel — stays clear of radar (bottom-right) and step badge.
    px = pad
    py = pad + badge_h + 10
    pw = int(min(w * 0.27, 320)) if layout == "talk" else int(w * 0.46)
    penalties = [
        (label, pen)
        for label, pen in (
            ("TM", breakdown.teammate_penalty),
            ("RUN", breakdown.backward_run_penalty),
            ("BACK", breakdown.backward_attack_penalty),
        )
        if label == "RUN" or pen > 0.005
    ]
    row_count = 3 + len(penalties)
    ph = int(36 * scale + row_count * 26 * scale)
    ph = min(ph, int(h * 0.52) - py)
    overlay = canvas.copy()
    cv2.rectangle(overlay, (px, py), (px + pw, py + ph), _PANEL_BG, -1)
    cv2.rectangle(overlay, (px, py), (px + pw, py + ph), (55, 55, 65), 1)
    canvas[:] = cv2.addWeighted(overlay, 0.92, canvas, 0.08, 0)

    draw_text_shadow(
        canvas,
        f"{breakdown.total:.2f}",
        (px + 12, py + int(28 * scale)),
        font_scale=0.72 * scale,
        color_bgr=_RANK_BGR[0],
        thickness=2,
    )
    y = py + int(44 * scale)
    label_w = 52
    bar_w = pw - label_w - 58
    y = _draw_score_bar(
        canvas, px + 12, y, "OPEN", breakdown.openness_term, weights.openness,
        color=_RANK_BGR[0], label_w=label_w, bar_w=bar_w, scale=scale,
    )
    y = _draw_score_bar(
        canvas, px + 12, y, "FWD", breakdown.forward_term, weights.forward,
        color=(0, 200, 255), label_w=label_w, bar_w=bar_w, scale=scale,
    )
    y = _draw_score_bar(
        canvas, px + 12, y, "SPACE", breakdown.space_term, weights.space,
        color=(200, 140, 255), label_w=label_w, bar_w=bar_w, scale=scale,
    )
    for label, pen in penalties:
        y = _draw_score_bar(
            canvas, px + 12, y, label, pen, 0.22,
            color=_REJECT_BGR, label_w=label_w, bar_w=bar_w, scale=scale,
            negative=pen > 0.005,
            panel_right=px + pw,
        )


def _ensure_lane_debug(
    ctx: ExplainContext,
    options: list[PassOption],
) -> list[PassOption]:
    """Guarantee corridor polygons exist for explain overlays."""
    if not ctx.metric or ctx.transformer is None:
        return options
    if options and all(o.lane_debug is not None for o in options):
        return options
    feet_img = feet_xy(ctx.dets)
    pitch_feet = image_to_pitch_m(feet_img, ctx.transformer)
    pitch_cm = image_to_pitch_cm(feet_img, ctx.transformer)
    body_pitch_m = image_to_pitch_m(bbox_center_xy(ctx.dets), ctx.transformer)
    if pitch_feet is None or pitch_cm is None or ctx.attack_dir is None:
        return options
    refreshed = top_pass_options(
        ctx.dets,
        ctx.carrier,
        k=max(len(options), 6),
        weights=ctx.weights,
        attack_dir=ctx.attack_dir,
        positions=pitch_feet,
        carrier_motion_dir=ctx.carrier_motion_dir,
        pitch_cm=pitch_cm,
        body_pitch_m=body_pitch_m,
    )
    by_receiver = {o.receiver_index: o for o in refreshed}
    return [by_receiver.get(o.receiver_index, o) for o in options]


def _attach_radar(
    canvas: np.ndarray,
    ctx: ExplainContext,
    options: list[PassOption],
    *,
    candidate_indices: list[int] | None = None,
    pitch_confidence: float = 0.5,
) -> np.ndarray:
    """Minimap + lanes — uses tracker radar H (orientation-locked) when available."""
    if not ctx.metric or ctx.keypoints is None:
        return canvas
    radar_h = ctx.radar_transformer
    if radar_h is None:
        radar_h = homography_from_keypoints_radar(
            ctx.keypoints, confidence=pitch_confidence
        )
    if radar_h is None:
        return canvas
    radar = render_radar_simple(
        ctx.dets,
        ctx.keypoints,
        confidence=pitch_confidence,
        transformer=radar_h,
        debug_keypoints=False,
    )
    if radar is None:
        return canvas
    feet_img = feet_xy(ctx.dets)
    pitch_cm = image_to_pitch_cm(feet_img, radar_h)
    if pitch_cm is not None:
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
            radar = draw_pass_lanes_on_radar(radar, remapped, pitch_cm)
    return draw_radar_minimap(
        canvas,
        ctx.dets,
        ctx.keypoints,
        prebuilt_radar=radar,
        debug_keypoints=False,
        scale_frac=0.33,
    )


def render_step1_candidates(
    ctx: ExplainContext,
    *,
    layout: Literal["talk", "social"] = "talk",
) -> np.ndarray:
    """Step 1 — identify ball carrier and teammate pass candidates."""
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
        out,
        step=1,
        total=4,
        title="WHO CAN RECEIVE?",
        subtitle="",
        layout=layout,
    )
    out = _attach_radar(out, ctx, [], candidate_indices=teammate_idxs)
    return draw_branding_tag(out, "Roboflow · pass lane AI")


def render_step2_corridors(
    ctx: ExplainContext,
    *,
    layout: Literal["talk", "social"] = "talk",
) -> np.ndarray:
    """Step 2 — pass corridors + blockers + arrows (same stack as freeze video)."""
    out = _dim_frame(ctx.frame, 0.32)
    options = _ensure_lane_debug(
        ctx, list(ctx.top3) if ctx.top3 else list(ctx.options[:3])
    )
    feet = feet_xy(ctx.dets)
    if ctx.metric and ctx.transformer is not None and options:
        out = apply_pass_lane_geometry(out, options, ctx.transformer, feet)
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
        )

    _draw_step_badge(
        out,
        step=2,
        total=4,
        title="CHECK EACH LANE",
        subtitle="",
        layout=layout,
    )
    out = _attach_radar(out, ctx, options)
    return draw_branding_tag(out, "Roboflow · pass lane AI")


def render_step3_scoring(
    ctx: ExplainContext,
    *,
    layout: Literal["talk", "social"] = "talk",
    rank: int = 0,
) -> np.ndarray:
    """Step 3 — decompose one ranked lane into openness / forward / space."""
    out = _dim_frame(ctx.frame, 0.32)
    if not ctx.top3:
        return out

    rank = max(0, min(rank, len(ctx.top3) - 1))
    best_opts = _ensure_lane_debug(ctx, [ctx.top3[rank]])
    best = best_opts[0]
    feet = feet_xy(ctx.dets)
    breakdown = decompose_lane_score(
        best,
        ctx.weights,
        carrier_feet=feet[ctx.carrier.index],
        attack_dir=ctx.attack_dir,
        carrier_motion_dir=ctx.carrier_motion_dir,
    )
    if ctx.metric and ctx.transformer is not None:
        out = apply_pass_lane_geometry(out, best_opts, ctx.transformer, feet)
    out = annotate_players(out, ctx.dets, show_tracker_ids=True)
    out = annotate_ball(out, ctx.dets)
    out = draw_carrier_spotlight(
        out, ctx.frame, (int(feet[ctx.carrier.index, 0]), int(feet[ctx.carrier.index, 1]))
    )
    out = draw_pass_arrows_on_frame(
        out,
        feet,
        ctx.carrier.index,
        best_opts,
        metric=ctx.metric,
        show_chips=False,
        show_length=ctx.metric,
    )
    _draw_scoring_panel(out, breakdown, ctx.weights, layout=layout)

    rank_title = "SCORE THE LANE"
    if len(ctx.top3) > 1:
        rank_title = f"SCORE LANE #{rank + 1}"
    _draw_step_badge(
        out,
        step=3,
        total=4,
        title=rank_title,
        subtitle="",
        layout=layout,
    )
    out = _attach_radar(out, ctx, best_opts)
    return draw_branding_tag(out, "Roboflow · pass lane AI")


def render_step4_ranking(
    ctx: ExplainContext,
    *,
    layout: Literal["talk", "social"] = "talk",
) -> np.ndarray:
    """Step 4 — top three lanes plus rejection reasons for the rest."""
    out = _dim_frame(ctx.frame)
    feet = feet_xy(ctx.dets)
    cx, cy = int(feet[ctx.carrier.index, 0]), int(feet[ctx.carrier.index, 1])

    top3 = _ensure_lane_debug(ctx, list(ctx.top3))
    if ctx.metric and ctx.transformer is not None and top3:
        out = apply_pass_lane_geometry(out, top3, ctx.transformer, feet)
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
        )

    _draw_step_badge(
        out,
        step=4,
        total=4,
        title="PICK TOP 3",
        subtitle="",
        layout=layout,
    )
    out = _attach_radar(out, ctx, top3)
    return draw_branding_tag(out, "Roboflow · pass lane AI")


def _fit_social_square(frame: np.ndarray, size: int = 1080) -> np.ndarray:
    """Center-crop to square for Instagram / X carousel posts."""
    h, w = frame.shape[:2]
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    crop = frame[y0 : y0 + side, x0 : x0 + side]
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)


def render_explain_steps(
    ctx: ExplainContext,
    *,
    layout: Literal["talk", "social"] = "talk",
) -> dict[str, np.ndarray]:
    """Return all four step images keyed by filename stem."""
    steps = {
        "01_candidates": render_step1_candidates(ctx, layout=layout),
        "02_corridors": render_step2_corridors(ctx, layout=layout),
        "03_scoring": render_step3_scoring(ctx, layout=layout),
        "04_ranking": render_step4_ranking(ctx, layout=layout),
    }
    if layout == "social":
        steps = {k: _fit_social_square(v) for k, v in steps.items()}
    return steps


def render_explain_timeline(
    steps: dict[str, np.ndarray],
    *,
    gap: int = 14,
    bg: tuple[int, int, int] = (12, 12, 18),
) -> np.ndarray:
    """Vertical stack of the four steps — poster / slide handout."""
    order = ["01_candidates", "02_corridors", "03_scoring", "04_ranking"]
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


@dataclass(frozen=True)
class LaneExplainVideoTiming:
    """Pacing for pass-lane scoring walkthrough on a single freeze frame."""

    output_fps: float = 6.0
    step_hold_seconds: float = 2.8
    rank_hold_seconds: float = 2.4
    summary_hold_seconds: float = 3.2
    crossfade_frames: int = 6
    crf: int = 16


def _blend_frames(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
    return cv2.addWeighted(a, 1.0 - alpha, b, alpha, 0)


def build_lane_explain_video_sequence(
    ctx: ExplainContext,
    *,
    layout: Literal["talk", "social"] = "talk",
    timing: LaneExplainVideoTiming = LaneExplainVideoTiming(),
) -> list[np.ndarray]:
    """Stepped clip: candidates → corridors → score each rank → top 3."""
    fps = timing.output_fps
    step_hold = max(1, int(round(timing.step_hold_seconds * fps)))
    rank_hold = max(1, int(round(timing.rank_hold_seconds * fps)))
    summary_hold = max(1, int(round(timing.summary_hold_seconds * fps)))
    fade_n = max(0, timing.crossfade_frames)

    segments: list[np.ndarray] = [
        render_step1_candidates(ctx, layout=layout),
        render_step2_corridors(ctx, layout=layout),
    ]
    for rank in range(min(3, len(ctx.top3))):
        segments.append(render_step3_scoring(ctx, layout=layout, rank=rank))
    segments.append(render_step4_ranking(ctx, layout=layout))

    holds = [step_hold, step_hold] + [rank_hold] * min(3, len(ctx.top3)) + [summary_hold]
    out: list[np.ndarray] = []
    for i, (seg, hold) in enumerate(zip(segments, holds)):
        if i > 0 and fade_n > 0:
            prev = segments[i - 1]
            for f in range(1, fade_n + 1):
                alpha = f / (fade_n + 1)
                out.append(_blend_frames(prev, seg, alpha))
        out.extend([seg] * hold)
    return out


def write_lane_explain_video(
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


def write_lane_explain_gif(
    path: str | Path,
    mp4_path: str | Path,
    *,
    fps: float = 6.0,
    width: int | None = 1280,
) -> Path:
    return write_gif_from_mp4(path, mp4_path, fps=fps, width=width)


def render_explain_grid(steps: dict[str, np.ndarray], *, cell_size: tuple[int, int] = (960, 540)) -> np.ndarray:
    """2×2 composite slide for a single talk frame."""
    order = ["01_candidates", "02_corridors", "03_scoring", "04_ranking"]
    cells = [
        cv2.resize(steps[key], cell_size, interpolation=cv2.INTER_AREA)
        for key in order
        if key in steps
    ]
    if len(cells) != 4:
        raise ValueError("Need four step images for grid composite")
    top = np.hstack(cells[:2])
    bottom = np.hstack(cells[2:])
    return np.vstack([top, bottom])


def build_explain_context(
    sequence,
    dets: sv.Detections,
    frame_idx: int,
    frame: np.ndarray,
    *,
    weights: PassWeights | None = None,
    metric: bool = True,
    pitch_device: str = "cpu",
    transformer: ViewTransformer | None = None,
    radar_transformer: ViewTransformer | None = None,
    keypoints: sv.KeyPoints | None = None,
) -> ExplainContext | None:
    """Score all lanes for one frame; return None if no carrier."""
    w = weights or PassWeights.metric()
    carrier = find_control_carrier(dets, transformer=transformer)
    if carrier is None:
        return None

    motion_dir = None
    if w.use_carrier_motion:
        motion_dir = carrier_kalman_direction(
            dets, carrier.index, transformer=transformer if metric else None
        )
    attack_dir = None

    if metric and transformer is not None:
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
            weights=w,
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
            weights=w,
            carrier_motion_dir=motion_dir,
        )

    pmask = player_mask(dets)
    teams = dets.data["team"]
    teammates = pmask & (teams == carrier.team)
    teammates[carrier.index] = False

    if keypoints is None and metric:
        model = load_pitch_model(device=pitch_device)
        keypoints = detect_pitch_keypoints(frame, model)
    # Radar inset uses homography_from_keypoints_radar at draw time (see _attach_radar).
    if radar_transformer is None and keypoints is not None:
        radar_transformer = homography_from_keypoints_radar(keypoints, confidence=0.5)

    return ExplainContext(
        frame_idx=frame_idx,
        frame=frame,
        dets=dets,
        carrier=carrier,
        options=tuple(options),
        top3=tuple(options[:3]),
        weights=w,
        transformer=transformer,
        radar_transformer=radar_transformer,
        keypoints=keypoints,
        metric=metric,
        teammate_count=int(teammates.sum()),
        attack_dir=attack_dir,
        carrier_motion_dir=motion_dir,
    )


def write_explain_frames(
    out_dir: str | Path,
    steps: dict[str, np.ndarray],
    *,
    grid: bool = True,
    timeline: bool = True,
) -> list[Path]:
    """Write step PNGs and optional composites to ``out_dir``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for stem, image in steps.items():
        path = out_dir / f"pass_lane_{stem}.png"
        cv2.imwrite(str(path), image)
        written.append(path)
    if grid:
        grid_path = out_dir / "pass_lane_grid_2x2.png"
        cv2.imwrite(str(grid_path), render_explain_grid(steps))
        written.append(grid_path)
    if timeline:
        timeline_path = out_dir / "pass_lane_timeline.png"
        cv2.imwrite(str(timeline_path), render_explain_timeline(steps))
        written.append(timeline_path)
    return written
