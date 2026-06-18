"""Decomposed metric visuals for conference pass-lane explain (OPEN / FWD / SPACE / penalties)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np
from world_cup_projects.common.geometry import (
    pass_corridor_polygon,
    point_to_segment_distance_and_t,
    unit,
)
from world_cup_projects.common.pitch import (
    image_to_pitch_cm,
    image_to_pitch_m,
    pitch_circle_polygon_cm,
    pitch_circle_to_image,
    pitch_cm_to_image,
    pitch_polygon_extent_px,
)
from world_cup_projects.common.possession import bbox_center_xy, feet_xy, player_mask
from world_cup_projects.explain.pass_alternatives_visual import (
    ExplainContext,
    _PANEL_BG,
    _RANK_BGR,
    _draw_step_badge,
    _ui_scale,
)
from world_cup_projects.pass_alternatives.lane_visual import (
    _corridor_image_polygon,
    pass_line_label_xy,
    pitch_cm_to_radar_px,
)
from world_cup_projects.pass_alternatives.pass_options import (
    PassOption,
    ScoreBreakdown,
    _lane_width_for_pass,
    _teammate_lane_width,
)

_OPEN_BGR = _RANK_BGR[0]
_FWD_BGR = (0, 200, 255)
_SPACE_BGR = (200, 140, 255)
_RIVAL_BGR = (60, 80, 255)
_TEAMMATE_BGR = (0, 210, 255)
_PENALTY_BGR = (70, 70, 220)
_MOTION_BGR = (180, 255, 120)
_ANGLE_ARC_BGR = (100, 220, 255)
_ATTACK_AXIS_BGR = (190, 190, 200)
_CENTROID_LINE_BGR = (140, 140, 155)


@dataclass(frozen=True)
class AttackAxisGeometry:
    """Team centroids on the pitch used to derive the attack direction."""

    own_centroid_m: np.ndarray
    opp_centroid_m: np.ndarray
    attack_dir: np.ndarray
    axis_end_m: np.ndarray
    n_own: int
    n_opp: int


@dataclass(frozen=True)
class AttackAxisRadarDraw:
    own_cm: np.ndarray
    opp_cm: np.ndarray
    axis_end_cm: np.ndarray


@dataclass(frozen=True)
class RadarMetricDraw:
    """Points in sports-radar pitch cm — same frame as ``draw_pass_lanes_on_radar``."""

    rank_color: tuple[int, int, int]
    carrier_cm: np.ndarray
    receiver_cm: np.ndarray
    openness_rival_cm: np.ndarray | None
    openness_proj_cm: np.ndarray | None
    openness_m: float
    opponent_openness_m: float
    teammate_openness_m: float
    teammate_openness_cm: np.ndarray | None
    teammate_openness_proj_cm: np.ndarray | None
    teammate_in_corridor: bool
    teammate_lane_width_m: float | None
    space_rival_cm: np.ndarray | None
    space_m: float
    space_ref_m: float
    forward_gain_m: float
    forward_tip_cm: np.ndarray
    attack_end_cm: np.ndarray


@dataclass(frozen=True)
class LaneMetricGeometry:
    """Pitch-space geometry for one lane's three score terms."""

    carrier_m: np.ndarray
    receiver_m: np.ndarray
    attack_dir: np.ndarray
    lane_width_m: float | None
    openness_m: float
    opponent_openness_m: float
    teammate_openness_m: float
    openness_rival_idx: int | None
    openness_rival_m: np.ndarray | None
    openness_proj_m: np.ndarray | None
    teammate_openness_idx: int | None
    teammate_openness_m_pt: np.ndarray | None
    teammate_openness_proj_m: np.ndarray | None
    teammate_lane_width_m: float | None
    teammate_in_corridor: bool
    forward_gain_m: float
    forward_tip_m: np.ndarray
    attack_display_m: np.ndarray
    space_m: float
    space_ref_m: float
    space_rival_idx: int | None
    space_rival_m: np.ndarray | None


@dataclass(frozen=True)
class PenaltyGeometry:
    teammate_idx: int | None
    teammate_m: np.ndarray | None
    teammate_proj_m: np.ndarray | None
    teammate_openness_m: float
    motion_dir_m: np.ndarray | None
    pass_align: float
    backward_tip_m: np.ndarray | None


@dataclass(frozen=True)
class PenaltyRadarDraw:
    rank_color: tuple[int, int, int]
    carrier_cm: np.ndarray
    receiver_cm: np.ndarray
    teammate_cm: np.ndarray | None
    teammate_proj_cm: np.ndarray | None
    motion_end_cm: np.ndarray | None
    backward_tip_cm: np.ndarray | None
    attack_end_cm: np.ndarray
    pass_align: float | None = None


def _m_per_px(ctx: ExplainContext, option: PassOption) -> float:
    feet = feet_xy(ctx.dets)
    px = float(np.linalg.norm(feet[option.receiver_index] - feet[ctx.carrier.index]))
    if px < 1e-3 or option.length < 1e-6:
        return 0.01
    return option.length / px


def _m_to_img(
    pt_m: np.ndarray, transformer, *, m_per_px: float | None = None
) -> tuple[int, int] | None:
    if transformer is None:
        return None
    cm = np.asarray(pt_m, dtype=np.float64).reshape(1, 2) * 100.0
    img = pitch_cm_to_image(cm, transformer)
    if img is None:
        return None
    return int(img[0, 0]), int(img[0, 1])


def _proj_on_segment_cm(
    point_cm: np.ndarray, a_cm: np.ndarray, b_cm: np.ndarray
) -> np.ndarray:
    ab = b_cm.astype(np.float64) - a_cm.astype(np.float64)
    denom = float(ab @ ab)
    if denom < 1e-9:
        return a_cm.copy()
    t = float(np.clip(((point_cm.astype(np.float64) - a_cm) @ ab) / denom, 0.0, 1.0))
    return a_cm + t * ab


def compute_attack_axis_geometry(
    ctx: ExplainContext,
    transformer=None,
) -> AttackAxisGeometry | None:
    """Rebuild own/opp centroids and the unit attack vector used by scoring."""
    t = transformer if transformer is not None else ctx.transformer
    if t is None:
        return None
    feet = feet_xy(ctx.dets)
    pitch_m = image_to_pitch_m(feet, t)
    if pitch_m is None:
        return None
    pmask = player_mask(ctx.dets)
    teams = ctx.dets.data["team"]
    carrier_team = ctx.carrier.team
    own_mask = pmask & (teams == carrier_team)
    opp_mask = pmask & (teams != carrier_team)
    if not own_mask.any() or not opp_mask.any():
        return None
    own = pitch_m[own_mask]
    opp = pitch_m[opp_mask]
    own_c = own.mean(axis=0)
    opp_c = opp.mean(axis=0)
    delta = opp_c - own_c
    attack = unit(delta) if float(np.linalg.norm(delta)) > 1e-6 else np.array(
        [1.0, 0.0], dtype=np.float64
    )
    span_m = float(np.linalg.norm(delta))
    axis_end = own_c + delta * 0.85 if span_m > 1e-6 else own_c + attack * 20.0
    return AttackAxisGeometry(
        own_centroid_m=own_c,
        opp_centroid_m=opp_c,
        attack_dir=attack,
        axis_end_m=axis_end,
        n_own=int(own_mask.sum()),
        n_opp=int(opp_mask.sum()),
    )


def build_attack_axis_radar_draw(
    axis: AttackAxisGeometry, pitch_cm: np.ndarray, dets, carrier_team: int
) -> AttackAxisRadarDraw:
    pmask = player_mask(dets)
    teams = dets.data["team"]
    own_cm = pitch_cm[pmask & (teams == carrier_team)].mean(axis=0)
    opp_cm = pitch_cm[pmask & (teams != carrier_team)].mean(axis=0)
    delta = opp_cm.astype(np.float64) - own_cm.astype(np.float64)
    norm = float(np.linalg.norm(delta))
    axis_end_cm = own_cm + delta * 0.85 if norm > 1e-6 else own_cm.copy()
    return AttackAxisRadarDraw(own_cm=own_cm, opp_cm=opp_cm, axis_end_cm=axis_end_cm)


def _attack_unit_from_pitch_cm(
    pitch_cm: np.ndarray, dets, carrier_team: int
) -> np.ndarray:
    pmask = player_mask(dets)
    teams = dets.data["team"]
    mask = pmask & np.isin(teams, (0, 1))
    if not mask.any():
        return np.array([1.0, 0.0], dtype=np.float64)
    pts_cm = pitch_cm[mask]
    tms = teams[mask]
    own = pts_cm[tms == carrier_team]
    opp = pts_cm[tms != carrier_team]
    if len(own) == 0 or len(opp) == 0:
        return np.array([1.0, 0.0], dtype=np.float64)
    return unit((opp.mean(axis=0) - own.mean(axis=0)) / 100.0)


def _radar_pt(point_cm: np.ndarray) -> tuple[int, int]:
    return tuple(pitch_cm_to_radar_px(point_cm.reshape(1, 2))[0])


def _carrier_motion_end_cm(
    ctx: ExplainContext,
    carrier_cm: np.ndarray,
    transformer,
    *,
    span_m: float = 10.0,
) -> np.ndarray | None:
    """Carrier run tip in the same pitch-cm frame as the sports-radar minimap."""
    if transformer is None:
        return None
    from world_cup_projects.common.tracking_facing import carrier_kalman_direction

    motion_m = carrier_kalman_direction(
        ctx.dets, ctx.carrier.index, transformer=transformer
    )
    if motion_m is None:
        return None
    return carrier_cm + motion_m * (span_m * 100.0)


def build_radar_metric_draw(
    ctx: ExplainContext,
    option: PassOption,
    geom: LaneMetricGeometry,
    pitch_cm: np.ndarray,
    *,
    rank_color: tuple[int, int, int],
) -> RadarMetricDraw:
    """Map metric geometry into the same pitch-cm frame used by the minimap."""
    carrier_cm = pitch_cm[ctx.carrier.index]
    receiver_cm = pitch_cm[option.receiver_index]

    openness_rival_cm = openness_proj_cm = None
    if geom.openness_rival_idx is not None:
        openness_rival_cm = pitch_cm[geom.openness_rival_idx]
        openness_proj_cm = _proj_on_segment_cm(
            openness_rival_cm, carrier_cm, receiver_cm
        )

    teammate_openness_cm = teammate_openness_proj_cm = None
    if geom.teammate_openness_idx is not None:
        teammate_openness_cm = pitch_cm[geom.teammate_openness_idx]
        teammate_openness_proj_cm = _proj_on_segment_cm(
            teammate_openness_cm, carrier_cm, receiver_cm
        )

    space_rival_cm = None
    if geom.space_rival_idx is not None:
        space_rival_cm = pitch_cm[geom.space_rival_idx]

    attack_m = _attack_unit_from_pitch_cm(pitch_cm, ctx.dets, ctx.carrier.team)
    gain_cm = max(geom.forward_gain_m, 0.0) * 100.0
    forward_tip_cm = carrier_cm + attack_m * gain_cm
    attack_end_cm = carrier_cm + attack_m * (28.0 * 100.0)

    return RadarMetricDraw(
        rank_color=rank_color,
        carrier_cm=carrier_cm,
        receiver_cm=receiver_cm,
        openness_rival_cm=openness_rival_cm,
        openness_proj_cm=openness_proj_cm,
        openness_m=geom.openness_m,
        opponent_openness_m=geom.opponent_openness_m,
        teammate_openness_m=geom.teammate_openness_m,
        teammate_openness_cm=teammate_openness_cm,
        teammate_openness_proj_cm=teammate_openness_proj_cm,
        teammate_in_corridor=geom.teammate_in_corridor,
        teammate_lane_width_m=geom.teammate_lane_width_m,
        space_rival_cm=space_rival_cm,
        space_m=geom.space_m,
        space_ref_m=geom.space_ref_m,
        forward_gain_m=geom.forward_gain_m,
        forward_tip_cm=forward_tip_cm,
        attack_end_cm=attack_end_cm,
    )


def compute_lane_metric_geometry(
    ctx: ExplainContext,
    option: PassOption,
) -> LaneMetricGeometry | None:
    """Rebuild rival indices + projection points used by scoring."""
    if ctx.attack_dir is None:
        return None
    feet = feet_xy(ctx.dets)
    pitch_m = image_to_pitch_m(feet, ctx.transformer)
    if pitch_m is None:
        return None

    pmask = player_mask(ctx.dets)
    teams = ctx.dets.data["team"]
    opponents = pmask & (teams != ctx.carrier.team)
    opp_global = np.flatnonzero(opponents)
    carrier_m = pitch_m[ctx.carrier.index]
    receiver_m = pitch_m[option.receiver_index]
    attack = unit(ctx.attack_dir.astype(np.float64))

    pass_len_px = float(np.linalg.norm(feet[option.receiver_index] - feet[ctx.carrier.index]))
    lane_w = _lane_width_for_pass(
        ctx.weights, pass_length=option.length, pass_length_px=pass_len_px
    )

    openness_rival_idx: int | None = None
    openness_rival_m: np.ndarray | None = None
    openness_proj_m: np.ndarray | None = None
    openness_m = option.openness

    if len(opp_global):
        body = image_to_pitch_m(bbox_center_xy(ctx.dets), ctx.transformer)
        opp_feet = pitch_m[opp_global]
        opp_body = body[opp_global] if body is not None else opp_feet
        d_f, t_f = point_to_segment_distance_and_t(opp_feet, carrier_m, receiver_m)
        d_b, t_b = point_to_segment_distance_and_t(opp_body, carrier_m, receiver_m)
        on_seg = ((t_f >= ctx.weights.lane_t_min) & (t_f <= ctx.weights.lane_t_max)) | (
            (t_b >= ctx.weights.lane_t_min) & (t_b <= ctx.weights.lane_t_max)
        )
        per_player = np.where(on_seg, np.minimum(d_f, d_b), np.inf)
        if lane_w is not None and lane_w > 0:
            in_lane = on_seg & (per_player <= lane_w / 2.0)
        else:
            in_lane = on_seg
        if in_lane.any():
            local_i = int(np.argmin(per_player[in_lane]))
            global_i = int(opp_global[np.flatnonzero(in_lane)[local_i]])
            openness_rival_idx = global_i
            openness_rival_m = pitch_m[global_i]
            ab = receiver_m - carrier_m
            denom = float(ab @ ab)
            if denom > 1e-9:
                t = float(((openness_rival_m - carrier_m) @ ab) / denom)
                t = float(np.clip(t, 0.0, 1.0))
                openness_proj_m = carrier_m + t * ab

    teammate_openness_idx: int | None = None
    teammate_openness_m_pt: np.ndarray | None = None
    teammate_openness_proj_m: np.ndarray | None = None
    teammate_in_corridor = False
    tm_lane_w = _teammate_lane_width(
        ctx.weights, pass_length=option.length, pass_length_px=pass_len_px
    )
    blockers = pmask & (teams == ctx.carrier.team)
    blockers[ctx.carrier.index] = False
    blockers[option.receiver_index] = False
    team_global = np.flatnonzero(blockers)
    if len(team_global):
        team_xy = pitch_m[team_global]
        tm_dists, tm_t = point_to_segment_distance_and_t(
            team_xy, carrier_m, receiver_m
        )
        local_i = int(np.argmin(tm_dists))
        on_segment = (
            ctx.weights.lane_t_min <= tm_t[local_i] <= ctx.weights.lane_t_max
        )
        in_width = tm_lane_w is None or tm_lane_w <= 0 or tm_dists[local_i] <= tm_lane_w / 2.0
        if on_segment and in_width:
            global_i = int(team_global[local_i])
            teammate_openness_idx = global_i
            teammate_openness_m_pt = pitch_m[global_i]
            teammate_openness_proj_m = _proj_on_segment_cm(
                teammate_openness_m_pt * 100.0,
                carrier_m * 100.0,
                receiver_m * 100.0,
            ) / 100.0
            teammate_in_corridor = True

    space_rival_idx: int | None = None
    space_rival_m: np.ndarray | None = None
    space_m = option.receiver_space
    if len(opp_global):
        dists = np.linalg.norm(pitch_m[opp_global] - receiver_m, axis=1)
        local_i = int(np.argmin(dists))
        space_rival_idx = int(opp_global[local_i])
        space_rival_m = pitch_m[space_rival_idx]

    fwd = float(option.forward_gain)
    forward_tip_m = carrier_m + attack * max(fwd, 0.0)
    attack_display_m = carrier_m + attack * min(ctx.weights.forward_ref, 28.0)

    return LaneMetricGeometry(
        carrier_m=carrier_m,
        receiver_m=receiver_m,
        attack_dir=attack,
        lane_width_m=lane_w,
        openness_m=openness_m,
        opponent_openness_m=float(option.opponent_openness),
        teammate_openness_m=float(option.teammate_openness),
        openness_rival_idx=openness_rival_idx,
        openness_rival_m=openness_rival_m,
        openness_proj_m=openness_proj_m,
        teammate_openness_idx=teammate_openness_idx,
        teammate_openness_m_pt=teammate_openness_m_pt,
        teammate_openness_proj_m=teammate_openness_proj_m,
        teammate_lane_width_m=tm_lane_w,
        teammate_in_corridor=teammate_in_corridor,
        forward_gain_m=fwd,
        forward_tip_m=forward_tip_m,
        attack_display_m=attack_display_m,
        space_m=space_m,
        space_ref_m=ctx.weights.space_ref,
        space_rival_idx=space_rival_idx,
        space_rival_m=space_rival_m,
    )


def compute_penalty_geometry(
    ctx: ExplainContext,
    option: PassOption,
    breakdown: ScoreBreakdown,
) -> PenaltyGeometry | None:
    if ctx.attack_dir is None:
        return None
    feet = feet_xy(ctx.dets)
    pitch_m = image_to_pitch_m(feet, ctx.transformer)
    if pitch_m is None:
        return None

    carrier_m = pitch_m[ctx.carrier.index]
    receiver_m = pitch_m[option.receiver_index]
    attack = unit(ctx.attack_dir.astype(np.float64))
    pass_len_px = float(np.linalg.norm(feet[option.receiver_index] - feet[ctx.carrier.index]))

    teammate_idx: int | None = None
    teammate_m: np.ndarray | None = None
    teammate_proj_m: np.ndarray | None = None
    if breakdown.teammate_penalty > 0.005:
        pmask = player_mask(ctx.dets)
        teams = ctx.dets.data["team"]
        blockers = pmask & (teams == ctx.carrier.team)
        blockers[ctx.carrier.index] = False
        blockers[option.receiver_index] = False
        team_global = np.flatnonzero(blockers)
        if len(team_global):
            team_xy = pitch_m[team_global]
            tm_lane_w = _teammate_lane_width(
                ctx.weights, pass_length=option.length, pass_length_px=pass_len_px
            )
            dists, t = point_to_segment_distance_and_t(team_xy, carrier_m, receiver_m)
            in_corridor = (t >= ctx.weights.lane_t_min) & (t <= ctx.weights.lane_t_max)
            if tm_lane_w is not None and tm_lane_w > 0:
                in_corridor &= dists <= tm_lane_w / 2.0
            if in_corridor.any():
                local_i = int(np.argmin(dists[in_corridor]))
                global_i = int(team_global[np.flatnonzero(in_corridor)[local_i]])
                teammate_idx = global_i
                teammate_m = pitch_m[global_i]
                teammate_proj_m = _proj_on_segment_cm(
                    teammate_m * 100.0, carrier_m * 100.0, receiver_m * 100.0
                ) / 100.0

    motion_dir_m = None
    pass_align = float(option.motion_alignment)
    if ctx.carrier_motion_dir is not None:
        motion_dir_m = unit(ctx.carrier_motion_dir.astype(np.float64))

    backward_tip_m = None
    if breakdown.backward_attack_penalty > 0.005 and option.forward_gain < 0:
        backward_tip_m = carrier_m + attack * float(option.forward_gain)

    return PenaltyGeometry(
        teammate_idx=teammate_idx,
        teammate_m=teammate_m,
        teammate_proj_m=teammate_proj_m,
        teammate_openness_m=float(option.teammate_openness),
        motion_dir_m=motion_dir_m,
        pass_align=pass_align,
        backward_tip_m=backward_tip_m,
    )


def build_penalty_radar_draw(
    ctx: ExplainContext,
    option: PassOption,
    penalty: PenaltyGeometry,
    pitch_cm: np.ndarray,
    *,
    rank_color: tuple[int, int, int],
    radar_transformer=None,
) -> PenaltyRadarDraw:
    carrier_cm = pitch_cm[ctx.carrier.index]
    receiver_cm = pitch_cm[option.receiver_index]
    attack_m = _attack_unit_from_pitch_cm(pitch_cm, ctx.dets, ctx.carrier.team)
    attack_end_cm = carrier_cm + attack_m * (28.0 * 100.0)
    teammate_cm = teammate_proj_cm = None
    if penalty.teammate_idx is not None:
        teammate_cm = pitch_cm[penalty.teammate_idx]
        teammate_proj_cm = _proj_on_segment_cm(teammate_cm, carrier_cm, receiver_cm)
    motion_end_cm = None
    if penalty.motion_dir_m is not None:
        motion_end_cm = _carrier_motion_end_cm(
            ctx, carrier_cm, radar_transformer, span_m=10.0
        )
    backward_tip_cm = None
    if penalty.backward_tip_m is not None:
        backward_tip_cm = carrier_cm + attack_m * (float(option.forward_gain) * 100.0)
    return PenaltyRadarDraw(
        rank_color=rank_color,
        carrier_cm=carrier_cm,
        receiver_cm=receiver_cm,
        teammate_cm=teammate_cm,
        teammate_proj_cm=teammate_proj_cm,
        motion_end_cm=motion_end_cm,
        backward_tip_cm=backward_tip_cm,
        attack_end_cm=attack_end_cm,
        pass_align=penalty.pass_align,
    )


def draw_focus_zone(
    dimmed: np.ndarray,
    original: np.ndarray,
    centers: list[tuple[int, int]],
    *,
    radius: int = 130,
    strength: float = 0.68,
) -> np.ndarray:
    if not centers:
        return dimmed
    h, w = dimmed.shape[:2]
    mask = np.zeros((h, w), dtype=np.float32)
    for cx, cy in centers:
        cv2.circle(mask, (int(cx), int(cy)), radius, 1.0, -1, cv2.LINE_AA)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=max(radius * 0.36, 1.0))
    mask = (mask[..., None] * strength).astype(np.float32)
    out = dimmed.astype(np.float32) * (1.0 - mask) + original.astype(np.float32) * mask
    return np.clip(out, 0, 255).astype(np.uint8)


def _label_point(
    img: np.ndarray,
    pt: tuple[int, int],
    text: str,
    color: tuple[int, int, int],
    *,
    offset: tuple[int, int] = (0, -22),
    font_scale: float = 0.46,
) -> None:
    from world_cup_projects.common.visual import draw_text_shadow

    draw_text_shadow(
        img,
        text,
        (pt[0] + offset[0], pt[1] + offset[1]),
        font_scale=font_scale,
        color_bgr=color,
        thickness=2,
    )


def _label_segment(
    img: np.ndarray,
    p0: tuple[int, int],
    p1: tuple[int, int],
    text: str,
    color: tuple[int, int, int],
    *,
    along: float = 0.45,
    offset_px: int = 16,
    font_scale: float = 0.46,
) -> None:
    """Short label beside a line/arrow on the broadcast frame."""
    from world_cup_projects.common.visual import draw_text_shadow

    lx, ly = pass_line_label_xy(p0, p1, along=along, offset_px=offset_px)
    draw_text_shadow(
        img,
        text,
        (lx, ly),
        font_scale=font_scale,
        color_bgr=color,
        thickness=2,
    )


def _draw_metric_line(
    img: np.ndarray,
    p0: tuple[int, int],
    p1: tuple[int, int],
    color: tuple[int, int, int],
    *,
    thickness: int = 2,
) -> None:
    cv2.line(img, p0, p1, color, thickness, cv2.LINE_AA)


def _draw_metric_arrow(
    img: np.ndarray,
    p0: tuple[int, int],
    p1: tuple[int, int],
    color: tuple[int, int, int],
    *,
    thickness: int = 2,
    head_len: float = 8.0,
    head_width: float = 4.5,
) -> None:
    """Thin shaft + small filled head — reads clearly at broadcast and radar scale."""
    x0, y0 = p0
    x1, y1 = p1
    dx, dy = float(x1 - x0), float(y1 - y0)
    length = float(np.hypot(dx, dy))
    if length < 5.0:
        cv2.line(img, p0, p1, color, thickness, cv2.LINE_AA)
        return
    ux, uy = dx / length, dy / length
    px, py = -uy, ux
    base_x = x1 - ux * head_len
    base_y = y1 - uy * head_len
    cv2.line(img, p0, (int(base_x), int(base_y)), color, thickness, cv2.LINE_AA)
    tip = np.array(
        [
            [x1, y1],
            [base_x + px * head_width, base_y + py * head_width],
            [base_x - px * head_width, base_y - py * head_width],
        ],
        dtype=np.int32,
    )
    cv2.fillConvexPoly(img, tip, color, lineType=cv2.LINE_AA)


def _draw_metric_chevron(
    img: np.ndarray,
    p0: tuple[int, int],
    p1: tuple[int, int],
    color: tuple[int, int, int],
    *,
    thickness: int = 2,
    head_len: float = 8.0,
    wing: float = 0.52,
) -> None:
    """Backward-compatible alias — prefer the filled-head metric arrow."""
    _draw_metric_arrow(
        img,
        p0,
        p1,
        color,
        thickness=thickness,
        head_len=head_len,
        head_width=max(3.5, head_len * wing),
    )


def _label_radar_segment(
    radar: np.ndarray,
    p0_cm: np.ndarray,
    p1_cm: np.ndarray,
    text: str,
    color: tuple[int, int, int],
    *,
    along: float = 0.45,
    offset_px: int = 10,
) -> None:
    p0 = _radar_pt(p0_cm)
    p1 = _radar_pt(p1_cm)
    lx, ly = pass_line_label_xy(p0, p1, along=along, offset_px=offset_px)
    from world_cup_projects.common.visual import draw_text_shadow

    draw_text_shadow(
        radar, text, (lx, ly), font_scale=0.34, color_bgr=color, thickness=1,
    )


def _draw_projected_pitch_circle(
    img: np.ndarray,
    center_m: np.ndarray,
    radius_m: float,
    transformer,
    color: tuple[int, int, int],
    *,
    thickness: int = 2,
    dashed: bool = False,
    gap: int = 12,
) -> np.ndarray | None:
    """Draw a pitch-space circle; homography maps it to a perspective ellipse."""
    poly = pitch_circle_to_image(center_m, radius_m, transformer)
    if poly is None or len(poly) < 3:
        return None
    if dashed:
        for i in range(len(poly)):
            p0 = tuple(int(v) for v in poly[i])
            p1 = tuple(int(v) for v in poly[(i + 1) % len(poly)])
            _dashed_line(img, p0, p1, color, thickness=thickness, gap=gap)
    else:
        cv2.polylines(
            img, [poly], isClosed=True, color=color, thickness=thickness, lineType=cv2.LINE_AA
        )
    return poly


def _draw_projected_radar_circle(
    radar: np.ndarray,
    center_cm: np.ndarray,
    radius_m: float,
    color: tuple[int, int, int],
    *,
    thickness: int = 1,
    dashed: bool = False,
    gap: int = 6,
) -> np.ndarray | None:
    """Project a pitch-space circle onto the sports-radar minimap."""
    poly_cm = pitch_circle_polygon_cm(center_cm / 100.0, radius_m)
    poly_px = pitch_cm_to_radar_px(poly_cm).astype(np.int32)
    if len(poly_px) < 3:
        return None
    if dashed:
        for i in range(len(poly_px)):
            p0 = tuple(int(v) for v in poly_px[i])
            p1 = tuple(int(v) for v in poly_px[(i + 1) % len(poly_px)])
            _dashed_line(radar, p0, p1, color, thickness=thickness, gap=gap)
    else:
        cv2.polylines(
            radar, [poly_px], isClosed=True, color=color, thickness=thickness, lineType=cv2.LINE_AA
        )
    return poly_px


def _dashed_line(
    img: np.ndarray,
    p0: tuple[int, int],
    p1: tuple[int, int],
    color: tuple[int, int, int],
    *,
    thickness: int = 2,
    gap: int = 10,
) -> None:
    x0, y0 = p0
    x1, y1 = p1
    dist = int(np.hypot(x1 - x0, y1 - y0))
    if dist < 1:
        return
    for i in range(0, dist, gap * 2):
        t0 = i / dist
        t1 = min(1.0, (i + gap) / dist)
        a = (int(x0 + (x1 - x0) * t0), int(y0 + (y1 - y0) * t0))
        b = (int(x0 + (x1 - x0) * t1), int(y0 + (y1 - y0) * t1))
        cv2.line(img, a, b, color, thickness, cv2.LINE_AA)


def _align_angle_deg(cos_align: float) -> float:
    """Angle between pass and carrier run from pitch-space cosine alignment."""
    return float(np.degrees(np.arccos(np.clip(cos_align, -1.0, 1.0))))


def _draw_angle_arc(
    img: np.ndarray,
    vertex: tuple[int, int],
    pt_a: tuple[int, int],
    pt_b: tuple[int, int],
    *,
    color: tuple[int, int, int],
    radius_px: float | None = None,
    thickness: int = 2,
    num_pts: int = 28,
) -> None:
    """Draw a circular arc at ``vertex`` between rays toward ``pt_a`` and ``pt_b``."""
    v0 = np.array(vertex, dtype=np.float64)
    va = np.array(pt_a, dtype=np.float64) - v0
    vb = np.array(pt_b, dtype=np.float64) - v0
    na, nb = float(np.linalg.norm(va)), float(np.linalg.norm(vb))
    if na < 1e-6 or nb < 1e-6:
        return
    va = va / na
    vb = vb / nb
    dot = float(np.clip(va @ vb, -1.0, 1.0))
    sweep = float(np.arccos(dot))
    if sweep < np.radians(3.0):
        return
    cross = float(va[0] * vb[1] - va[1] * vb[0])
    if cross < 0.0:
        sweep = -sweep
    if abs(sweep) > np.pi:
        sweep -= np.sign(sweep) * 2.0 * np.pi

    if radius_px is None:
        radius_px = min(na, nb) * 0.32
    radius_px = float(np.clip(radius_px, 16.0, 52.0))

    arc_pts: list[tuple[int, int]] = []
    for t in np.linspace(0.0, 1.0, num_pts):
        theta = sweep * float(t)
        c, s = np.cos(theta), np.sin(theta)
        vr = np.array(
            (c * va[0] - s * va[1], s * va[0] + c * va[1]),
            dtype=np.float64,
        )
        pt = v0 + vr * radius_px
        arc_pts.append((int(round(pt[0])), int(round(pt[1]))))
    if len(arc_pts) >= 2:
        cv2.polylines(img, [np.array(arc_pts, dtype=np.int32)], False, color, thickness, cv2.LINE_AA)


def _draw_step_panel(
    canvas: np.ndarray,
    *,
    label: str,
    justification: str,
    color: tuple[int, int, int],
    layout: Literal["talk", "social"],
    score: float | None = None,
    negative: bool = False,
) -> None:
    """Compact upper-left panel: title, one-line justification, optional score."""
    from world_cup_projects.common.visual import draw_text_shadow

    scale = _ui_scale(layout)
    pad = 14
    badge_h = int(46 + 6 * scale)
    px, py = pad, pad + badge_h + 10
    pw = int(min(canvas.shape[1] * 0.27, 280))
    ph = int(72 * scale)
    overlay = canvas.copy()
    cv2.rectangle(overlay, (px, py), (px + pw, py + ph), _PANEL_BG, -1)
    cv2.rectangle(overlay, (px, py), (px + pw, py + ph), (55, 55, 65), 1)
    canvas[:] = cv2.addWeighted(overlay, 0.92, canvas, 0.08, 0)
    draw_text_shadow(
        canvas,
        label,
        (px + 12, py + int(20 * scale)),
        font_scale=0.46 * scale,
        color_bgr=color,
        thickness=2,
    )
    draw_text_shadow(
        canvas,
        justification,
        (px + 12, py + int(40 * scale)),
        font_scale=0.34 * scale,
        color_bgr=(165, 165, 175),
        thickness=1,
    )
    if score is not None:
        score_color = _PENALTY_BGR if negative else color
        score_text = f"-{abs(score):.2f}" if negative else f"+{abs(score):.2f}"
        font_scale = 0.54 * scale
        thickness = 2
        (tw, _), _ = cv2.getTextSize(
            score_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        draw_text_shadow(
            canvas,
            score_text,
            (px + pw - 12 - tw, py + int(62 * scale)),
            font_scale=font_scale,
            color_bgr=score_color,
            thickness=thickness,
        )


def _draw_attack_axis_panel(
    canvas: np.ndarray,
    _axis: AttackAxisGeometry,
    *,
    layout: Literal["talk", "social"],
) -> None:
    _draw_step_panel(
        canvas,
        label="ATTACK AXIS",
        justification="own centroid → opponent centroid",
        color=_ATTACK_AXIS_BGR,
        layout=layout,
    )


def _draw_centroid_marker(
    img: np.ndarray,
    pt: tuple[int, int],
    color: tuple[int, int, int],
    *,
    radius: int = 20,
) -> None:
    cv2.circle(img, pt, radius + 3, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.circle(img, pt, radius, color, -1, cv2.LINE_AA)
    cv2.circle(img, pt, radius, (255, 255, 255), 1, cv2.LINE_AA)


def draw_attack_axis_on_frame(
    frame: np.ndarray,
    original: np.ndarray,
    axis: AttackAxisGeometry,
    transformer,
    *,
    own_color: tuple[int, int, int],
    opp_color: tuple[int, int, int],
) -> np.ndarray:
    own = _m_to_img(axis.own_centroid_m, transformer)
    opp = _m_to_img(axis.opp_centroid_m, transformer)
    axis_end = _m_to_img(axis.axis_end_m, transformer)
    if own is None or opp is None or axis_end is None:
        return frame
    centers = [own, opp, axis_end]
    _dashed_line(frame, own, opp, _CENTROID_LINE_BGR, thickness=2, gap=14)
    _draw_metric_arrow(frame, own, axis_end, _ATTACK_AXIS_BGR, thickness=2, head_len=9.0)
    _draw_centroid_marker(frame, own, own_color)
    _draw_centroid_marker(frame, opp, opp_color)
    out = draw_focus_zone(frame, original, centers, radius=200, strength=0.70)
    _label_segment(
        out, own, opp, "TEAM CENTROIDS", _CENTROID_LINE_BGR, along=0.5, offset_px=-18
    )
    _label_segment(
        out, own, axis_end, "ATTACK AXIS", _ATTACK_AXIS_BGR, along=0.62, offset_px=18
    )
    _label_point(out, own, "OWN TEAM", own_color, offset=(0, -30))
    _label_point(out, opp, "OPP TEAM", opp_color, offset=(0, -30))
    return out


def draw_attack_axis_on_radar(
    radar: np.ndarray,
    draw: AttackAxisRadarDraw,
    *,
    own_color: tuple[int, int, int],
    opp_color: tuple[int, int, int],
) -> np.ndarray:
    out = radar.copy()
    own_pt = _radar_pt(draw.own_cm)
    opp_pt = _radar_pt(draw.opp_cm)
    axis_pt = _radar_pt(draw.axis_end_cm)
    _dashed_line(out, own_pt, opp_pt, _CENTROID_LINE_BGR, thickness=1, gap=8)
    _draw_metric_arrow(
        out, own_pt, axis_pt, _ATTACK_AXIS_BGR, thickness=1, head_len=6.0, head_width=3.5
    )
    cv2.circle(out, own_pt, 8, own_color, -1, cv2.LINE_AA)
    cv2.circle(out, opp_pt, 8, opp_color, -1, cv2.LINE_AA)
    _label_radar_segment(out, draw.own_cm, draw.opp_cm, "CENTROIDS", _CENTROID_LINE_BGR)
    _label_radar_segment(out, draw.own_cm, draw.axis_end_cm, "ATTACK", _ATTACK_AXIS_BGR)
    return out


def _teammate_corridor_polygon_cm(
    geom: LaneMetricGeometry,
    *,
    t_min: float = 0.0,
    t_max: float = 1.0,
) -> np.ndarray | None:
    if geom.teammate_lane_width_m is None or geom.teammate_lane_width_m <= 0:
        return None
    return pass_corridor_polygon(
        geom.carrier_m * 100.0,
        geom.receiver_m * 100.0,
        geom.teammate_lane_width_m / 2.0 * 100.0,
        t_min=t_min,
        t_max=t_max,
    )


def _draw_teammate_corridor_outline(
    frame: np.ndarray,
    geom: LaneMetricGeometry,
    transformer,
    *,
    color: tuple[int, int, int],
) -> None:
    poly_cm = _teammate_corridor_polygon_cm(geom)
    if poly_cm is None:
        return
    poly = _corridor_image_polygon(poly_cm, transformer, frame.shape)
    if poly is None:
        return
    for i in range(4):
        p0 = tuple(poly[i])
        p1 = tuple(poly[(i + 1) % 4])
        _dashed_line(frame, p0, p1, color, thickness=2, gap=10)


def _draw_teammate_corridor_on_radar(
    radar: np.ndarray,
    draw: RadarMetricDraw,
    *,
    teammate_lane_width_m: float | None,
) -> None:
    if teammate_lane_width_m is None or teammate_lane_width_m <= 0:
        return
    poly_cm = pass_corridor_polygon(
        draw.carrier_cm,
        draw.receiver_cm,
        teammate_lane_width_m / 2.0 * 100.0,
    )
    poly = pitch_cm_to_radar_px(poly_cm)
    for i in range(4):
        p0 = tuple(poly[i])
        p1 = tuple(poly[(i + 1) % 4])
        _dashed_line(radar, p0, p1, draw.rank_color, thickness=1, gap=6)


def draw_openness_on_frame(
    frame: np.ndarray,
    original: np.ndarray,
    geom: LaneMetricGeometry,
    transformer,
    *,
    m_per_px: float,
    rank_color: tuple[int, int, int],
    feet_xy: np.ndarray | None = None,
    blocking_teammate_indices: tuple[int, ...] = (),
) -> np.ndarray:
    c = _m_to_img(geom.carrier_m, transformer, m_per_px=m_per_px)
    r = _m_to_img(geom.receiver_m, transformer, m_per_px=m_per_px)
    if c is None or r is None:
        return frame
    _draw_teammate_corridor_outline(frame, geom, transformer, color=rank_color)
    centers = [c, r]
    rival = proj = None
    if geom.openness_rival_m is not None:
        rival = _m_to_img(geom.openness_rival_m, transformer, m_per_px=m_per_px)
        if geom.openness_proj_m is not None:
            proj = _m_to_img(geom.openness_proj_m, transformer, m_per_px=m_per_px)
        if rival is not None:
            centers.append(rival)
            cv2.circle(frame, rival, 16, _RIVAL_BGR, -1, cv2.LINE_AA)
            if proj is not None:
                centers.append(proj)
                _dashed_line(frame, rival, proj, _OPEN_BGR, thickness=2)
                cv2.circle(frame, proj, 8, _OPEN_BGR, -1, cv2.LINE_AA)
    tm = tm_proj = None
    if geom.teammate_openness_m_pt is not None:
        tm = _m_to_img(geom.teammate_openness_m_pt, transformer, m_per_px=m_per_px)
        if geom.teammate_openness_proj_m is not None:
            tm_proj = _m_to_img(
                geom.teammate_openness_proj_m, transformer, m_per_px=m_per_px
            )
        if tm is not None:
            centers.append(tm)
            cv2.circle(frame, tm, 16, _TEAMMATE_BGR, -1, cv2.LINE_AA)
            if tm_proj is not None:
                centers.append(tm_proj)
                _dashed_line(frame, tm, tm_proj, _TEAMMATE_BGR, thickness=2)
                cv2.circle(frame, tm_proj, 8, _TEAMMATE_BGR, -1, cv2.LINE_AA)
    if feet_xy is not None:
        drawn_tm: set[int] = set()
        for idx in blocking_teammate_indices:
            if idx in drawn_tm or idx >= len(feet_xy):
                continue
            drawn_tm.add(idx)
            tx, ty = int(feet_xy[idx, 0]), int(feet_xy[idx, 1])
            centers.append((tx, ty))
            cv2.circle(frame, (tx, ty), 14, _TEAMMATE_BGR, -1, cv2.LINE_AA)
    out = draw_focus_zone(frame, original, centers, radius=150, strength=0.72)
    _draw_metric_line(out, c, r, rank_color, thickness=3)
    return out


def draw_openness_on_radar(radar: np.ndarray, draw: RadarMetricDraw) -> np.ndarray:
    out = radar.copy()
    c_pt = _radar_pt(draw.carrier_cm)
    r_pt = _radar_pt(draw.receiver_cm)
    _draw_teammate_corridor_on_radar(
        out, draw, teammate_lane_width_m=draw.teammate_lane_width_m
    )
    cv2.line(out, c_pt, r_pt, draw.rank_color, 3, cv2.LINE_AA)
    if draw.openness_rival_cm is not None and draw.openness_proj_cm is not None:
        rival_pt = _radar_pt(draw.openness_rival_cm)
        proj_pt = _radar_pt(draw.openness_proj_cm)
        cv2.circle(out, rival_pt, 10, _RIVAL_BGR, -1, cv2.LINE_AA)
        _dashed_line(out, rival_pt, proj_pt, _OPEN_BGR, thickness=2)
        cv2.circle(out, proj_pt, 6, _OPEN_BGR, -1, cv2.LINE_AA)
    if (
        draw.teammate_openness_cm is not None
        and draw.teammate_openness_proj_cm is not None
    ):
        tm_pt = _radar_pt(draw.teammate_openness_cm)
        tm_proj_pt = _radar_pt(draw.teammate_openness_proj_cm)
        cv2.circle(out, tm_pt, 10, _TEAMMATE_BGR, -1, cv2.LINE_AA)
        _dashed_line(out, tm_pt, tm_proj_pt, _TEAMMATE_BGR, thickness=2)
        cv2.circle(out, tm_proj_pt, 6, _TEAMMATE_BGR, -1, cv2.LINE_AA)
    return out




def draw_forward_on_frame(
    frame: np.ndarray,
    original: np.ndarray,
    geom: LaneMetricGeometry,
    transformer,
    *,
    m_per_px: float,
    rank_color: tuple[int, int, int],
) -> np.ndarray:
    c = _m_to_img(geom.carrier_m, transformer, m_per_px=m_per_px)
    r = _m_to_img(geom.receiver_m, transformer, m_per_px=m_per_px)
    attack_end = _m_to_img(geom.attack_display_m, transformer, m_per_px=m_per_px)
    tip = _m_to_img(geom.forward_tip_m, transformer, m_per_px=m_per_px)
    if c is None or r is None or attack_end is None:
        return frame
    centers = [c, r]
    if tip is not None:
        centers.append(tip)
    _dashed_line(frame, c, attack_end, _ATTACK_AXIS_BGR, thickness=2, gap=12)
    _draw_metric_chevron(frame, c, attack_end, _ATTACK_AXIS_BGR, thickness=2, head_len=9.0)
    _draw_metric_line(frame, c, r, (110, 110, 120), thickness=2)
    if tip is not None and geom.forward_gain_m > 0.05:
        _draw_metric_line(frame, c, tip, _FWD_BGR, thickness=3)
        cv2.circle(frame, tip, 8, _FWD_BGR, -1, cv2.LINE_AA)
    out = draw_focus_zone(frame, original, centers, radius=160, strength=0.72)
    _draw_metric_line(out, c, r, rank_color, thickness=3)
    _label_segment(out, c, attack_end, "ATTACK AXIS", _ATTACK_AXIS_BGR, along=0.6, offset_px=-18)
    _label_segment(out, c, r, "PASS LANE", rank_color, along=0.38, offset_px=20)
    if tip is not None and geom.forward_gain_m > 0.05:
        _label_segment(
            out,
            c,
            tip,
            f"FORWARD +{geom.forward_gain_m:.1f} m",
            _FWD_BGR,
            along=0.55,
            offset_px=-16,
        )
    return out


def draw_forward_on_radar(radar: np.ndarray, draw: RadarMetricDraw) -> np.ndarray:
    out = radar.copy()
    c_pt = _radar_pt(draw.carrier_cm)
    r_pt = _radar_pt(draw.receiver_cm)
    attack_pt = _radar_pt(draw.attack_end_cm)
    tip_pt = _radar_pt(draw.forward_tip_cm)
    _dashed_line(out, c_pt, attack_pt, (180, 180, 190), thickness=1, gap=8)
    _draw_metric_chevron(
        out, c_pt, attack_pt, (180, 180, 190), thickness=1, head_len=6.0, wing=0.5
    )
    _draw_metric_line(out, c_pt, r_pt, draw.rank_color, thickness=2)
    _draw_metric_line(out, c_pt, tip_pt, _FWD_BGR, thickness=2)
    cv2.circle(out, tip_pt, 6, _FWD_BGR, -1, cv2.LINE_AA)
    _label_radar_segment(out, draw.carrier_cm, draw.attack_end_cm, "ATTACK", (180, 180, 190))
    _label_radar_segment(out, draw.carrier_cm, draw.receiver_cm, "PASS", draw.rank_color)
    _label_radar_segment(
        out,
        draw.carrier_cm,
        draw.forward_tip_cm,
        f"FWD +{draw.forward_gain_m:.1f} m",
        _FWD_BGR,
    )
    return out


def draw_space_on_frame(
    frame: np.ndarray,
    original: np.ndarray,
    geom: LaneMetricGeometry,
    transformer,
    *,
    m_per_px: float,
) -> np.ndarray:
    recv = _m_to_img(geom.receiver_m, transformer, m_per_px=m_per_px)
    if recv is None:
        return frame
    centers = [recv]
    rival = None
    if geom.space_rival_m is not None:
        rival = _m_to_img(geom.space_rival_m, transformer, m_per_px=m_per_px)
        if rival is not None:
            centers.append(rival)
            cv2.line(frame, recv, rival, _SPACE_BGR, 3, cv2.LINE_AA)
            cv2.circle(frame, rival, 14, _RIVAL_BGR, -1, cv2.LINE_AA)
    ref_poly = _draw_projected_pitch_circle(
        frame,
        geom.receiver_m,
        geom.space_ref_m,
        transformer,
        (170, 170, 185),
        thickness=2,
        dashed=True,
    )
    actual_poly = None
    if geom.space_m < geom.space_ref_m - 0.05:
        actual_poly = _draw_projected_pitch_circle(
            frame,
            geom.receiver_m,
            geom.space_m,
            transformer,
            _SPACE_BGR,
            thickness=2,
        )
    focus_r = (
        max(
            pitch_polygon_extent_px(ref_poly),
            pitch_polygon_extent_px(actual_poly),
            40,
        )
        + 40
    )
    out = draw_focus_zone(frame, original, centers, radius=focus_r, strength=0.75)
    if ref_poly is not None and len(ref_poly) > 0:
        label_idx = len(ref_poly) // 8
        ref_label_pt = tuple(int(v) for v in ref_poly[label_idx])
        _label_point(
            out,
            ref_label_pt,
            f"SEARCH RADIUS {geom.space_ref_m:.0f} m",
            (170, 170, 185),
            offset=(0, -18),
            font_scale=0.42,
        )
    return out


def _space_distance_label_xy(
    recv: tuple[int, int],
    rival: tuple[int, int],
    avoid_pts: list[tuple[int, int]],
) -> tuple[int, int]:
    """Pick a spot beside the line that clears player feet and ID tags."""
    best_score = -1.0
    best_xy = pass_line_label_xy(recv, rival, along=0.65, offset_px=34)
    for along in (0.52, 0.64, 0.76, 0.88):
        for sign in (1, -1):
            for offset_px in (26, 34, 42):
                lx, ly = pass_line_label_xy(
                    recv, rival, along=along, offset_px=sign * offset_px
                )
                endpoint_clear = min(
                    float(np.hypot(lx - recv[0], ly - recv[1])),
                    float(np.hypot(lx - rival[0], ly - rival[1])),
                )
                if avoid_pts:
                    feet_clear = min(
                        float(np.hypot(lx - px, ly - py)) for px, py in avoid_pts
                    )
                else:
                    feet_clear = endpoint_clear
                # Tags sit below feet; bias labels upward on screen.
                score = min(endpoint_clear, feet_clear) + max(0.0, (ly * -0.04))
                if score > best_score:
                    best_score = score
                    best_xy = (lx, ly)
    return best_xy


def _label_with_bg(
    img: np.ndarray,
    text: str,
    org: tuple[int, int],
    *,
    color: tuple[int, int, int],
    font_scale: float = 0.46,
) -> None:
    from world_cup_projects.common.visual import draw_text_shadow

    (tw, th), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2
    )
    x, y = org
    pad_x, pad_y = 6, 4
    x0 = x - pad_x
    y0 = y - th - pad_y
    x1 = x + tw + pad_x
    y1 = y + baseline + pad_y
    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (16, 16, 20), -1)
    cv2.addWeighted(overlay, 0.82, img, 0.18, 0, img)
    draw_text_shadow(
        img, text, (x, y), font_scale=font_scale, color_bgr=color, thickness=2
    )


def draw_space_distance_label(
    frame: np.ndarray,
    geom: LaneMetricGeometry,
    transformer,
    *,
    m_per_px: float,
    avoid_pts: list[tuple[int, int]] | None = None,
) -> None:
    """Draw after player tags so the distance label stays readable."""
    if geom.space_rival_m is None:
        return
    recv = _m_to_img(geom.receiver_m, transformer, m_per_px=m_per_px)
    rival = _m_to_img(geom.space_rival_m, transformer, m_per_px=m_per_px)
    if recv is None or rival is None:
        return
    lx, ly = _space_distance_label_xy(recv, rival, avoid_pts or [])
    _label_with_bg(
        frame,
        f"NEAREST RIVAL {geom.space_m:.1f} m",
        (lx, ly),
        color=_SPACE_BGR,
    )


def draw_space_on_radar(radar: np.ndarray, draw: RadarMetricDraw) -> np.ndarray:
    out = radar.copy()
    c_pt = _radar_pt(draw.carrier_cm)
    recv_pt = _radar_pt(draw.receiver_cm)
    cv2.line(out, c_pt, recv_pt, draw.rank_color, 2, cv2.LINE_AA)
    ref_poly = _draw_projected_radar_circle(
        out,
        draw.receiver_cm,
        draw.space_ref_m,
        (170, 170, 185),
        thickness=1,
        dashed=True,
    )
    if draw.space_m < draw.space_ref_m - 0.05:
        _draw_projected_radar_circle(
            out, draw.receiver_cm, draw.space_m, _SPACE_BGR, thickness=1
        )
    if ref_poly is not None and len(ref_poly) > 0:
        label_pt = tuple(int(v) for v in ref_poly[len(ref_poly) // 8])
        _label_point(
            out,
            label_pt,
            f"REF {draw.space_ref_m:.0f} m",
            (170, 170, 185),
            offset=(0, -10),
            font_scale=0.30,
        )
    if draw.space_rival_cm is not None:
        rival_pt = _radar_pt(draw.space_rival_cm)
        cv2.line(out, recv_pt, rival_pt, _SPACE_BGR, 2, cv2.LINE_AA)
        cv2.circle(out, rival_pt, 8, _RIVAL_BGR, -1, cv2.LINE_AA)
        _label_radar_segment(
            out,
            draw.receiver_cm,
            draw.space_rival_cm,
            f"NEAREST RIVAL {draw.space_m:.1f} m",
            _SPACE_BGR,
        )
    return out


def draw_penalty_teammate_on_frame(
    frame: np.ndarray,
    original: np.ndarray,
    penalty: PenaltyGeometry,
    geom: LaneMetricGeometry,
    transformer,
    *,
    m_per_px: float,
    rank_color: tuple[int, int, int],
) -> np.ndarray:
    c = _m_to_img(geom.carrier_m, transformer, m_per_px=m_per_px)
    r = _m_to_img(geom.receiver_m, transformer, m_per_px=m_per_px)
    if c is None or r is None:
        return frame
    centers = [c, r]
    tm = proj = None
    _draw_metric_line(frame, c, r, rank_color, thickness=3)
    if penalty.teammate_m is not None:
        tm = _m_to_img(penalty.teammate_m, transformer, m_per_px=m_per_px)
        if penalty.teammate_proj_m is not None:
            proj = _m_to_img(penalty.teammate_proj_m, transformer, m_per_px=m_per_px)
        if tm is not None:
            centers.append(tm)
            cv2.circle(frame, tm, 16, _TEAMMATE_BGR, -1, cv2.LINE_AA)
            if proj is not None:
                centers.append(proj)
                _dashed_line(frame, tm, proj, _PENALTY_BGR, thickness=2)
    out = draw_focus_zone(frame, original, centers, radius=150, strength=0.72)
    _label_segment(out, c, r, "PASS LANE", rank_color, offset_px=18)
    if tm is not None:
        _label_point(out, tm, "TEAMMATE BLOCKING", _TEAMMATE_BGR, offset=(0, -24))
        if proj is not None:
            _label_segment(
                out,
                tm,
                proj,
                f"TM CLEARANCE {penalty.teammate_openness_m:.1f} m",
                _PENALTY_BGR,
                along=0.5,
                offset_px=-14,
            )
    return out


def draw_penalty_teammate_on_radar(radar: np.ndarray, draw: PenaltyRadarDraw) -> np.ndarray:
    out = radar.copy()
    c_pt = _radar_pt(draw.carrier_cm)
    r_pt = _radar_pt(draw.receiver_cm)
    cv2.line(out, c_pt, r_pt, draw.rank_color, 2, cv2.LINE_AA)
    _label_radar_segment(out, draw.carrier_cm, draw.receiver_cm, "PASS", draw.rank_color)
    if draw.teammate_cm is not None and draw.teammate_proj_cm is not None:
        tm_pt = _radar_pt(draw.teammate_cm)
        proj_pt = _radar_pt(draw.teammate_proj_cm)
        cv2.circle(out, tm_pt, 10, _TEAMMATE_BGR, -1, cv2.LINE_AA)
        _dashed_line(out, tm_pt, proj_pt, _PENALTY_BGR, thickness=2)
        _label_radar_segment(out, draw.teammate_cm, draw.teammate_proj_cm, "TM BLOCK", _PENALTY_BGR)
    return out


def draw_penalty_run_on_frame(
    frame: np.ndarray,
    original: np.ndarray,
    penalty: PenaltyGeometry,
    geom: LaneMetricGeometry,
    transformer,
    *,
    m_per_px: float,
    rank_color: tuple[int, int, int],
) -> np.ndarray:
    c = _m_to_img(geom.carrier_m, transformer, m_per_px=m_per_px)
    r = _m_to_img(geom.receiver_m, transformer, m_per_px=m_per_px)
    if c is None or r is None:
        return frame
    centers = [c, r]
    _draw_metric_line(frame, c, r, rank_color, thickness=3)
    if penalty.motion_dir_m is not None:
        motion_end_m = geom.carrier_m + penalty.motion_dir_m * 10.0
        motion_pt = _m_to_img(motion_end_m, transformer, m_per_px=m_per_px)
        if motion_pt is not None:
            centers.append(motion_pt)
            _draw_metric_chevron(frame, c, motion_pt, _MOTION_BGR, thickness=2, head_len=10.0)
    out = draw_focus_zone(frame, original, centers, radius=150, strength=0.72)
    _label_segment(out, c, r, "PASS DIRECTION", rank_color, along=0.42, offset_px=18)
    if penalty.motion_dir_m is not None:
        motion_end_m = geom.carrier_m + penalty.motion_dir_m * 10.0
        motion_pt = _m_to_img(motion_end_m, transformer, m_per_px=m_per_px)
        if motion_pt is not None:
            _label_segment(
                out, c, motion_pt, "CARRIER RUN", _MOTION_BGR, along=0.55, offset_px=-18,
            )
            _draw_angle_arc(
                out, c, r, motion_pt, color=_ANGLE_ARC_BGR, radius_px=38.0, thickness=2
            )
    return out


def draw_penalty_run_on_radar(radar: np.ndarray, draw: PenaltyRadarDraw) -> np.ndarray:
    out = radar.copy()
    c_pt = _radar_pt(draw.carrier_cm)
    r_pt = _radar_pt(draw.receiver_cm)
    cv2.line(out, c_pt, r_pt, draw.rank_color, 2, cv2.LINE_AA)
    _label_radar_segment(out, draw.carrier_cm, draw.receiver_cm, "PASS", draw.rank_color)
    if draw.motion_end_cm is not None:
        m_pt = _radar_pt(draw.motion_end_cm)
        _draw_metric_chevron(out, c_pt, m_pt, _MOTION_BGR, thickness=1, head_len=6.0)
        _label_radar_segment(out, draw.carrier_cm, draw.motion_end_cm, "RUN", _MOTION_BGR)
        _draw_angle_arc(
            out, c_pt, r_pt, m_pt, color=_ANGLE_ARC_BGR, radius_px=14.0, thickness=1
        )
    return out


def draw_penalty_back_on_frame(
    frame: np.ndarray,
    original: np.ndarray,
    penalty: PenaltyGeometry,
    geom: LaneMetricGeometry,
    transformer,
    *,
    m_per_px: float,
    rank_color: tuple[int, int, int],
) -> np.ndarray:
    c = _m_to_img(geom.carrier_m, transformer, m_per_px=m_per_px)
    r = _m_to_img(geom.receiver_m, transformer, m_per_px=m_per_px)
    attack_end = _m_to_img(geom.attack_display_m, transformer, m_per_px=m_per_px)
    if c is None or r is None:
        return frame
    centers = [c, r]
    if attack_end is not None:
        _dashed_line(frame, c, attack_end, _ATTACK_AXIS_BGR, thickness=2, gap=10)
        _draw_metric_chevron(frame, c, attack_end, _ATTACK_AXIS_BGR, thickness=2, head_len=9.0)
    _draw_metric_line(frame, c, r, rank_color, thickness=3)
    if penalty.backward_tip_m is not None:
        back_pt = _m_to_img(penalty.backward_tip_m, transformer, m_per_px=m_per_px)
        if back_pt is not None:
            centers.append(back_pt)
            cv2.line(frame, c, back_pt, _PENALTY_BGR, 4, cv2.LINE_AA)
    out = draw_focus_zone(frame, original, centers, radius=160, strength=0.72)
    if attack_end is not None:
        _label_segment(out, c, attack_end, "ATTACK AXIS", _ATTACK_AXIS_BGR, along=0.6, offset_px=-18)
    _label_segment(out, c, r, "PASS LANE", rank_color, along=0.4, offset_px=20)
    if penalty.backward_tip_m is not None:
        back_pt = _m_to_img(penalty.backward_tip_m, transformer, m_per_px=m_per_px)
        if back_pt is not None:
            _label_segment(
                out, c, back_pt,
                f"BACKWARD {geom.forward_gain_m:.1f} m",
                _PENALTY_BGR, along=0.5, offset_px=-16,
            )
    return out


def draw_penalty_back_on_radar(radar: np.ndarray, draw: PenaltyRadarDraw) -> np.ndarray:
    out = radar.copy()
    c_pt = _radar_pt(draw.carrier_cm)
    r_pt = _radar_pt(draw.receiver_cm)
    attack_pt = _radar_pt(draw.attack_end_cm)
    _dashed_line(out, c_pt, attack_pt, _ATTACK_AXIS_BGR, thickness=1, gap=8)
    cv2.line(out, c_pt, r_pt, draw.rank_color, 2, cv2.LINE_AA)
    _label_radar_segment(out, draw.carrier_cm, draw.attack_end_cm, "ATTACK", (180, 180, 190))
    _label_radar_segment(out, draw.carrier_cm, draw.receiver_cm, "PASS", draw.rank_color)
    if draw.backward_tip_cm is not None:
        back_pt = _radar_pt(draw.backward_tip_cm)
        cv2.line(out, c_pt, back_pt, _PENALTY_BGR, 3, cv2.LINE_AA)
        cv2.circle(out, back_pt, 7, _PENALTY_BGR, -1, cv2.LINE_AA)
        _label_radar_segment(out, draw.carrier_cm, draw.backward_tip_cm, "BACK", _PENALTY_BGR)
    return out
