"""Draw pass-lane corridors (pitch/radar space) on the video and minimap."""

from __future__ import annotations

import cv2
import numpy as np
import supervision as sv

from world_cup_projects.common.pitch import (
    PITCH_CONFIG,
    SoccerPitchConfiguration,
    ViewTransformer,
    pitch_cm_to_image,
)
from world_cup_projects.common.visual import draw_glow_arrow, draw_score_chip, draw_text_shadow
from world_cup_projects.pass_alternatives.pass_options import PassOption

RANK_LABELS = ["BEST", "2ND", "3RD"]

_RADAR_SCALE = 0.1
_RADAR_PADDING = 50
_CORRIDOR_ALPHA_RADAR = 0.45
_CORRIDOR_ALPHA_VIDEO = 0.38
_CORRIDOR_ALPHA_PRESENTATION = 0.62
_CANDIDATE_BGR = (100, 180, 255)
_RANK_COLORS = [
    sv.Color.from_hex("#3CDC3C"),  # best (Green)
    sv.Color.from_hex("#FFD700"),  # 2nd (Yellow instead of Cyan/blue shadow)
    sv.Color.from_hex("#FF8C28"),  # 3rd (Orange)
]
_RANK_BGR = [c.as_bgr() for c in _RANK_COLORS]
_BLOCKER_BGR = (40, 40, 255)


def pass_line_label_xy(
    start: tuple[int, int],
    end: tuple[int, int],
    *,
    along: float = 0.42,
    offset_px: int = 14,
) -> tuple[int, int]:
    """Point beside the pass segment for a distance label (freeze-frame style)."""
    sx, sy = start
    ex, ey = end
    px = sx + along * (ex - sx)
    py = sy + along * (ey - sy)
    dx, dy = ex - sx, ey - sy
    norm = float(np.hypot(dx, dy)) or 1.0
    return (
        int(px - dy / norm * offset_px),
        int(py + dx / norm * offset_px),
    )


def apply_pass_lane_geometry(
    frame: np.ndarray,
    options: list[PassOption],
    transformer: ViewTransformer,
    feet_xy: np.ndarray,
    *,
    display_ranks: list[int] | None = None,
) -> np.ndarray:
    """Shaded corridors + rival blockers — same as pass-alternatives freeze video."""
    out = draw_pass_corridors_on_frame(
        frame, options, transformer, display_ranks=display_ranks
    )
    return draw_blocking_rivals_on_frame(out, options, feet_xy=feet_xy)


def draw_receiver_highlight(
    frame: np.ndarray,
    center: tuple[int, int],
    rank: int,
    color_bgr: tuple[int, int, int],
    *,
    alpha: float = 1.0,
) -> None:
    """Ring at the receiver feet (BEST gets a white outer ring)."""
    rx, ry = center
    ring = 20 if rank == 0 else 14
    layer = frame.copy()
    cv2.circle(layer, (rx, ry), ring, color_bgr, 3 if rank == 0 else 2, cv2.LINE_AA)
    if rank == 0:
        cv2.circle(layer, (rx, ry), ring + 6, (255, 255, 255), 1, cv2.LINE_AA)
    if alpha >= 0.99:
        frame[:] = layer
    else:
        frame[:] = cv2.addWeighted(layer, alpha, frame, 1.0 - alpha, 0)


def draw_pass_arrows_on_frame(
    frame: np.ndarray,
    feet_xy: np.ndarray,
    carrier_index: int,
    options: list[PassOption],
    *,
    metric: bool = False,
    show_chips: bool = True,
    show_length: bool = True,
    arrow_thickness: int = 5,
    arrow_alpha: float = 1.0,
    display_ranks: list[int] | None = None,
) -> np.ndarray:
    """Glow arrows, receiver rings, and optional chips — matches freeze overlay."""
    cx, cy = int(feet_xy[carrier_index, 0]), int(feet_xy[carrier_index, 1])
    for idx, option in enumerate(options):
        rank = display_ranks[idx] if display_ranks is not None else idx
        color = _RANK_BGR[min(rank, len(_RANK_BGR) - 1)]
        rx, ry = int(feet_xy[option.receiver_index, 0]), int(feet_xy[option.receiver_index, 1])
        draw_glow_arrow(
            frame, (cx, cy), (rx, ry), color, thickness=arrow_thickness, alpha=arrow_alpha
        )
        if arrow_alpha > 0.2:
            draw_receiver_highlight(frame, (rx, ry), rank, color, alpha=arrow_alpha)
        if arrow_alpha < 0.85:
            continue
        if show_chips:
            chip = f"{RANK_LABELS[rank]}  {option.score:.2f}"
            if metric:
                chip += f"  {option.length:.1f} m"
            if option.rivals_in_lane:
                chip += f"  ({option.rivals_in_lane} riv)"
            if option.lane_debug and option.lane_debug.blocking_rival_indices:
                chip += f"  !{len(option.lane_debug.blocking_rival_indices)}"
            midx, midy = (cx + rx) // 2, (cy + ry) // 2
            draw_score_chip(frame, chip, (midx, midy), bg_bgr=color)
        if metric and show_length:
            lx, ly = pass_line_label_xy((cx, cy), (rx, ry))
            draw_text_shadow(
                frame,
                f"{option.length:.1f} m",
                (lx - 18, ly - 6),
                font_scale=0.58,
                color_bgr=color,
                thickness=2,
            )
    return frame


def pitch_cm_to_radar_px(
    points_cm: np.ndarray,
    *,
    config: SoccerPitchConfiguration = PITCH_CONFIG,
    scale: float = _RADAR_SCALE,
    padding: int = _RADAR_PADDING,
) -> np.ndarray:
    pts = np.asarray(points_cm, dtype=np.float64).reshape(-1, 2)
    return np.stack(
        [
            (pts[:, 0] * scale + padding).astype(np.int32),
            (pts[:, 1] * scale + padding).astype(np.int32),
        ],
        axis=1,
    )


def _draw_blocker_markers(
    out: np.ndarray,
    pitch_cm: np.ndarray,
    indices: tuple[int, ...],
    *,
    config: SoccerPitchConfiguration,
    drawn: set[int] | None = None,
) -> set[int]:
    """Small red badge with ! — no white halo (matches freeze video)."""
    seen = drawn if drawn is not None else set()
    for idx in indices:
        if idx in seen or idx >= len(pitch_cm):
            continue
        seen.add(idx)
        pt = pitch_cm_to_radar_px(pitch_cm[idx : idx + 1], config=config)[0]
        cv2.circle(out, tuple(pt), 12, _BLOCKER_BGR, -1, cv2.LINE_AA)
        cv2.putText(
            out,
            "!",
            (pt[0] - 5, pt[1] + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return seen


def draw_candidate_lanes_on_radar(
    radar: np.ndarray,
    pitch_cm: np.ndarray,
    carrier_index: int,
    receiver_indices: list[int],
    *,
    config: SoccerPitchConfiguration = PITCH_CONFIG,
) -> np.ndarray:
    """Simple carrier→teammate segments on the minimap (step 1 explain)."""
    out = radar.copy()
    if carrier_index >= len(pitch_cm):
        return out
    c_pt = pitch_cm_to_radar_px(pitch_cm[carrier_index : carrier_index + 1], config=config)[0]
    for idx in receiver_indices:
        if idx >= len(pitch_cm):
            continue
        r_pt = pitch_cm_to_radar_px(pitch_cm[idx : idx + 1], config=config)[0]
        cv2.line(out, tuple(c_pt), tuple(r_pt), _CANDIDATE_BGR, 3, cv2.LINE_AA)
        cv2.circle(out, tuple(r_pt), 10, _CANDIDATE_BGR, -1, cv2.LINE_AA)
    cv2.circle(out, tuple(c_pt), 12, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def draw_pass_lanes_on_radar(
    radar: np.ndarray,
    options: list[PassOption],
    pitch_cm: np.ndarray,
    *,
    config: SoccerPitchConfiguration = PITCH_CONFIG,
    display_ranks: list[int] | None = None,
) -> np.ndarray:
    """Overlay ranked pass corridors and highlight blocking rivals (pitch cm coords)."""
    out = radar.copy()
    drawn: set[int] = set()
    for idx, option in enumerate(options):
        rank = display_ranks[idx] if display_ranks is not None else idx
        debug = option.lane_debug
        if debug is None:
            continue
        color = _RANK_COLORS[min(rank, len(_RANK_COLORS) - 1)]
        poly = pitch_cm_to_radar_px(debug.corridor_polygon_cm, config=config)
        overlay = out.copy()
        cv2.fillPoly(overlay, [poly], color.as_bgr())
        out = cv2.addWeighted(overlay, _CORRIDOR_ALPHA_RADAR, out, 1.0 - _CORRIDOR_ALPHA_RADAR, 0)
        cv2.polylines(out, [poly], isClosed=True, color=color.as_bgr(), thickness=2)

        drawn = _draw_blocker_markers(
            out,
            pitch_cm,
            debug.blocking_rival_indices,
            config=config,
            drawn=drawn,
        )

    return out


def _corridor_image_polygon(
    corridor_polygon_cm: np.ndarray,
    transformer: ViewTransformer,
    frame_shape: tuple[int, int],
) -> np.ndarray | None:
    """Project a pitch corridor (cm) onto image pixels, clipped to the frame."""
    img = pitch_cm_to_image(corridor_polygon_cm, transformer)
    if img is None or len(img) < 3:
        return None
    h, w = frame_shape[:2]
    poly = img.astype(np.int32)
    if not ((poly[:, 0] >= -w) & (poly[:, 0] <= 2 * w) & (poly[:, 1] >= -h) & (poly[:, 1] <= 2 * h)).any():
        return None
    return poly


def draw_pass_corridors_on_frame(
    frame: np.ndarray,
    options: list[PassOption],
    transformer: ViewTransformer,
    *,
    presentation: bool = False,
    display_ranks: list[int] | None = None,
) -> np.ndarray:
    """Draw pitch-space pass corridors on the main camera view (H^-1)."""
    out = frame
    alpha = _CORRIDOR_ALPHA_PRESENTATION if presentation else _CORRIDOR_ALPHA_VIDEO
    thickness = 5 if presentation else 3
    for idx, option in enumerate(options):
        rank = display_ranks[idx] if display_ranks is not None else idx
        debug = option.lane_debug
        if debug is None:
            continue
        poly = _corridor_image_polygon(
            debug.corridor_polygon_cm, transformer, out.shape
        )
        if poly is None:
            continue
        color = _RANK_BGR[min(rank, len(_RANK_BGR) - 1)]
        overlay = out.copy()
        cv2.fillPoly(overlay, [poly], color)
        out = cv2.addWeighted(overlay, alpha, out, 1.0 - alpha, 0)
        cv2.polylines(out, [poly], isClosed=True, color=color, thickness=thickness)

    return out


def draw_blocking_rivals_on_frame(
    frame: np.ndarray,
    options: list[PassOption],
    *,
    feet_xy: np.ndarray,
) -> np.ndarray:
    """Mark rivals counted inside a corridor — compact red ! at feet."""
    out = frame
    drawn: set[int] = set()
    for option in options:
        debug = option.lane_debug
        if debug is None:
            continue
        for idx in debug.blocking_rival_indices:
            if idx in drawn or idx >= len(feet_xy):
                continue
            drawn.add(idx)
            x, y = int(feet_xy[idx, 0]), int(feet_xy[idx, 1])
            cv2.circle(out, (x, y), 14, _BLOCKER_BGR, -1, cv2.LINE_AA)
            cv2.putText(
                out,
                "!",
                (x - 7, y + 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
    return out


def draw_pass_lane_legend(frame: np.ndarray) -> np.ndarray:
    """Legend for pass-lane debug overlays."""
    lines = [
        "shaded = pass corridor (pitch/radar; wider on long passes)",
        "green/cyan/orange = BEST / 2ND / 3RD",
        "red ! = rival inside corridor",
    ]
    y = 78
    for line in lines:
        cv2.putText(
            frame,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )
        y += 18
    return frame


from dataclasses import dataclass

from world_cup_projects.common.pitch import (
    ViewTransformer,
    homography_from_keypoints_radar,
    image_to_pitch_cm,
    image_to_pitch_m,
    pitch_attack_direction,
    render_radar_simple,
)
from world_cup_projects.common.possession import (
    Carrier,
    bbox_center_xy,
    feet_xy,
    player_mask,
)
from world_cup_projects.common.visual import (
    ROBOFLOW_PURPLE_BGR,
    annotate_ball,
    annotate_players,
    draw_branding_tag,
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


@dataclass(frozen=True)
class PassEvent:
    frame_idx: int
    carrier: Carrier
    options: list[PassOption]
    top_score: float


def _options_with_lane_debug(
    dets: sv.Detections,
    event: PassEvent,
    *,
    weights: PassWeights,
    transformer: ViewTransformer,
    pitch_cm: np.ndarray | None = None,
) -> list[PassOption]:
    """Re-score if needed so freeze frames always carry pitch corridor geometry."""
    if event.options and all(o.lane_debug is not None for o in event.options):
        return event.options
    pitch_feet = image_to_pitch_m(feet_xy(dets), transformer)
    if pitch_cm is None:
        pitch_cm = image_to_pitch_cm(feet_xy(dets), transformer)
    body_pitch_m = image_to_pitch_m(bbox_center_xy(dets), transformer)
    if pitch_feet is None or pitch_cm is None:
        return event.options
    attack_dir = pitch_attack_direction(
        dets,
        event.carrier.team,
        transformer,
        player_mask_fn=player_mask,
        feet_fn=feet_xy,
    )
    return top_pass_options(
        dets,
        event.carrier,
        k=3,
        weights=weights,
        attack_dir=attack_dir,
        positions=pitch_feet,
        pitch_cm=pitch_cm,
        body_pitch_m=body_pitch_m,
    )


def draw_pass_overlay(
    frame: np.ndarray,
    dets: sv.Detections,
    event: PassEvent,
    *,
    weights: PassWeights = PassWeights(),
    metric: bool = False,
    keypoints: sv.KeyPoints | None = None,
    pitch_confidence: float = 0.9,
    transformer: ViewTransformer | None = None,
    show_lane_debug: bool = True,
    show_radar: bool = True,
    locked_goal_defenders: tuple[int, int] | None = None,
    debug_pitch_keypoints: bool = False,
    revealed_options: int | None = None,
    reveal_progress: float = 1.0,
    facing: np.ndarray | None = None,
    facing_motion: np.ndarray | None = None,
    facing_kalman: np.ndarray | None = None,
) -> np.ndarray:
    """Dim the frame and draw ranked pass arrows from the carrier.

    ``revealed_options``: how many top options to show (0 = carrier only, None = all).
    ``reveal_progress``: 0–1 animation within the current reveal phase.
    """
    dim = (frame.astype(np.float32) * 0.32).astype(np.uint8)
    if debug_pitch_keypoints and keypoints is not None:
        dim = draw_pitch_keypoints_debug(
            dim, keypoints, confidence_threshold=pitch_confidence
        )

    options = event.options
    feet_img = feet_xy(dets)
    if show_lane_debug and transformer is not None:
        pitch_cm_vis = image_to_pitch_cm(feet_img, transformer)
        options = _options_with_lane_debug(
            dets,
            event,
            weights=weights,
            transformer=transformer,
            pitch_cm=pitch_cm_vis,
        )

    visible = options
    if revealed_options is not None:
        visible = options[: max(0, revealed_options)]

    if show_lane_debug and transformer is not None and visible:
        dim = apply_pass_lane_geometry(dim, visible, transformer, feet_img)

    dim = annotate_players(
        dim,
        dets,
        facing=facing,
        facing_motion=facing_motion,
        facing_kalman=facing_kalman,
        show_tracker_ids=True,
    )
    dim = annotate_ball(dim, dets)

    feet = feet_xy(dets)
    carrier_xy = feet[event.carrier.index]
    cx, cy = int(carrier_xy[0]), int(carrier_xy[1])
    dim = draw_carrier_spotlight(dim, frame, (cx, cy))

    n_total = min(3, len(options))
    phase_revealed = 0 if revealed_options == 0 else revealed_options

    if revealed_options == 0:
        draw_carrier_pulse(dim, (cx, cy), reveal_progress)
        draw_score_chip(dim, "ON BALL", (cx, cy - 42), bg_bgr=ROBOFLOW_PURPLE_BGR)
        dim = draw_pass_analysis_panel(
            dim,
            progress=reveal_progress,
            revealed=0,
            total=n_total,
        )
        dim = draw_hud_bar(dim, "PASS ALTERNATIVES")
        return draw_branding_tag(dim)

    draw_carrier_pulse(dim, (cx, cy), min(1.0, reveal_progress * 0.35 + 0.65))

    for rank, option in enumerate(visible):
        color = RANK_COLORS_BGR[rank]
        recv_xy = feet[option.receiver_index]
        rx, ry = int(recv_xy[0]), int(recv_xy[1])
        is_new = rank == len(visible) - 1
        alpha = ease_out_cubic(reveal_progress) if is_new else 1.0
        draw_glow_arrow(dim, (cx, cy), (rx, ry), color, thickness=5, alpha=alpha)
        if alpha > 0.2:
            draw_receiver_highlight(dim, (rx, ry), rank, color, alpha=alpha)
        if alpha < 0.85:
            continue
        midx, midy = (cx + rx) // 2, (cy + ry) // 2
        chip = f"{RANK_LABELS[rank]}  {option.score:.2f}"
        if metric:
            chip += f"  {option.length:.1f} m"
        if option.rivals_in_lane:
            chip += f"  ({option.rivals_in_lane} riv)"
        if option.teammates_in_lane:
            chip += f"  ({option.teammates_in_lane} tm)"
        if option.lane_debug and option.lane_debug.blocking_rival_indices:
            chip += f"  !{len(option.lane_debug.blocking_rival_indices)}"
        draw_score_chip(dim, chip, (midx, midy), bg_bgr=color)
        if metric:
            lx, ly = pass_line_label_xy((cx, cy), (rx, ry))
            draw_text_shadow(
                dim,
                f"{option.length:.1f} m",
                (lx - 18, ly - 6),
                font_scale=0.58,
                color_bgr=color,
                thickness=2,
            )

    latest_rank = len(visible) - 1
    dim = draw_pass_analysis_panel(
        dim,
        progress=reveal_progress,
        revealed=phase_revealed,
        total=n_total,
        rank_label=RANK_LABELS[latest_rank] if visible else None,
        rank_color=RANK_COLORS_BGR[latest_rank] if visible else None,
    )

    if show_radar and keypoints is not None:
        radar_h = homography_from_keypoints_radar(
            keypoints, confidence=pitch_confidence
        )
        radar = render_radar_simple(
            dets,
            keypoints,
            confidence=pitch_confidence,
            transformer=radar_h,
            locked_goal_defenders=locked_goal_defenders,
            debug_keypoints=True,
        )
        if radar is not None and show_lane_debug and visible and radar_h is not None:
            pitch_cm_radar = image_to_pitch_cm(feet_img, radar_h)
            if pitch_cm_radar is not None:
                radar_visible = remap_lane_debug_to_pitch_cm(
                    visible,
                    event.carrier,
                    pitch_cm_radar,
                    feet_img,
                    weights=weights,
                )
                radar = draw_pass_lanes_on_radar(
                    radar, radar_visible, pitch_cm_radar
                )
            dim = draw_pass_lane_legend(dim)
        if radar is not None:
            dim = draw_radar_minimap(
                dim,
                dets,
                keypoints,
                pitch_confidence=pitch_confidence,
                locked_goal_defenders=locked_goal_defenders,
                prebuilt_radar=radar,
            )

    dim = draw_hud_bar(dim, "PASS ALTERNATIVES  -  top 3 open lanes")
    return draw_branding_tag(dim)
