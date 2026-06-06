"""Render the speed & distance demo: Roboflow Football-AI styled overlays.

Live frames show team-colored player ellipses with rounded speed-label chips, a
downward ball triangle, and a bottom-center radar minimap. The clip closes on a
leaderboard end-card. Visuals are shared with the pass demo via ``common.visual``.
"""

from __future__ import annotations

import cv2
import numpy as np
import supervision as sv

from world_cup_projects.common.pitch import (
    image_to_pitch_m,
    warmup_goal_defenders,
)
from world_cup_projects.common.possession import feet_xy, player_mask
from world_cup_projects.common.soccernet import SoccerNetSequence
from world_cup_projects.common.visual import (
    ROBOFLOW_PURPLE_BGR,
    TEAM_COLORS,
    annotate_ball,
    annotate_players,
    draw_branding_tag,
    draw_hud_bar,
    draw_homography_feet_debug,
    draw_pitch_keypoints_debug,
    draw_radar_minimap,
    draw_text_shadow,
)
from world_cup_projects.player_stats.speed_distance import (
    MS_TO_KMH,
    PlayerTrack,
    format_speed_kmh,
    speed_at_frame,
)

TEAM_COLORS_BGR = [c.as_bgr() for c in TEAM_COLORS[:2]]
NEUTRAL_BGR = (200, 200, 200)


def _team_color(team: int) -> tuple[int, int, int]:
    return TEAM_COLORS_BGR[team] if team in (0, 1) else NEUTRAL_BGR


def _player_labels(
    dets: sv.Detections, tracks: dict[int, PlayerTrack], frame_idx: int
) -> list[str]:
    """Per-player chip text aligned with outfield ``ROLE_PLAYER`` rows in ``dets``."""
    from world_cup_projects.common.soccernet import ROLE_PLAYER

    labels: list[str] = []
    for i in np.flatnonzero(dets.class_id == ROLE_PLAYER):
        tid = int(dets.tracker_id[i]) if dets.tracker_id is not None else -1
        track = tracks.get(tid)
        speed = speed_at_frame(track, frame_idx) if track is not None else None
        if speed is None:
            labels.append(f"#{tid}")
        else:
            labels.append(f"#{tid}  {format_speed_kmh(speed)}")
    return labels


def _leaderboard_card(
    size: tuple[int, int], tracks: dict[int, PlayerTrack], *, calibration: str
) -> np.ndarray:
    w, h = size
    card = np.full((h, w, 3), 22, np.uint8)
    # subtle purple top accent band
    cv2.rectangle(card, (0, 0), (w, 6), ROBOFLOW_PURPLE_BGR, -1)

    draw_text_shadow(card, "PLAYER SPEED & DISTANCE", (40, 72),
                     font_scale=1.1, color_bgr=ROBOFLOW_PURPLE_BGR, thickness=2)
    draw_text_shadow(card, f"calibration: {calibration}", (42, 108),
                     font_scale=0.55, color_bgr=(170, 170, 170), thickness=1)

    ranked = sorted(tracks.values(), key=lambda t: t.distance_m, reverse=True)[:8]
    y = 180
    draw_text_shadow(card, "TOP DISTANCE COVERED", (40, y - 18),
                     font_scale=0.72, color_bgr=(120, 230, 120), thickness=2)
    for rank, t in enumerate(ranked, 1):
        color = _team_color(t.team)
        line = (
            f"{rank:2d}.  #{t.track_id:<4d} {t.distance_m:6.1f} m"
            f"   peak {format_speed_kmh(t.top_speed_ms)}"
        )
        draw_text_shadow(card, line, (44, y + rank * 40),
                         font_scale=0.68, color_bgr=color, thickness=2)

    fastest = max(tracks.values(), key=lambda t: t.top_speed_ms, default=None)
    if fastest is not None:
        draw_text_shadow(
            card,
            f"FASTEST SPRINT:  #{fastest.track_id}  {format_speed_kmh(fastest.top_speed_ms)}",
            (44, y + 9 * 40 + 30), font_scale=0.8, color_bgr=(40, 220, 240), thickness=2,
        )
    return draw_branding_tag(card)


def render_demo(
    sequence: SoccerNetSequence,
    detections_iter,
    tracks: dict[int, PlayerTrack],
    out_path: str,
    *,
    frame_loader,
    calibration: str = "height",
    leaderboard_seconds: float = 4.0,
    frame_transforms: dict | None = None,
    show_radar: bool = True,
    frame_keypoints: dict | None = None,
    pitch_kp_debug: bool = False,
    pitch_confidence: float = 0.98,
    pitch_tracker=None,
) -> dict:
    """Render the speed/distance MP4. ``frame_loader(frame_idx) -> bgr image``."""
    frames_list = list(detections_iter)
    locked_goals: tuple[int, int] | None = None
    if pitch_tracker is not None and frame_transforms is not None:
        locked_goals = warmup_goal_defenders(
            pitch_tracker, frames_list, frame_transforms
        )

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        out_path, fourcc, sequence.frame_rate, (sequence.width, sequence.height)
    )
    for frame_idx, dets in frames_list:
        image = frame_loader(frame_idx)
        if image is None:
            image = np.full((sequence.height, sequence.width, 3), 30, np.uint8)

        labels = _player_labels(dets, tracks, frame_idx)
        image = annotate_players(image, dets, labels=labels)
        image = annotate_ball(image, dets)
        kps = frame_keypoints.get(frame_idx) if frame_keypoints is not None else None
        h_t = (
            frame_transforms.get(frame_idx)
            if frame_transforms is not None
            else None
        )
        if pitch_tracker is not None and h_t is not None and locked_goals is None:
            from world_cup_projects.common.soccernet import ROLE_PLAYER

            omask = dets.class_id == ROLE_PLAYER
            if omask.any():
                pitch_m = image_to_pitch_m(feet_xy(dets)[omask], h_t)
                if pitch_m is not None:
                    teams = dets.data.get("team", np.zeros(len(dets), dtype=int))[
                        omask
                    ]
                    if pitch_tracker.register_reliable_goal_vote(pitch_m, teams):
                        locked_goals = pitch_tracker.locked_goal_defenders
        if show_radar and kps is not None:
            image = draw_radar_minimap(
                image,
                dets,
                kps,
                pitch_confidence=pitch_confidence,
                transformer=h_t,
                locked_goal_defenders=locked_goals,
                debug_keypoints=pitch_kp_debug,
            )
        if pitch_kp_debug and frame_keypoints is not None:
            image = draw_pitch_keypoints_debug(
                image,
                kps,
                confidence_threshold=pitch_confidence,
            )
            image = draw_homography_feet_debug(
                image, dets, kps, confidence_threshold=pitch_confidence
            )
        title = (
            "PLAYER SPEED & DISTANCE  ·  PITCH DEBUG"
            if pitch_kp_debug
            else "PLAYER SPEED & DISTANCE"
        )
        image = draw_hud_bar(image, title)
        image = draw_branding_tag(image)
        writer.write(image)

    card = _leaderboard_card(
        (sequence.width, sequence.height), tracks, calibration=calibration
    )
    for _ in range(int(leaderboard_seconds * sequence.frame_rate)):
        writer.write(card)
    writer.release()

    ranked = sorted(tracks.values(), key=lambda t: t.distance_m, reverse=True)[:5]
    return {
        "sequence": sequence.name,
        "output": out_path,
        "calibration": calibration,
        "n_tracks": len(tracks),
        "top_distance": [
            {
                "id": t.track_id,
                "distance_m": round(t.distance_m, 1),
                "top_speed_kmh": round(t.top_speed_ms * MS_TO_KMH, 1),
                "top_speed_ms": round(t.top_speed_ms, 2),
            }
            for t in ranked
        ],
    }
