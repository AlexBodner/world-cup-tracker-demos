"""Render pass-network demo video with a stats end-card."""

from __future__ import annotations

import cv2
import numpy as np
import supervision as sv

from world_cup_projects.common.pitch import image_to_pitch_m, warmup_goal_defenders
from world_cup_projects.common.possession import bbox_center_xy, feet_xy
from world_cup_projects.common.soccernet import ROLE_PLAYER, SoccerNetSequence
from world_cup_projects.common.visual import (
    ROBOFLOW_PURPLE_BGR,
    TEAM_COLORS,
    annotate_ball,
    annotate_players,
    draw_branding_tag,
    draw_carrier_pulse,
    draw_glow_arrow,
    draw_hud_bar,
    draw_pitch_keypoints_debug,
    draw_radar_minimap,
    draw_text_shadow,
)
from world_cup_projects.player_stats.pass_events import InferredPass
from world_cup_projects.player_stats.pass_network import PassNetwork

TEAM_COLORS_BGR = [c.as_bgr() for c in TEAM_COLORS[:2]]
NEUTRAL_BGR = (200, 200, 200)


def _team_color(team: int) -> tuple[int, int, int]:
    return TEAM_COLORS_BGR[team] if team in (0, 1) else NEUTRAL_BGR


def _get_player_box(dets: sv.Detections, tid: int) -> np.ndarray | None:
    if dets.tracker_id is None:
        return None
    idx = np.flatnonzero(dets.tracker_id == tid)
    if len(idx) == 0:
        return None
    return dets.xyxy[idx[0]]


def _draw_ground_highlight(
    image: np.ndarray,
    box: np.ndarray,
    color_bgr: tuple[int, int, int],
    *,
    alpha: float = 1.0,
    scale: float = 1.0,
) -> None:
    """Draw a perspective-aware ellipse on the ground at the player's feet."""
    x0, y0, x1, y1 = box
    cx, cy = int((x0 + x1) / 2), int(y1)
    w = int((x1 - x0) * 0.8 * scale)
    h = int(w / 3)
    
    overlay = image.copy()
    # Fill the ellipse with a semi-transparent version of the team color
    cv2.ellipse(overlay, (cx, cy), (w, h), 0, 0, 360, color_bgr, -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha * 0.4, image, 1.0 - alpha * 0.4, 0, image)
    
    # Draw a solid outline
    overlay_outline = image.copy()
    cv2.ellipse(overlay_outline, (cx, cy), (w, h), 0, 0, 360, color_bgr, 3, cv2.LINE_AA)
    cv2.addWeighted(overlay_outline, alpha, image, 1.0 - alpha, 0, image)


def _draw_pass_highlights(
    image: np.ndarray,
    dets: sv.Detections,
    frame_idx: int,
    passes: tuple[InferredPass, ...],
    frame_rate: float,
) -> None:
    """Highlight active passes: pulse passer/receiver and draw the lane arrow."""
    for p in passes:
        # Window: from release frame to 0.5s after reception
        receive_idx = p.frame_idx + p.gap_frames
        twinkle_duration = int(frame_rate * 0.5)
        end_idx = receive_idx + twinkle_duration

        if p.frame_idx <= frame_idx <= end_idx:
            passer_box = _get_player_box(dets, p.passer_tid)
            receiver_box = _get_player_box(dets, p.receiver_tid)
            color = _team_color(p.team)

            if frame_idx <= receive_idx:
                # 1. Flight Phase: Pulse both players and draw arrow following the ball
                t = (frame_idx - p.frame_idx) / p.gap_frames if p.gap_frames > 0 else 1.0
                pulse_alpha = 0.5 + 0.3 * np.sin(t * np.pi * 4) # fast pulse
                
                if passer_box is not None:
                    _draw_ground_highlight(image, passer_box, color, alpha=pulse_alpha)
                
                if receiver_box is not None:
                    _draw_ground_highlight(image, receiver_box, color, alpha=pulse_alpha)

                if passer_box is not None and receiver_box is not None:
                    # Arrow from passer feet to actual ball position (or interpolated if ball missing)
                    p_feet = (int((passer_box[0] + passer_box[2]) / 2), int(passer_box[3]))
                    r_feet = (int((receiver_box[0] + receiver_box[2]) / 2), int(receiver_box[3]))
                    
                    from world_cup_projects.common.possession import ball_xy
                    ball_pos = ball_xy(dets)
                    
                    if ball_pos is not None:
                        current_tip_x, current_tip_y = int(ball_pos[0]), int(ball_pos[1])
                    else:
                        current_tip_x = int(p_feet[0] + (r_feet[0] - p_feet[0]) * t)
                        current_tip_y = int(p_feet[1] + (r_feet[1] - p_feet[1]) * t)
                        
                    current_tip = (current_tip_x, current_tip_y)
                    
                    # Only draw if the arrow has some length
                    if np.hypot(current_tip_x - p_feet[0], current_tip_y - p_feet[1]) > 5:
                        draw_glow_arrow(image, p_feet, current_tip, color, alpha=0.8)

            elif frame_idx <= end_idx:
                # 2. Reception Phase: Twinkle 2 times, arrow removed
                if receiver_box is not None:
                    # Twinkle logic: 2 cycles over twinkle_duration
                    prog = (frame_idx - receive_idx) / twinkle_duration
                    # 2 cycles of sine wave (0 to pi to 0 twice)
                    twinkle_val = np.sin(prog * 2 * 2 * np.pi) 
                    # Map -1..1 to 0..1 for twinkle intensity
                    twinkle_alpha = max(0.0, twinkle_val)
                    
                    # Also scale the ellipse slightly for the "twinkle" pop
                    twinkle_scale = 1.0 + 0.4 * twinkle_alpha
                    
                    _draw_ground_highlight(
                        image, 
                        receiver_box, 
                        color, 
                        alpha=twinkle_alpha * 0.9, 
                        scale=twinkle_scale
                    )


def _draw_collaboration_web(
    image: np.ndarray,
    dets: sv.Detections,
    frame_idx: int,
    passes: tuple[InferredPass, ...],
) -> None:
    """Draw a dynamic network graph on the pitch representing completed passes."""
    # 1. Aggregate connection strength for passes completed *before* this frame
    connections: dict[tuple[int, int], dict] = {}
    for p in passes:
        receive_idx = p.frame_idx + p.gap_frames
        if receive_idx <= frame_idx:
            # Undirected link between teammates
            pair = tuple(sorted([p.passer_tid, p.receiver_tid]))
            if pair not in connections:
                connections[pair] = {"count": 0, "team": p.team}
            connections[pair]["count"] += 1

    if not connections:
        return

    # 2. Draw lines for active pairs currently visible
    overlay = image.copy()
    max_count = max(c["count"] for c in connections.values())

    for (t1, t2), data in connections.items():
        box1 = _get_player_box(dets, t1)
        box2 = _get_player_box(dets, t2)
        
        if box1 is not None and box2 is not None:
            # Feet positions
            p1 = (int((box1[0] + box1[2]) / 2), int(box1[3]))
            p2 = (int((box2[0] + box2[2]) / 2), int(box2[3]))
            
            # Visual mapping based on interaction strength
            count = data["count"]
            intensity = min(count / max(max_count, 1), 1.0)
            
            # Base alpha 0.15 for 1 pass, scaling up to 0.6 for max passes
            alpha = 0.15 + (intensity * 0.45)
            # Thickness 1 to 4
            thickness = 1 + int(intensity * 3)
            color = _team_color(data["team"])
            
            # Draw line directly on the alpha-blended overlay
            temp_overlay = image.copy()
            cv2.line(temp_overlay, p1, p2, color, thickness, cv2.LINE_AA)
            cv2.addWeighted(temp_overlay, alpha, overlay, 1.0 - alpha, 0, overlay)

    # Blend the entire web back onto the main image
    image[:] = overlay


def _player_teams(network: PassNetwork) -> dict[int, int]:
    teams: dict[int, int] = {}
    for player in network.players:
        teams[player.tracker_id] = player.team
    for link in network.links:
        teams.setdefault(link.passer_tid, link.team)
        teams.setdefault(link.receiver_tid, link.team)
    return teams


def _collaboration_graph_panel(
    card: np.ndarray,
    network: PassNetwork,
    *,
    origin: tuple[int, int],
    size: tuple[int, int],
) -> None:
    """Draw directed pass links as a node graph (mutates ``card`` in place)."""
    ox, oy = origin
    gw, gh = size
    if gw < 80 or gh < 80:
        return

    overlay = card.copy()
    cv2.rectangle(overlay, (ox, oy), (ox + gw, oy + gh), (28, 28, 34), -1)
    cv2.rectangle(overlay, (ox, oy), (ox + gw, oy + gh), (55, 55, 65), 1)
    card[:] = cv2.addWeighted(overlay, 0.92, card, 0.08, 0)

    draw_text_shadow(
        card,
        "COLLABORATION GRAPH",
        (ox + 14, oy + 28),
        font_scale=0.55,
        color_bgr=(200, 200, 210),
        thickness=1,
    )

    if not network.links:
        draw_text_shadow(
            card,
            "no inferred links",
            (ox + 14, oy + gh // 2),
            font_scale=0.5,
            color_bgr=(140, 140, 150),
            thickness=1,
        )
        return

    teams = _player_teams(network)
    nodes = sorted({link.passer_tid for link in network.links} | {link.receiver_tid for link in network.links})
    cx, cy = ox + gw // 2, oy + gh // 2 + 12
    radius = min(gw, gh) * 0.34
    positions: dict[int, tuple[int, int]] = {}
    for i, tid in enumerate(nodes):
        angle = 2.0 * np.pi * i / max(len(nodes), 1) - np.pi / 2
        positions[tid] = (
            int(cx + radius * np.cos(angle)),
            int(cy + radius * np.sin(angle)),
        )

    max_count = max(link.count for link in network.links)
    for link in network.links:
        p0 = positions.get(link.passer_tid)
        p1 = positions.get(link.receiver_tid)
        if p0 is None or p1 is None:
            continue
        thickness = 1 + int(3 * link.count / max(max_count, 1))
        color = _team_color(link.team)
        cv2.arrowedLine(
            card, p0, p1, (20, 20, 24), thickness + 2, cv2.LINE_AA, tipLength=0.22
        )
        cv2.arrowedLine(
            card, p0, p1, color, thickness, cv2.LINE_AA, tipLength=0.22
        )
        mid = ((p0[0] + p1[0]) // 2, (p0[1] + p1[1]) // 2)
        draw_text_shadow(
            card,
            str(link.count),
            (mid[0] - 6, mid[1] - 4),
            font_scale=0.38,
            color_bgr=(230, 230, 230),
            thickness=1,
        )

    for tid, pos in positions.items():
        team = teams.get(tid, -1)
        color = _team_color(team)
        cv2.circle(card, pos, 16, (16, 16, 20), -1, cv2.LINE_AA)
        cv2.circle(card, pos, 16, color, 2, cv2.LINE_AA)
        draw_text_shadow(
            card,
            f"#{tid}",
            (pos[0] - 14, pos[1] + 5),
            font_scale=0.42,
            color_bgr=(255, 255, 255),
            thickness=1,
        )


def _stats_end_card(size: tuple[int, int], network: PassNetwork) -> np.ndarray:
    """Full-frame summary of collaboration links and player pass volume."""
    w, h = size
    card = np.full((h, w, 3), 22, np.uint8)
    cv2.rectangle(card, (0, 0), (w, 6), ROBOFLOW_PURPLE_BGR, -1)

    draw_text_shadow(
        card,
        "PASS NETWORK",
        (40, 72),
        font_scale=1.1,
        color_bgr=ROBOFLOW_PURPLE_BGR,
        thickness=2,
    )
    mode = "metric lanes" if network.metric else "image-space lanes"
    draw_text_shadow(
        card,
        f"{network.n_passes} inferred passes  |  {mode}",
        (42, 108),
        font_scale=0.55,
        color_bgr=(170, 170, 170),
        thickness=1,
    )

    y_links = 170
    draw_text_shadow(
        card,
        "TOP COLLABORATORS",
        (40, y_links),
        font_scale=0.72,
        color_bgr=(120, 230, 120),
        thickness=2,
    )
    for rank, link in enumerate(network.links[:6], 1):
        color = _team_color(link.team)
        line = (
            f"{rank}.  #{link.passer_tid} -> #{link.receiver_tid}"
            f"   {link.count} passes  |  Avg Quality: {link.avg_quality:.2f}"
        )
        draw_text_shadow(
            card,
            line,
            (44, y_links + rank * 38),
            font_scale=0.62,
            color_bgr=color,
            thickness=2,
        )

    y_players = y_links + 8 * 38 + 20
    draw_text_shadow(
        card,
        "MOST ACTIVE PLAYERS",
        (40, y_players),
        font_scale=0.72,
        color_bgr=(40, 220, 240),
        thickness=2,
    )
    for rank, player in enumerate(network.players[:6], 1):
        color = _team_color(player.team)
        line = (
            f"{rank}.  #{player.tracker_id}"
            f"   Made: {player.passes_made}  |  Received: {player.passes_received}"
            f"  |  Avg Quality Made: {player.avg_quality_made:.2f}"
        )
        draw_text_shadow(
            card,
            line,
            (44, y_players + rank * 38),
            font_scale=0.62,
            color_bgr=color,
            thickness=2,
        )

    graph_x = int(w * 0.52)
    _collaboration_graph_panel(
        card,
        network,
        origin=(graph_x, 150),
        size=(w - graph_x - 40, h - 220),
    )

    if network.links:
        best = network.links[0]
        draw_text_shadow(
            card,
            f"STRONGEST LINK:  #{best.passer_tid} -> #{best.receiver_tid}"
            f"  ({best.count} passes, Avg Quality: {best.avg_quality:.2f})",
            (44, h - 80),
            font_scale=0.72,
            color_bgr=(255, 220, 80),
            thickness=2,
        )

    return draw_branding_tag(card)


def render_pass_network_demo(
    sequence: SoccerNetSequence,
    frames: list[tuple[int, sv.Detections]],
    network: PassNetwork,
    out_path: str,
    *,
    frame_loader,
    metric: bool = False,
    show_radar: bool = True,
    frame_transforms: dict | None = None,
    frame_keypoints: dict | None = None,
    pitch_confidence: float = 0.99,
    pitch_tracker=None,
    stats_seconds: float = 5.0,
    debug_pitch_keypoints: bool = False,
) -> dict:
    """Render tracked clip plus a stats end-card."""
    locked_goals: tuple[int, int] | None = None
    if pitch_tracker is not None and frame_transforms is not None:
        locked_goals = warmup_goal_defenders(pitch_tracker, frames, frame_transforms)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        out_path, fourcc, sequence.frame_rate, (sequence.width, sequence.height)
    )

    for frame_idx, dets in frames:
        image = frame_loader(frame_idx)
        if image is None:
            image = np.full((sequence.height, sequence.width, 3), 30, np.uint8)

        image = annotate_players(image, dets, show_tracker_ids=True)
        image = annotate_ball(image, dets)

        _draw_collaboration_web(image, dets, frame_idx, network.passes)

        _draw_pass_highlights(
            image, dets, frame_idx, network.passes, sequence.frame_rate
        )

        kps = frame_keypoints.get(frame_idx) if frame_keypoints is not None else None
        transformer = (
            frame_transforms.get(frame_idx) if frame_transforms is not None else None
        )
        if pitch_tracker is not None and transformer is not None and locked_goals is None:
            omask = dets.class_id == ROLE_PLAYER
            if omask.any():
                pitch_m = image_to_pitch_m(feet_xy(dets)[omask], transformer)
                if pitch_m is not None:
                    teams = dets.data.get("team", np.zeros(len(dets), dtype=int))[omask]
                    if pitch_tracker.register_reliable_goal_vote(pitch_m, teams):
                        locked_goals = pitch_tracker.locked_goal_defenders

        if debug_pitch_keypoints and kps is not None:
            image = draw_pitch_keypoints_debug(
                image, kps, confidence_threshold=pitch_confidence
            )

        if show_radar and metric and kps is not None:
            image = draw_radar_minimap(
                image,
                dets,
                kps,
                pitch_confidence=pitch_confidence,
                transformer=transformer,
                locked_goal_defenders=locked_goals,
                debug_keypoints=debug_pitch_keypoints,
            )

        title = (
            "PASS NETWORK  |  PITCH KP DEBUG"
            if debug_pitch_keypoints
            else "PASS NETWORK"
        )
        image = draw_hud_bar(image, title)
        image = draw_branding_tag(image)
        writer.write(image)

    card = _stats_end_card((sequence.width, sequence.height), network)
    for _ in range(int(stats_seconds * sequence.frame_rate)):
        writer.write(card)
    writer.release()

    return {
        "sequence": sequence.name,
        "output": out_path,
        "metric": metric,
        "n_passes": network.n_passes,
        "top_collaborators": [link.to_dict() for link in network.links[:5]],
        "top_players": [player.to_dict() for player in network.players[:5]],
    }
