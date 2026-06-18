#!/usr/bin/env python3
"""Debug carrier touches around frames 486-539 for 08fd33_0.mp4."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import supervision as sv

from world_cup_projects.common.possession import (
    ball_xy,
    feet_xy,
    find_control_carrier,
    find_reception_carrier,
    player_mask,
)
from world_cup_projects.common.possession_touch import (
    is_aerial_touch,
    is_valid_possession_touch,
    nearest_player_tid,
)
from world_cup_projects.common.video import load_video_sequence
from world_cup_projects.player_stats.pass_events import PassDetectionConfig, _active_carrier
import pickle

from world_cup_projects import PACKAGE_ROOT
from world_cup_projects.common.pipeline import load_detections_source
from world_cup_projects.common.pitch import load_pitch_homography_cache


def main() -> None:
    start, end = 430, 560
    sequence = load_video_sequence("bundesliga_videos/08fd33_0.mp4")
    config = PassDetectionConfig().for_frame_rate(sequence.frame_rate)

    # Load cached pitch transforms
    cache_path = PACKAGE_ROOT / ".cache/pitch" / f"{sequence.name}_cpu_{sequence.length}_0.5.pkl"
    if not cache_path.exists():
        pitch_dir = PACKAGE_ROOT / ".cache/pitch"
        for p in pitch_dir.glob(f"{sequence.name}_*"):
            cache_path = p
            break
    frame_transforms = {}
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            frame_transforms = pickle.load(f)["transforms"]
        print(f"Using pitch cache: {cache_path.name}")

    class Args:
        source = "football"
        device = "cpu"
        refresh_detections_cache = False
        pitch_confidence = 0.5
        tracker = "botsort"

    detections_source = load_detections_source(Args(), sequence)
    frames = list(detections_source(sequence, start=1, end=sequence.length))
    frames = [(fi, d) for fi, d in frames if start <= fi <= end]

    print(f"\n{'frame':>5} {'nearest':>7} {'ctrl':>5} {'recv':>5} {'active':>6} {'kind':>9} "
          f"{'valid':>5} {'aerial':>6} {'dy':>6} {'dist_px':>7}")
    print("-" * 80)

    for frame_idx, dets in frames:
        transformer = frame_transforms.get(frame_idx)
        ball = ball_xy(dets)
        if ball is None:
            print(f"{frame_idx:5d}  NO BALL")
            continue

        pmask = player_mask(dets)
        feet = feet_xy(dets)[pmask]
        tids = dets.tracker_id[pmask]
        dists = np.hypot(feet[:, 0] - ball[0], feet[:, 1] - ball[1])
        nearest_local = int(np.argmin(dists))
        nearest_tid = int(tids[nearest_local])
        nearest_dist = float(dists[nearest_local])

        carrier, touch_kind = _active_carrier(dets, transformer=transformer, config=config)
        touch_kind = touch_kind or "reception"

        active_tid = -1
        aerial = False
        dy = 0.0
        valid = False
        ctrl_tid = -1
        recv_tid = -1

        if carrier is not None and dets.tracker_id is not None:
            active_tid = int(dets.tracker_id[carrier.index])
            touch_cfg = config.touch_validation_config()
            aerial = is_aerial_touch(
                dets, carrier, threshold_px=touch_cfg.aerial_dy_threshold_px
            )
            feet_pt = feet_xy(dets)[carrier.index]
            dy = float(ball[1] - feet_pt[1])
            valid = is_valid_possession_touch(
                dets, carrier, touch_kind=touch_kind, config=touch_cfg
            )

        control = find_control_carrier(
            dets,
            max_distance_px=config.control_max_distance_px,
            transformer=transformer,
            max_distance_m=config.control_max_distance_m,
        )
        reception = find_reception_carrier(
            dets,
            max_distance_px=config.reception_max_distance_px,
            transformer=transformer,
            max_distance_m=config.reception_max_distance_m,
        )
        if control is not None and dets.tracker_id is not None:
            ctrl_tid = int(dets.tracker_id[control.index])
        if reception is not None and dets.tracker_id is not None:
            recv_tid = int(dets.tracker_id[reception.index])

        # Highlight players 1, 3, 27
        marker = ""
        if active_tid in (1, 3, 27) or nearest_tid in (1, 3, 27):
            marker = " *"

        print(
            f"{frame_idx:5d} #{nearest_tid:5d} #{ctrl_tid:4d} #{recv_tid:4d} "
            f"#{active_tid:5d} {touch_kind:>9} {str(valid):>5} {str(aerial):>6} "
            f"{dy:6.1f} {nearest_dist:7.1f}{marker}"
        )

        # Detail for player #1 when they're active or reception
        if active_tid == 1 or (nearest_tid == 1 and active_tid >= 0):
            nearest_check = nearest_player_tid(dets, ball)
            print(
                f"       #1 detail: active={active_tid} nearest_check={nearest_check} "
                f"kind={touch_kind} valid={valid} aerial={aerial} dy={dy:.1f}"
            )


if __name__ == "__main__":
    main()
