#!/usr/bin/env python3
"""Trace team-1 possession state around the #3→#27 pass."""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

_pkg_root = Path(__file__).resolve().parent
_repo_root = _pkg_root.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from world_cup_projects import PACKAGE_ROOT
from world_cup_projects.common.pipeline import load_detections_source
from world_cup_projects.common.pitch import load_pitch_homography_cache
from world_cup_projects.common.possession_touch import is_valid_possession_touch
from world_cup_projects.common.video import load_video_sequence
from world_cup_projects.player_stats.pass_events import (
    PassDetectionConfig,
    _TeamPossessionState,
    _active_carrier,
    _min_control_frames_for,
    _on_confirmed_possession,
    _promote_pre_flight_release,
    ball_xy,
)


def main() -> None:
    start, end = 450, 525
    sequence = load_video_sequence("bundesliga_videos/08fd33_0.mp4")
    config = PassDetectionConfig().for_frame_rate(sequence.frame_rate)

    maps = load_pitch_homography_cache(
        sequence.name,
        end=sequence.length,
        device="cpu",
        pitch_confidence=0.99,
    )
    if maps is None:
        cache_path = PACKAGE_ROOT / ".cache/pitch/08fd33_0_cpu_750_0.99.pkl"
        with open(cache_path, "rb") as f:
            frame_transforms = pickle.load(f)["transforms"]
    else:
        frame_transforms = maps.transforms

    class Args:
        source = "football"
        device = "cpu"
        refresh_detections_cache = False
        pitch_confidence = 0.5
        tracker = "botsort"

    detections_source = load_detections_source(Args(), sequence)
    frames = [
        (fi, d)
        for fi, d in detections_source(sequence, start=1, end=sequence.length)
        if start <= fi <= end
    ]

    state = _TeamPossessionState()

    def update_control_streak(s, tid):
        if tid == s.possession_tid:
            s.control_streak += 1
        else:
            s.possession_tid = tid
            s.control_streak = 1

    print(f"{'frame':>5} {'tid':>4} {'kind':>9} {'valid':>5} {'release':>8} {'last':>6} {'in_fl':>5} {'arr':>4} {'str':>3}")
    for frame_idx, dets in frames:
        transformer = frame_transforms.get(frame_idx)
        carrier, touch_kind = _active_carrier(dets, transformer=transformer, config=config)
        touch_kind = touch_kind or "reception"
        valid = False
        tid = -1
        if carrier is not None:
            valid = is_valid_possession_touch(
                dets,
                carrier,
                touch_kind=touch_kind,
                config=config.touch_validation_config(),
            )
            if dets.tracker_id is not None:
                tid = int(dets.tracker_id[carrier.index])

        rel = state.release[3] if state.release else None
        rel_f = state.release[0] if state.release else None
        last = state.last_touch[3] if state.last_touch else None
        last_f = state.last_touch[0] if state.last_touch else None

        note = ""
        if carrier is None or not valid:
            if ball_xy(dets) is not None:
                if not state.in_flight:
                    before = state.release
                    _promote_pre_flight_release(state, frame_idx, config=config)
                    if state.release != before:
                        note = " PROMOTED"
                state.in_flight = True
        else:
            state.in_flight = False
            state.last_touch = (frame_idx, dets, carrier, tid)
            min_control = _min_control_frames_for(dets, carrier, config=config)
            if touch_kind == "control":
                update_control_streak(state, tid)
                if state.control_streak >= min_control:
                    _on_confirmed_possession(state, frame_idx, dets, carrier, tid)
                    note = " CONFIRM"
            else:
                state.possession_tid = tid
                state.control_streak = 0

            release = state.release
            if release is not None and tid != release[3]:
                if tid == state.arrival_candidate_tid:
                    state.arrival_streak += 1
                else:
                    state.arrival_candidate_tid = tid
                    state.arrival_streak = 1
                if state.arrival_streak >= config.min_arrival_frames:
                    note += f" ARRIVE({tid})"

            if release is not None and tid == release[3]:
                state.release = (frame_idx, dets, carrier, tid)
                note += " ADVANCE"

        rel = state.release[3] if state.release else None
        rel_f = state.release[0] if state.release else None
        if tid in (1, 3, 27) or rel in (1, 3, 27):
            print(
                f"{frame_idx:5d} {tid:4d} {touch_kind:>9} {str(valid):>5} "
                f"#{rel!s:>7}@{rel_f} #{last!s:>5}@{last_f} {state.in_flight!s:>5} "
                f"{state.arrival_candidate_tid:4d} {state.arrival_streak:3d}{note}"
            )


if __name__ == "__main__":
    main()
