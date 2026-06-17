"""Tests for one-touch reception release logic."""

from __future__ import annotations

import numpy as np
import pytest
import supervision as sv

from world_cup_projects.common.possession_touch import (
    ball_departed_for_one_touch,
    ball_redirected_at_touch,
    is_gravity_arc_flyby_at_touch,
)
from world_cup_projects.common.soccernet import ROLE_BALL, ROLE_PLAYER


def test_ball_departed_requires_movement_when_ball_visible():
    touch = np.array([100.0, 200.0])
    assert ball_departed_for_one_touch(
        touch,
        np.array([140.0, 200.0]),
        touch_frame=10,
        in_flight_frame=11,
        depart_min_px=35.0,
    )
    assert not ball_departed_for_one_touch(
        touch,
        np.array([110.0, 200.0]),
        touch_frame=10,
        in_flight_frame=11,
        depart_min_px=35.0,
    )


def test_ball_departed_allows_missing_ball_on_in_flight_frame():
    touch = np.array([100.0, 200.0])
    assert ball_departed_for_one_touch(
        touch,
        None,
        touch_frame=10,
        in_flight_frame=11,
        depart_min_px=35.0,
    )


def _player_ball_detections(
    *,
    passer_tid: int,
    receiver_tid: int,
    team: int,
    passer_feet: tuple[float, float],
    receiver_feet: tuple[float, float],
    ball: tuple[float, float],
) -> sv.Detections:
    def _box(feet_x: float, feet_y: float) -> np.ndarray:
        return np.array([[feet_x - 12, feet_y - 50, feet_x + 12, feet_y]], dtype=np.float32)

    passer_box = _box(*passer_feet)
    receiver_box = _box(*receiver_feet)
    ball_box = np.array(
        [[ball[0] - 6, ball[1] - 6, ball[0] + 6, ball[1]]],
        dtype=np.float32,
    )
    return sv.Detections(
        xyxy=np.vstack([passer_box, receiver_box, ball_box]),
        class_id=np.array([ROLE_PLAYER, ROLE_PLAYER, ROLE_BALL], dtype=int),
        tracker_id=np.array([passer_tid, receiver_tid, -1], dtype=int),
        data={"team": np.array([team, team, -1], dtype=int)},
    )


def test_one_touch_reception_pass_without_control():
    from world_cup_projects.player_stats.pass_events import (
        PassDetectionConfig,
        PassQualityScorer,
        scan_possession_events,
    )

    config = PassDetectionConfig(
        min_one_touch_reception_frames=2,
        one_touch_release_window_frames=6,
        one_touch_depart_min_px=25.0,
        min_ball_travel_px=10.0,
        min_ball_travel_m=0.1,
        min_arrival_frames=2,
        min_reception_arrival_frames=2,
        min_control_frames=2,
    )
    frames: list[tuple[int, sv.Detections]] = []
    # Two reception frames at the passer, ball rolls out, teammate collects.
    specs = [
        ((160.0, 215.0), (400.0, 220.0)),
        ((160.0, 215.0), (400.0, 220.0)),
        ((220.0, 215.0), (400.0, 220.0)),
        ((280.0, 215.0), (400.0, 220.0)),
        ((340.0, 215.0), (400.0, 220.0)),
        ((360.0, 215.0), (360.0, 220.0)),
        ((360.0, 215.0), (360.0, 220.0)),
        ((360.0, 215.0), (360.0, 220.0)),
    ]
    for fi, (ball, receiver_feet) in enumerate(specs, start=1):
        frames.append(
            (
                fi,
                _player_ball_detections(
                    passer_tid=10,
                    receiver_tid=20,
                    team=0,
                    passer_feet=(100.0, 220.0),
                    receiver_feet=receiver_feet,
                    ball=ball,
                ),
            )
        )

    result = scan_possession_events(
        iter(frames),
        scorer=PassQualityScorer(),
        config=config,
        metric=False,
    )
    assert len(result.passes) == 1
    assert result.passes[0].passer_tid == 10
    assert result.passes[0].receiver_tid == 20


def test_fast_flyby_reception_does_not_emit_one_touch_pass():
    from world_cup_projects.player_stats.pass_events import (
        PassDetectionConfig,
        PassQualityScorer,
        scan_possession_events,
    )

    config = PassDetectionConfig(
        min_one_touch_reception_frames=2,
        one_touch_release_window_frames=4,
        one_touch_depart_min_px=20.0,
    )
    frames: list[tuple[int, sv.Detections]] = []
    balls = [(100.0, 200.0), (180.0, 200.0), (260.0, 200.0), (340.0, 200.0)]
    for fi, ball in enumerate(balls, start=1):
        frames.append(
            (
                fi,
                _player_ball_detections(
                    passer_tid=10,
                    receiver_tid=20,
                    team=0,
                    passer_feet=(100.0, 220.0),
                    receiver_feet=(420.0, 220.0),
                    ball=ball,
                ),
            )
        )

    result = scan_possession_events(
        iter(frames),
        scorer=PassQualityScorer(),
        config=config,
        metric=False,
    )
    assert list(result.passes) == []


def test_ball_redirected_detects_direction_change():
    frames_by_idx: dict[int, sv.Detections] = {}
    for fi, ball in enumerate(
        [
            (100.0, 200.0),
            (130.0, 200.0),
            (160.0, 200.0),
            (190.0, 210.0),
            (170.0, 240.0),
            (140.0, 260.0),
        ],
        start=1,
    ):
        frames_by_idx[fi] = _player_ball_detections(
            passer_tid=10,
            receiver_tid=20,
            team=0,
            passer_feet=(100.0, 220.0),
            receiver_feet=(420.0, 220.0),
            ball=ball,
        )
    assert ball_redirected_at_touch(frames_by_idx, 4, lookback=3, lookahead=2)
    assert not ball_redirected_at_touch(frames_by_idx, 2, lookback=2, lookahead=2)


def test_gravity_arc_flyby_rejects_unredirected_touch():
    frames_by_idx: dict[int, sv.Detections] = {}
    balls = [
        (100.0, 180.0),
        (130.0, 200.0),
        (160.0, 230.0),
        (190.0, 270.0),
        (220.0, 320.0),
        (250.0, 380.0),
    ]
    for fi, ball in enumerate(balls, start=1):
        frames_by_idx[fi] = _player_ball_detections(
            passer_tid=10,
            receiver_tid=20,
            team=0,
            passer_feet=(100.0, 220.0),
            receiver_feet=(420.0, 220.0),
            ball=ball,
        )
    assert is_gravity_arc_flyby_at_touch(frames_by_idx, 3, lookback=2, lookahead=2)
