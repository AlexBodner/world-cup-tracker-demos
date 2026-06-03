"""Auto-select the best SoccerNet clips for the demos.

A "good" pass-alternatives clip has: the ball visible most of the time, frequent clear
possession (ball glued to a player's feet), both teams well represented, and plenty of
players on screen (so there are real passing options). We score every sequence on those
axes and rank them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from world_cup_projects.common.possession import (
    ball_xy,
    find_ball_carrier,
    player_mask,
)
from world_cup_projects.common.soccernet import iter_gt_detections, load_sequence


@dataclass(frozen=True)
class ClipScore:
    name: str
    path: Path
    frames: int
    ball_frames: int
    carrier_frames: int
    avg_players: float
    both_teams: bool
    score: float

    @property
    def reason(self) -> str:
        return (
            f"ball visible {self.ball_frames}/{self.frames} frames, "
            f"clear possession in {self.carrier_frames} frames, "
            f"~{self.avg_players:.0f} players/frame, "
            f"{'both teams' if self.both_teams else 'single team'}"
        )


def score_sequence(seq_dir: Path, *, carrier_max_distance: float = 80.0) -> ClipScore:
    seq = load_sequence(seq_dir)
    ball_frames = carrier_frames = 0
    player_counts: list[int] = []
    teams: set[int] = set()

    for _, dets in iter_gt_detections(seq):
        player_counts.append(int(player_mask(dets).sum()))
        if find_ball_carrier(dets, max_distance_px=carrier_max_distance) is not None:
            carrier_frames += 1
        if ball_xy(dets) is not None:
            ball_frames += 1
        teams.update(int(t) for t in dets.data["team"] if t in (0, 1))

    avg_players = float(np.mean(player_counts)) if player_counts else 0.0
    both_teams = {0, 1}.issubset(teams)
    # Possession density dominates; reward player count, require both teams.
    score = (
        carrier_frames
        + 0.3 * ball_frames
        + 5.0 * avg_players
        + (200.0 if both_teams else 0.0)
    )
    return ClipScore(
        name=seq.name,
        path=Path(seq_dir),
        frames=seq.length,
        ball_frames=ball_frames,
        carrier_frames=carrier_frames,
        avg_players=avg_players,
        both_teams=both_teams,
        score=score,
    )


def rank_clips(seq_dirs: list[Path]) -> list[ClipScore]:
    scores = [score_sequence(p) for p in seq_dirs]
    scores.sort(key=lambda s: s.score, reverse=True)
    return scores
