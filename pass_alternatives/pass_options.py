"""Score teammate passing lanes for a ball carrier.

Each candidate lane (carrier -> teammate) is scored on three football-sense axes:

* **openness** - how far the nearest blocker is from the pass line (opponent or teammate).
* **forward progress** - how much the pass advances toward the attacking direction.
* **receiver space** - how much room the receiver has from the nearest opponent.

A short-range/very-long-range penalty keeps the suggestions realistic. Scores are
normalized to ``[0, 1]`` against tunable reference distances (in the same units as the
input coordinates: pixels for v1, meters for v2), then combined with weights.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import supervision as sv

from world_cup_projects.common.geometry import point_to_segment_distance, unit
from world_cup_projects.common.possession import Carrier, feet_xy, player_mask


@dataclass(frozen=True)
class PassWeights:
    openness: float = 0.45
    forward: float = 0.30
    space: float = 0.25
    # reference scales for normalization (pixels for v1)
    open_ref: float = 120.0
    space_ref: float = 120.0
    forward_ref: float = 600.0
    min_length: float = 60.0
    max_length: float = 1100.0
    length_penalty: float = 0.25

    @classmethod
    def metric(cls) -> PassWeights:
        """Reference distances in pitch meters (homography-calibrated scoring).

        Tuned so lane scores spread across [0, 1]: a *very* open lane keeps the
        nearest opponent ~8 m off the pass line, a tight one ~1-2 m; a strong
        forward pass advances ~25 m toward the opponent goal.
        """
        return cls(
            open_ref=8.0,
            space_ref=8.0,
            forward_ref=25.0,
            min_length=2.0,
            max_length=45.0,
        )


@dataclass(frozen=True)
class PassOption:
    receiver_index: int
    receiver_xy: np.ndarray
    score: float
    openness: float        # effective lane clearance (min opp / teammate distances)
    opponent_openness: float
    teammate_openness: float
    forward_gain: float    # raw projection onto attack direction
    receiver_space: float  # raw distance, receiver to nearest opponent
    length: float


def attack_direction(
    detections: sv.Detections, carrier: Carrier
) -> np.ndarray:
    """Image-space proxy for the carrier team's attacking direction.

    Without pitch calibration (v1) we point from the carrier-team centroid toward the
    opponent-team centroid: teams attack toward where the opposition is massed / their
    goal. v2 replaces this with the true direction to the opponent goal via homography.
    """
    pmask = player_mask(detections)
    feet = feet_xy(detections)
    teams = detections.data["team"]
    own = pmask & (teams == carrier.team)
    opp = pmask & (teams == (1 - carrier.team))
    if not own.any() or not opp.any():
        return np.array([1.0, 0.0])
    return unit(feet[opp].mean(axis=0) - feet[own].mean(axis=0))


def score_pass_options(
    detections: sv.Detections,
    carrier: Carrier,
    *,
    weights: PassWeights = PassWeights(),
    attack_dir: np.ndarray | None = None,
    positions: np.ndarray | None = None,
) -> list[PassOption]:
    """Rank every teammate as a passing option (best first).

    Pass *positions* (e.g. pitch coordinates in meters) to score in metric space
    instead of image pixels.
    """
    pmask = player_mask(detections)
    feet = positions if positions is not None else feet_xy(detections)
    teams = detections.data["team"]

    teammates = pmask & (teams == carrier.team)
    teammates[carrier.index] = False
    opponents = pmask & (teams == (1 - carrier.team))

    carrier_xy = feet[carrier.index]
    opp_xy = feet[opponents]
    attack = attack_direction(detections, carrier) if attack_dir is None else attack_dir

    options: list[PassOption] = []
    for idx in np.flatnonzero(teammates):
        receiver_xy = feet[idx]
        delta = receiver_xy - carrier_xy
        length = float(np.linalg.norm(delta))
        if length < 1e-6:
            continue

        if len(opp_xy):
            opponent_openness = float(
                point_to_segment_distance(opp_xy, carrier_xy, receiver_xy).min()
            )
            receiver_space = float(np.linalg.norm(opp_xy - receiver_xy, axis=1).min())
        else:
            opponent_openness = receiver_space = weights.open_ref

        blockers = teammates.copy()
        blockers[carrier.index] = False
        blockers[idx] = False
        team_xy = feet[blockers]
        if len(team_xy):
            teammate_openness = float(
                point_to_segment_distance(team_xy, carrier_xy, receiver_xy).min()
            )
        else:
            teammate_openness = weights.open_ref

        openness = min(opponent_openness, teammate_openness)

        forward_gain = float(delta @ attack)

        n_open = min(openness / weights.open_ref, 1.0)
        n_space = min(receiver_space / weights.space_ref, 1.0)
        n_forward = float(np.clip(forward_gain / weights.forward_ref, -1.0, 1.0))

        score = (
            weights.openness * n_open
            + weights.forward * max(n_forward, 0.0)
            + weights.space * n_space
        )
        if length < weights.min_length or length > weights.max_length:
            score -= weights.length_penalty

        options.append(
            PassOption(
                receiver_index=int(idx),
                receiver_xy=receiver_xy,
                score=float(score),
                openness=openness,
                opponent_openness=opponent_openness,
                teammate_openness=teammate_openness,
                forward_gain=forward_gain,
                receiver_space=receiver_space,
                length=length,
            )
        )

    options.sort(key=lambda o: o.score, reverse=True)
    return options


def top_pass_options(
    detections: sv.Detections,
    carrier: Carrier,
    *,
    k: int = 3,
    weights: PassWeights = PassWeights(),
    attack_dir: np.ndarray | None = None,
    positions: np.ndarray | None = None,
) -> list[PassOption]:
    return score_pass_options(
        detections,
        carrier,
        weights=weights,
        attack_dir=attack_dir,
        positions=positions,
    )[:k]
