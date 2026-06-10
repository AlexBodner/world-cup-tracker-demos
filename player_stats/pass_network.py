"""Aggregate inferred passes into collaboration links and player summaries."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass

import numpy as np

from world_cup_projects.player_stats.pass_events import InferredPass, InferredTurnover


@dataclass(frozen=True)
class CollaborationLink:
    """Directed pass count between two teammates."""

    passer_tid: int
    receiver_tid: int
    team: int
    count: int
    avg_quality: float | None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class PlayerPassSummary:
    """Pass activity for one tracked player."""

    tracker_id: int
    team: int
    passes_made: int
    passes_received: int
    avg_quality_made: float | None
    avg_quality_received: float | None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class PassNetwork:
    """Full v1 pass-interaction snapshot for a sequence."""

    sequence: str
    metric: bool
    n_passes: int
    n_turnovers: int
    passes: tuple[InferredPass, ...]
    turnovers: tuple[InferredTurnover, ...]
    links: tuple[CollaborationLink, ...]
    players: tuple[PlayerPassSummary, ...]

    def to_dict(self) -> dict:
        return {
            "sequence": self.sequence,
            "metric": self.metric,
            "n_passes": self.n_passes,
            "n_turnovers": self.n_turnovers,
            "passes": [p.to_dict() for p in self.passes],
            "turnovers": [t.to_dict() for t in self.turnovers],
            "collaboration_links": [link.to_dict() for link in self.links],
            "player_summaries": [player.to_dict() for player in self.players],
            "top_collaborators": [
                link.to_dict() for link in sorted(self.links, key=lambda l: l.count, reverse=True)[:10]
            ],
        }


def build_collaboration_links(events: list[InferredPass]) -> list[CollaborationLink]:
    """Count directed A -> B passes and average lane quality per link."""
    counts: dict[tuple[int, int, int], int] = defaultdict(int)
    quality_sums: dict[tuple[int, int, int], float] = defaultdict(float)
    quality_counts: dict[tuple[int, int, int], int] = defaultdict(int)

    for event in events:
        key = (event.passer_tid, event.receiver_tid, event.team)
        counts[key] += 1
        if event.quality_score is not None:
            quality_sums[key] += event.quality_score
            quality_counts[key] += 1

    links: list[CollaborationLink] = []
    for (passer, receiver, team), count in counts.items():
        scored = quality_counts[(passer, receiver, team)]
        links.append(
            CollaborationLink(
                passer_tid=passer,
                receiver_tid=receiver,
                team=team,
                count=count,
                avg_quality=(
                    quality_sums[(passer, receiver, team)] / scored
                    if scored
                    else None
                ),
            )
        )
    links.sort(key=lambda link: link.count, reverse=True)
    return links


def build_player_summaries(events: list[InferredPass]) -> list[PlayerPassSummary]:
    """Per-player pass counts and average quality as passer vs receiver."""
    teams: dict[int, int] = {}
    made_counts: dict[int, int] = defaultdict(int)
    recv_counts: dict[int, int] = defaultdict(int)
    made_quality: dict[int, list[float | None]] = defaultdict(list)
    recv_quality: dict[int, list[float | None]] = defaultdict(list)

    for event in events:
        teams[event.passer_tid] = event.team
        teams[event.receiver_tid] = event.team
        made_counts[event.passer_tid] += 1
        recv_counts[event.receiver_tid] += 1
        if event.quality_score is not None:
            made_quality[event.passer_tid].append(event.quality_score)
        if event.quality_score is not None:
            recv_quality[event.receiver_tid].append(event.quality_score)

    summaries: list[PlayerPassSummary] = []
    for tid in sorted(teams):
        summaries.append(
            PlayerPassSummary(
                tracker_id=tid,
                team=teams[tid],
                passes_made=made_counts[tid],
                passes_received=recv_counts[tid],
                avg_quality_made=_mean(made_quality[tid]),
                avg_quality_received=_mean(recv_quality[tid]),
            )
        )
    summaries.sort(key=lambda row: row.passes_made + row.passes_received, reverse=True)
    return summaries


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def build_pass_network(
    sequence_name: str,
    events: list[InferredPass],
    turnovers: list[InferredTurnover] | None = None,
    *,
    metric: bool,
) -> PassNetwork:
    """Build the v1 collaboration snapshot from inferred pass events."""
    turnovers = turnovers or []
    links = build_collaboration_links(events)
    players = build_player_summaries(events)
    return PassNetwork(
        sequence=sequence_name,
        metric=metric,
        n_passes=len(events),
        n_turnovers=len(turnovers),
        passes=tuple(events),
        turnovers=tuple(turnovers),
        links=tuple(links),
        players=tuple(players),
    )
