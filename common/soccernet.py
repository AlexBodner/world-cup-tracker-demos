"""Shared SoccerNet sequence loader for the World Cup tracker demos.

SoccerNet game-state / tracking sequences live under::

    <SOCCERNET_TRACKING_ROOT>/<split>/SNMOT-XXX/
        seqinfo.ini      # resolution, frame rate, length
        gameinfo.ini     # tracklet_id -> role / team / jersey mapping
        img1/000001.jpg  # 6-digit, 1-indexed frames
        gt/gt.txt        # MOT: frame,track_id,x,y,w,h,conf,-1,-1,-1
        det/det.txt      # detector boxes (track_id == -1)

Set ``SOCCERNET_TRACKING_ROOT`` to your tracking split root. When unset, the
loader uses the Roboflow monorepo mirror if present, otherwise
``data/soccernet/tracking``.

The ground-truth tracks already carry stable IDs plus role/team/jersey labels
(via ``gameinfo.ini``), so the demos can run end-to-end without any model
weights. Swap :func:`iter_gt_detections` for an RF-DETR + ByteTrack pipeline
when you want the "from raw pixels" version.
"""

from __future__ import annotations

import configparser
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import supervision as sv

_MONOREPO_TRACKING = Path("trackers metrics/soccernet/soccernet_data/tracking")
_STANDALONE_TRACKING = Path("data/soccernet/tracking")


def _default_tracking_root() -> str:
    env = os.environ.get("SOCCERNET_TRACKING_ROOT")
    if env:
        return env
    if _MONOREPO_TRACKING.is_dir():
        return str(_MONOREPO_TRACKING)
    return str(_STANDALONE_TRACKING)


DEFAULT_TRACKING_ROOT = _default_tracking_root()

# Roles encoded as class ids so they survive inside sv.Detections.class_id.
ROLE_PLAYER = 0
ROLE_GOALKEEPER = 1
ROLE_REFEREE = 2
ROLE_BALL = 3

ROLE_NAMES = {
    ROLE_PLAYER: "player",
    ROLE_GOALKEEPER: "goalkeeper",
    ROLE_REFEREE: "referee",
    ROLE_BALL: "ball",
}

# team: 0 = left, 1 = right, -1 = not a team object (ball / referee)
TEAM_LEFT = 0
TEAM_RIGHT = 1
TEAM_NONE = -1


@dataclass(frozen=True)
class TrackletMeta:
    """Parsed ``gameinfo.ini`` entry for a single ground-truth track id."""

    role: int
    team: int
    jersey: str


@dataclass(frozen=True)
class SoccerNetSequence:
    name: str
    root: Path
    img_dir: Path
    gt_path: Path
    width: int
    height: int
    frame_rate: float
    length: int
    tracklets: dict[int, TrackletMeta]

    def frame_path(self, frame_idx: int) -> Path:
        return self.img_dir / f"{frame_idx:06d}.jpg"


_GAMEINFO_RE = re.compile(r"^trackletID_(\d+)\s*=\s*(.+)$")


def _parse_tracklet_value(value: str) -> TrackletMeta:
    """Parse a value such as ``player team left;9`` or ``ball;1``."""
    descriptor, _, jersey = value.partition(";")
    descriptor = descriptor.strip().lower()
    jersey = jersey.strip()

    if descriptor.startswith("ball"):
        return TrackletMeta(ROLE_BALL, TEAM_NONE, jersey)
    if descriptor.startswith("referee"):
        return TrackletMeta(ROLE_REFEREE, TEAM_NONE, jersey)

    team = TEAM_NONE
    if "team left" in descriptor:
        team = TEAM_LEFT
    elif "team right" in descriptor:
        team = TEAM_RIGHT

    role = ROLE_GOALKEEPER if descriptor.startswith("goalkeeper") else ROLE_PLAYER
    return TrackletMeta(role, team, jersey)


def _parse_gameinfo(path: Path) -> dict[int, TrackletMeta]:
    tracklets: dict[int, TrackletMeta] = {}
    if not path.is_file():
        return tracklets
    for line in path.read_text().splitlines():
        match = _GAMEINFO_RE.match(line.strip())
        if match:
            tracklets[int(match.group(1))] = _parse_tracklet_value(match.group(2))
    return tracklets


def load_sequence(seq_dir: str | Path) -> SoccerNetSequence:
    """Load metadata for a single ``SNMOT-XXX`` directory."""
    root = Path(seq_dir)
    cfg = configparser.ConfigParser()
    cfg.read(root / "seqinfo.ini")
    seq = cfg["Sequence"]
    return SoccerNetSequence(
        name=seq.get("name", root.name),
        root=root,
        img_dir=root / seq.get("imDir", "img1"),
        gt_path=root / "gt" / "gt.txt",
        width=int(seq.get("imWidth", 1920)),
        height=int(seq.get("imHeight", 1080)),
        frame_rate=float(seq.get("frameRate", 25)),
        length=int(seq.get("seqLength", 750)),
        tracklets=_parse_gameinfo(root / "gameinfo.ini"),
    )


def _load_gt_rows(gt_path: Path) -> dict[int, list[tuple[int, float, float, float, float]]]:
    """Group ground-truth rows by frame index.

    Returns ``{frame_idx: [(track_id, x, y, w, h), ...]}`` where (x, y) is the
    top-left corner in pixels.
    """
    frames: dict[int, list[tuple[int, float, float, float, float]]] = {}
    with gt_path.open() as handle:
        for line in handle:
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue
            frame_idx = int(parts[0])
            track_id = int(parts[1])
            x, y, w, h = (float(v) for v in parts[2:6])
            frames.setdefault(frame_idx, []).append((track_id, x, y, w, h))
    return frames


def iter_gt_detections(
    sequence: SoccerNetSequence,
    *,
    start: int = 1,
    end: int | None = None,
) -> Iterator[tuple[int, sv.Detections]]:
    """Yield ``(frame_idx, sv.Detections)`` built from the ground-truth file.

    Each ``Detections`` carries ``tracker_id``, ``class_id`` (role) and a
    ``data`` dict with ``team`` and ``jersey`` arrays so downstream demos can
    filter by team / pick out the ball without re-deriving anything.
    """
    rows_by_frame = _load_gt_rows(sequence.gt_path)
    last = sequence.length if end is None else min(end, sequence.length)

    for frame_idx in range(start, last + 1):
        rows = rows_by_frame.get(frame_idx, [])
        if not rows:
            yield frame_idx, sv.Detections.empty()
            continue

        xyxy = np.empty((len(rows), 4), dtype=np.float32)
        tracker_id = np.empty(len(rows), dtype=int)
        class_id = np.empty(len(rows), dtype=int)
        team = np.empty(len(rows), dtype=int)
        jersey: list[str] = []

        for i, (tid, x, y, w, h) in enumerate(rows):
            xyxy[i] = (x, y, x + w, y + h)
            tracker_id[i] = tid
            meta = sequence.tracklets.get(tid)
            class_id[i] = meta.role if meta else ROLE_PLAYER
            team[i] = meta.team if meta else TEAM_NONE
            jersey.append(meta.jersey if meta else "")

        yield frame_idx, sv.Detections(
            xyxy=xyxy,
            tracker_id=tracker_id,
            class_id=class_id,
            data={"team": team, "jersey": np.asarray(jersey, dtype=object)},
        )


def find_sequences(tracking_root: str | Path, split: str = "test") -> list[Path]:
    """Return sorted ``SNMOT-XXX`` directories for a split."""
    root = Path(tracking_root) / split
    return sorted(p for p in root.glob("SNMOT-*") if p.is_dir())
