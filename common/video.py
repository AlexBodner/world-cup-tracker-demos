"""Load arbitrary MP4 clips with the same frame interface as SoccerNet sequences."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoSequence:
    """Duck-compatible with :class:`SoccerNetSequence` for demo renderers."""

    name: str
    video_path: Path
    width: int
    height: int
    frame_rate: float
    length: int

    def frame_path(self, frame_idx: int) -> Path:
        raise NotImplementedError(
            f"{self.name}: use read_sequence_frame() for video sources"
        )


def load_video_sequence(path: str | Path) -> VideoSequence:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    if length <= 0:
        raise RuntimeError(f"Video has no frames: {path}")
    return VideoSequence(
        name=path.stem,
        video_path=path.resolve(),
        width=width,
        height=height,
        frame_rate=fps,
        length=length,
    )


def read_sequence_frame(sequence, frame_idx: int) -> np.ndarray | None:
    """Read one 1-indexed BGR frame from a SoccerNet dir or an MP4 sequence."""
    video_path = getattr(sequence, "video_path", None)
    if video_path is not None:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(frame_idx - 1, 0))
        ok, frame = cap.read()
        cap.release()
        return frame if ok else None
    return cv2.imread(str(sequence.frame_path(frame_idx)))
