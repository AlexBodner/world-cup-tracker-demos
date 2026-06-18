"""Compare ball visibility across ball-threshold settings on one clip.

Example::

    PYTHONPATH=. python -m world_cup_projects.dev.compare_ball_detection \\
        --video world_cup_projects/bundesliga_videos/08fd33_0.mp4 \\
        --ball-thresholds 0.5,0.35,0.25,0.20,0.15
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_pkg_root = Path(__file__).resolve().parents[1]
_repo_root = _pkg_root.parent
if (_repo_root / "world_cup_projects" / "__init__.py").is_file():
    if str(_repo_root) not in sys.path:
        sys.path.insert(0, str(_repo_root))

import numpy as np

from world_cup_projects.common.detect import (
    DEFAULT_BALL_DETECTION_THRESHOLD,
    iter_football_model_detections,
)
from world_cup_projects.common.possession import ball_xy
from world_cup_projects.common.video import load_video_sequence


def _ball_rate(frames) -> float:
    hits = sum(1 for _fi, d in frames if ball_xy(d) is not None)
    return hits / max(len(frames), 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare ball detection thresholds")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tracker", default="botsort")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument(
        "--ball-thresholds",
        default="0.5,0.35,0.25,0.20,0.15",
        help="Comma-separated ball class thresholds (players stay at --detection-threshold)",
    )
    parser.add_argument("--detection-threshold", type=float, default=0.5)
    args = parser.parse_args()

    sequence = load_video_sequence(args.video)
    end = args.max_frames if args.max_frames is not None else sequence.length
    thresholds = [float(x.strip()) for x in args.ball_thresholds.split(",") if x.strip()]

    print(f"{sequence.name}  frames=1..{end}  player_thr={args.detection_threshold}")
    print(f"{'ball_thr':>8}  {'visible':>8}  {'rate':>7}")
    print("-" * 28)
    for ball_thr in thresholds:
        frames = list(
            iter_football_model_detections(
                sequence,
                start=1,
                end=end,
                device=args.device,
                threshold=args.detection_threshold,
                ball_threshold=ball_thr,
                tracker=args.tracker,
            )
        )
        rate = _ball_rate(frames)
        print(f"{ball_thr:>8.2f}  {int(rate * len(frames)):>8}  {rate:>6.1%}")

    print(f"\nDefault ball threshold in pipeline: {DEFAULT_BALL_DETECTION_THRESHOLD}")


if __name__ == "__main__":
    main()
