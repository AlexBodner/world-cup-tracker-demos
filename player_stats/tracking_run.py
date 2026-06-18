"""Tracking-only demo — team ellipses + tracker IDs + ball, no motion overlay.

From the repo root::

    PYTHONPATH=. python -m world_cup_projects.player_stats.tracking_run \\
        --video world_cup_projects/bundesliga_videos/08fd33_0.mp4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_pkg_root = Path(__file__).resolve().parents[1]
_repo_root = _pkg_root.parent
if (_repo_root / "world_cup_projects" / "__init__.py").is_file():
    if str(_repo_root) not in sys.path:
        sys.path.insert(0, str(_repo_root))

from world_cup_projects import DEFAULT_ASSETS_DIR
from world_cup_projects.common.device import default_torch_device
from world_cup_projects.common.pipeline import load_football_detections_cached
from world_cup_projects.common.video import load_video_sequence
from world_cup_projects.player_stats.tracking_render import render_tracking_video


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tracking-only overlay (per-track colors + tracker IDs + ball)"
    )
    parser.add_argument(
        "--video",
        required=True,
        help="Input MP4 (e.g. bundesliga_videos/08fd33_0.mp4)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output MP4 path (default: assets/tracking_<clip>.mp4)",
    )
    parser.add_argument(
        "--device",
        default=default_torch_device(),
        help="Detector + pitch keypoint device (default: mps when available)",
    )
    parser.add_argument(
        "--tracker",
        default="botsort",
        choices=("bytetrack", "botsort", "botsort_nocmc"),
    )
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--refresh-detections-cache", action="store_true")
    args = parser.parse_args()

    sequence = load_video_sequence(args.video)
    detections_source = load_football_detections_cached(args, sequence)

    out_dir = DEFAULT_ASSETS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out or str(out_dir / f"tracking_{sequence.name}.mp4")

    manifest = render_tracking_video(
        sequence,
        out_path,
        detections_source=detections_source,
        max_frames=args.max_frames,
        pitch_device=args.device,
    )
    manifest["output"] = out_path
    print(json.dumps(manifest, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
