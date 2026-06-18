"""Debug render: ball velocity arrow + speed label from raw detection positions.

Ball speed is **not** from Kalman — it is displacement of ``ball_xy()`` between frames.
Players still show tracker ids; their Kalman joystick demo is separate.

From ``world_cup_projects/``::

    PYTHONPATH=.. python -m player_stats.ball_speed_debug_run \\
        --video bundesliga_videos/08fd33_0.mp4 --start 350 --end 500
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_pkg_root = Path(__file__).resolve().parents[1]
_repo_root = _pkg_root.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from world_cup_projects import DEFAULT_ASSETS_DIR
from world_cup_projects.common.pipeline import load_football_detections_cached
from world_cup_projects.common.video import load_video_sequence
from world_cup_projects.player_stats.ball_speed_debug_render import render_ball_speed_debug_video


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, help="Input MP4")
    parser.add_argument("--out", default=None, help="Output MP4 path")
    parser.add_argument("--device", default="mps", help="Detector device")
    parser.add_argument("--tracker", default="botsort", choices=("bytetrack", "botsort"))
    parser.add_argument("--ball-threshold", type=float, default=DEFAULT_BALL_DETECTION_THRESHOLD)
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--lookback", type=int, default=6, help="Frames for speed window")
    parser.add_argument(
        "--smooth-alpha",
        type=float,
        default=0.22,
        help="EMA weight on velocity (lower = smoother arrow/label)",
    )
    parser.add_argument("--refresh-detections-cache", action="store_true")
    args = parser.parse_args()

    sequence = load_video_sequence(args.video)
    end = args.end if args.end is not None else sequence.length

    detections_source = load_football_detections_cached(args, sequence)

    out_dir = DEFAULT_ASSETS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.start}_{end}" if args.start > 1 or end < sequence.length else ""
    out_path = args.out or str(out_dir / f"ball_speed_debug_{sequence.name}{suffix}.mp4")

    manifest = render_ball_speed_debug_video(
        sequence,
        out_path,
        detections_source=detections_source,
        start_frame=args.start,
        end_frame=end,
        lookback_frames=args.lookback,
        smooth_alpha=args.smooth_alpha,
        pitch_device=args.device,
    )
    manifest["output"] = out_path
    print(json.dumps(manifest, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
