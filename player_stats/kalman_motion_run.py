"""Kalman motion joystick demo — team ellipses + direction/speed dots, no radar.

From the repo root::

    PYTHONPATH=. python -m world_cup_projects.player_stats.kalman_motion_run \\
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
from world_cup_projects.player_stats.kalman_motion_render import (
    compare_kalman_motion_speeds,
    render_kalman_motion_video,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Kalman joystick motion overlay (team ellipses + direction dots)"
    )
    parser.add_argument(
        "--video",
        required=True,
        help="Input MP4 (e.g. bundesliga_videos/08fd33_0.mp4)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output MP4 path (default: assets/kalman_motion_<clip>.mp4)",
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
    parser.add_argument(
        "--min-speed-px",
        type=float,
        default=0.5,
        help="Hide dot below this Kalman speed (px/frame)",
    )
    parser.add_argument(
        "--max-speed-px",
        type=float,
        default=4.0,
        help="Kalman speed mapped to full ellipse edge (px/frame; lower = dot reaches edge sooner)",
    )
    parser.add_argument(
        "--smooth-alpha",
        type=float,
        default=0.28,
        help="EMA weight on Kalman velocity (lower = smoother direction/speed)",
    )
    parser.add_argument(
        "--dot-smooth-alpha",
        type=float,
        default=0.32,
        help="EMA weight on dot offset from ellipse center (lower = smoother motion)",
    )
    parser.add_argument(
        "--width-smooth-alpha",
        type=float,
        default=0.22,
        help="EMA weight on bbox width for ellipse size (lower = steadier ellipses)",
    )
    parser.add_argument(
        "--team-flip-after",
        type=int,
        default=16,
        help="(Legacy) ignored — Kalman overlay uses per-tracklet majority shirt-color lock",
    )
    parser.add_argument(
        "--show-speed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show speed (m/s) on every player; unit explained once in corner legend",
    )
    parser.add_argument(
        "--min-speed-ms",
        type=float,
        default=0.0,
        help="Hide speed badge below this ground speed (m/s); 0 = always show",
    )
    parser.add_argument(
        "--max-speed-labels",
        type=int,
        default=0,
        help="Cap badges per frame to N fastest (0 = show all players)",
    )
    parser.add_argument(
        "--speed-smooth-alpha",
        type=float,
        default=0.22,
        help="EMA weight on displayed m/s labels (lower = smoother)",
    )
    parser.add_argument(
        "--speed-source",
        choices=("kalman", "multilag", "compare"),
        default="kalman",
        help="Speed label source: Kalman homography, multi-lag pitch history, or compare both",
    )
    parser.add_argument(
        "--speed-k-frames",
        type=int,
        default=5,
        help="Lag window for multi-lag homography speed (compare / multilag only)",
    )
    parser.add_argument("--refresh-detections-cache", action="store_true")
    args = parser.parse_args()

    sequence = load_video_sequence(args.video)
    detections_source = load_football_detections_cached(args, sequence)

    out_dir = DEFAULT_ASSETS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    render_kwargs = dict(
        detections_source=detections_source,
        max_frames=args.max_frames,
        tracker_kind=args.tracker,
        min_speed_px=args.min_speed_px,
        max_speed_px=args.max_speed_px,
        pitch_device=args.device,
        smooth_alpha=args.smooth_alpha,
        dot_smooth_alpha=args.dot_smooth_alpha,
        width_smooth_alpha=args.width_smooth_alpha,
        team_flip_after=args.team_flip_after,
        show_speed=args.show_speed,
        min_speed_ms=args.min_speed_ms,
        max_speed_labels=args.max_speed_labels,
        speed_smooth_alpha=args.speed_smooth_alpha,
    )

    if args.speed_source == "compare":
        result = compare_kalman_motion_speeds(
            sequence,
            str(out_dir),
            speed_k_frames=args.speed_k_frames,
            **render_kwargs,
        )
        print(json.dumps(result, indent=2))
        print(f"\nWrote {result['outputs']['kalman']}")
        print(f"Wrote {result['outputs']['multilag']}")
        print(f"Wrote {result['outputs']['stats']}")
        return

    suffix = "" if args.speed_source == "kalman" else f"_{args.speed_source}"
    out_path = args.out or str(out_dir / f"kalman_motion_{sequence.name}{suffix}.mp4")

    manifest = render_kalman_motion_video(
        sequence,
        out_path,
        speed_source=args.speed_source,
        speed_k_frames=args.speed_k_frames,
        **render_kwargs,
    )
    manifest["output"] = out_path
    print(json.dumps(manifest, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
