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
from world_cup_projects.common.detection_cache import wrap_detections_cache
from world_cup_projects.common.detect import DEFAULT_BALL_DETECTION_THRESHOLD
from world_cup_projects.common.video import load_video_sequence
from world_cup_projects.player_stats.kalman_motion_render import render_kalman_motion_video


def _load_detections(args, sequence):
    from world_cup_projects.common.detect import iter_football_model_detections

    ball_thr = getattr(args, "ball_threshold", DEFAULT_BALL_DETECTION_THRESHOLD)
    return wrap_detections_cache(
        iter_football_model_detections,
        source_name="football",
        refresh=args.refresh_detections_cache,
        device=args.device,
        threshold=0.5,
        ball_threshold=ball_thr,
        tracker=args.tracker,
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
    parser.add_argument("--device", default="cpu", help="Detector + pitch keypoint device")
    parser.add_argument("--tracker", default="bytetrack", choices=("bytetrack", "botsort"))
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
        help="Outfield team flip only after N consecutive jersey disagrees (tracklet lock)",
    )
    parser.add_argument("--refresh-detections-cache", action="store_true")
    args = parser.parse_args()

    sequence = load_video_sequence(args.video)
    detections_source = _load_detections(args, sequence)

    out_dir = DEFAULT_ASSETS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out or str(out_dir / f"kalman_motion_{sequence.name}.mp4")

    manifest = render_kalman_motion_video(
        sequence,
        out_path,
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
    )
    manifest["output"] = out_path
    print(json.dumps(manifest, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
