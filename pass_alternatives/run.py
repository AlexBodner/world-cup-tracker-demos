"""Run the pass-alternatives demo.

Default (must-have): GT detections, image-space scoring (pixels)::

    PYTHONPATH=. python -m world_cup_projects.pass_alternatives.run --sequence SNMOT-194

Metric scoring (GT detections -> pitch homography -> meters)::

    PYTHONPATH=. python -m world_cup_projects.pass_alternatives.run \\
        --sequence SNMOT-194 --metric --device cpu

``--source`` swaps the detector. ``gt`` (default) uses the SoccerNet ground-truth
tracks; ``rfdetr`` uses the optional RF-DETR + ByteTrack pipeline (``common/detect.py``).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from world_cup_projects.common.clips import rank_clips
from world_cup_projects.common.possession import CARRIER_MAX_DISTANCE_M, CARRIER_MAX_DISTANCE_PX
from world_cup_projects import DEFAULT_ASSETS_DIR
from world_cup_projects.common.soccernet import (
    DEFAULT_TRACKING_ROOT,
    find_sequences,
    iter_gt_detections,
    load_sequence,
)
from world_cup_projects.pass_alternatives.pass_options import PassWeights
from world_cup_projects.pass_alternatives.render import render_demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Pass-alternatives demo")
    parser.add_argument("--data", default=DEFAULT_TRACKING_ROOT)
    parser.add_argument("--split", default="test")
    parser.add_argument("--sequence", default=None)
    parser.add_argument("--out", default=str(DEFAULT_ASSETS_DIR))
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--max-events", type=int, default=4)
    parser.add_argument(
        "--source",
        choices=("gt", "rfdetr"),
        default="gt",
        help="Detection source. gt = SoccerNet GT (default); rfdetr = optional RF-DETR.",
    )
    parser.add_argument(
        "--metric",
        action="store_true",
        help="Score in pitch meters via the pitch-keypoint homography.",
    )
    parser.add_argument("--device", default="cpu", help="Pitch keypoint model device.")
    parser.add_argument(
        "--carrier-max-px",
        type=float,
        default=CARRIER_MAX_DISTANCE_PX,
        help="Ball-to-feet limit in pixels when not using --metric (default 80).",
    )
    parser.add_argument(
        "--carrier-max-m",
        type=float,
        default=CARRIER_MAX_DISTANCE_M,
        help="Ball-to-feet limit in meters on the pitch when using --metric (default 1.0).",
    )
    parser.add_argument(
        "--rank-only",
        action="store_true",
        help="Just print the clip ranking and exit (no render).",
    )
    parser.add_argument(
        "--debug-pitch-keypoints",
        action="store_true",
        help="Draw pitch keypoints + confidence on each frame (homography debug).",
    )
    parser.add_argument(
        "--pitch-confidence",
        type=float,
        default=0.5,
        help="Keypoint confidence threshold (overlay legend + homography filter).",
    )
    args = parser.parse_args()

    seq_dirs = find_sequences(args.data, args.split)
    if not seq_dirs:
        raise SystemExit(f"No SNMOT-* sequences under {args.data}/{args.split}")

    if args.rank_only or args.sequence is None:
        ranking = rank_clips(seq_dirs)
        print("Clip ranking (best first):")
        for i, clip in enumerate(ranking[:10], 1):
            print(f"  {i:2d}. {clip.name}  score={clip.score:7.1f}  | {clip.reason}")
        if args.rank_only:
            return
        chosen_name = ranking[0].name
        print(f"\nAuto-picked: {chosen_name} -> {ranking[0].reason}")
        seq_dir = next(p for p in seq_dirs if p.name == chosen_name)
    else:
        seq_dir = next(p for p in seq_dirs if p.name == args.sequence)

    sequence = load_sequence(seq_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.source == "rfdetr":
        from world_cup_projects.common.detect import iter_model_detections

        detections_source = iter_model_detections
    else:
        detections_source = iter_gt_detections

    weights = PassWeights.metric() if args.metric else PassWeights()
    tag = args.source + ("_metric" if args.metric else "")
    if args.debug_pitch_keypoints:
        tag += "_pitch_kp_debug"

    out_path = out_dir / f"pass_alternatives_{tag}_{sequence.name}.mp4"
    manifest = render_demo(
        sequence,
        str(out_path),
        max_frames=args.max_frames,
        max_events=args.max_events,
        weights=weights,
        detections_source=detections_source,
        metric=args.metric,
        pitch_device=args.device,
        version_tag=tag,
        carrier_max_distance_px=args.carrier_max_px,
        carrier_max_distance_m=args.carrier_max_m,
        debug_pitch_keypoints=args.debug_pitch_keypoints,
        pitch_confidence=args.pitch_confidence,
    )
    json_path = out_dir / f"pass_alternatives_{tag}_{sequence.name}.json"
    json_path.write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
