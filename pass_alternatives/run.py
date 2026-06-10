"""Run the pass-alternatives demo.

Default (must-have): GT detections, image-space scoring (pixels)::

    PYTHONPATH=. python -m world_cup_projects.pass_alternatives.run --sequence SNMOT-194

Metric scoring (GT detections -> pitch homography -> meters)::

    PYTHONPATH=. python -m world_cup_projects.pass_alternatives.run \\
        --sequence SNMOT-194 --metric --device cpu

``--source`` swaps the detector. ``gt`` (default) uses the SoccerNet ground-truth
tracks; ``football`` uses football-players-detection YOLO (default for ``--video``);
``rfdetr`` is a generic COCO fallback (poor team/role split).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from world_cup_projects.common.clips import (
    PITCH_HOMOGRAPHY_DEMO_CLIP,
    PITCH_KEYPOINT_AVOID,
    PITCH_KEYPOINT_AVOID_NOTES,
    pick_homography_demo_clip,
    pitch_keypoints_unreliable,
    rank_clips,
)
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
    parser.add_argument(
        "--video",
        default=None,
        help="MP4 path instead of SNMOT (implies --source football).",
    )
    parser.add_argument("--out", default=str(DEFAULT_ASSETS_DIR))
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument(
        "--max-events",
        type=int,
        default=0,
        help="Cap freeze count; 0 = auto-detect all good moments (default).",
    )
    parser.add_argument(
        "--source",
        choices=("gt", "football", "rfdetr"),
        default="gt",
        help="Detection source. gt = SoccerNet GT; football = DFL player detector; rfdetr = COCO.",
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
        help="Draw pitch keypoints on the video + radar (green=used, blue=rejected).",
    )
    parser.add_argument(
        "--pitch-confidence",
        type=float,
        default=0.9,
        help="Keypoint confidence threshold (overlay legend + homography filter).",
    )
    parser.add_argument(
        "--facing-mode",
        choices=("motion", "kalman", "both"),
        default="kalman",
        help="Player facing arrows: kalman (default), displacement motion, or both.",
    )
    parser.add_argument(
        "--tracker",
        choices=("bytetrack", "botsort", "botsort_nocmc"),
        default="bytetrack",
        help="Player tracker for --source football/rfdetr. botsort = BoT-SORT + CMC.",
    )
    parser.add_argument(
        "--refresh-detections-cache",
        action="store_true",
        help="Re-run YOLO/RF-DETR instead of loading .cache/detections/.",
    )
    parser.add_argument(
        "--freeze-min-pick-score",
        type=float,
        default=None,
        help="Min combined freeze score (pass + control); default from PassWeights.",
    )
    parser.add_argument(
        "--freeze-min-pass-score",
        type=float,
        default=None,
        help="Min score of the best pass option; default from PassWeights.",
    )
    parser.add_argument(
        "--no-freeze-local-peaks",
        action="store_true",
        help="Accept any frame above score thresholds (skip local-peak filter).",
    )
    parser.add_argument(
        "--pass-segment-openness",
        action="store_true",
        help="Use old openness (any defender near the pass line). Default is lane-only.",
    )
    parser.add_argument(
        "--pass-lane-image",
        action="store_true",
        help="Score rival corridors in image pixels instead of pitch/radar meters.",
    )
    parser.add_argument(
        "--pass-lane-width",
        type=float,
        default=None,
        metavar="W",
        help="Intercept corridor full width in m (--metric) or px. Default 2.5 m with --metric; 0 disables.",
    )
    parser.add_argument(
        "--pass-teammate-lane-width",
        type=float,
        default=None,
        metavar="W",
        help="Teammate corridor width (default 0.5 m with --metric). 0 uses rival width.",
    )
    parser.add_argument(
        "--force-unreliable-pitch",
        action="store_true",
        help="Allow --metric on SNMOT-132 / SNMOT-189 (pitch keypoints usually bad).",
    )
    parser.add_argument(
        "--pass-teammate-penalty",
        type=float,
        default=None,
        metavar="P",
        help="Max score deduction for a blocking teammate (default 0.10 with --metric).",
    )
    args = parser.parse_args()

    if args.video:
        if args.rank_only:
            raise SystemExit("--rank-only is for SNMOT clip ranking only.")
        from world_cup_projects.common.video import load_video_sequence

        sequence = load_video_sequence(args.video)
        if args.source == "gt":
            print("Note: --video uses football-players-detection (--source football).")
            args.source = "football"
    else:
        sequence = None

    seq_dirs = find_sequences(args.data, args.split)
    if sequence is None and not seq_dirs:
        raise SystemExit(f"No SNMOT-* sequences under {args.data}/{args.split}")

    if sequence is None and (args.rank_only or args.sequence is None):
        ranking = (
            rank_clips(
                seq_dirs,
                assess_pitch=True,
                pitch_confidence=args.pitch_confidence,
                homography_demo=args.metric,
            )
            if args.metric
            else rank_clips(seq_dirs)
        )
        print("Clip ranking (best first):")
        for i, clip in enumerate(ranking[:10], 1):
            print(f"  {i:2d}. {clip.name}  score={clip.score:7.1f}  | {clip.reason}")
        if args.rank_only:
            return
        if args.metric:
            pick = pick_homography_demo_clip(
                seq_dirs,
                pitch_device=args.device,
                pitch_confidence=args.pitch_confidence,
            )
            chosen_name = pick.name
            print(
                f"\nAuto-picked (homography): {chosen_name} -> {pick.reason}"
            )
        else:
            chosen_name = ranking[0].name
            print(f"\nAuto-picked: {chosen_name} -> {ranking[0].reason}")
        seq_dir = next(p for p in seq_dirs if p.name == chosen_name)
    elif sequence is None:
        seq_dir = next(p for p in seq_dirs if p.name == args.sequence)
        sequence = load_sequence(seq_dir)

    if pitch_keypoints_unreliable(sequence.name):
        note = PITCH_KEYPOINT_AVOID_NOTES.get(sequence.name, "Pitch keypoints unreliable.")
        if args.metric and not args.force_unreliable_pitch:
            raise SystemExit(
                f"{sequence.name}: {note}\n"
                f"Omit --metric (pixel-space demo) or use {PITCH_HOMOGRAPHY_DEMO_CLIP} / SNMOT-194. "
                "Pass --force-unreliable-pitch to render metric anyway."
            )
        if args.metric:
            print(f"Warning: {sequence.name} — {note} (--force-unreliable-pitch set).")
        else:
            print(f"Note: {sequence.name} — {note} Pixel-space mode is fine.")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.source == "football":
        from world_cup_projects.common.detect import iter_football_model_detections
        from world_cup_projects.common.detection_cache import wrap_detections_cache

        detections_source = wrap_detections_cache(
            iter_football_model_detections,
            source_name="football",
            refresh=args.refresh_detections_cache,
            device=args.device,
            threshold=0.5,
            tracker=args.tracker,
        )
    elif args.source == "rfdetr":
        from world_cup_projects.common.detect import iter_model_detections
        from world_cup_projects.common.detection_cache import wrap_detections_cache

        detections_source = wrap_detections_cache(
            iter_model_detections,
            source_name="rfdetr",
            refresh=args.refresh_detections_cache,
            device=args.device,
        )
    else:
        detections_source = iter_gt_detections

    weights = PassWeights.metric() if args.metric else PassWeights()
    if args.pass_segment_openness:
        weights = replace(weights, use_lane_openness=False)
    if args.pass_lane_image:
        weights = replace(weights, lane_in_image_space=True)
    if args.pass_lane_width is not None:
        w = args.pass_lane_width
        weights = replace(weights, lane_width=None if w <= 0 else w)
    if args.pass_teammate_lane_width is not None:
        tw = args.pass_teammate_lane_width
        weights = replace(
            weights,
            teammate_lane_width=None if tw <= 0 else tw,
        )
    if args.pass_teammate_penalty is not None:
        weights = replace(weights, teammate_lane_penalty=max(0.0, args.pass_teammate_penalty))
    if args.freeze_min_pick_score is not None:
        weights = replace(weights, freeze_min_pick_score=args.freeze_min_pick_score)
    if args.freeze_min_pass_score is not None:
        weights = replace(weights, freeze_min_pass_score=args.freeze_min_pass_score)
    if args.no_freeze_local_peaks:
        weights = replace(weights, freeze_detect_local_peaks=False)
    max_events = args.max_events if args.max_events > 0 else None
    tag = args.source + ("_metric" if args.metric else "")
    if args.tracker != "bytetrack":
        tag += f"_{args.tracker}"
    tag += f"_facing_{args.facing_mode}"
    if args.debug_pitch_keypoints:
        tag += "_pitch_kp_debug"

    out_path = out_dir / f"pass_alternatives_{tag}_{sequence.name}.mp4"
    manifest = render_demo(
        sequence,
        str(out_path),
        max_frames=args.max_frames,
        max_events=max_events,
        weights=weights,
        detections_source=detections_source,
        metric=args.metric,
        pitch_device=args.device,
        version_tag=tag,
        carrier_max_distance_px=args.carrier_max_px,
        carrier_max_distance_m=args.carrier_max_m,
        debug_pitch_keypoints=args.debug_pitch_keypoints,
        pitch_confidence=args.pitch_confidence,
        facing_mode=args.facing_mode,
        tracker_kind=args.tracker,
    )
    json_path = out_dir / f"pass_alternatives_{tag}_{sequence.name}.json"
    json_path.write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    n_events = len(manifest.get("events", []))
    print(f"\nWrote {out_path}  ({n_events} pass moment{'s' if n_events != 1 else ''})")


if __name__ == "__main__":
    main()
