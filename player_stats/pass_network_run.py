"""Run v1 pass-network analysis (inferred passes + collaboration links).

From the monorepo root::

    PYTHONPATH=. python -m world_cup_projects.player_stats.pass_network_run \\
        --video world_cup_projects/bundesliga_videos/08fd33_0.mp4 \\
        --metric --source football --tracker botsort

From inside ``world_cup_projects/``::

    PYTHONPATH=. python -m player_stats.pass_network_run \\
        --video bundesliga_videos/08fd33_0.mp4 --metric --render --debug-carrier

SoccerNet GT tracks::

    PYTHONPATH=. python -m world_cup_projects.player_stats.pass_network_run \\
        --sequence SNMOT-194 --metric
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow ``python -m player_stats.pass_network_run`` from the package directory
# without ``pip install -e .`` (adds the parent repo root to sys.path).
_pkg_root = Path(__file__).resolve().parents[1]
_repo_root = _pkg_root.parent
if (_repo_root / "world_cup_projects" / "__init__.py").is_file():
    if str(_repo_root) not in sys.path:
        sys.path.insert(0, str(_repo_root))

import argparse
import json
from pathlib import Path

from world_cup_projects import DEFAULT_ASSETS_DIR
from world_cup_projects.common.cli import (
    FootballDetectionDefaults,
    add_football_detection_args,
)
from world_cup_projects.common.pipeline import (
    load_detections_source,
    load_metric_context,
    prepare_model_frames,
)
from world_cup_projects.common.soccernet import (
    DEFAULT_TRACKING_ROOT,
    find_sequences,
    iter_gt_detections,
    load_sequence,
)
from world_cup_projects.pass_alternatives.pass_options import PassWeights
from world_cup_projects.player_stats.pass_events import (
    PassDetectionConfig,
    PassQualityScorer,
    build_pass_carrier_timeline,
    scan_possession_events,
)
from world_cup_projects.player_stats.pass_network import PassNetwork, build_pass_network
from world_cup_projects.player_stats.pass_network_render import render_pass_network_demo


_load_detections_source = load_detections_source


def _pitch_transformers(sequence, *, max_frames, device, pitch_confidence):
    from world_cup_projects.common.pitch import iter_pitch_transformers

    end = max_frames if max_frames is not None else sequence.length
    return {
        frame_idx: speed_t
        for frame_idx, speed_t, _radar_t in iter_pitch_transformers(
            sequence,
            device=device,
            end=end,
            confidence=pitch_confidence,
        )
    }


def analyze_pass_network(
    sequence,
    frames: list[tuple[int, object]],
    *,
    metric: bool,
    scorer: PassQualityScorer,
    config: PassDetectionConfig,
) -> PassNetwork:
    """Infer passes and turnovers; build a collaboration snapshot."""
    scan = scan_possession_events(
        iter(frames),
        scorer=scorer,
        config=config,
        metric=metric,
        transformers=scorer._transformers,
        fps=float(sequence.frame_rate),
    )
    return build_pass_network(
        sequence.name,
        list(scan.passes),
        list(scan.turnovers),
        metric=metric,
    )


def main() -> None:
    from world_cup_projects.common.model_ids import (
        DEFAULT_FOOTBALL_BALL_MODEL_ID,
        DEFAULT_FOOTBALL_PLAYERS_MODEL_ID,
    )

    parser = argparse.ArgumentParser(description="Pass network v1: inferred passes + collaborator links")
    parser.add_argument("--data", default=DEFAULT_TRACKING_ROOT)
    parser.add_argument("--split", default="test")
    parser.add_argument("--out", default=str(DEFAULT_ASSETS_DIR))
    add_football_detection_args(
        parser,
        defaults=FootballDetectionDefaults(tracker="botsort"),
    )
    parser.add_argument(
        "--control-max-m",
        type=float,
        default=PassDetectionConfig.control_max_distance_m,
        help="Tight feet distance for sustained control.",
    )
    parser.add_argument(
        "--reception-max-m",
        type=float,
        default=PassDetectionConfig.reception_max_distance_m,
        help="Looser distance for one-touch reception.",
    )
    parser.add_argument(
        "--min-ball-travel-m",
        type=float,
        default=PassDetectionConfig.min_ball_travel_m,
        help="Minimum ball travel between passer and receiver.",
    )
    parser.add_argument(
        "--min-pass-gap",
        type=int,
        default=PassDetectionConfig.min_carrier_gap_frames,
        help="Min frames between passer and receiver possession.",
    )
    parser.add_argument(
        "--max-pass-gap",
        type=int,
        default=PassDetectionConfig.max_pass_gap_frames,
        help="Max frames between passer and receiver possession.",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render MP4 with stats end-card (default when --video is set).",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="JSON only; skip video render.",
    )
    parser.add_argument(
        "--stats-seconds",
        type=float,
        default=5.0,
        help="Duration of the stats end-card.",
    )
    parser.add_argument(
        "--show-predictions",
        action="store_true",
        help="Overlay real-time pass alternatives (predictions) alongside the pass network.",
    )
    parser.add_argument(
        "--freeze-quality-threshold",
        type=float,
        default=0.0,
        help="Only freeze for passes with quality score above this threshold.",
    )
    parser.add_argument(
        "--min-arrival-frames",
        type=int,
        default=PassDetectionConfig.min_arrival_frames,
        help="Frames a receiver must hold the ball after in-flight before pass counts.",
    )
    parser.add_argument(
        "--min-control-frames",
        type=int,
        default=PassDetectionConfig.min_control_frames,
        help="Consecutive control frames before crediting a player as passer.",
    )
    parser.add_argument(
        "--missing-ball-tolerance",
        type=int,
        default=PassDetectionConfig.missing_ball_tolerance,
        help="Frames without ball detection to bridge as in-flight (~0.4s default).",
    )
    parser.add_argument(
        "--debug-carrier",
        action="store_true",
        help="Overlay per-frame ball-carrier HUD (control/reception, anchor, in-flight).",
    )
    parser.add_argument(
        "--tag-suffix",
        default=None,
        help="Extra tag in output filenames (e.g. inference_v20). Auto-set for --detector-backend inference.",
    )
    args = parser.parse_args()

    if args.video:
        from world_cup_projects.common.video import load_video_sequence

        sequence = load_video_sequence(args.video)
        if args.source == "gt":
            args.source = "football"
    else:
        if args.sequence is None:
            raise SystemExit("Provide --sequence or --video")
        seq_dir = next(
            p for p in find_sequences(args.data, args.split) if p.name == args.sequence
        )
        sequence = load_sequence(seq_dir)

    if args.player_model_id is None:
        from world_cup_projects.common.detect import DEFAULT_FOOTBALL_PLAYERS_MODEL_ID

        args.player_model_id = DEFAULT_FOOTBALL_PLAYERS_MODEL_ID
    if args.ball_threshold is None:
        from world_cup_projects.common.detect import DEFAULT_BALL_DETECTION_THRESHOLD

        args.ball_threshold = DEFAULT_BALL_DETECTION_THRESHOLD
    if args.ball_model_id is None:
        from world_cup_projects.common.detect import DEFAULT_FOOTBALL_BALL_MODEL_ID

        args.ball_model_id = DEFAULT_FOOTBALL_BALL_MODEL_ID

    config = PassDetectionConfig(
        min_carrier_gap_frames=args.min_pass_gap,
        max_pass_gap_frames=args.max_pass_gap,
        control_max_distance_m=args.control_max_m,
        reception_max_distance_m=args.reception_max_m,
        min_ball_travel_m=args.min_ball_travel_m,
        min_arrival_frames=args.min_arrival_frames,
        min_control_frames=args.min_control_frames,
        missing_ball_tolerance=args.missing_ball_tolerance,
    ).for_frame_rate(sequence.frame_rate)
    detections_source = load_detections_source(args, sequence)
    end = args.max_frames if args.max_frames is not None else sequence.length
    frames = list(detections_source(sequence, start=1, end=end))
    print(
        f"Possession scan: sequence={sequence.name} "
        f"tracker={args.tracker} device={args.device} frames={len(frames)}"
    )
    if args.source in ("football", "rfdetr"):
        frames = prepare_model_frames(frames, frame_width=float(sequence.width))

    frame_transforms: dict = {}
    frame_radar_transforms: dict = {}
    frame_keypoints: dict = {}
    locked_goals = None
    if args.metric:
        ctx = load_metric_context(
            sequence,
            frames,
            device=args.device,
            pitch_confidence=args.pitch_confidence,
            end=end,
            source=args.source,
            frame_width=float(sequence.width),
        )
        frame_transforms = ctx.transforms
        frame_radar_transforms = ctx.radar_transforms
        frame_keypoints = ctx.keypoints
        locked_goals = ctx.locked_goals

    weights = PassWeights.metric() if args.metric else PassWeights()
    scorer = PassQualityScorer(weights=weights, metric=args.metric, transformers=frame_transforms)

    network = analyze_pass_network(
        sequence,
        frames,
        metric=args.metric,
        scorer=scorer,
        config=config,
    )
    manifest = network.to_dict()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.source + ("_metric" if args.metric else "")
    if args.detector_backend == "inference" and args.player_model_id:
        ver = args.player_model_id.rsplit("/", 1)[-1]
        tag += f"_inference_v{ver}"
    if args.tag_suffix:
        tag += f"_{args.tag_suffix}"
    if args.debug_pitch_keypoints:
        tag += "_pitch_kp_debug"
    if args.debug_carrier:
        tag += "_carrier_debug"
    json_path = out_dir / f"pass_network_{tag}_{sequence.name}.json"
    json_path.write_text(json.dumps(manifest, indent=2))

    carrier_timeline = None
    if args.debug_carrier:
        carrier_timeline = {
            state.frame_idx: state
            for state in build_pass_carrier_timeline(
                iter(frames),
                config=config,
                metric=args.metric,
                transformers=frame_transforms if args.metric else None,
            )
        }

    should_render = args.render or (args.video is not None and not args.no_render)
    if should_render:
        from world_cup_projects.common.video import read_sequence_frame

        video_path = out_dir / f"pass_network_{tag}_{sequence.name}.mp4"
        render_manifest = render_pass_network_demo(
            sequence,
            frames,
            network,
            str(video_path),
            frame_loader=lambda fi: read_sequence_frame(sequence, fi),
            metric=args.metric,
            frame_transforms=frame_transforms,
            frame_radar_transforms=frame_radar_transforms,
            frame_keypoints=frame_keypoints,
            pitch_confidence=args.pitch_confidence,
            locked_goal_defenders=locked_goals,
            stats_seconds=args.stats_seconds,
            debug_pitch_keypoints=args.debug_pitch_keypoints,
            scorer=scorer,
            show_predictions=args.show_predictions,
            freeze_quality_threshold=args.freeze_quality_threshold,
            debug_carrier=args.debug_carrier,
            carrier_timeline=carrier_timeline,
        )
        manifest["video"] = render_manifest["output"]
        json_path.write_text(json.dumps(manifest, indent=2))
        print(f"Wrote {video_path}")

    print(json.dumps(manifest, indent=2))
    print(f"\nWrote {json_path}  ({manifest['n_passes']} inferred passes)")


if __name__ == "__main__":
    main()
