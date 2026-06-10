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
from world_cup_projects.common.pitch import iter_pitch_transformers
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


def _load_detections_source(args, sequence):
    if args.source == "football":
        from world_cup_projects.common.detect import iter_football_model_detections
        from world_cup_projects.common.detection_cache import wrap_detections_cache

        return wrap_detections_cache(
            iter_football_model_detections,
            source_name="football",
            refresh=args.refresh_detections_cache,
            device=args.device,
            threshold=0.5,
            tracker=args.tracker,
        )
    if args.source == "rfdetr":
        from world_cup_projects.common.detect import iter_model_detections
        from world_cup_projects.common.detection_cache import wrap_detections_cache

        return wrap_detections_cache(
            iter_model_detections,
            source_name="rfdetr",
            refresh=args.refresh_detections_cache,
            device=args.device,
        )
    return iter_gt_detections


def _pitch_transformers(sequence, *, max_frames, device, pitch_confidence):
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
    )
    return build_pass_network(
        sequence.name,
        list(scan.passes),
        list(scan.turnovers),
        metric=metric,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Pass network v1: inferred passes + collaborator links")
    parser.add_argument("--data", default=DEFAULT_TRACKING_ROOT)
    parser.add_argument("--split", default="test")
    parser.add_argument("--sequence", default=None)
    parser.add_argument("--video", default=None, help="MP4 path (implies --source football).")
    parser.add_argument("--out", default=str(DEFAULT_ASSETS_DIR))
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--source", choices=("gt", "football", "rfdetr"), default="gt")
    parser.add_argument("--tracker", choices=("bytetrack", "botsort", "botsort_nocmc"), default="botsort")
    parser.add_argument("--metric", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--pitch-confidence", type=float, default=0.9)
    parser.add_argument("--refresh-detections-cache", action="store_true")
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
        "--debug-pitch-keypoints",
        action="store_true",
        help="Draw pitch keypoints on video + radar (green=used, red=rejected).",
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
    detections_source = _load_detections_source(args, sequence)
    end = args.max_frames if args.max_frames is not None else sequence.length
    frames = list(detections_source(sequence, start=1, end=end))

    # Perform pitch transformation and goalkeeper stabilization early
    frame_transforms: dict = {}
    frame_radar_transforms: dict = {}
    frame_keypoints: dict = {}
    pitch_tracker = None
    if args.metric:
        from world_cup_projects.common.pitch import (
            iter_pitch_transformers,
            warmup_goal_defenders,
        )
        from world_cup_projects.common.teams import stabilize_goalkeeper_teams

        # Check for cached pitch data
        cache_dir = Path(".cache/pitch")
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_name = f"{sequence.name}_{args.device}_{end}_{args.pitch_confidence}.pkl"
        cache_path = cache_dir / cache_name

        from world_cup_projects.common.pitch import resolve_radar_anchor

        detections_by_frame = {int(fi): d for fi, d in frames}
        radar_anchor = None

        if not args.refresh_detections_cache and cache_path.exists():
            import pickle
            with open(cache_path, "rb") as f:
                cached_data = pickle.load(f)
                frame_transforms = cached_data["transforms"]
                frame_radar_transforms = cached_data["radar_transforms"]
                frame_keypoints = cached_data["keypoints"]
                locked_goals = cached_data.get("locked_goals")
                radar_anchor = cached_data.get("radar_anchor")
            print(f"Loaded cached pitch homography: {cache_path.name}")
        else:
            print(f"Running pitch homography model (will cache to {cache_path.name})...")
            for frame_idx, speed_t, radar_t, kps, tracker in iter_pitch_transformers(
                sequence,
                device=args.device,
                end=end,
                confidence=args.pitch_confidence,
                yield_keypoints=True,
                yield_tracker=True,
                detections_by_frame=detections_by_frame,
            ):
                frame_transforms[frame_idx] = speed_t
                frame_radar_transforms[frame_idx] = radar_t
                frame_keypoints[frame_idx] = kps
                pitch_tracker = tracker

            locked_goals = None
            if args.source in ("football", "rfdetr"):
                locked_goals = warmup_goal_defenders(pitch_tracker, frames, frame_transforms)

            radar_anchor = resolve_radar_anchor(
                frames,
                frame_keypoints,
                confidence=args.pitch_confidence,
            )

            import pickle
            with open(cache_path, "wb") as f:
                pickle.dump(
                    {
                        "transforms": frame_transforms,
                        "radar_transforms": frame_radar_transforms,
                        "keypoints": frame_keypoints,
                        "locked_goals": locked_goals,
                        "radar_anchor": radar_anchor,
                    },
                    f,
                )
            print(f"Wrote pitch cache: {cache_path}")

        if radar_anchor is None:
            radar_anchor = resolve_radar_anchor(
                frames,
                frame_keypoints,
                confidence=args.pitch_confidence,
            )

        if args.source in ("football", "rfdetr"):
            stabilize_goalkeeper_teams(frames, frame_transforms, locked_goals)

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
            radar_anchor=radar_anchor,
        )
        manifest["video"] = render_manifest["output"]
        json_path.write_text(json.dumps(manifest, indent=2))
        print(f"Wrote {video_path}")

    print(json.dumps(manifest, indent=2))
    print(f"\nWrote {json_path}  ({manifest['n_passes']} inferred passes)")


if __name__ == "__main__":
    main()
