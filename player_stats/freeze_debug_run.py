"""Debug why pass-alternative freezes skip detected passes (per-pass gate report + video).

Example::

    PYTHONPATH=. python -m world_cup_projects.player_stats.freeze_debug_run \\
        --video world_cup_projects/bundesliga_videos/08fd33_0.mp4 \\
        --metric --device cpu
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
from world_cup_projects.common.video import load_video_sequence, read_sequence_frame
from world_cup_projects.pass_alternatives.pass_options import PassWeights
from world_cup_projects.player_stats.freeze_debug import (
    diagnose_all_passes,
    render_freeze_debug_video,
)
from world_cup_projects.player_stats.pass_events import (
    PassDetectionConfig,
    PassQualityScorer,
    scan_possession_events,
)
from world_cup_projects.player_stats.pass_network_run import (
    _load_detections_source,
    analyze_pass_network,
)


def _load_pitch_maps(sequence, frames, *, device, end, pitch_confidence, refresh: bool):
    from world_cup_projects.common.pitch import iter_pitch_transformers, warmup_goal_defenders_radar
    from world_cup_projects.common.teams import stabilize_goalkeeper_teams

    cache_dir = Path(".cache/pitch")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{sequence.name}_{device}_{end}_{pitch_confidence}.pkl"
    detections_by_frame = {int(fi): d for fi, d in frames}

    frame_transforms: dict = {}
    frame_radar_transforms: dict = {}
    frame_keypoints: dict = {}
    locked_goals = None

    cache_ok = False
    cache_candidates = [cache_path]
    if not refresh:
        cache_candidates.extend(
            sorted(
                cache_dir.glob(f"{sequence.name}_*_{end}_{pitch_confidence}.pkl"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
        )
    import pickle

    for candidate in cache_candidates:
        if not candidate.is_file():
            continue
        with candidate.open("rb") as f:
            cached = pickle.load(f)
        if {"transforms", "radar_transforms", "keypoints"} <= cached.keys():
            frame_transforms = cached["transforms"]
            frame_radar_transforms = cached["radar_transforms"]
            frame_keypoints = cached["keypoints"]
            cache_ok = True
            print(f"Loaded cached pitch homography: {candidate.name}")
            break

    if not cache_ok:
        print(f"Running pitch homography (cache → {cache_path.name})...")
        for frame_idx, speed_t, radar_t, kps, _trk in iter_pitch_transformers(
            sequence,
            device=device,
            end=end,
            confidence=pitch_confidence,
            yield_keypoints=True,
            yield_tracker=True,
            detections_by_frame=detections_by_frame,
        ):
            frame_transforms[frame_idx] = speed_t
            frame_radar_transforms[frame_idx] = radar_t
            frame_keypoints[frame_idx] = kps
        with cache_path.open("wb") as f:
            pickle.dump(
                {
                    "transforms": frame_transforms,
                    "radar_transforms": frame_radar_transforms,
                    "keypoints": frame_keypoints,
                },
                f,
            )

    if frame_keypoints:
        locked_goals = warmup_goal_defenders_radar(
            frames, frame_keypoints, confidence=pitch_confidence
        )
        stabilize_goalkeeper_teams(
            frames,
            locked_goal_defenders=locked_goals,
            keypoints_by_frame=frame_keypoints,
            pitch_confidence=pitch_confidence,
        )

    return frame_transforms, frame_radar_transforms, frame_keypoints, locked_goals


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze gate debug video per detected pass")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--metric", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tracker", default="botsort")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--pitch-confidence", type=float, default=0.9)
    parser.add_argument("--refresh-detections-cache", action="store_true")
    parser.add_argument("--refresh-pitch-cache", action="store_true")
    parser.add_argument("--freeze-quality-threshold", type=float, default=0.0)
    parser.add_argument("--hold-seconds", type=float, default=3.5)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    sequence = load_video_sequence(args.video)
    end = args.max_frames if args.max_frames is not None else sequence.length

    class _Args:
        source = "football"
        device = args.device
        tracker = args.tracker
        refresh_detections_cache = args.refresh_detections_cache

    detections_source = _load_detections_source(_Args, sequence)
    frames = list(detections_source(sequence, start=1, end=end))

    from world_cup_projects.common.teams import stabilize_teams_by_tracklet

    frames = stabilize_teams_by_tracklet(frames)

    frame_transforms: dict = {}
    frame_keypoints: dict = {}
    if args.metric:
        frame_transforms, _radar, frame_keypoints, _goals = _load_pitch_maps(
            sequence,
            frames,
            device=args.device,
            end=end,
            pitch_confidence=args.pitch_confidence,
            refresh=args.refresh_pitch_cache,
        )

    config = PassDetectionConfig().for_frame_rate(sequence.frame_rate)
    weights = PassWeights.metric() if args.metric else PassWeights()
    scorer = PassQualityScorer(
        weights=weights, metric=args.metric, transformers=frame_transforms
    )

    scan = scan_possession_events(
        iter(frames),
        scorer=scorer,
        config=config,
        metric=args.metric,
        transformers=frame_transforms,
    )
    passes = list(scan.passes)
    frames_by_idx = {int(fi): d for fi, d in frames}

    diagnoses = diagnose_all_passes(
        passes,
        frames_by_idx,
        scorer=scorer,
        weights=weights,
        freeze_quality_threshold=args.freeze_quality_threshold,
        show_predictions=True,
        keypoints_by_frame=frame_keypoints,
    )

    out_dir = args.out_dir or (DEFAULT_ASSETS_DIR / "freeze_debug")
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = "metric" if args.metric else "pixel"
    json_path = out_dir / f"freeze_debug_{tag}_{sequence.name}.json"
    video_path = out_dir / f"freeze_debug_{tag}_{sequence.name}.mp4"

    manifest = render_freeze_debug_video(
        sequence,
        frames,
        passes,
        diagnoses,
        str(video_path),
        frame_loader=lambda fi: read_sequence_frame(sequence, fi),
        hold_seconds=args.hold_seconds,
    )
    json_path.write_text(json.dumps(manifest, indent=2))

    print(f"\nPasses: {manifest['n_passes']}  would freeze: {manifest['n_would_freeze']}  blocked: {manifest['n_blocked']}")
    print(f"Wrote {video_path}")
    print(f"Wrote {json_path}\n")

    for d in diagnoses:
        status = "FREEZE" if d.would_freeze else "SKIP "
        block = d.blockers[0] if d.blockers else "ok"
        qs = f"{d.quality_score:.2f}" if d.quality_score is not None else "null"
        print(
            f"  [{status}] frame {d.frame_idx:4d}  "
            f"#{d.passer_tid}→#{d.receiver_tid}  qs={qs}  — {block}"
        )


if __name__ == "__main__":
    main()
