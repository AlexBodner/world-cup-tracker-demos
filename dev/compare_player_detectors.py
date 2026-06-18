"""Compare football player detectors (YOLO local vs Inference v11/v20).

Runs the same clip through each detector backend + model id and reports detection
quality proxies plus downstream pass / freeze stats.

Example::

    export ROBOFLOW_API_KEY=your_key

    PYTHONPATH=. python -m world_cup_projects.dev.compare_player_detectors \\
        --video world_cup_projects/bundesliga_videos/08fd33_0.mp4 \\
        --metric --device cpu \\
        --configs yolo:11,inference:11,inference:20
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

_pkg_root = Path(__file__).resolve().parents[1]
_repo_root = _pkg_root.parent
if (_repo_root / "world_cup_projects" / "__init__.py").is_file():
    if str(_repo_root) not in sys.path:
        sys.path.insert(0, str(_repo_root))

import numpy as np

from world_cup_projects import DEFAULT_ASSETS_DIR
from world_cup_projects.common.detect import (
    DEFAULT_BALL_DETECTION_THRESHOLD,
    DEFAULT_FOOTBALL_PLAYERS_MODEL_ID,
    FOOTBALL_PLAYERS_INFERENCE_V11,
    FOOTBALL_PLAYERS_INFERENCE_V20,
    iter_football_model_detections,
)
from world_cup_projects.common.possession import ball_xy, carrier_from_tracker_id, player_mask
from world_cup_projects.common.soccernet import ROLE_BALL
from world_cup_projects.common.teams import stabilize_teams_by_tracklet
from world_cup_projects.common.video import load_video_sequence
from world_cup_projects.pass_alternatives.pass_options import PassWeights
from world_cup_projects.dev.freeze_debug import diagnose_all_passes
from world_cup_projects.player_stats.pass_events import (
    PassDetectionConfig,
    PassQualityScorer,
    scan_possession_events,
)


@dataclass(frozen=True)
class DetectorConfig:
    label: str
    backend: str
    model_id: str


@dataclass
class DetectorReport:
    label: str
    backend: str
    model_id: str
    frames: int
    ball_detection_rate: float
    mean_trackable_players: float
    mean_total_detections: float
    n_passes: int
    n_quality_scored: int
    n_would_freeze: int
    n_receiver_missing_at_release: int
    passes: list[dict]


def _parse_config(spec: str) -> DetectorConfig:
    """Parse ``backend:version`` e.g. ``inference:20`` or ``yolo:11``."""
    if ":" not in spec:
        raise ValueError(f"Expected backend:version, got {spec!r}")
    backend, version = spec.split(":", 1)
    backend = backend.strip().lower()
    version = version.strip()
    if backend not in ("yolo", "inference"):
        raise ValueError(f"Unknown backend {backend!r} in {spec!r}")
    model_id = f"football-players-detection-3zvbc/{version}"
    return DetectorConfig(label=spec, backend=backend, model_id=model_id)


def _load_pitch_transforms(sequence, frames, *, device, end, pitch_confidence):
    from pathlib import Path
    import pickle

    from world_cup_projects.common.pitch import iter_pitch_transformers, warmup_goal_defenders_radar
    from world_cup_projects.common.teams import stabilize_goalkeeper_teams

    from world_cup_projects import PACKAGE_ROOT

    cache_dir = PACKAGE_ROOT / ".cache" / "pitch"
    detections_by_frame = {int(fi): d for fi, d in frames}
    frame_transforms: dict = {}
    frame_keypoints: dict = {}

    for candidate in sorted(
        cache_dir.glob(f"{sequence.name}_*_{end}_{pitch_confidence}.pkl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    ):
        with candidate.open("rb") as f:
            cached = pickle.load(f)
        if {"transforms", "keypoints"} <= cached.keys():
            frame_transforms = cached["transforms"]
            frame_keypoints = cached["keypoints"]
            print(f"  pitch cache: {candidate.name}")
            break
    else:
        for frame_idx, speed_t, _radar, kps, _trk in iter_pitch_transformers(
            sequence,
            device=device,
            end=end,
            confidence=pitch_confidence,
            yield_keypoints=True,
            yield_tracker=True,
            detections_by_frame=detections_by_frame,
        ):
            frame_transforms[frame_idx] = speed_t
            frame_keypoints[frame_idx] = kps

    if frame_keypoints:
        locked = warmup_goal_defenders_radar(
            frames, frame_keypoints, confidence=pitch_confidence
        )
        stabilize_goalkeeper_teams(
            frames,
            locked_goal_defenders=locked,
            keypoints_by_frame=frame_keypoints,
            pitch_confidence=pitch_confidence,
        )
    return frame_transforms, frame_keypoints


def _detection_stats(frames: list[tuple[int, object]]) -> tuple[float, float, float, int]:
    ball_frames = 0
    trackable_counts: list[int] = []
    total_counts: list[int] = []
    for _fi, dets in frames:
        total_counts.append(len(dets))
        if ball_xy(dets) is not None:
            ball_frames += 1
        pmask = player_mask(dets)
        trackable_counts.append(int(pmask.sum()))
    n = max(len(frames), 1)
    return (
        ball_frames / n,
        float(np.mean(trackable_counts)) if trackable_counts else 0.0,
        float(np.mean(total_counts)) if total_counts else 0.0,
        n,
    )


def evaluate_detector(
    sequence,
    cfg: DetectorConfig,
    *,
    end: int,
    device: str,
    threshold: float,
    ball_threshold: float,
    tracker: str,
    metric: bool,
    pitch_confidence: float,
    refresh: bool,
) -> DetectorReport:
    from world_cup_projects.common.detection_cache import wrap_detections_cache

    print(f"\n=== {cfg.label} ({cfg.backend}, {cfg.model_id}) ===")
    source_name = "football" if cfg.backend == "yolo" else f"football_{cfg.backend}"
    cache_params: dict = {
        "device": device,
        "threshold": threshold,
        "ball_threshold": ball_threshold,
        "tracker": tracker,
    }
    if cfg.backend == "inference":
        cache_params["backend"] = cfg.backend
        cache_params["model_id"] = cfg.model_id
    detections_source = wrap_detections_cache(
        iter_football_model_detections,
        source_name=source_name,
        refresh=refresh,
        **cache_params,
    )
    frames = list(detections_source(sequence, start=1, end=end))
    frames = stabilize_teams_by_tracklet(frames)

    ball_rate, mean_players, mean_dets, n_frames = _detection_stats(frames)
    print(f"  ball visible: {ball_rate:.1%}  trackable players/frame: {mean_players:.1f}")

    frame_transforms: dict = {}
    frame_keypoints: dict = {}
    if metric:
        frame_transforms, frame_keypoints = _load_pitch_transforms(
            sequence, frames, device=device, end=end, pitch_confidence=pitch_confidence
        )

    weights = PassWeights.metric() if metric else PassWeights()
    scorer = PassQualityScorer(
        weights=weights, metric=metric, transformers=frame_transforms
    )
    config = PassDetectionConfig().for_frame_rate(sequence.frame_rate)
    scan = scan_possession_events(
        iter(frames),
        scorer=scorer,
        config=config,
        metric=metric,
        transformers=frame_transforms,
    )
    passes = list(scan.passes)
    frames_by_idx = {int(fi): d for fi, d in frames}
    diagnoses = diagnose_all_passes(
        passes,
        frames_by_idx,
        scorer=scorer,
        weights=weights,
        show_predictions=True,
        keypoints_by_frame=frame_keypoints,
    )

    n_quality = sum(1 for p in passes if p.quality_score is not None)
    n_freeze = sum(1 for d in diagnoses if d.would_freeze)
    n_recv_missing = 0
    for p in passes:
        dets = frames_by_idx.get(p.frame_idx)
        if dets is None:
            continue
        if carrier_from_tracker_id(dets, p.receiver_tid) is None:
            n_recv_missing += 1

    print(
        f"  passes: {len(passes)}  quality_scored: {n_quality}  "
        f"would_freeze: {n_freeze}  recv_missing@release: {n_recv_missing}"
    )

    pass_rows = []
    for p, d in zip(passes, diagnoses):
        pass_rows.append(
            {
                "frame": p.frame_idx,
                "passer": p.passer_tid,
                "receiver": p.receiver_tid,
                "quality_score": p.quality_score,
                "pass_length_m": p.pass_length_m,
                "would_freeze": d.would_freeze,
                "blockers": d.blockers,
            }
        )

    return DetectorReport(
        label=cfg.label,
        backend=cfg.backend,
        model_id=cfg.model_id,
        frames=n_frames,
        ball_detection_rate=ball_rate,
        mean_trackable_players=mean_players,
        mean_total_detections=mean_dets,
        n_passes=len(passes),
        n_quality_scored=n_quality,
        n_would_freeze=n_freeze,
        n_receiver_missing_at_release=n_recv_missing,
        passes=pass_rows,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare football player detector configs")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--metric", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tracker", default="botsort")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--pitch-confidence", type=float, default=0.9)
    parser.add_argument("--detection-threshold", type=float, default=0.5)
    parser.add_argument(
        "--ball-threshold",
        type=float,
        default=DEFAULT_BALL_DETECTION_THRESHOLD,
    )
    parser.add_argument("--refresh-detections-cache", action="store_true")
    parser.add_argument(
        "--configs",
        default=f"yolo:11,inference:11,inference:20",
        help="Comma-separated backend:version list (e.g. yolo:11,inference:20)",
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    sequence = load_video_sequence(args.video)
    end = args.max_frames if args.max_frames is not None else sequence.length
    configs = [_parse_config(s.strip()) for s in args.configs.split(",") if s.strip()]

    reports: list[DetectorReport] = []
    for cfg in configs:
        if cfg.backend == "inference":
            from world_cup_projects.common.detect import ensure_football_players_inference

            ensure_football_players_inference(model_id=cfg.model_id)
        reports.append(
            evaluate_detector(
                sequence,
                cfg,
                end=end,
                device=args.device,
                threshold=args.detection_threshold,
                ball_threshold=args.ball_threshold,
                tracker=args.tracker,
                metric=args.metric,
                pitch_confidence=args.pitch_confidence,
                refresh=args.refresh_detections_cache,
            )
        )

    out_dir = args.out_dir or (DEFAULT_ASSETS_DIR / "detector_compare")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"compare_{sequence.name}.json"
    payload = {
        "sequence": sequence.name,
        "metric": args.metric,
        "default_model_id": DEFAULT_FOOTBALL_PLAYERS_MODEL_ID,
        "models_compared": [FOOTBALL_PLAYERS_INFERENCE_V11, FOOTBALL_PLAYERS_INFERENCE_V20],
        "reports": [asdict(r) for r in reports],
    }
    out_path.write_text(json.dumps(payload, indent=2))

    print(f"\n{'label':<16} {'ball%':>6} {'players':>8} {'passes':>7} {'qs':>4} {'freeze':>7} {'recv@rel':>9}")
    print("-" * 70)
    for r in reports:
        print(
            f"{r.label:<16} {r.ball_detection_rate:>6.1%} {r.mean_trackable_players:>8.1f} "
            f"{r.n_passes:>7} {r.n_quality_scored:>4} {r.n_would_freeze:>7} "
            f"{r.n_receiver_missing_at_release:>9}"
        )
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
