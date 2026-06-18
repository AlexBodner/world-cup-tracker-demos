"""Shared CLI flags for football detection runs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, fields


@dataclass(frozen=True)
class FootballDetectionDefaults:
    source: str = "gt"
    tracker: str = "botsort"
    device: str = "cpu"
    detector_backend: str = "yolo"
    player_model_id: str | None = None
    detection_threshold: float = 0.5
    ball_threshold: float | None = None
    ball_detector_backend: str = "yolo"
    ball_ensemble_mode: str = "fallback"
    ball_model_id: str | None = None
    pitch_confidence: float = 0.9
    refresh_detections_cache: bool = False
    legacy_detections_cache: bool = False


@dataclass
class FootballRunConfig:
    source: str
    tracker: str
    device: str
    detector_backend: str
    player_model_id: str | None
    detection_threshold: float
    ball_threshold: float | None
    ball_detector_backend: str
    ball_ensemble_mode: str
    ball_model_id: str | None
    pitch_confidence: float
    refresh_detections_cache: bool
    legacy_detections_cache: bool
    metric: bool = False
    debug_pitch_keypoints: bool = False
    sequence: str | None = None
    video: str | None = None
    max_frames: int | None = None

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> FootballRunConfig:
        kwargs = {}
        for f in fields(cls):
            if hasattr(args, f.name):
                kwargs[f.name] = getattr(args, f.name)
        return cls(**kwargs)


def add_football_detection_args(
    parser: argparse.ArgumentParser,
    *,
    defaults: FootballDetectionDefaults | None = None,
    include_metric: bool = True,
    include_pitch_debug: bool = True,
    include_sequence: bool = True,
    ball_detector_choices: tuple[str, ...] | None = None,
    include_ball_ensemble: bool = True,
) -> None:
    """Register shared football detection flags on ``parser``."""
    d = defaults or FootballDetectionDefaults()
    if include_sequence:
        parser.add_argument("--sequence", default=None)
        parser.add_argument(
            "--video",
            default=None,
            help="MP4 path (implies --source football when applicable).",
        )
        parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument(
        "--source",
        choices=("gt", "football", "rfdetr"),
        default=d.source,
    )
    parser.add_argument(
        "--tracker",
        choices=("bytetrack", "botsort", "botsort_nocmc"),
        default=d.tracker,
    )
    if include_metric:
        parser.add_argument("--metric", action="store_true")
    parser.add_argument("--device", default=d.device)
    parser.add_argument("--pitch-confidence", type=float, default=d.pitch_confidence)
    parser.add_argument(
        "--refresh-detections-cache",
        action="store_true",
    )
    parser.add_argument(
        "--legacy-detections-cache",
        action="store_true",
    )
    parser.add_argument(
        "--detector-backend",
        choices=("yolo", "inference"),
        default=d.detector_backend,
        help="football source: local Ultralytics .pt (yolo) or Roboflow Inference",
    )
    parser.add_argument(
        "--player-model-id",
        default=d.player_model_id,
        help="Universe model id for --detector-backend inference",
    )
    parser.add_argument(
        "--detection-threshold",
        type=float,
        default=d.detection_threshold,
        help="Player / GK / referee detection confidence threshold",
    )
    parser.add_argument(
        "--ball-threshold",
        type=float,
        default=d.ball_threshold,
        help="Ball class confidence threshold (default 0.20)",
    )
    ball_choices = ball_detector_choices or ("none", "yolo", "inference")
    parser.add_argument(
        "--ball-detector-backend",
        choices=ball_choices,
        default=d.ball_detector_backend,
        help="Dedicated ball model stacked on the player detector",
    )
    if include_ball_ensemble and "yolo" in ball_choices:
        parser.add_argument(
            "--ball-ensemble",
            choices=("fallback", "merge"),
            default=d.ball_ensemble_mode,
            dest="ball_ensemble_mode",
            help="How to combine player-model ball with --ball-detector-backend",
        )
    parser.add_argument(
        "--ball-model-id",
        default=d.ball_model_id,
        help="Universe ball model id",
    )
    if include_pitch_debug:
        parser.add_argument(
            "--debug-pitch-keypoints",
            action="store_true",
            help="Draw pitch keypoints on main video",
        )
