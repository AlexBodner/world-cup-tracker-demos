"""Fast ball-detection benchmark: one YOLO pass, many ball thresholds.

Also tests optional Inference dedicated ball model on a sample of missing frames.
"""

from __future__ import annotations

import argparse
import pickle
import random
import sys
from pathlib import Path

_pkg_root = Path(__file__).resolve().parents[1]
_repo_root = _pkg_root.parent
if (_repo_root / "world_cup_projects" / "__init__.py").is_file():
    if str(_repo_root) not in sys.path:
        sys.path.insert(0, str(_repo_root))

import numpy as np
import supervision as sv
from ultralytics import YOLO

from world_cup_projects.common.detect import (
    DEFAULT_BALL_DETECTION_THRESHOLD,
    DEFAULT_FOOTBALL_BALL_MODEL_ID,
    FootballBallInferenceDetector,
    _apply_fp_class_thresholds,
    _map_fp_detections,
    ensure_football_players_model,
)
from world_cup_projects.common.detection_cache import _record_to_detections
from world_cup_projects.common.possession import ball_xy
from world_cup_projects.common.soccernet import ROLE_BALL
from world_cup_projects.common.video import load_video_sequence, read_sequence_frame


def _ball_rate_from_preds(preds: list[sv.Detections], ball_thr: float, player_thr: float) -> float:
    hits = 0
    for det in preds:
        filtered = _apply_fp_class_thresholds(
            det, player_threshold=player_thr, ball_threshold=ball_thr
        )
        if (filtered.class_id == ROLE_BALL).any():
            hits += 1
    return hits / max(len(preds), 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=750)
    parser.add_argument("--player-threshold", type=float, default=0.5)
    parser.add_argument(
        "--ball-thresholds",
        default="0.5,0.35,0.25,0.20,0.15,0.10",
    )
    parser.add_argument("--test-dedicated-ball", action="store_true")
    parser.add_argument("--dedicated-sample", type=int, default=30)
    args = parser.parse_args()

    seq = load_video_sequence(args.video)
    end = min(args.max_frames, seq.length)
    model = YOLO(str(ensure_football_players_model()))

    preds: list[sv.Detections] = []
    for fi in range(1, end + 1):
        img = read_sequence_frame(seq, fi)
        if img is None:
            preds.append(sv.Detections.empty())
            continue
        raw = sv.Detections.from_ultralytics(
            model.predict(img, conf=0.05, verbose=False, device="cpu")[0]
        )
        preds.append(_map_fp_detections(raw))

    thresholds = [float(x.strip()) for x in args.ball_thresholds.split(",") if x.strip()]
    print(f"{seq.name} frames 1..{end}  player_thr={args.player_threshold}")
    print(f"{'ball_thr':>8}  {'visible':>8}  {'rate':>7}")
    print("-" * 28)
    for thr in thresholds:
        rate = _ball_rate_from_preds(preds, thr, args.player_threshold)
        print(f"{thr:>8.2f}  {int(rate * len(preds)):>8}  {rate:>6.1%}")

    print(f"\nPipeline default ball_thr={DEFAULT_BALL_DETECTION_THRESHOLD}")

    if args.test_dedicated_ball:
        import os

        if not os.environ.get("ROBOFLOW_API_KEY"):
            print("\nSkip dedicated ball model: ROBOFLOW_API_KEY not set")
            return

        missing = []
        for fi, det in enumerate(preds, start=1):
            filtered = _apply_fp_class_thresholds(
                det,
                player_threshold=args.player_threshold,
                ball_threshold=DEFAULT_BALL_DETECTION_THRESHOLD,
            )
            if ball_xy(filtered) is None:
                missing.append(fi)

        ball_det = FootballBallInferenceDetector(model_id=DEFAULT_FOOTBALL_BALL_MODEL_ID)
        sample = sorted(random.sample(missing, min(args.dedicated_sample, len(missing))))
        recovered = 0
        for fi in sample:
            img = read_sequence_frame(seq, fi)
            if ball_xy(ball_det.detect(img)) is not None:
                recovered += 1
        print(
            f"\nDedicated ball model ({DEFAULT_FOOTBALL_BALL_MODEL_ID}) "
            f"on {len(sample)} frames still missing at ball_thr={DEFAULT_BALL_DETECTION_THRESHOLD}: "
            f"recovered {recovered}/{len(sample)} ({100*recovered/max(len(sample),1):.0f}%)"
        )


if __name__ == "__main__":
    main()
