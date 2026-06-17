#!/usr/bin/env python3
"""Scan all GT clips with dedicated ball YOLO fallback ensemble and compare."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

_pkg_root = Path(__file__).resolve().parents[1]
_repo_root = _pkg_root.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from world_cup_projects.common.detection_cache import cache_path, load_cached_detections
from world_cup_projects.common.player_tracker import tracker_cache_key_params
from world_cup_projects.common.pitch import warmup_goal_defenders_radar
from world_cup_projects.common.teams import (
    enforce_one_goalkeeper_per_team_frames,
    stabilize_goalkeeper_teams,
    stabilize_teams_by_tracklet,
)
from world_cup_projects.common.video import load_video_sequence
from world_cup_projects.ground_truth.compare_passes import compare, load_ground_truth
from world_cup_projects.player_stats.pass_events import (
    PassDetectionConfig,
    PassQualityScorer,
    scan_possession_events,
)

_CACHE_DET = _pkg_root / ".cache/detections"
_CACHE_PITCH = _pkg_root / ".cache/pitch"
_GT_DIR = Path(__file__).resolve().parent / "passes"
_VIDEOS = _pkg_root / "bundesliga_videos"


def _detection_cache_path(sequence: str, *, ball_threshold: float) -> Path:
    video = _VIDEOS / f"{sequence}.mp4"
    if not video.is_file():
        raise FileNotFoundError(video)
    seq = load_video_sequence(video)
    end = seq.length
    return cache_path(
        seq,
        "football",
        start=1,
        end=end,
        device="cpu",
        threshold=0.5,
        tracker="botsort",
        ball_threshold=ball_threshold,
        ball_backend="yolo",
        ball_mid="4",
        **tracker_cache_key_params(),
    )


def scan_sequence(
    sequence: str,
    *,
    ball_threshold: float = 0.20,
    fps: float = 25.0,
) -> dict:
    det_path = _detection_cache_path(sequence, ball_threshold=ball_threshold)
    if not det_path.is_file():
        legacy = sorted(
            _CACHE_DET.glob(
                f"{sequence}*ball_backend=yolo*ball_threshold={ball_threshold}*end=750*botsort.pkl"
            )
        )
        if not legacy:
            raise FileNotFoundError(det_path)
        det_path = legacy[-1]

    pitch_paths = sorted(_CACHE_PITCH.glob(f"{sequence}*cpu*750*.pkl"))
    if not pitch_paths:
        pitch_paths = sorted(_CACHE_PITCH.glob(f"{sequence}*mps*750*.pkl"))
    if not pitch_paths:
        raise FileNotFoundError(f"No pitch cache for {sequence}")
    pitch_path = pitch_paths[-1]
    pitch_data = pickle.load(pitch_path.open("rb"))
    transformers = pitch_data["transforms"]
    keypoints = pitch_data.get("keypoints")

    _, frames = load_cached_detections(det_path)
    frames = stabilize_teams_by_tracklet(frames)
    frames = enforce_one_goalkeeper_per_team_frames(frames, frame_width=1920)
    if keypoints is not None:
        locked = warmup_goal_defenders_radar(frames, keypoints, confidence=0.9)
        stabilize_goalkeeper_teams(
            frames,
            locked_goal_defenders=locked,
            keypoints_by_frame=keypoints,
            pitch_confidence=0.9,
            frame_width=1920,
        )

    scan = scan_possession_events(
        iter(frames),
        scorer=PassQualityScorer(),
        config=PassDetectionConfig(),
        metric=True,
        transformers=transformers,
        fps=fps,
    )
    return {
        "sequence": sequence,
        "detection_cache": det_path.name,
        "passes": [
            {"passer_tid": p.passer_tid, "receiver_tid": p.receiver_tid}
            for p in scan.passes
        ],
        "turnovers": [
            {
                "passer_tid": t.passer_tid,
                "interceptor_tid": t.interceptor_tid,
            }
            for t in scan.turnovers
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ball-threshold",
        type=float,
        default=0.20,
        help="Ball threshold used when building/using detection cache",
    )
    parser.add_argument(
        "--sequences",
        default="0bfacc_1,08fd33_0,08fd33_8,08fd33_4",
        help="Comma-separated GT clip names",
    )
    args = parser.parse_args()

    sequences = [s.strip() for s in args.sequences.split(",") if s.strip()]
    summary: list[dict] = []

    for seq in sequences:
        gt_path = _GT_DIR / f"{seq}.json"
        if not gt_path.is_file():
            print(f"SKIP {seq}: no GT file")
            continue
        gt = load_ground_truth(seq)
        try:
            detected = scan_sequence(seq, ball_threshold=args.ball_threshold)
        except FileNotFoundError as exc:
            video = _VIDEOS / f"{seq}.mp4"
            print(f"MISSING CACHE {seq}: {exc}")
            if video.is_file():
                print(
                    f"  Run: PYTHONPATH=.. python -m player_stats.pass_network_run "
                    f'--video bundesliga_videos/{seq}.mp4 --metric --no-render '
                    f"--ball-detector-backend yolo --ball-threshold {args.ball_threshold} "
                    f"--ball-ensemble fallback --tracker botsort --device cpu"
                )
            continue

        report = compare(gt, detected)
        summary.append(report)
        print(f"=== {seq} ===")
        print(f"  cache: {detected['detection_cache']}")
        print(
            f"  passes: {len(report['passes']['matched'])}/{report['passes']['expected']} matched "
            f"(detected {report['passes']['detected']})"
        )
        if report["passes"]["missing"]:
            print(f"    missing: {report['passes']['missing']}")
        if report["passes"]["extra"]:
            print(f"    extra:   {report['passes']['extra']}")
        det_passes = [(p["passer_tid"], p["receiver_tid"]) for p in detected["passes"]]
        print(f"    detected list: {det_passes}")
        print(
            f"  turnovers: {len(report['turnovers']['matched'])}/{report['turnovers']['expected']} matched "
            f"(detected {report['turnovers']['detected']})"
        )
        if report["turnovers"]["missing"]:
            print(f"    missing: {report['turnovers']['missing']}")
        if report["turnovers"]["extra"]:
            print(f"    extra:   {report['turnovers']['extra']}")
        det_to = [(t["passer_tid"], t["interceptor_tid"]) for t in detected["turnovers"]]
        print(f"    detected list: {det_to}")
        print()

    out = _pkg_root / "assets" / "gt_ball_ensemble_scan.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
