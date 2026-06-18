#!/usr/bin/env python3
"""Compare inferred pass-network output against ground-truth pass lists."""

from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter
from pathlib import Path

from world_cup_projects import PACKAGE_ROOT

_GT_DIR = Path(__file__).resolve().parent / "passes"
_CACHE_DET = PACKAGE_ROOT / ".cache/detections"
_CACHE_PITCH = PACKAGE_ROOT / ".cache/pitch"


def _pair(passer: int, receiver: int) -> tuple[int, int]:
    return (passer, receiver)


def load_ground_truth(sequence: str) -> dict:
    path = _GT_DIR / f"{sequence}.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def load_detection(path: Path) -> dict:
    return json.loads(path.read_text())


def _pick_cache(paths: list[Path], *, prefer_substr: str = "__football__") -> Path:
    """Pick the canonical full-sequence cache (prefer football over inference)."""
    preferred = [p for p in paths if prefer_substr in p.name and "inference" not in p.name]
    if preferred:
        return preferred[-1]
    return paths[-1]


def scan_live(sequence: str, *, fps: float = 25.0) -> dict:
    """Run pass detection on cached detections + pitch homography."""
    from world_cup_projects.common.detection_cache import load_cached_detections
    from world_cup_projects.common.pipeline import prepare_model_frames
    from world_cup_projects.common.pitch import warmup_goal_defenders_radar
    from world_cup_projects.common.teams import stabilize_goalkeeper_teams
    from world_cup_projects.player_stats.pass_events import (
        PassDetectionConfig,
        PassQualityScorer,
        scan_possession_events,
    )

    det_paths = sorted(_CACHE_DET.glob(f"{sequence}*device=mps*end=750*botsort.pkl"))
    pitch_paths = sorted(_CACHE_PITCH.glob(f"{sequence}*mps*750*.pkl"))
    if not det_paths:
        det_paths = sorted(_CACHE_DET.glob(f"{sequence}*device=cpu*end=750*botsort.pkl"))
    if not pitch_paths:
        pitch_paths = sorted(_CACHE_PITCH.glob(f"{sequence}*cpu*750*.pkl"))
    if not det_paths or not pitch_paths:
        raise FileNotFoundError(
            f"No mps cache for {sequence} under {_CACHE_DET} / {_CACHE_PITCH}"
        )
    det_path = _pick_cache(det_paths)
    pitch_path = pitch_paths[-1]
    pitch_data = pickle.load(pitch_path.open("rb"))
    transformers = pitch_data["transforms"]
    keypoints = pitch_data.get("keypoints")
    _, frames = load_cached_detections(det_path)
    frames = prepare_model_frames(frames, frame_width=1920)
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
        "passes": [
            {"passer_tid": p.passer_tid, "receiver_tid": p.receiver_tid}
            for p in scan.passes
        ],
        "turnovers": [
            {"passer_tid": t.passer_tid, "interceptor_tid": t.interceptor_tid}
            for t in scan.turnovers
        ],
    }


def compare(gt: dict, detected: dict) -> dict:
    gt_passes = Counter(
        _pair(p["passer_tid"], p["receiver_tid"]) for p in gt.get("passes", [])
    )
    det_passes = Counter(
        _pair(p["passer_tid"], p["receiver_tid"]) for p in detected.get("passes", [])
    )
    gt_turnovers = Counter(
        _pair(t["passer_tid"], t["interceptor_tid"]) for t in gt.get("turnovers", [])
    )
    det_turnovers = Counter(
        _pair(t["passer_tid"], t["interceptor_tid"])
        for t in detected.get("turnovers", [])
    )

    def _diff(expected: Counter, found: Counter) -> tuple[list, list, list]:
        matched: list[tuple[int, int]] = []
        missing: list[tuple[int, int]] = []
        extra: list[tuple[int, int]] = []
        all_keys = set(expected) | set(found)
        for key in sorted(all_keys):
            m = min(expected[key], found[key])
            if m:
                matched.extend([key] * m)
            missing.extend([key] * max(0, expected[key] - found[key]))
            extra.extend([key] * max(0, found[key] - expected[key]))
        return matched, missing, extra

    pass_matched, pass_missing, pass_extra = _diff(gt_passes, det_passes)
    to_matched, to_missing, to_extra = _diff(gt_turnovers, det_turnovers)

    return {
        "sequence": gt.get("sequence", detected.get("sequence")),
        "passes": {
            "expected": sum(gt_passes.values()),
            "detected": sum(det_passes.values()),
            "matched": pass_matched,
            "missing": pass_missing,
            "extra": pass_extra,
        },
        "turnovers": {
            "expected": sum(gt_turnovers.values()),
            "detected": sum(det_turnovers.values()),
            "matched": to_matched,
            "missing": to_missing,
            "extra": to_extra,
        },
    }


def _print_report(report: dict) -> None:
    seq = report["sequence"]
    print(f"=== {seq} ===")
    for kind in ("passes", "turnovers"):
        block = report[kind]
        print(
            f"{kind}: {len(block['matched'])}/{block['expected']} matched "
            f"(detected {block['detected']})"
        )
        if block["missing"]:
            print(f"  missing: {block['missing']}")
        if block["extra"]:
            print(f"  extra:   {block['extra']}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "sequence",
        help="Clip name without extension (e.g. 08fd33_0, 0bfacc_1)",
    )
    parser.add_argument(
        "--detected",
        type=Path,
        help="Path to pass_network JSON (default: assets/pass_network_football_metric_<seq>.json)",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Scan cached detections with current pass_events logic (ignores --detected)",
    )
    args = parser.parse_args()

    gt = load_ground_truth(args.sequence)
    if args.live:
        detected = scan_live(args.sequence)
    else:
        detected_path = args.detected or (
            Path("assets") / f"pass_network_football_metric_{args.sequence}.json"
        )
        detected = load_detection(detected_path)
    report = compare(gt, detected)
    _print_report(report)


if __name__ == "__main__":
    main()
