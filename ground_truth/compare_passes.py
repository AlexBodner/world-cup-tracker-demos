#!/usr/bin/env python3
"""Compare inferred pass-network output against ground-truth pass lists."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

_GT_DIR = Path(__file__).resolve().parent / "passes"


def _pair(passer: int, receiver: int) -> tuple[int, int]:
    return (passer, receiver)


def load_ground_truth(sequence: str) -> dict:
    path = _GT_DIR / f"{sequence}.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def load_detection(path: Path) -> dict:
    return json.loads(path.read_text())


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
    args = parser.parse_args()

    detected_path = args.detected or (
        Path("assets") / f"pass_network_football_metric_{args.sequence}.json"
    )
    gt = load_ground_truth(args.sequence)
    detected = load_detection(detected_path)
    report = compare(gt, detected)
    _print_report(report)


if __name__ == "__main__":
    main()
