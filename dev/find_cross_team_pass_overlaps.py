#!/usr/bin/env python3
"""Find cross-team overlapping pass flight windows in pass network manifests."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def flight_window(pass_entry: dict) -> tuple[int, int]:
    start = int(pass_entry["frame_idx"])
    gap = int(pass_entry["gap_frames"])
    end = start + gap  # inclusive reception frame (release..reception)
    return start, end


def overlap(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int] | None:
    lo = max(a[0], b[0])
    hi = min(a[1], b[1])
    if lo <= hi:
        return lo, hi
    return None


def pass_label(p: dict, idx: int) -> str:
    return (
        f"pass[{idx}] team={p['team']} "
        f"#{p['passer_tid']}→#{p['receiver_tid']} "
        f"release={p['frame_idx']} gap={p['gap_frames']} "
        f"recv={p['frame_idx'] + p['gap_frames']}"
    )


def analyze_manifest(path: Path) -> list[dict]:
    with path.open() as f:
        data = json.load(f)
    passes = data.get("passes") or []
    pairs: list[dict] = []
    for i, pa in enumerate(passes):
        wa = flight_window(pa)
        for j in range(i + 1, len(passes)):
            pb = passes[j]
            if "team" not in pa or "team" not in pb:
                continue
            if int(pa["team"]) == int(pb["team"]):
                continue
            wb = flight_window(pb)
            ov = overlap(wa, wb)
            if ov is None:
                continue
            pairs.append(
                {
                    "manifest": str(path),
                    "sequence": data.get("sequence"),
                    "pass_a": pa,
                    "pass_b": pb,
                    "idx_a": i,
                    "idx_b": j,
                    "window_a": wa,
                    "window_b": wb,
                    "overlap": ov,
                }
            )
    return pairs


def frames_both_teams_in_flight(passes: list[dict]) -> list[dict]:
    """Frames where at least one team-0 and one team-1 pass are in flight (release..recv)."""
    if not passes:
        return []
    max_frame = max(p["frame_idx"] + p["gap_frames"] for p in passes)
    min_frame = min(p["frame_idx"] for p in passes)
    spans: list[dict] = []
    run_start: int | None = None
    for f in range(min_frame, max_frame + 1):
        teams_in_flight = set()
        active = []
        for i, p in enumerate(passes):
            s, e = flight_window(p)
            if s <= f <= e:
                if "team" not in p:
                    continue
                teams_in_flight.add(int(p["team"]))
                active.append(i)
        both = 0 in teams_in_flight and 1 in teams_in_flight
        if both:
            if run_start is None:
                run_start = f
        elif run_start is not None:
            spans.append({"start": run_start, "end": f - 1, "note": "both teams in-flight"})
            run_start = None
    if run_start is not None:
        spans.append({"start": run_start, "end": max_frame, "note": "both teams in-flight"})
    return spans


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--assets",
        type=Path,
        default=Path(__file__).resolve().parent / "assets",
    )
    ap.add_argument("--glob", default="pass_network_football_metric_*.json")
    ap.add_argument("--sequence", default=None, help="Filter to one sequence id e.g. 08fd33_8")
    args = ap.parse_args()

    paths = sorted(args.assets.glob(args.glob))
    # exclude debug / inference sidecars
    paths = [
        p
        for p in paths
        if "carrier_debug" not in p.name
        and "pitch_kp_debug" not in p.name
        and "inference_v20" not in p.name
    ]

    all_pairs: list[dict] = []
    for path in paths:
        with path.open() as f:
            seq = json.load(f).get("sequence")
        if args.sequence and seq != args.sequence and args.sequence not in path.name:
            continue
        all_pairs.extend(analyze_manifest(path))

    print("=== Cross-team overlapping pass flight windows ===\n")
    if not all_pairs:
        print("No overlapping cross-team pass pairs found.")
    for item in all_pairs:
        pa, pb = item["pass_a"], item["pass_b"]
        ia, ib = item["idx_a"], item["idx_b"]
        wa, wb = item["window_a"], item["window_b"]
        ov = item["overlap"]
        print(f"File: {Path(item['manifest']).name}  sequence={item['sequence']}")
        print(f"  {pass_label(pa, ia)}")
        print(f"  {pass_label(pb, ib)}")
        print(f"  Flight A: [{wa[0]}, {wa[1]}]  Flight B: [{wb[0]}, {wb[1]}]")
        print(f"  Overlap:  [{ov[0]}, {ov[1]}]  ({ov[1] - ov[0] + 1} frames)\n")

    # Per-sequence frame spans (manifest-only)
    seen_seq: set[str] = set()
    for path in paths:
        with path.open() as f:
            data = json.load(f)
        seq = data.get("sequence") or path.stem
        if args.sequence and seq != args.sequence and args.sequence not in path.name:
            continue
        key = f"{path.name}:{seq}"
        if key in seen_seq:
            continue
        seen_seq.add(key)
        spans = frames_both_teams_in_flight(data.get("passes") or [])
        if spans:
            print(f"--- Frames with BOTH team 0 & 1 in-flight arrows ({path.name}) ---")
            for sp in spans:
                print(f"  frames [{sp['start']}, {sp['end']}]")
            print()


if __name__ == "__main__":
    main()
