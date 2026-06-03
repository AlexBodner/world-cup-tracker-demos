"""Compare pass-demo possession thresholds across SoccerNet sequences.

Counts carrier frames, pass-scoring candidates, and final freeze events for each
``--carrier-max-m`` (metric / homography) and the pixel baseline (80 px).

Example::

    MPLCONFIGDIR=/tmp/mpl PYTHONPATH=. python -m world_cup_projects.pass_alternatives.compare_thresholds \\
        --top 5 --thresholds 0.6,0.7,0.8,1.0,1.2,2.0

Optional short MP4s for one sequence::

    ... --render-sequence SNMOT-194 --render-thresholds 0.7,1.0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from world_cup_projects.common.clips import rank_clips
from world_cup_projects.common.possession import (
    CARRIER_MAX_DISTANCE_PX,
    find_ball_carrier,
)
from world_cup_projects import DEFAULT_ASSETS_DIR
from world_cup_projects.common.soccernet import (
    DEFAULT_TRACKING_ROOT,
    find_sequences,
    iter_gt_detections,
    load_sequence,
)
from world_cup_projects.pass_alternatives.pass_options import PassWeights
from world_cup_projects.pass_alternatives.render import (
    _pitch_transform_map,
    _score_options,
    plan_events,
    render_demo,
)


def _count_frames(
    sequence,
    *,
    max_frames: int | None,
    transformers: dict,
    carrier_max_m: float | None,
    carrier_max_px: float,
    metric: bool,
) -> dict:
    carrier_frames = 0
    candidate_frames = 0
    total = 0

    for frame_idx, dets in iter_gt_detections(sequence, end=max_frames):
        total += 1
        transformer = transformers.get(frame_idx) if metric else None
        carrier = find_ball_carrier(
            dets,
            max_distance_px=carrier_max_px,
            transformer=transformer,
            max_distance_m=carrier_max_m if carrier_max_m is not None else 1.0,
        )
        if carrier is None:
            continue
        carrier_frames += 1
        options = _score_options(
            dets,
            carrier,
            weights=PassWeights.metric() if metric else PassWeights(),
            transformer=transformer,
            metric=metric,
        )
        if len(options) >= 3:
            candidate_frames += 1

    return {
        "frames": total,
        "carrier_frames": carrier_frames,
        "candidate_frames": candidate_frames,
        "carrier_pct": round(100 * carrier_frames / max(total, 1), 1),
        "candidate_pct": round(100 * candidate_frames / max(total, 1), 1),
    }


def compare_sequence(
    sequence,
    *,
    thresholds_m: list[float],
    max_frames: int | None,
    pitch_device: str,
) -> dict:
    print(f"  loading homography for {sequence.name} ...", flush=True)
    transformers = _pitch_transform_map(
        sequence, max_frames=max_frames, pitch_device=pitch_device
    )

    rows: list[dict] = []

    px_counts = _count_frames(
        sequence,
        max_frames=max_frames,
        transformers=transformers,
        carrier_max_m=None,
        carrier_max_px=CARRIER_MAX_DISTANCE_PX,
        metric=False,
    )
    events_px = plan_events(
        sequence,
        max_frames=max_frames,
        metric=False,
        carrier_max_distance_px=CARRIER_MAX_DISTANCE_PX,
    )
    rows.append(
        {
            "mode": "pixel",
            "threshold": f"{CARRIER_MAX_DISTANCE_PX:g}px",
            "threshold_m": None,
            **px_counts,
            "freeze_events": len(events_px),
            "event_frames": [e.frame_idx for e in events_px],
        }
    )

    for m in thresholds_m:
        counts = _count_frames(
            sequence,
            max_frames=max_frames,
            transformers=transformers,
            carrier_max_m=m,
            carrier_max_px=CARRIER_MAX_DISTANCE_PX,
            metric=True,
        )
        events = plan_events(
            sequence,
            max_frames=max_frames,
            metric=True,
            pitch_device=pitch_device,
            frame_transforms=transformers,
            carrier_max_distance_m=m,
        )
        rows.append(
            {
                "mode": "metric",
                "threshold": f"{m:g}m",
                "threshold_m": m,
                **counts,
                "freeze_events": len(events),
                "event_frames": [e.frame_idx for e in events],
            }
        )

    return {"sequence": sequence.name, "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare carrier distance thresholds")
    parser.add_argument("--data", default=DEFAULT_TRACKING_ROOT)
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--sequences",
        default=None,
        help="Comma-separated SNMOT ids; default: top N from rank_clips",
    )
    parser.add_argument("--top", type=int, default=5, help="If --sequences omitted, use top N clips")
    parser.add_argument(
        "--thresholds",
        default="0.6,0.7,0.8,1.0,1.2,2.0",
        help="Comma-separated carrier-max-m values for metric mode",
    )
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--out",
        default=str(DEFAULT_ASSETS_DIR / "carrier_threshold_comparison.json"),
    )
    parser.add_argument(
        "--render-sequence",
        default=None,
        help="Also render metric MP4s for this sequence at --render-thresholds",
    )
    parser.add_argument(
        "--render-thresholds",
        default="0.7,1.0",
        help="Comma-separated m values to render when --render-sequence is set",
    )
    args = parser.parse_args()

    thresholds_m = [float(x.strip()) for x in args.thresholds.split(",")]
    seq_dirs = find_sequences(args.data, args.split)
    if args.sequences:
        names = [s.strip() for s in args.sequences.split(",")]
        chosen = [p for p in seq_dirs if p.name in names]
    else:
        ranking = rank_clips(seq_dirs)
        names = [r.name for r in ranking[: args.top]]
        chosen = [p for p in seq_dirs if p.name in names]

    report = {
        "sequences": names,
        "thresholds_m": thresholds_m,
        "pixel_baseline_px": CARRIER_MAX_DISTANCE_PX,
        "max_frames": args.max_frames,
        "results": [],
    }

    print(f"Comparing {len(chosen)} sequences: {', '.join(names)}")
    print(f"Metric thresholds (m): {thresholds_m}\n")

    for seq_dir in chosen:
        sequence = load_sequence(seq_dir)
        print(f"\n=== {sequence.name} ({sequence.length} frames) ===")
        result = compare_sequence(
            sequence,
            thresholds_m=thresholds_m,
            max_frames=args.max_frames,
            pitch_device=args.device,
        )
        report["results"].append(result)

        print(f"{'threshold':>10}  {'carrier':>8}  {'cand':>8}  {'freezes':>7}  event_frames")
        for row in result["rows"]:
            print(
                f"{row['threshold']:>10}  "
                f"{row['carrier_frames']:>4}/{row['frames']}  "
                f"{row['candidate_frames']:>4}/{row['frames']}  "
                f"{row['freeze_events']:>7}  "
                f"{row['event_frames']}"
            )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nWrote {out_path}")

    if args.render_sequence:
        seq = load_sequence(next(p for p in seq_dirs if p.name == args.render_sequence))
        render_ms = [float(x.strip()) for x in args.render_thresholds.split(",")]
        out_dir = out_path.parent
        for m in render_ms:
            tag = f"pass_alternatives_gt_metric_{seq.name}_carrier{m:g}m"
            mp4 = out_dir / f"{tag}.mp4"
            print(f"\nRendering {mp4.name} (carrier_max_m={m}) ...", flush=True)
            manifest = render_demo(
                seq,
                str(mp4),
                max_frames=args.max_frames,
                metric=True,
                pitch_device=args.device,
                version_tag=f"metric_carrier{m:g}m",
                carrier_max_distance_m=m,
            )
            (out_dir / f"{tag}.json").write_text(json.dumps(manifest, indent=2))
            print(f"  -> {mp4}")


if __name__ == "__main__":
    main()
