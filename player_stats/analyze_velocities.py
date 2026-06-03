"""Analyze homography-based velocities for a SoccerNet clip.

Computes the same speed model as the demo (median of last K instantaneous
pitch speeds, per-step ``H_{i-1}`` / ``H_i`` warps) plus raw 1-frame speeds
for comparison. Writes JSON under the package ``assets/`` directory by default.

Example::

    PYTHONPATH=. python -m world_cup_projects.player_stats.analyze_velocities \\
        --sequence SNMOT-197
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from world_cup_projects.common.pitch import iter_pitch_transformers
from world_cup_projects import DEFAULT_ASSETS_DIR
from world_cup_projects.common.soccernet import (
    DEFAULT_TRACKING_ROOT,
    find_sequences,
    iter_gt_detections,
    load_sequence,
)
from world_cup_projects.player_stats.speed_distance import (
    DEFAULT_SPEED_K_FRAMES,
    SOFT_INST_SPEED_CAP_MS,
    PlayerTrack,
    _instantaneous_speed_homography,
    _smooth_xy,
    HOMOGRAPHY_XY_SMOOTH,
    collect_tracks,
    compute_kinematics,
)

# FIFA / STATSports-style bands (m/s).
SPEED_ZONES_MS = (
    ("standing_walk", 0.0, 2.0),
    ("jog", 2.0, 4.0),
    ("run", 4.0, 5.5),
    ("hsr", 5.5, 7.0),
    ("sprint", 7.0, float("inf")),
)


def _zone_fractions(speeds: np.ndarray) -> dict[str, float]:
    speeds = speeds[np.isfinite(speeds) & (speeds >= 0)]
    if len(speeds) == 0:
        return {name: 0.0 for name, _, _ in SPEED_ZONES_MS}
    total = float(len(speeds))
    out: dict[str, float] = {}
    for name, lo, hi in SPEED_ZONES_MS:
        mask = (speeds >= lo) & (speeds < hi)
        out[name] = round(float(mask.sum()) / total, 4)
    return out


def _percentiles(speeds: np.ndarray) -> dict[str, float]:
    speeds = speeds[np.isfinite(speeds) & (speeds > 0.05)]
    if len(speeds) == 0:
        return {}
    return {
        "p50": round(float(np.percentile(speeds, 50)), 2),
        "p75": round(float(np.percentile(speeds, 75)), 2),
        "p90": round(float(np.percentile(speeds, 90)), 2),
        "p95": round(float(np.percentile(speeds, 95)), 2),
        "max": round(float(np.max(speeds)), 2),
        "mean": round(float(np.mean(speeds)), 2),
    }


def _collect_inst_speeds(
    track: PlayerTrack,
    frame_transforms: dict,
    fps: float,
) -> tuple[np.ndarray, np.ndarray]:
    xy = _smooth_xy(np.asarray(track.xy, dtype=np.float64), HOMOGRAPHY_XY_SMOOTH)
    frames = np.asarray(track.frames)
    raw: list[float] = []
    capped: list[float] = []
    for i in range(1, len(frames)):
        v = _instantaneous_speed_homography(xy, frames, frame_transforms, i, fps)
        if v is None:
            continue
        raw.append(v)
        capped.append(min(v, SOFT_INST_SPEED_CAP_MS))
    return np.asarray(raw, dtype=np.float64), np.asarray(capped, dtype=np.float64)


def analyze_sequence(
    sequence,
    *,
    frame_transforms: dict,
    speed_k_frames: int = DEFAULT_SPEED_K_FRAMES,
) -> dict:
    frames = list(iter_gt_detections(sequence))
    tracks = collect_tracks(iter(frames))
    compute_kinematics(
        tracks,
        sequence.frame_rate,
        mode="homography",
        frame_transforms=frame_transforms,
        speed_k_frames=speed_k_frames,
    )

    players = []
    all_display: list[float] = []
    all_inst_raw: list[float] = []
    all_inst_capped: list[float] = []

    for track in sorted(tracks.values(), key=lambda t: t.track_id):
        if track.speed_ms is None or len(track.frames) < 10:
            continue
        display = track.speed_ms[track.speed_ms > 0.05]
        inst_raw, inst_capped = _collect_inst_speeds(track, frame_transforms, sequence.frame_rate)
        all_display.extend(display.tolist())
        all_inst_raw.extend(inst_raw.tolist())
        all_inst_capped.extend(inst_capped.tolist())

        players.append(
            {
                "track_id": track.track_id,
                "team": track.team,
                "distance_m": round(track.distance_m, 1),
                "top_speed_ms": round(track.top_speed_ms, 2),
                "display_speed": _percentiles(display),
                "display_zones": _zone_fractions(display),
                "inst_1frame_raw": _percentiles(inst_raw[inst_raw <= 15.0]),
                "inst_1frame_capped": _percentiles(inst_capped),
                "inst_zones_capped": _zone_fractions(inst_capped),
                "pct_display_zero": round(
                    float((track.speed_ms < 0.1).mean()) * 100, 1
                ),
            }
        )

    return {
        "sequence": sequence.name,
        "fps": sequence.frame_rate,
        "n_frames": sequence.length,
        "speed_model": {
            "k_frames": speed_k_frames,
            "warp": "per-step H_{i-1}(xy_{i-1}), H_i(xy_i); no H matrix chain",
            "display": f"mean(v_{{j,j-1}}..v_{{j,j-K}}), soft cap {SOFT_INST_SPEED_CAP_MS} m/s per lag",
            "post_smooth_frames": 11,
        },
        "reference_bands_ms": {
            name: {"min": lo, "max": None if hi == float("inf") else hi}
            for name, lo, hi in SPEED_ZONES_MS
        },
        "elite_benchmarks_ms": {
            "hsr_threshold": 5.5,
            "sprint_threshold": 7.0,
            "typical_elite_peak": "8.3-9.2 (30-33 km/h)",
            "wc_top_peak": "~9.9 (35.7 km/h)",
        },
        "players": players,
        "clip_summary": {
            "n_tracks": len(players),
            "display_speed": _percentiles(np.asarray(all_display)),
            "display_zones": _zone_fractions(np.asarray(all_display)),
            "inst_1frame_raw_sane": _percentiles(
                np.asarray([v for v in all_inst_raw if v <= 15.0])
            ),
            "inst_1frame_glitch_steps": int(sum(1 for v in all_inst_raw if v > 15.0)),
            "inst_1frame_capped": _percentiles(np.asarray(all_inst_capped)),
            "tracks_peak_over_7ms": int(
                sum(1 for p in players if p["top_speed_ms"] >= 7.0)
            ),
            "tracks_peak_over_9ms": int(
                sum(1 for p in players if p["top_speed_ms"] >= 9.0)
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Velocity analysis for one SNMOT clip")
    parser.add_argument("--data", default=DEFAULT_TRACKING_ROOT)
    parser.add_argument("--split", default="test")
    parser.add_argument("--sequence", default="SNMOT-197")
    parser.add_argument("--out", default=str(DEFAULT_ASSETS_DIR))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--speed-k-frames", type=int, default=DEFAULT_SPEED_K_FRAMES)
    args = parser.parse_args()

    seq_dir = next(
        p for p in find_sequences(args.data, args.split) if p.name == args.sequence
    )
    sequence = load_sequence(seq_dir)
    end = args.max_frames

    print(f"Loading homography for {sequence.name} ...", flush=True)
    frame_transforms = {
        idx: t
        for idx, t in iter_pitch_transformers(
            sequence, device=args.device, end=end
        )
    }

    print("Computing tracks + velocities ...", flush=True)
    report = analyze_sequence(
        sequence,
        frame_transforms=frame_transforms,
        speed_k_frames=args.speed_k_frames,
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"velocity_analysis_{sequence.name}.json"
    json_path.write_text(json.dumps(report, indent=2))

    s = report["clip_summary"]
    print(f"\n=== {sequence.name} velocity analysis ===")
    print(f"Tracks: {s['n_tracks']}")
    print(f"Display speed (label): {s['display_speed']}")
    print(f"Display zones: {s['display_zones']}")
    print(f"1-frame inst (sane <=15 m/s): {s.get('inst_1frame_raw_sane', {})}")
    print(f"1-frame homography glitches (>15 m/s): {s.get('inst_1frame_glitch_steps', 0)} steps")
    print(f"1-frame inst (capped @ {SOFT_INST_SPEED_CAP_MS} m/s): {s['inst_1frame_capped']}")
    print(f"Peaks >= 7 m/s: {s['tracks_peak_over_7ms']} tracks, >= 9 m/s: {s['tracks_peak_over_9ms']}")
    print(f"\nWrote {json_path}")


if __name__ == "__main__":
    main()
