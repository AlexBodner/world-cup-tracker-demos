"""Generate step-by-step passing-lane explanation frames for talks and social posts.

From the repo root::

    PYTHONPATH=. python -m world_cup_projects.explain.pass_alternatives_run \\
        --video world_cup_projects/bundesliga_videos/08fd33_0.mp4 \\
        --metric --layout talk --auto-frame

Outputs PNGs under ``assets/explain_frames/`` (flat, ``pass_lane_*`` filenames).
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
from world_cup_projects.common.detection_cache import wrap_detections_cache
from world_cup_projects.common.pitch import (
    ensure_pitch_homography_maps,
    iter_pitch_transformers,
    load_pitch_homography_cache,
)
from world_cup_projects.common.video import load_video_sequence, read_sequence_frame
from world_cup_projects.common.detect import DEFAULT_BALL_DETECTION_THRESHOLD
from world_cup_projects.explain.pass_alternatives_conference import (
    ConferenceVideoTiming,
    build_conference_context,
    build_conference_video_sequence,
    render_conference_steps,
    write_conference_gif,
    write_conference_video,
    write_conference_frames,
)
from world_cup_projects.pass_alternatives.pass_options import PassWeights
from world_cup_projects.pass_alternatives.render import plan_events


def _load_detections(args, sequence):
    from world_cup_projects.common.detect import iter_football_model_detections

    ball_thr = getattr(args, "ball_threshold", DEFAULT_BALL_DETECTION_THRESHOLD)
    return wrap_detections_cache(
        iter_football_model_detections,
        source_name="football",
        refresh=args.refresh_detections_cache,
        device=args.device,
        threshold=0.5,
        ball_threshold=ball_thr,
        tracker=args.tracker,
    )


def _pick_frame_auto(sequence, detections_source, *, metric: bool, pitch_device: str) -> int:
    """Choose a strong possession moment for the explain sequence."""
    transformers = None
    if metric:
        maps = load_pitch_homography_cache(
            sequence.name, end=sequence.length, device=pitch_device
        )
        if maps is not None:
            transformers = maps.transforms
        else:
            try:
                transformers = {
                    fi: t
                    for fi, t, _ in iter_pitch_transformers(
                        sequence, device=pitch_device, confidence=0.5
                    )
                }
            except RuntimeError as exc:
                print(
                    f"  pitch homography unavailable ({exc}); using pixel scoring for frame pick"
                )
                metric = False
    events = plan_events(
        sequence,
        max_events=1,
        min_gap_frames=30,
        weights=PassWeights.metric(),
        detections_source=detections_source,
        metric=metric,
        pitch_device=pitch_device,
        frame_transforms=transformers,
    )
    if events:
        fi = events[0].frame_idx
        if transformers is None or transformers.get(fi) is not None:
            return fi

    for frame_idx, dets in detections_source(sequence, start=1, end=min(300, sequence.length)):
        if transformers is not None and transformers.get(frame_idx) is None:
            continue
        from world_cup_projects.common.possession import find_control_carrier

        if find_control_carrier(dets) is not None and len(dets) >= 8:
            return frame_idx
    return max(1, sequence.length // 2)


def _pick_frame_midfield(
    sequence,
    detections_source,
    *,
    metric: bool,
    pitch_device: str,
) -> int:
    """Pick a strong possession moment near the middle of the clip (midfield)."""
    mid = sequence.length // 2
    window = max(90, sequence.length // 4)
    transformers = None
    if metric:
        maps = load_pitch_homography_cache(
            sequence.name, end=sequence.length, device=pitch_device
        )
        if maps is not None:
            transformers = maps.transforms
    events = plan_events(
        sequence,
        max_events=40,
        min_gap_frames=20,
        weights=PassWeights.metric(),
        detections_source=detections_source,
        metric=metric,
        pitch_device=pitch_device,
        frame_transforms=transformers,
    )
    in_window = [
        e
        for e in events
        if mid - window <= e.frame_idx <= mid + window
        and (transformers is None or transformers.get(e.frame_idx) is not None)
    ]
    if in_window:
        return max(in_window, key=lambda e: e.top_score).frame_idx
    if events:
        return min(events, key=lambda e: abs(e.frame_idx - mid)).frame_idx
    return _pick_frame_auto(
        sequence, detections_source, metric=metric, pitch_device=pitch_device
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render 4 step-by-step passing-lane explanation PNGs"
    )
    parser.add_argument("--video", type=Path, required=True, help="Bundesliga / broadcast MP4")
    parser.add_argument(
        "--frame",
        type=int,
        default=None,
        help="1-indexed frame (default: --auto-frame picks a good possession moment)",
    )
    parser.add_argument(
        "--auto-frame",
        action="store_true",
        help="Pick the best freeze moment automatically (default when --frame omitted)",
    )
    parser.add_argument(
        "--midfield",
        action="store_true",
        help="Pick a possession moment near the middle of the clip (midfield)",
    )
    parser.add_argument("--metric", action="store_true", help="Pitch homography scoring + radar")
    parser.add_argument("--device", default="cpu", help="Torch device for YOLO + pitch model")
    parser.add_argument("--tracker", default="botsort", choices=["bytetrack", "botsort", "botsort_nocmc"])
    parser.add_argument(
        "--layout",
        choices=["talk", "social"],
        default="talk",
        help="talk = 16:9 with side panel; social = 1080×1080 square crop",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: assets/explain_frames/)",
    )
    parser.add_argument("--no-grid", action="store_true", help="Skip 2×2 composite PNG")
    parser.add_argument("--no-timeline", action="store_true", help="Skip vertical timeline PNG")
    parser.add_argument(
        "--ball-threshold",
        type=float,
        default=DEFAULT_BALL_DETECTION_THRESHOLD,
        help="Ball detection confidence (default 0.20)",
    )
    parser.add_argument(
        "--explain-video",
        action="store_true",
        help="Write pass_lane_explain.mp4 — stepped scoring walkthrough",
    )
    parser.add_argument(
        "--gif",
        action="store_true",
        help="Also write pass_lane_explain.gif (requires --explain-video or implies it)",
    )
    parser.add_argument(
        "--video-fps",
        type=float,
        default=6.0,
        help="Output FPS for --explain-video (default 6)",
    )
    parser.add_argument("--refresh-detections-cache", action="store_true")
    args = parser.parse_args()

    if args.gif:
        args.explain_video = True

    if args.frame is None:
        args.auto_frame = True

    sequence = load_video_sequence(args.video)
    detections_source = _load_detections(args, sequence)

    frame_idx = args.frame
    if args.midfield:
        print("Selecting midfield explain frame (middle of clip)...")
        frame_idx = _pick_frame_midfield(
            sequence,
            detections_source,
            metric=args.metric,
            pitch_device=args.device,
        )
        print(f"  → frame {frame_idx}")
    elif args.auto_frame or frame_idx is None:
        print("Selecting explain frame (scanning for strong possession moment)...")
        frame_idx = _pick_frame_auto(
            sequence,
            detections_source,
            metric=args.metric,
            pitch_device=args.device,
        )
        print(f"  → frame {frame_idx}")

    frame = read_sequence_frame(sequence, frame_idx)
    if frame is None:
        raise SystemExit(f"Could not read frame {frame_idx}")

    dets = None
    # Load from full cached clip when available (faster than re-running YOLO per frame).
    for fi, det in detections_source(sequence, start=1, end=sequence.length):
        if fi == frame_idx:
            dets = det
            break
    if dets is None or len(dets) == 0:
        raise SystemExit(f"No detections for frame {frame_idx}")

    transformer = None
    radar_transformer = None
    frame_keypoints = None
    pitch_maps = None
    metric = args.metric
    if metric:
        try:
            maps = load_pitch_homography_cache(
                sequence.name, end=sequence.length, device=args.device
            )
            if maps is None:
                maps = ensure_pitch_homography_maps(
                    sequence,
                    device=args.device,
                    detections_by_frame={frame_idx: dets},
                )
            pitch_maps = maps
            transformer = maps.transforms.get(frame_idx)
            radar_transformer = maps.radar_transforms.get(frame_idx)
            frame_keypoints = maps.keypoints.get(frame_idx)
            if transformer is None:
                print(f"Warning: no homography at frame {frame_idx}; metric disabled")
                metric = False
        except RuntimeError as exc:
            print(f"Warning: metric mode disabled ({exc})")
            metric = False

    weights = PassWeights.metric() if metric else PassWeights()
    warmup_frames: list[tuple[int, object]] | None = None
    if metric and pitch_maps is not None:
        warmup_frames = []
        for fi, det in detections_source(sequence, start=1, end=sequence.length):
            if fi % 8 != 0:
                continue
            if pitch_maps.keypoints.get(fi) is None:
                continue
            warmup_frames.append((fi, det))
            if len(warmup_frames) >= 40:
                break
    ctx = build_conference_context(
        sequence,
        dets,
        frame_idx,
        frame,
        weights=weights,
        metric=metric,
        pitch_device=args.device,
        transformer=transformer,
        radar_transformer=radar_transformer,
        keypoints=frame_keypoints,
        keypoints_by_frame=pitch_maps.keypoints if pitch_maps else None,
        warmup_frames=warmup_frames,
    )
    if ctx is None:
        raise SystemExit(
            f"No ball carrier or scorable lanes at frame {frame_idx}. "
            "Try another --frame or a clip with clear possession."
        )

    out_dir = args.out_dir or DEFAULT_ASSETS_DIR / "explain_frames"
    steps = render_conference_steps(ctx, layout=args.layout)
    paths = write_conference_frames(
        out_dir,
        steps,
        grid=not args.no_grid,
        timeline=not args.no_timeline,
    )

    video_path: Path | None = None
    gif_path: Path | None = None
    if args.explain_video:
        timing = ConferenceVideoTiming(output_fps=args.video_fps)
        video_frames = build_conference_video_sequence(
            ctx, layout=args.layout, timing=timing
        )
        video_path = write_conference_video(
            out_dir / "pass_lane_explain.mp4",
            video_frames,
            fps=timing.output_fps,
            crf=timing.crf,
        )
        paths.append(video_path)
        print(f"  explain video: {len(video_frames)} frames @ {timing.output_fps} fps")
        if args.gif:
            gif_path = write_conference_gif(
                out_dir / "pass_lane_explain.gif",
                video_path,
                fps=timing.output_fps,
            )
            paths.append(gif_path)

    manifest = {
        "sequence": sequence.name,
        "frame_idx": frame_idx,
        "layout": args.layout,
        "metric": metric,
        "teammate_count": ctx.teammate_count,
        "scored_lanes": len(ctx.options),
        "top3_scores": [round(o.score, 3) for o in ctx.top3],
        "explain_video": str(video_path) if video_path else None,
        "explain_gif": str(gif_path) if gif_path else None,
        "outputs": [str(p) for p in paths],
    }
    manifest_path = out_dir / "pass_lane_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f"\nWrote {len(paths)} images to {out_dir}/")
    for p in paths:
        print(f"  {p.name}")
    print(f"  {manifest_path.name}")


if __name__ == "__main__":
    main()
