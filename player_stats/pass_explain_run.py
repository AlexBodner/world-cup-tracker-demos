"""Generate pass-detection filmstrip explain frames for talks and social posts.

From the repo root::

    PYTHONPATH=. python -m world_cup_projects.player_stats.pass_explain_run \\
        --video world_cup_projects/bundesliga_videos/08fd33_0.mp4 \\
        --metric --layout talk

Outputs PNGs under ``assets/explain_frames/`` (``pass_detect_strip_*`` filenames).
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
from world_cup_projects.common.video import load_video_sequence, read_sequence_frame
from world_cup_projects.pass_alternatives.pass_options import PassWeights
from world_cup_projects.player_stats.pass_events import (
    PassDetectionConfig,
    PassQualityScorer,
    build_pass_carrier_timeline,
    scan_possession_events,
)
from world_cup_projects.player_stats.pass_explain_visual import (
    PassExplainVideoTiming,
    build_pass_explain_context,
    build_pass_explain_video_sequence,
    build_strip_plan,
    frames_needed_for_explain,
    pick_explain_pass,
    pick_midfield_explain_pass,
    render_pass_explain_strips,
    write_pass_explain_frames,
    write_pass_explain_gif,
    write_pass_explain_video,
)
from world_cup_projects.player_stats.pass_network_run import _load_detections_source


def _load_pitch_maps(sequence, *, device: str, end: int, detections_by_frame):
    from world_cup_projects.common.pitch import (
        ensure_pitch_homography_maps,
        load_pitch_homography_cache,
    )

    maps = load_pitch_homography_cache(sequence.name, end=end, device=device)
    if maps is None:
        maps = ensure_pitch_homography_maps(
            sequence,
            device=device,
            end=end,
            detections_by_frame=detections_by_frame,
        )
    return maps


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render pass-detection filmstrip explanation PNGs"
    )
    parser.add_argument("--video", type=Path, required=True, help="Bundesliga / broadcast MP4")
    parser.add_argument("--metric", action="store_true", help="Pitch homography + radar on summary")
    parser.add_argument("--device", default="cpu", help="Torch device for YOLO + pitch model")
    parser.add_argument("--tracker", default="botsort", choices=["bytetrack", "botsort", "botsort_nocmc"])
    parser.add_argument(
        "--layout",
        choices=["talk", "social"],
        default="talk",
        help="talk = 16:9; social = 1080×1080 square crop",
    )
    parser.add_argument(
        "--pass-index",
        type=int,
        default=None,
        help="Pick Nth best pass by quality (default: auto-pick best)",
    )
    parser.add_argument(
        "--midfield",
        action="store_true",
        help="Pick the best explainable pass near the middle of the clip",
    )
    parser.add_argument(
        "--passer-tid",
        type=int,
        default=None,
        help="Pick a specific pass link (use with --receiver-tid)",
    )
    parser.add_argument(
        "--receiver-tid",
        type=int,
        default=None,
        help="Pick a specific pass link (use with --passer-tid)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory (default: assets/explain_frames/)",
    )
    parser.add_argument("--max-frames", type=int, default=None, help="Limit scan to first N frames")
    parser.add_argument("--no-timeline", action="store_true", help="Skip stacked timeline PNG")
    parser.add_argument(
        "--explain-video",
        action="store_true",
        help="Write annotated slow-motion MP4 (real consecutive frames)",
    )
    parser.add_argument(
        "--video-fps",
        type=float,
        default=8.0,
        help="Output FPS for --explain-video (default 8)",
    )
    parser.add_argument(
        "--video-hold",
        type=float,
        default=1.2,
        help="Extra hold seconds on passer/receiver locked beats",
    )
    parser.add_argument(
        "--video-crf",
        type=int,
        default=16,
        help="h264 quality (lower = sharper, default 16)",
    )
    parser.add_argument(
        "--video-lock-nudge",
        type=int,
        default=5,
        help="Hold locked beats this many frames before pass credit (default 5)",
    )
    parser.add_argument("--gif", action="store_true", help="Also write slow-motion explain GIF")
    parser.add_argument(
        "--gif-width",
        type=int,
        default=1280,
        help="GIF width in px (default 1280; 0 = full resolution)",
    )
    parser.add_argument("--no-gif-summary", action="store_true", help="Omit summary beat from GIF/video")
    parser.add_argument("--refresh-detections-cache", action="store_true")
    args = parser.parse_args()

    sequence = load_video_sequence(args.video)
    end = args.max_frames if args.max_frames is not None else sequence.length
    config = PassDetectionConfig().for_frame_rate(sequence.frame_rate)

    class _Args:
        source = "football"
        device = args.device
        tracker = args.tracker
        refresh_detections_cache = args.refresh_detections_cache

    detections_source = _load_detections_source(_Args, sequence)
    frames = list(detections_source(sequence, start=1, end=end))

    from world_cup_projects.common.teams import stabilize_teams_by_tracklet

    frames = stabilize_teams_by_tracklet(frames)
    dets_by_frame = {int(fi): d for fi, d in frames}

    pitch_maps = None
    metric = args.metric
    if metric:
        try:
            pitch_maps = _load_pitch_maps(
                sequence,
                device=args.device,
                end=end,
                detections_by_frame=dets_by_frame,
            )
        except RuntimeError as exc:
            print(f"Warning: metric mode disabled ({exc})")
            metric = False
    frame_transforms = pitch_maps.transforms if pitch_maps is not None else {}

    weights = PassWeights.metric() if metric else PassWeights()
    scorer = PassQualityScorer(
        weights=weights, metric=metric, transformers=frame_transforms
    )

    print("Scanning for inferred passes...")
    scan = scan_possession_events(
        iter(frames),
        scorer=scorer,
        config=config,
        metric=metric,
        transformers=frame_transforms,
    )
    passes = list(scan.passes)
    if not passes:
        raise SystemExit("No passes detected in clip. Try a longer clip or different video.")

    print(f"  found {len(passes)} passes:")
    for p in passes:
        print(
            f"    #{p.passer_tid} -> #{p.receiver_tid} "
            f"rel={p.frame_idx} gap={p.gap_frames}f arr={p.frame_idx + p.gap_frames}"
        )

    print("Building carrier timeline...")
    timeline = build_pass_carrier_timeline(
        iter(frames),
        config=config,
        metric=metric,
        transformers=frame_transforms,
    )
    timeline_by_frame = {st.frame_idx: st for st in timeline}

    if args.passer_tid is not None or args.receiver_tid is not None:
        link_passes = [
            p
            for p in passes
            if (args.passer_tid is None or p.passer_tid == args.passer_tid)
            and (args.receiver_tid is None or p.receiver_tid == args.receiver_tid)
        ]
        if not link_passes:
            raise SystemExit(
                f"No pass matching passer={args.passer_tid} receiver={args.receiver_tid}"
            )
        pass_event = link_passes[0]
    elif args.midfield:
        pass_event = pick_midfield_explain_pass(
            passes,
            sequence_length=sequence.length,
            timeline_by_frame=timeline_by_frame,
            dets_by_frame=dets_by_frame,
            min_control_frames=config.min_control_frames,
        )
    else:
        pass_event = pick_explain_pass(
            passes,
            pass_index=args.pass_index,
            timeline_by_frame=timeline_by_frame,
            dets_by_frame=dets_by_frame,
            min_control_frames=config.min_control_frames,
        )
    print(
        f"  → pass #{pass_event.passer_tid} → #{pass_event.receiver_tid} "
        f"at frame {pass_event.frame_idx} (gap {pass_event.gap_frames}f, "
        f"quality={pass_event.quality_score})"
    )

    strip_plan = build_strip_plan(
        pass_event,
        timeline_by_frame,
        dets_by_frame=dets_by_frame,
        min_control_frames=config.min_control_frames,
        min_arrival_frames=config.min_arrival_frames,
    )
    need_video = args.explain_video or args.gif
    needed = frames_needed_for_explain(strip_plan, include_video=need_video)

    frames_by_idx: dict[int, object] = {}
    for fi in needed:
        frame = read_sequence_frame(sequence, fi)
        if frame is None:
            raise SystemExit(f"Could not read frame {fi}")
        frames_by_idx[fi] = frame

    keypoints_by_frame: dict = {}
    radar_transformers: dict = {}
    if pitch_maps is not None:
        keypoints_by_frame = {fi: pitch_maps.keypoints.get(fi) for fi in needed}
        radar_transformers = {fi: pitch_maps.radar_transforms.get(fi) for fi in needed}

    ctx = build_pass_explain_context(
        pass_event,
        frame_rate=sequence.frame_rate,
        frames_by_idx=frames_by_idx,
        dets_by_frame=dets_by_frame,
        timeline_by_frame=timeline_by_frame,
        keypoints_by_frame=keypoints_by_frame,
        radar_transformers=radar_transformers,
        metric=metric,
        min_control_frames=config.min_control_frames,
        min_arrival_frames=config.min_arrival_frames,
    )

    out_dir = args.out or (DEFAULT_ASSETS_DIR / "explain_frames")
    strips = render_pass_explain_strips(ctx, layout=args.layout)
    paths = write_pass_explain_frames(
        out_dir, strips, timeline=not args.no_timeline
    )

    video_timing = PassExplainVideoTiming(
        output_fps=args.video_fps,
        hold_locked_seconds=args.video_hold,
        lock_nudge_frames=args.video_lock_nudge,
        crf=args.video_crf,
    )
    include_summary = not args.no_gif_summary

    if args.explain_video or args.gif:
        video_frames = build_pass_explain_video_sequence(
            ctx,
            timing=video_timing,
            include_summary=include_summary,
        )
        video_path = write_pass_explain_video(
            out_dir / "pass_detect_explain.mp4",
            video_frames,
            fps=video_timing.output_fps,
            crf=video_timing.crf,
        )
        dur = len(video_frames) / video_timing.output_fps
        paths.append(video_path)
        print(
            f"  Video: {len(video_frames)} frames @ {video_timing.output_fps:.1f} fps "
            f"(~{dur:.1f}s, h264)"
        )
        if args.gif:
            gif_width = args.gif_width if args.gif_width > 0 else None
            gif_path = write_pass_explain_gif(
                out_dir / "pass_detect_explain.gif",
                video_path,
                fps=video_timing.output_fps,
                width=gif_width,
            )
            paths.append(gif_path)
            w_label = gif_width if gif_width else "full"
            print(f"  GIF: {gif_path.name} @ {video_timing.output_fps:.1f} fps, width={w_label}")

    manifest = {
        "sequence": sequence.name,
        "pass": pass_event.to_dict(),
        "layout": args.layout,
        "metric": metric,
        "strips": {
            "passer": list(ctx.strip_plan.passer_frames),
            "flight": list(ctx.strip_plan.flight_frames),
            "receiver": list(ctx.strip_plan.receiver_frames),
            "summary": ctx.strip_plan.summary_frame,
        },
        "outputs": [str(p) for p in paths],
    }
    manifest_path = out_dir / "pass_detect_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f"\nWrote {len(paths)} images to {out_dir}/")
    for p in paths:
        print(f"  {p.name}")
    print(f"  {manifest_path.name}")


if __name__ == "__main__":
    main()
