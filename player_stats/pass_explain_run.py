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
from world_cup_projects.common.detect import (
    BEST_FOOTBALL_PLAYERS_YOLO_MODEL_ID,
    DEFAULT_BALL_DETECTION_THRESHOLD,
    DEFAULT_FOOTBALL_BALL_MODEL_ID,
    DEFAULT_FOOTBALL_PLAYERS_MODEL_ID,
)
from world_cup_projects.common.video import load_video_sequence, read_sequence_frame
from world_cup_projects.pass_alternatives.pass_options import PassWeights
from world_cup_projects.player_stats.pass_events import (
    InferredPass,
    PassDetectionConfig,
    PassQualityScorer,
    build_pass_carrier_timeline,
    scan_possession_events,
)
from world_cup_projects.player_stats.pass_explain_visual import (
    PassExplainVideoTiming,
    PassStripPlan,
    _find_passer_demo_confirm_frame,
    _find_receiver_visual_confirm_frame,
    _receiver_arrival_panels,
    build_fixed_strip_plan,
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
from world_cup_projects.common.pipeline import load_detections_source


def _resolve_roboflow_api_key(cli_key: str | None) -> str | None:
    import os

    if cli_key:
        return cli_key.strip()
    env = os.environ.get("ROBOFLOW_API_KEY")
    if env:
        return env.strip()
    for path in (_repo_root / ".env", _pkg_root / ".env"):
        if not path.is_file():
            continue
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key.strip() == "ROBOFLOW_API_KEY":
                return value.strip().strip('"').strip("'")
    return None


def _feet_xy(dets, tid: int) -> tuple[float, float] | None:
    if dets is None or dets.tracker_id is None:
        return None
    mask = dets.tracker_id == tid
    if not mask.any():
        return None
    box = dets.xyxy[mask][0]
    return (float((box[0] + box[2]) / 2), float(box[3]))


def _match_tid_at_feet(dets, feet: tuple[float, float], *, max_px: float = 150.0) -> int | None:
    import numpy as np

    from world_cup_projects.common.possession import player_mask

    if dets is None or dets.tracker_id is None:
        return None
    pm = player_mask(dets)
    if not pm.any():
        return None
    boxes = dets.xyxy[pm]
    player_feet = np.column_stack(((boxes[:, 0] + boxes[:, 2]) / 2, boxes[:, 3]))
    dists = np.linalg.norm(player_feet - np.asarray(feet, dtype=np.float32), axis=1)
    idx = int(dists.argmin())
    if float(dists[idx]) > max_px:
        return None
    return int(dets.tracker_id[pm][idx])


def _load_dets_from_manifest_detector(
    sequence,
    manifest: dict,
    *,
    device: str,
    tracker: str,
) -> dict[int, object]:
    det_cfg = manifest["detector"]
    dev, trk = device, tracker

    class _RefArgs:
        source = "football"
        device = dev
        tracker = trk
        refresh_detections_cache = False
        detector_backend = det_cfg["backend"]
        player_model_id = det_cfg["player_model_id"]
        detection_threshold = det_cfg["detection_threshold"]
        ball_threshold = det_cfg["ball_threshold"]
        ball_detector_backend = det_cfg["ball_detector_backend"]
        ball_model_id = det_cfg["ball_model_id"]

    from world_cup_projects.common.teams import stabilize_teams_by_tracklet

    src = load_detections_source(_RefArgs, sequence)
    frames = stabilize_teams_by_tracklet(list(src(sequence, start=1, end=sequence.length)))
    return {int(fi): d for fi, d in frames}


def _match_pass_from_manifest(
    manifest_path: Path,
    *,
    sequence,
    target_dets_by_frame: dict[int, object],
    device: str,
    tracker: str,
) -> tuple[InferredPass, PassStripPlan]:
    """Map passer/receiver by pitch position and lock strip frames from a reference run."""
    manifest = json.loads(manifest_path.read_text())
    ref_pass = manifest["pass"]
    strips = manifest["strips"]
    release = int(ref_pass["frame_idx"])
    gap = int(ref_pass["gap_frames"])
    summary = int(strips["summary"])

    ref_dets = _load_dets_from_manifest_detector(
        sequence, manifest, device=device, tracker=tracker
    )
    ref_release = ref_dets.get(release)
    ref_summary = ref_dets.get(summary)
    tgt_release = target_dets_by_frame.get(release)
    tgt_summary = target_dets_by_frame.get(summary)

    passer_feet = _feet_xy(ref_release, int(ref_pass["passer_tid"]))
    receiver_feet = _feet_xy(ref_summary, int(ref_pass["receiver_tid"]))
    passer_tid = _match_tid_at_feet(tgt_release, passer_feet) if passer_feet else None
    receiver_tid = _match_tid_at_feet(tgt_summary, receiver_feet) if receiver_feet else None
    if passer_tid is None or receiver_tid is None:
        raise SystemExit(
            f"Could not map reference pass #{ref_pass['passer_tid']}→#{ref_pass['receiver_tid']} "
            f"onto current detections at frame {release}"
        )

    team = 0
    if tgt_release is not None and tgt_release.tracker_id is not None and tgt_release.data:
        mask = tgt_release.tracker_id == passer_tid
        team_arr = tgt_release.data.get("team")
        if mask.any() and team_arr is not None:
            team = int(team_arr[mask][0])

    pass_event = InferredPass(
        frame_idx=release,
        passer_tid=passer_tid,
        receiver_tid=receiver_tid,
        team=team,
        gap_frames=gap,
        pass_length_m=ref_pass.get("pass_length_m"),
        quality_score=ref_pass.get("quality_score"),
        openness=ref_pass.get("openness"),
        forward_gain=ref_pass.get("forward_gain"),
        rivals_in_lane=ref_pass.get("rivals_in_lane"),
        motion_alignment=ref_pass.get("motion_alignment"),
        receiver_space=ref_pass.get("receiver_space"),
        touch_kind=ref_pass.get("touch_kind", "control"),
    )
    config = PassDetectionConfig()
    timeline = {
        st.frame_idx: st
        for st in build_pass_carrier_timeline(
            iter(sorted(target_dets_by_frame.items())),
            config=config,
            metric=False,
            transformers={},
        )
    }
    passer_confirm = _find_passer_demo_confirm_frame(
        pass_event, timeline, min_control_frames=config.min_control_frames
    )
    confirm = _find_receiver_visual_confirm_frame(
        pass_event,
        target_dets_by_frame,
        timeline_by_frame=timeline,
        min_control_frames=config.min_control_frames,
    )
    receiver_frames = _receiver_arrival_panels(
        release, confirm, min_arrival_frames=config.min_arrival_frames
    )
    strip_plan = PassStripPlan(
        passer_frames=tuple(int(f) for f in strips["passer"]),
        flight_frames=tuple(int(f) for f in strips["flight"]),
        receiver_frames=receiver_frames,
        summary_frame=summary,
        passer_confirm_frame=passer_confirm,
        receiver_confirm_frame=confirm,
    )
    print(
        f"  Matched reference #{ref_pass['passer_tid']}→#{ref_pass['receiver_tid']} "
        f"→ #{passer_tid}→#{receiver_tid} (same frames {release}..{summary})"
    )
    return pass_event, strip_plan


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
    from world_cup_projects.common.model_ids import (
        BEST_FOOTBALL_PLAYERS_YOLO_MODEL_ID,
        DEFAULT_FOOTBALL_BALL_MODEL_ID,
        DEFAULT_FOOTBALL_PLAYERS_MODEL_ID,
    )

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
        "--release-frame",
        type=int,
        default=None,
        help="Force explain at this release frame (with --passer-tid, --receiver-tid, --gap-frames)",
    )
    parser.add_argument(
        "--gap-frames",
        type=int,
        default=None,
        help="Ball-flight gap in frames when using --release-frame",
    )
    parser.add_argument(
        "--match-pass-from",
        type=Path,
        default=None,
        help=(
            "Reference pass_detect_manifest.json: map passer/receiver by position "
            "and use the same strip frames (for cross-model comparison)"
        ),
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
    parser.add_argument(
        "--lane-corridor",
        action="store_true",
        help="Also write pass-corridor explain PNGs/video (pass line + intercept threat)",
    )
    parser.add_argument(
        "--lane-only",
        action="store_true",
        help="With --lane-corridor, skip pass-detection strips and video",
    )
    parser.add_argument(
        "--lane-video",
        action="store_true",
        help="Write pass_lane_detect_explain.mp4 (implies --lane-corridor)",
    )
    parser.add_argument("--refresh-detections-cache", action="store_true")
    parser.add_argument(
        "--universe-best",
        action="store_true",
        help=(
            f"Use newer Universe YOLO ({BEST_FOOTBALL_PLAYERS_YOLO_MODEL_ID}) via Inference "
            f"+ dedicated ball model ({DEFAULT_FOOTBALL_BALL_MODEL_ID}). Needs ROBOFLOW_API_KEY."
        ),
    )
    parser.add_argument(
        "--detector-backend",
        choices=("yolo", "inference"),
        default=None,
        help="football detector: local YOLO .pt (v11) or Roboflow Inference (Universe model id)",
    )
    parser.add_argument(
        "--player-model-id",
        default=None,
        help=(
            f"Universe player model id (default: {DEFAULT_FOOTBALL_PLAYERS_MODEL_ID} local; "
            f"{BEST_FOOTBALL_PLAYERS_YOLO_MODEL_ID} with --universe-best)"
        ),
    )
    parser.add_argument(
        "--detection-threshold",
        type=float,
        default=0.5,
        help="Player / GK / referee confidence threshold",
    )
    parser.add_argument(
        "--ball-threshold",
        type=float,
        default=None,
        help="Ball class confidence threshold (default 0.20; try 0.15 for blur / air balls)",
    )
    parser.add_argument(
        "--ball-detector-backend",
        choices=("none", "yolo", "inference"),
        default=None,
        help="Dedicated ball model: local YOLO .pt or Inference (recommended with --universe-best)",
    )
    parser.add_argument(
        "--ball-model-id",
        default=None,
        help=f"Universe ball model id (default {DEFAULT_FOOTBALL_BALL_MODEL_ID})",
    )
    parser.add_argument(
        "--roboflow-api-key",
        default=None,
        help="Roboflow API key (else ROBOFLOW_API_KEY env or .env in repo root)",
    )
    args = parser.parse_args()

    import os

    api_key = _resolve_roboflow_api_key(args.roboflow_api_key)
    if api_key:
        os.environ["ROBOFLOW_API_KEY"] = api_key

    if args.universe_best:
        if args.ball_threshold is None:
            args.ball_threshold = 0.15
        if api_key:
            args.detector_backend = "inference"
            args.player_model_id = BEST_FOOTBALL_PLAYERS_YOLO_MODEL_ID
            args.ball_detector_backend = "inference"
            args.ball_model_id = DEFAULT_FOOTBALL_BALL_MODEL_ID
        else:
            print(
                "Note: ROBOFLOW_API_KEY not set — using local YOLO v11 players "
                "+ dedicated local ball model (football-ball-detection.pt)."
            )
            args.detector_backend = "yolo"
            args.player_model_id = DEFAULT_FOOTBALL_PLAYERS_MODEL_ID
            args.ball_detector_backend = "yolo"
            args.ball_model_id = DEFAULT_FOOTBALL_BALL_MODEL_ID
    if args.detector_backend is None:
        args.detector_backend = "yolo"
    if args.player_model_id is None:
        args.player_model_id = DEFAULT_FOOTBALL_PLAYERS_MODEL_ID
    if args.ball_threshold is None:
        args.ball_threshold = DEFAULT_BALL_DETECTION_THRESHOLD
    if args.ball_detector_backend is None:
        args.ball_detector_backend = "none"
    if args.ball_model_id is None:
        args.ball_model_id = DEFAULT_FOOTBALL_BALL_MODEL_ID

    sequence = load_video_sequence(args.video)
    end = args.max_frames if args.max_frames is not None else sequence.length
    config = PassDetectionConfig().for_frame_rate(sequence.frame_rate)

    class _Args:
        source = "football"
        device = args.device
        tracker = args.tracker
        refresh_detections_cache = args.refresh_detections_cache
        detector_backend = args.detector_backend
        player_model_id = args.player_model_id
        detection_threshold = args.detection_threshold
        ball_threshold = args.ball_threshold
        ball_detector_backend = args.ball_detector_backend
        ball_model_id = args.ball_model_id

    needs_inference_api = (
        args.detector_backend == "inference" or args.ball_detector_backend == "inference"
    ) and not os.environ.get("ROBOFLOW_API_KEY")
    if needs_inference_api and args.refresh_detections_cache:
        raise SystemExit(
            "Inference detection needs ROBOFLOW_API_KEY when refreshing cache. "
            "Export it, add to .env, or pass --roboflow-api-key. "
            "Use --universe-best for automatic local fallback."
        )
    if needs_inference_api:
        print(
            "Note: ROBOFLOW_API_KEY not set — using cached Inference detections if available."
        )

    print(
        f"Detections: backend={args.detector_backend} "
        f"players={args.player_model_id} "
        f"ball_thr={args.ball_threshold} "
        f"ball_backend={args.ball_detector_backend} "
        f"ball_model={args.ball_model_id if args.ball_detector_backend == 'inference' else '—'}"
    )

    detections_source = load_detections_source(_Args, sequence)
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

    strip_plan = None
    if args.match_pass_from is not None:
        pass_event, strip_plan = _match_pass_from_manifest(
            args.match_pass_from,
            sequence=sequence,
            target_dets_by_frame=dets_by_frame,
            device=args.device,
            tracker=args.tracker,
        )
    elif args.release_frame is not None:
        if args.passer_tid is None or args.receiver_tid is None:
            raise SystemExit("--release-frame requires --passer-tid and --receiver-tid")
        gap = args.gap_frames if args.gap_frames is not None else 44
        release_dets = dets_by_frame.get(args.release_frame)
        team = 0
        if release_dets is not None and release_dets.tracker_id is not None and release_dets.data:
            mask = release_dets.tracker_id == args.passer_tid
            team_arr = release_dets.data.get("team")
            if mask.any() and team_arr is not None:
                team = int(team_arr[mask][0])
        pass_event = InferredPass(
            frame_idx=args.release_frame,
            passer_tid=args.passer_tid,
            receiver_tid=args.receiver_tid,
            team=team,
            gap_frames=gap,
            pass_length_m=None,
            quality_score=None,
            openness=None,
            forward_gain=None,
            rivals_in_lane=None,
            motion_alignment=None,
            receiver_space=None,
        )
    elif args.passer_tid is not None or args.receiver_tid is not None:
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
    if args.match_pass_from is None:
        print(
            f"  → pass #{pass_event.passer_tid} → #{pass_event.receiver_tid} "
            f"at frame {pass_event.frame_idx} (gap {pass_event.gap_frames}f, "
            f"quality={pass_event.quality_score})"
        )

    if strip_plan is None and args.release_frame is not None:
        try:
            strip_plan = build_strip_plan(
                pass_event,
                timeline_by_frame,
                dets_by_frame=dets_by_frame,
                min_control_frames=config.min_control_frames,
                min_arrival_frames=config.min_arrival_frames,
            )
        except ValueError as exc:
            print(f"  Note: fixed strip plan ({exc})")
            strip_plan = build_fixed_strip_plan(pass_event)
    elif strip_plan is None:
        try:
            strip_plan = build_strip_plan(
                pass_event,
                timeline_by_frame,
                dets_by_frame=dets_by_frame,
                min_control_frames=config.min_control_frames,
                min_arrival_frames=config.min_arrival_frames,
            )
        except ValueError as exc:
            print(f"  Note: fixed strip plan ({exc})")
            strip_plan = build_fixed_strip_plan(pass_event)
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
        strip_plan=strip_plan,
    )

    out_dir = args.out or (DEFAULT_ASSETS_DIR / "explain_frames")
    lane_corridor = args.lane_corridor or args.lane_only or args.lane_video
    lane_only = args.lane_only
    paths: list[Path] = []

    if not lane_only:
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

    if not lane_only and (args.explain_video or args.gif):
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

    lane_manifest: dict | None = None
    if lane_corridor:
        from world_cup_projects.player_stats.pass_lane_detect_visual import (
            build_pass_lane_detect_context,
            build_pass_lane_detect_video_sequence,
            render_pass_lane_detect_strips,
            write_pass_lane_detect_frames,
            write_pass_lane_detect_gif,
            write_pass_lane_detect_video,
        )

        lane_ctx = build_pass_lane_detect_context(ctx, scorer, weights=weights)
        if lane_ctx is None:
            print(
                "Warning: could not score pass corridor at release "
                "(passer/receiver lane missing — try --metric or another pass)"
            )
        else:
            lane_strips = render_pass_lane_detect_strips(lane_ctx, layout=args.layout)
            lane_paths = write_pass_lane_detect_frames(
                out_dir, lane_strips, timeline=not args.no_timeline
            )
            paths.extend(lane_paths)
            need_lane_video = args.lane_video or args.explain_video or args.gif
            if need_lane_video:
                lane_frames = build_pass_lane_detect_video_sequence(
                    lane_ctx, timing=video_timing
                )
                lane_video = write_pass_lane_detect_video(
                    out_dir / "pass_lane_detect_explain.mp4",
                    lane_frames,
                    fps=video_timing.output_fps,
                    crf=video_timing.crf,
                )
                paths.append(lane_video)
                dur = len(lane_frames) / video_timing.output_fps
                print(
                    f"  Lane video: {len(lane_frames)} frames @ "
                    f"{video_timing.output_fps:.1f} fps (~{dur:.1f}s)"
                )
                if args.gif or args.lane_video:
                    gif_width = args.gif_width if args.gif_width > 0 else None
                    lane_gif = write_pass_lane_detect_gif(
                        out_dir / "pass_lane_detect_explain.gif",
                        lane_video,
                        fps=video_timing.output_fps,
                        width=gif_width,
                    )
                    paths.append(lane_gif)
            lane_manifest = {
                "release_frame": lane_ctx.bundle.release_frame,
                "openness_m": lane_ctx.pass_ctx.pass_event.openness,
                "quality_score": lane_ctx.pass_ctx.pass_event.quality_score,
                "pass_length_m": lane_ctx.bundle.option.length,
                "rivals_in_lane": lane_ctx.bundle.option.rivals_in_lane,
                "outputs": [str(p) for p in lane_paths],
            }
            lane_manifest_path = out_dir / "pass_lane_detect_manifest.json"
            lane_manifest_path.write_text(json.dumps(lane_manifest, indent=2))
            print(f"  {lane_manifest_path.name}")

    if not lane_only:
        manifest = {
            "sequence": sequence.name,
            "detector": {
                "backend": args.detector_backend,
                "player_model_id": args.player_model_id,
                "detection_threshold": args.detection_threshold,
                "ball_threshold": args.ball_threshold,
                "ball_detector_backend": args.ball_detector_backend,
                "ball_model_id": args.ball_model_id,
            },
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
        if lane_manifest is not None:
            manifest["lane_corridor"] = lane_manifest
        manifest_path = out_dir / "pass_detect_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f"\nWrote {len(paths)} file(s) to {out_dir}/")
    for p in paths:
        print(f"  {p.name}")
    if not lane_only:
        print(f"  {manifest_path.name}")


if __name__ == "__main__":
    main()
