"""Generate turnover filmstrip explain frames — pass tracked until interception.

From the repo root::

    PYTHONPATH=. python -m world_cup_projects.player_stats.pass_turnover_explain_run \\
        --video world_cup_projects/bundesliga_videos/08fd33_0.mp4 \\
        --metric --layout talk --explain-video
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

_pkg_root = Path(__file__).resolve().parents[1]
_repo_root = _pkg_root.parent
if (_repo_root / "world_cup_projects" / "__init__.py").is_file():
    if str(_repo_root) not in sys.path:
        sys.path.insert(0, str(_repo_root))

from world_cup_projects import DEFAULT_ASSETS_DIR
from world_cup_projects.common.detect import (
    DEFAULT_BALL_DETECTION_THRESHOLD,
    DEFAULT_FOOTBALL_BALL_MODEL_ID,
    DEFAULT_FOOTBALL_PLAYERS_MODEL_ID,
)
from world_cup_projects.common.video import load_video_sequence, read_sequence_frame
from world_cup_projects.pass_alternatives.pass_options import PassWeights
from world_cup_projects.player_stats.pass_events import (
    InferredTurnover,
    PassDetectionConfig,
    PassQualityScorer,
    build_pass_carrier_timeline,
    scan_possession_events,
)
from world_cup_projects.common.pipeline import load_detections_source
from world_cup_projects.player_stats.pass_explain_visual import (
    PassExplainVideoTiming,
    frames_needed_for_explain,
)
from world_cup_projects.player_stats.pass_turnover_explain_visual import (
    build_turnover_explain_context,
    build_turnover_explain_video_sequence,
    build_turnover_strip_plan,
    pick_explain_turnover,
    render_turnover_explain_strips,
    write_turnover_explain_frames,
    write_turnover_explain_gif,
    write_turnover_explain_video,
)


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


def _team_for_tid(
    dets_by_frame: dict[int, object],
    frame_idx: int,
    tid: int,
    *,
    search: int = 12,
) -> int | None:
    teams: list[int] = []
    for fi in range(frame_idx - search, frame_idx + search + 1):
        dets = dets_by_frame.get(fi)
        if dets is None or getattr(dets, "tracker_id", None) is None:
            continue
        team_arr = dets.data.get("team")
        if team_arr is None:
            continue
        mask = dets.tracker_id == tid
        if not mask.any():
            continue
        teams.append(int(team_arr[mask][0]))
    if not teams:
        return None
    # Mode team across nearby frames — stabilizes noisy single-frame labels.
    return max(set(teams), key=teams.count)


def _explain_only_turnover(
    *,
    passer_tid: int,
    interceptor_tid: int,
    release_frame: int,
    intercept_frame: int,
    dets_by_frame: dict[int, object],
) -> InferredTurnover:
    passer_team = _team_for_tid(dets_by_frame, release_frame, passer_tid)
    interceptor_team = _team_for_tid(dets_by_frame, intercept_frame, interceptor_tid)
    if passer_team is None or interceptor_team is None:
        raise SystemExit(
            "Could not resolve teams for explain-only turnover "
            f"(passer #{passer_tid} f{release_frame}, "
            f"interceptor #{interceptor_tid} f{intercept_frame})"
        )
    gap = intercept_frame - release_frame
    return InferredTurnover(
        release_frame=release_frame,
        interception_frame=intercept_frame,
        passer_tid=passer_tid,
        passer_team=passer_team,
        interceptor_tid=interceptor_tid,
        interceptor_team=interceptor_team,
        gap_frames=gap,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render turnover filmstrip (pass attempt → opponent intercept)"
    )
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--metric", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tracker", default="botsort", choices=["bytetrack", "botsort", "botsort_nocmc"])
    parser.add_argument("--layout", choices=["talk", "social"], default="talk")
    parser.add_argument("--turnover-index", type=int, default=None)
    parser.add_argument("--passer-tid", type=int, default=None, help="Pick turnover by passer")
    parser.add_argument("--interceptor-tid", type=int, default=None, help="Pick turnover by interceptor")
    parser.add_argument("--release-frame", type=int, default=None)
    parser.add_argument(
        "--intercept-frame",
        type=int,
        default=None,
        help="Explain-only: build turnover from frames when scan missed it",
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--no-timeline", action="store_true")
    parser.add_argument("--explain-video", action="store_true")
    parser.add_argument("--video-fps", type=float, default=8.0)
    parser.add_argument("--video-hold", type=float, default=1.2)
    parser.add_argument("--video-crf", type=int, default=16)
    parser.add_argument("--gif", action="store_true")
    parser.add_argument("--gif-width", type=int, default=1280)
    parser.add_argument("--refresh-detections-cache", action="store_true")
    parser.add_argument("--ball-threshold", type=float, default=0.15)
    parser.add_argument("--ball-detector-backend", choices=("none", "yolo", "inference"), default="yolo")
    args = parser.parse_args()

    sequence = load_video_sequence(args.video)
    end = args.max_frames if args.max_frames is not None else sequence.length
    config = PassDetectionConfig().for_frame_rate(sequence.frame_rate)

    class _Args:
        source = "football"
        device = args.device
        tracker = args.tracker
        refresh_detections_cache = args.refresh_detections_cache
        detector_backend = "yolo"
        player_model_id = DEFAULT_FOOTBALL_PLAYERS_MODEL_ID
        detection_threshold = 0.5
        ball_threshold = args.ball_threshold
        ball_detector_backend = args.ball_detector_backend
        ball_model_id = DEFAULT_FOOTBALL_BALL_MODEL_ID

    detections_source = load_detections_source(_Args, sequence)
    frames = list(detections_source(sequence, start=1, end=end))

    from world_cup_projects.common.teams import stabilize_teams_by_tracklet

    frames = stabilize_teams_by_tracklet(frames)
    dets_by_frame = {int(fi): d for fi, d in frames}

    metric = args.metric
    pitch_maps = None
    if metric:
        try:
            pitch_maps = _load_pitch_maps(
                sequence, device=args.device, end=end, detections_by_frame=dets_by_frame
            )
        except RuntimeError as exc:
            print(f"Warning: metric mode disabled ({exc})")
            metric = False
    frame_transforms = pitch_maps.transforms if pitch_maps is not None else {}

    weights = PassWeights.metric() if metric else PassWeights()
    scorer = PassQualityScorer(
        weights=weights, metric=metric, transformers=frame_transforms
    )

    print("Scanning for turnovers...")
    scan = scan_possession_events(
        iter(frames),
        scorer=scorer,
        config=config,
        metric=metric,
        transformers=frame_transforms,
        fps=sequence.frame_rate,
    )
    turnovers = list(scan.turnovers)
    if not turnovers:
        raise SystemExit("No turnovers detected. Try a longer clip or different video.")

    for t in turnovers:
        print(
            f"    #{t.passer_tid} lost → #{t.interceptor_tid} "
            f"rel={t.release_frame} intercept={t.interception_frame} gap={t.gap_frames}f"
        )

    timeline = build_pass_carrier_timeline(
        iter(frames),
        config=config,
        metric=metric,
        transformers=frame_transforms,
    )
    timeline_by_frame = {st.frame_idx: st for st in timeline}

    if args.passer_tid is not None or args.interceptor_tid is not None:
        matches = [
            t
            for t in turnovers
            if (args.passer_tid is None or t.passer_tid == args.passer_tid)
            and (args.interceptor_tid is None or t.interceptor_tid == args.interceptor_tid)
        ]
        if not matches:
            if (
                args.passer_tid is not None
                and args.interceptor_tid is not None
                and args.release_frame is not None
                and args.intercept_frame is not None
            ):
                turnover = _explain_only_turnover(
                    passer_tid=args.passer_tid,
                    interceptor_tid=args.interceptor_tid,
                    release_frame=args.release_frame,
                    intercept_frame=args.intercept_frame,
                    dets_by_frame=dets_by_frame,
                )
                print(
                    "  (explain-only turnover — not in current scan; "
                    "detection unchanged)"
                )
            else:
                raise SystemExit(
                    f"No turnover matching passer={args.passer_tid} "
                    f"interceptor={args.interceptor_tid}"
                )
        else:
            turnover = matches[0]
    elif args.release_frame is not None:
        matches = [t for t in turnovers if t.release_frame == args.release_frame]
        if not matches:
            raise SystemExit(f"No turnover at release frame {args.release_frame}")
        turnover = matches[0]
    else:
        turnover = pick_explain_turnover(
            turnovers,
            turnover_index=args.turnover_index,
            timeline_by_frame=timeline_by_frame,
            dets_by_frame=dets_by_frame,
            min_control_frames=config.min_control_frames,
        )

    print(
        f"  → turnover #{turnover.passer_tid} intercepted by #{turnover.interceptor_tid} "
        f"rel={turnover.release_frame} intercept={turnover.interception_frame}"
    )

    strip_plan = build_turnover_strip_plan(
        turnover,
        timeline_by_frame,
        dets_by_frame=dets_by_frame,
        config=config,
        transformers=frame_transforms,
        metric=metric,
        fps=sequence.frame_rate,
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

    ctx = build_turnover_explain_context(
        turnover,
        frame_rate=sequence.frame_rate,
        frames_by_idx=frames_by_idx,
        dets_by_frame=dets_by_frame,
        timeline_by_frame=timeline_by_frame,
        keypoints_by_frame=keypoints_by_frame,
        radar_transformers=radar_transformers,
        metric=metric,
        config=config,
        min_control_frames=config.min_control_frames,
        min_arrival_frames=config.min_arrival_frames,
        strip_plan=strip_plan,
    )

    out_dir = args.out or (DEFAULT_ASSETS_DIR / "explain_frames")
    strips = render_turnover_explain_strips(ctx, layout=args.layout)
    paths = write_turnover_explain_frames(out_dir, strips, timeline=not args.no_timeline)

    video_timing = PassExplainVideoTiming(
        output_fps=args.video_fps,
        hold_locked_seconds=args.video_hold,
        crf=args.video_crf,
    )

    if args.explain_video or args.gif:
        video_frames = build_turnover_explain_video_sequence(ctx, timing=video_timing)
        video_path = write_turnover_explain_video(
            out_dir / "pass_turnover_explain.mp4",
            video_frames,
            fps=video_timing.output_fps,
            crf=video_timing.crf,
        )
        paths.append(video_path)
        dur = len(video_frames) / video_timing.output_fps
        print(
            f"  Video: {len(video_frames)} frames @ {video_timing.output_fps:.1f} fps (~{dur:.1f}s)"
        )
        if args.gif:
            gif_width = args.gif_width if args.gif_width > 0 else None
            gif_path = write_turnover_explain_gif(
                out_dir / "pass_turnover_explain.gif",
                video_path,
                fps=video_timing.output_fps,
                width=gif_width,
            )
            paths.append(gif_path)

    manifest = {
        "sequence": sequence.name,
        "turnover": turnover.to_dict(),
        "logic": asdict(ctx.logic),
        "layout": args.layout,
        "metric": metric,
        "strips": {
            "passer": list(ctx.pass_ctx.strip_plan.passer_frames),
            "flight": list(ctx.pass_ctx.strip_plan.flight_frames),
            "intercept": list(ctx.pass_ctx.strip_plan.receiver_frames),
            "summary": ctx.pass_ctx.strip_plan.summary_frame,
        },
        "outputs": [str(p) for p in paths],
    }
    manifest_path = out_dir / "pass_turnover_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f"\nWrote {len(paths)} file(s) to {out_dir}/")
    for p in paths:
        print(f"  {p.name}")
    print(f"  {manifest_path.name}")


if __name__ == "__main__":
    main()
