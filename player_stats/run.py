"""Run the speed & distance demo end-to-end.

GT tracks + pitch homography (default, metric m/s and meters)::

    PYTHONPATH=. python -m world_cup_projects.player_stats.run --sequence SNMOT-194

RF-DETR + ByteTrack source (needs RF-DETR weights under ``world_cup_projects/.cache/rf``)::

    PYTHONPATH=. python -m world_cup_projects.player_stats.run \\
        --sequence SNMOT-194 --source rfdetr --max-frames 150
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import supervision as sv

from world_cup_projects import DEFAULT_ASSETS_DIR
from world_cup_projects.common.clips import rank_clips
from world_cup_projects.common.pitch import iter_pitch_transformers
from world_cup_projects.common.soccernet import (
    DEFAULT_TRACKING_ROOT,
    find_sequences,
    iter_gt_detections,
    load_sequence,
)
from world_cup_projects.player_stats.render import render_demo
from world_cup_projects.common.pitch import HOMOGRAPHY_RANSAC_REPROJ_THRESH
from world_cup_projects.player_stats.speed_distance import (
    DEFAULT_SPEED_K_FRAMES,
    collect_tracks,
    compute_kinematics,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="player speed & distance demo")
    parser.add_argument("--data", default=DEFAULT_TRACKING_ROOT)
    parser.add_argument("--split", default="test")
    parser.add_argument("--sequence", default=None)
    parser.add_argument("--out", default=str(DEFAULT_ASSETS_DIR))
    parser.add_argument("--source", choices=["gt", "rfdetr"], default="gt")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--rfdetr-model", default="nano")
    parser.add_argument(
        "--mode",
        choices=["homography", "height"],
        default="homography",
        help="homography = pitch keypoints (meters); height = bbox-height fallback.",
    )
    parser.add_argument("--device", default="cpu", help="Pitch keypoint model device.")
    parser.add_argument(
        "--speed-k-frames",
        type=int,
        default=DEFAULT_SPEED_K_FRAMES,
        help="Speed = displacement over this many frames / elapsed time (default 10).",
    )
    parser.add_argument(
        "--ransac-thresh",
        type=float,
        default=HOMOGRAPHY_RANSAC_REPROJ_THRESH,
        help="RANSAC reprojection threshold in pixels for pitch homography.",
    )
    parser.add_argument(
        "--max-reproj-px",
        type=float,
        default=8.0,
        help="Reject homography updates with mean reprojection error above this (px).",
    )
    parser.add_argument("--no-radar", action="store_true")
    parser.add_argument(
        "--debug-pitch-keypoints",
        action="store_true",
        help="Draw pitch keypoints + confidence on each frame (homography debug).",
    )
    parser.add_argument(
        "--pitch-confidence",
        type=float,
        default=0.5,
        help="Keypoint confidence threshold (overlay legend + homography filter).",
    )
    args = parser.parse_args()

    seq_dirs = find_sequences(args.data, args.split)
    if args.sequence:
        seq_dir = next(p for p in seq_dirs if p.name == args.sequence)
    else:
        seq_dir = next(p for p in seq_dirs if p.name == rank_clips(seq_dirs)[0].name)
    sequence = load_sequence(seq_dir)
    end = args.max_frames

    if args.source == "rfdetr":
        from world_cup_projects.common.detect import (
            RFDETRDetector,
            fit_team_classifier,
            iter_rfdetr_detections,
        )

        detector = RFDETRDetector(args.rfdetr_model, device=args.device)
        clf = fit_team_classifier(sequence, detector, max_frames=end or sequence.length)
        frames = list(iter_rfdetr_detections(sequence, detector, clf, end=end))
    else:
        frames = list(iter_gt_detections(sequence, end=end))

    frame_transforms: dict[int, object | None] = {}
    frame_keypoints: dict[int, sv.KeyPoints | None] = {}
    need_pitch_pass = args.mode == "homography" or args.debug_pitch_keypoints
    if need_pitch_pass:
        for idx, t_speed, _t_radar, kps in iter_pitch_transformers(
            sequence,
            device=args.device,
            end=end,
            ransac_thresh=args.ransac_thresh,
            max_reproj_px=args.max_reproj_px,
            confidence=args.pitch_confidence,
            yield_keypoints=True,
        ):
            frame_keypoints[idx] = kps
            if args.mode == "homography":
                frame_transforms[idx] = t_speed

    tracks = collect_tracks(iter(frames))
    compute_kinematics(
        tracks,
        sequence.frame_rate,
        mode=args.mode,
        frame_transforms=frame_transforms if args.mode == "homography" else None,
        speed_k_frames=args.speed_k_frames,
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.source}_{args.mode}"
    if args.debug_pitch_keypoints:
        tag += "_pitch_kp_debug"
    out_path = out_dir / f"player_stats_{tag}_{sequence.name}.mp4"

    calibration = (
        "pitch homography (meters, m/s)"
        if args.mode == "homography"
        else f"bbox-height (~1.8 m), source={args.source}"
    )

    manifest = render_demo(
        sequence,
        iter(frames),
        tracks,
        str(out_path),
        frame_loader=lambda fi: cv2.imread(str(sequence.frame_path(fi))),
        calibration=calibration,
        frame_transforms=frame_transforms if args.mode == "homography" else None,
        show_radar=not args.no_radar and args.mode == "homography",
        frame_keypoints=frame_keypoints if need_pitch_pass else None,
        pitch_kp_debug=args.debug_pitch_keypoints,
        pitch_confidence=args.pitch_confidence,
    )
    json_path = out_dir / f"player_stats_{tag}_{sequence.name}.json"
    json_path.write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
