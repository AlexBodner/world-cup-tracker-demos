"""Diagnose why pass-alternative freezes do or don't trigger on detected passes."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import cv2
import numpy as np
import supervision as sv

from world_cup_projects.common.pitch import image_to_pitch_m
from world_cup_projects.common.possession import (
    ball_xy,
    carrier_from_tracker_id,
    feet_xy,
    find_control_carrier,
    player_mask,
)
from world_cup_projects.common.visual import (
    ROBOFLOW_PURPLE_BGR,
    annotate_ball,
    annotate_players,
    draw_branding_tag,
    draw_carrier_halo,
    draw_glow_arrow,
    draw_hud_bar,
    draw_text_shadow,
)
from world_cup_projects.pass_alternatives.pass_options import PassWeights
from world_cup_projects.player_stats.pass_events import InferredPass, PassQualityScorer


@dataclass
class FreezeDiagnosis:
    """Per-pass report for ``--show-predictions`` freeze gating."""

    frame_idx: int
    passer_tid: int
    receiver_tid: int
    pass_length_m: float | None
    quality_score: float | None
    would_freeze: bool
    gates: dict[str, bool] = field(default_factory=dict)
    blockers: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _receiver_length_m(
    dets: sv.Detections,
    carrier_index: int,
    receiver_tid: int,
    *,
    transformer,
) -> float | None:
    if dets.tracker_id is None or transformer is None:
        return None
    pmask = player_mask(dets)
    rows = np.flatnonzero(pmask & (dets.tracker_id == receiver_tid))
    if len(rows) == 0:
        return None
    feet = feet_xy(dets)
    pitch = image_to_pitch_m(feet, transformer)
    if pitch is None:
        return None
    delta = pitch[int(rows[0])] - pitch[carrier_index]
    return float(np.linalg.norm(delta))


def _explain_quality_score_null(
    scorer: PassQualityScorer,
    frame_idx: int,
    dets: sv.Detections,
    carrier,
    receiver_tid: int,
    *,
    weights: PassWeights,
) -> list[str]:
    reasons: list[str] = []
    if receiver_tid < 0:
        return ["invalid receiver tracker id"]

    pmask = player_mask(dets)
    receiver_rows = np.flatnonzero(pmask & (dets.tracker_id == receiver_tid))
    if len(receiver_rows) == 0:
        return [f"receiver #{receiver_tid} not in detections at release frame"]

    transformer = scorer._transformers.get(frame_idx)
    if scorer._metric and transformer is None:
        reasons.append("no speed homography at release frame (transformer is None)")

    if scorer._metric and transformer is not None:
        length_m = _receiver_length_m(
            dets, carrier.index, receiver_tid, transformer=transformer
        )
        if length_m is None:
            reasons.append("could not project feet to pitch meters at release")
        elif length_m < weights.min_length:
            reasons.append(
                f"lane length {length_m:.2f} m < min {weights.min_length:.0f} m "
                "(receiver excluded from scored list)"
            )
        elif length_m > weights.max_length:
            reasons.append(
                f"lane length {length_m:.2f} m > max {weights.max_length:.0f} m "
                "(receiver excluded from scored list)"
            )

    option = scorer.option_for_receiver(frame_idx, dets, carrier, receiver_tid)
    if option is None and not reasons:
        reasons.append("receiver lane not in scored options (unknown scoring failure)")
    return reasons


def diagnose_pass_freeze(
    pass_event: InferredPass,
    dets: sv.Detections,
    *,
    scorer: PassQualityScorer,
    weights: PassWeights,
    freeze_quality_threshold: float = 0.0,
    show_predictions: bool = True,
    keypoints: sv.KeyPoints | None = None,
) -> FreezeDiagnosis:
    """Evaluate every freeze gate for one inferred pass at its release frame."""
    frame_idx = pass_event.frame_idx
    blockers: list[str] = []
    notes: list[str] = []
    gates: dict[str, bool] = {}

    gates["show_predictions"] = show_predictions
    if not show_predictions:
        blockers.append("--show-predictions is off")

    gates["pass_emitted"] = True

    passer_carrier = carrier_from_tracker_id(dets, pass_event.passer_tid)
    qs = pass_event.quality_score
    gates["quality_score_not_null"] = qs is not None
    if qs is None:
        diag_carrier = passer_carrier or find_control_carrier(
            dets, transformer=scorer._transformers.get(frame_idx)
        )
        if diag_carrier is None:
            blockers.append("quality_score is null (passer not in frame to re-diagnose)")
        else:
            for reason in _explain_quality_score_null(
                scorer,
                frame_idx,
                dets,
                diag_carrier,
                pass_event.receiver_tid,
                weights=weights,
            ):
                blockers.append(f"quality_score null: {reason}")
    elif qs < freeze_quality_threshold:
        gates["quality_score_above_threshold"] = False
        blockers.append(
            f"quality_score {qs:.3f} < threshold {freeze_quality_threshold:.3f}"
        )
    else:
        gates["quality_score_above_threshold"] = True

    transformer = scorer._transformers.get(frame_idx)
    gates["homography_at_release"] = transformer is not None or not scorer._metric
    if scorer._metric and transformer is None:
        notes.append("metric mode but no homography on release frame")

    gates["keypoints_at_release"] = keypoints is not None
    if keypoints is None:
        notes.append("no pitch keypoints at release (freeze may still run; no radar/corridors)")

    gates["passer_in_frame"] = passer_carrier is not None
    if passer_carrier is None:
        blockers.append(
            f"inferred passer #{pass_event.passer_tid} not in detections at release frame"
        )

    top_options = []
    if passer_carrier is not None:
        top_options = scorer.top_options(frame_idx, dets, passer_carrier, k=3)
    gates["top_options_non_empty"] = len(top_options) > 0
    if passer_carrier is not None and not top_options:
        blockers.append("top_options(3) empty — no scorable teammate lanes at release")
        if scorer._metric and transformer is None:
            notes.append("top_options likely empty because homography missing")

    if top_options:
        ranked_tids = [
            int(dets.tracker_id[o.receiver_index])
            for o in top_options
            if dets.tracker_id is not None
        ]
        receiver_in_top3 = pass_event.receiver_tid in ranked_tids
        gates["actual_receiver_in_top3"] = receiver_in_top3
        if not receiver_in_top3:
            notes.append(
                f"actual receiver #{pass_event.receiver_tid} not in displayed top 3 "
                f"(shown: {ranked_tids}) — freeze can still run"
            )
        else:
            rank = ranked_tids.index(pass_event.receiver_tid) + 1
            notes.append(f"actual receiver is rank #{rank} in top 3")

    above = qs is not None and qs >= freeze_quality_threshold
    gates.setdefault("quality_score_above_threshold", above)
    would_freeze = (
        show_predictions
        and gates.get("quality_score_not_null", False)
        and above
        and gates.get("passer_in_frame", False)
        and gates.get("top_options_non_empty", False)
    )

    return FreezeDiagnosis(
        frame_idx=frame_idx,
        passer_tid=pass_event.passer_tid,
        receiver_tid=pass_event.receiver_tid,
        pass_length_m=pass_event.pass_length_m,
        quality_score=qs,
        would_freeze=would_freeze,
        gates=gates,
        blockers=blockers,
        notes=notes,
    )


def diagnose_all_passes(
    passes: list[InferredPass],
    frames_by_idx: dict[int, sv.Detections],
    *,
    scorer: PassQualityScorer,
    weights: PassWeights,
    freeze_quality_threshold: float = 0.0,
    show_predictions: bool = True,
    keypoints_by_frame: dict[int, sv.KeyPoints | None] | None = None,
) -> list[FreezeDiagnosis]:
    kps_map = keypoints_by_frame or {}
    return [
        diagnose_pass_freeze(
            p,
            frames_by_idx[p.frame_idx],
            scorer=scorer,
            weights=weights,
            freeze_quality_threshold=freeze_quality_threshold,
            show_predictions=show_predictions,
            keypoints=kps_map.get(p.frame_idx),
        )
        for p in passes
        if p.frame_idx in frames_by_idx
    ]


_PASS_BGR = (80, 220, 60)
_FAIL_BGR = (80, 80, 255)
_OK_BGR = (80, 220, 120)


def _draw_debug_panel(
    frame: np.ndarray,
    diagnosis: FreezeDiagnosis,
    *,
    pass_index: int,
    pass_total: int,
) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]
    panel_w = min(520, w - 24)
    panel_h = min(420, h - 80)
    px, py = w - panel_w - 12, 52

    overlay = out.copy()
    cv2.rectangle(overlay, (px, py), (px + panel_w, py + panel_h), (14, 14, 18), -1)
    cv2.rectangle(overlay, (px, py), (px + panel_w, py + panel_h), (55, 55, 65), 1)
    status_color = _OK_BGR if diagnosis.would_freeze else _FAIL_BGR
    cv2.rectangle(overlay, (px, py), (px + panel_w, py + 4), status_color, -1)
    out[:] = cv2.addWeighted(overlay, 0.92, out, 0.08, 0)

    title = "FREEZE" if diagnosis.would_freeze else "NO FREEZE"
    draw_text_shadow(
        out,
        f"Pass {pass_index}/{pass_total}  frame {diagnosis.frame_idx}  →  {title}",
        (px + 12, py + 26),
        font_scale=0.52,
        color_bgr=status_color,
        thickness=2,
    )
    draw_text_shadow(
        out,
        f"#{diagnosis.passer_tid} → #{diagnosis.receiver_tid}  "
        f"len={diagnosis.pass_length_m:.1f}m"
        if diagnosis.pass_length_m is not None
        else f"#{diagnosis.passer_tid} → #{diagnosis.receiver_tid}",
        (px + 12, py + 48),
        font_scale=0.42,
        color_bgr=(210, 210, 220),
        thickness=1,
    )
    qs = diagnosis.quality_score
    draw_text_shadow(
        out,
        f"quality_score: {qs:.3f}" if qs is not None else "quality_score: null",
        (px + 12, py + 68),
        font_scale=0.42,
        color_bgr=(210, 210, 220),
        thickness=1,
    )

    y = py + 92
    gate_labels = [
        ("show_predictions", "show_predictions on"),
        ("quality_score_not_null", "quality_score not null"),
        ("quality_score_above_threshold", "quality_score ≥ threshold"),
        ("passer_in_frame", "passer in detections at release"),
        ("top_options_non_empty", "top_options(3) non-empty"),
        ("homography_at_release", "homography at release"),
        ("keypoints_at_release", "keypoints at release"),
    ]
    for key, label in gate_labels:
        if key not in diagnosis.gates:
            continue
        ok = diagnosis.gates[key]
        mark = "✓" if ok else "✗"
        color = _OK_BGR if ok else _FAIL_BGR
        draw_text_shadow(
            out,
            f"{mark} {label}",
            (px + 12, y),
            font_scale=0.40,
            color_bgr=color,
            thickness=1,
        )
        y += 20

    if diagnosis.blockers:
        y += 6
        draw_text_shadow(
            out, "Blocked by:", (px + 12, y), font_scale=0.42, color_bgr=_FAIL_BGR, thickness=1
        )
        y += 18
        for line in diagnosis.blockers[:5]:
            draw_text_shadow(
                out,
                f"• {line[:62]}",
                (px + 16, y),
                font_scale=0.36,
                color_bgr=(200, 200, 210),
                thickness=1,
            )
            y += 16

    if diagnosis.notes:
        y += 4
        draw_text_shadow(
            out, "Notes:", (px + 12, y), font_scale=0.40, color_bgr=(170, 170, 180), thickness=1
        )
        y += 16
        for line in diagnosis.notes[:3]:
            draw_text_shadow(
                out,
                f"· {line[:60]}",
                (px + 16, y),
                font_scale=0.34,
                color_bgr=(160, 160, 170),
                thickness=1,
            )
            y += 15

    return out


def _highlight_pass_players(
    frame: np.ndarray,
    dets: sv.Detections,
    passer_tid: int,
    receiver_tid: int,
) -> np.ndarray:
    out = annotate_players(frame, dets, show_tracker_ids=True)
    out = annotate_ball(out, dets)
    feet = feet_xy(dets)
    if dets.tracker_id is None:
        return out
    passer_xy = receiver_xy = None
    for tid in (passer_tid, receiver_tid):
        rows = np.flatnonzero(dets.tracker_id == tid)
        if len(rows):
            x, y = int(feet[int(rows[0]), 0]), int(feet[int(rows[0]), 1])
            draw_carrier_halo(out, (x, y))
            if tid == passer_tid:
                passer_xy = (x, y)
            else:
                receiver_xy = (x, y)
    if passer_xy and receiver_xy:
        draw_glow_arrow(
            out, passer_xy, receiver_xy, _PASS_BGR, thickness=3, alpha=0.45
        )
    return out


def render_freeze_debug_video(
    sequence,
    frames: list[tuple[int, sv.Detections]],
    passes: list[InferredPass],
    diagnoses: list[FreezeDiagnosis],
    out_path: str,
    *,
    frame_loader,
    hold_seconds: float = 3.5,
) -> dict:
    """Write a video that holds on each detected pass with freeze gate diagnostics."""
    if not diagnoses:
        raise ValueError("No pass diagnoses to render")

    diag_by_frame = {d.frame_idx: d for d in diagnoses}
    pass_by_frame = {p.frame_idx: p for p in passes}
    h, w = frame_loader(1).shape[:2]
    fps = float(sequence.frame_rate)
    hold_frames = max(1, int(round(hold_seconds * fps)))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {out_path}")

    ordered = sorted(diagnoses, key=lambda d: d.frame_idx)
    for i, diag in enumerate(ordered, start=1):
        frame = frame_loader(diag.frame_idx)
        if frame is None:
            continue
        dets = next(d for fi, d in frames if fi == diag.frame_idx)
        p = pass_by_frame[diag.frame_idx]
        vis = _highlight_pass_players(
            frame, dets, p.passer_tid, p.receiver_tid
        )
        vis = _draw_debug_panel(vis, diag, pass_index=i, pass_total=len(ordered))
        vis = draw_hud_bar(vis, "FREEZE DEBUG — pass alternative gates")
        vis = draw_branding_tag(vis, "Roboflow · freeze debug")
        for _ in range(hold_frames):
            writer.write(vis)

    writer.release()
    n_freeze = sum(1 for d in diagnoses if d.would_freeze)
    return {
        "output": out_path,
        "n_passes": len(diagnoses),
        "n_would_freeze": n_freeze,
        "n_blocked": len(diagnoses) - n_freeze,
        "diagnoses": [d.to_dict() for d in diagnoses],
    }
