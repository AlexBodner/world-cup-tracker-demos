"""Auto-select the best SoccerNet clips for the demos.

A "good" pass-alternatives clip has: the ball visible most of the time, frequent clear
possession (ball glued to a player's feet), both teams well represented, plenty of
players on screen, and **reliable pitch keypoints** for homography/radar.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from world_cup_projects.common.possession import (
    ball_xy,
    find_ball_carrier,
    player_mask,
)
from world_cup_projects.common.soccernet import iter_gt_detections, load_sequence

# Clips where pitch keypoints are often unusable (broken radar / homography).
PITCH_KEYPOINT_AVOID: frozenset[str] = frozenset(
    {"SNMOT-132", "SNMOT-189", "SNMOT-197"}
)

PITCH_KEYPOINT_AVOID_NOTES: dict[str, str] = {
    "SNMOT-189": "Pitch area keypoints are often wrong; homography and radar are unreliable.",
    "SNMOT-132": "Pitch area keypoints are often missing or wrong; homography and radar fail.",
    "SNMOT-197": "Pitch keypoints rarely reproject cleanly to the radar at high confidence.",
}

# Clips with poor gameplay for demos (e.g. play stopped — little live action).
PITCH_GAMEPLAY_AVOID: frozenset[str] = frozenset({"SNMOT-127"})

PITCH_GAMEPLAY_AVOID_NOTES: dict[str, str] = {
    "SNMOT-127": "Play is often stopped (referee); homography is fine but little live game.",
}

# Best homography/radar reprojection among active-game test clips @ confidence 0.9.
PITCH_HOMOGRAPHY_DEMO_CLIP: str = "SNMOT-117"


def pitch_keypoints_unreliable(sequence_name: str) -> bool:
    return sequence_name in PITCH_KEYPOINT_AVOID


@dataclass(frozen=True)
class PitchKeypointSummary:
    sampled_frames: int
    homography_ok_frames: int
    mean_reproj_px: float
    mean_accepted_kps: float

    @property
    def ok_ratio(self) -> float:
        if self.sampled_frames < 1:
            return 0.0
        return self.homography_ok_frames / self.sampled_frames

    @property
    def ok_for_demo(self) -> bool:
        if self.sampled_frames < 5:
            return False
        return self.ok_ratio >= 0.55 and self.mean_reproj_px <= 10.0

    @property
    def homography_usable(self) -> bool:
        """Relaxed gate for picking the least-bad homography clip."""
        if self.sampled_frames < 5:
            return False
        return self.ok_ratio >= 0.30 and self.mean_reproj_px <= 8.0


@dataclass(frozen=True)
class ClipScore:
    name: str
    path: Path
    frames: int
    ball_frames: int
    carrier_frames: int
    avg_players: float
    both_teams: bool
    score: float
    pitch_kp: PitchKeypointSummary | None = None

    @property
    def reason(self) -> str:
        base = (
            f"ball visible {self.ball_frames}/{self.frames} frames, "
            f"clear possession in {self.carrier_frames} frames, "
            f"~{self.avg_players:.0f} players/frame, "
            f"{'both teams' if self.both_teams else 'single team'}"
        )
        if self.pitch_kp is None:
            return base
        pk = self.pitch_kp
        tag = "pitch kp OK" if pk.ok_for_demo else "pitch kp POOR"
        return (
            f"{base}, {tag} "
            f"({pk.homography_ok_frames}/{pk.sampled_frames} sampled, "
            f"reproj {pk.mean_reproj_px:.1f}px)"
        )


def score_sequence(seq_dir: Path, *, carrier_max_distance: float = 80.0) -> ClipScore:
    seq = load_sequence(seq_dir)
    ball_frames = carrier_frames = 0
    player_counts: list[int] = []
    teams: set[int] = set()

    for _, dets in iter_gt_detections(seq):
        player_counts.append(int(player_mask(dets).sum()))
        if find_ball_carrier(dets, max_distance_px=carrier_max_distance) is not None:
            carrier_frames += 1
        if ball_xy(dets) is not None:
            ball_frames += 1
        teams.update(int(t) for t in dets.data["team"] if t in (0, 1))

    avg_players = float(np.mean(player_counts)) if player_counts else 0.0
    both_teams = {0, 1}.issubset(teams)
    # Possession density dominates; reward player count, require both teams.
    score = (
        carrier_frames
        + 0.3 * ball_frames
        + 5.0 * avg_players
        + (200.0 if both_teams else 0.0)
    )
    return ClipScore(
        name=seq.name,
        path=Path(seq_dir),
        frames=seq.length,
        ball_frames=ball_frames,
        carrier_frames=carrier_frames,
        avg_players=avg_players,
        both_teams=both_teams,
        score=score,
    )


def assess_pitch_keypoints(
    seq_dir: Path,
    *,
    device: str = "cpu",
    sample_step: int = 30,
    confidence: float = 0.5,
    max_reproj_px: float = 8.0,
) -> PitchKeypointSummary:
    """Sample frames and measure pitch-keypoint / homography reliability."""
    import supervision as sv

    from world_cup_projects.common.pitch import (
        DISPLAY_MAX_REPROJ_PX,
        detect_pitch_keypoints,
        homography_from_keypoints_simple,
        load_pitch_model,
        pitch_keypoint_accept_mask,
        pitch_keypoint_confidence,
        _mean_reproj_px,
    )

    seq = load_sequence(seq_dir)
    homography = load_pitch_model(device=device)
    homography.confidence = confidence

    reprojs: list[float] = []
    accepted: list[int] = []
    ok_frames = 0
    sampled = 0

    for frame_idx in range(1, seq.length + 1, sample_step):
        frame = cv2.imread(str(seq.frame_path(frame_idx)))
        if frame is None:
            continue
        sampled += 1
        kps = detect_pitch_keypoints(frame, homography)
        if kps.xy.shape[0] == 0:
            continue
        xy = kps.xy[0]
        conf = pitch_keypoint_confidence(kps, n_vertices=len(xy))
        mask = pitch_keypoint_accept_mask(xy, conf, confidence=confidence)
        accepted.append(int(mask.sum()))
        t = homography_from_keypoints_simple(
            kps,
            confidence=confidence,
            max_reproj_px=max_reproj_px,
        )
        if t is None:
            continue
        src = xy[mask].astype(np.float32)
        dst = np.array(homography.config.vertices, dtype=np.float32)[mask]
        err = _mean_reproj_px(t, src, dst)
        reprojs.append(err)
        if err <= min(max_reproj_px, DISPLAY_MAX_REPROJ_PX):
            ok_frames += 1

    return PitchKeypointSummary(
        sampled_frames=sampled,
        homography_ok_frames=ok_frames,
        mean_reproj_px=float(np.mean(reprojs)) if reprojs else 99.0,
        mean_accepted_kps=float(np.mean(accepted)) if accepted else 0.0,
    )


def rank_clips(
    seq_dirs: list[Path],
    *,
    skip_names: frozenset[str] | None = None,
    assess_pitch: bool = False,
    pitch_device: str = "cpu",
    pitch_confidence: float = 0.9,
    require_pitch_ok: bool = False,
    homography_demo: bool = False,
) -> list[ClipScore]:
    avoid = skip_names or (PITCH_KEYPOINT_AVOID | PITCH_GAMEPLAY_AVOID)
    scores: list[ClipScore] = []
    for p in seq_dirs:
        if p.name in avoid:
            continue
        base = score_sequence(p)
        pk = (
            assess_pitch_keypoints(
                p, device=pitch_device, confidence=pitch_confidence
            )
            if assess_pitch
            else None
        )
        if require_pitch_ok and pk is not None and not pk.ok_for_demo:
            continue
        if homography_demo and pk is not None and not pk.homography_usable:
            continue
        pitch_adjust = 0.0
        if pk is not None:
            pitch_adjust = pk.ok_ratio * 400.0 - pk.mean_reproj_px * 8.0
            if not pk.homography_usable:
                pitch_adjust -= 600.0
        scores.append(
            ClipScore(
                name=base.name,
                path=base.path,
                frames=base.frames,
                ball_frames=base.ball_frames,
                carrier_frames=base.carrier_frames,
                avg_players=base.avg_players,
                both_teams=base.both_teams,
                score=base.score + pitch_adjust,
                pitch_kp=pk,
            )
        )
    scores.sort(key=lambda s: s.score, reverse=True)
    return scores


def pick_homography_demo_clip(
    seq_dirs: list[Path],
    *,
    pitch_device: str = "cpu",
    pitch_confidence: float = 0.9,
) -> ClipScore:
    """Prefer clips where keypoints reproject cleanly to the radar."""
    blocked = PITCH_KEYPOINT_AVOID | PITCH_GAMEPLAY_AVOID
    preferred = next((p for p in seq_dirs if p.name == PITCH_HOMOGRAPHY_DEMO_CLIP), None)
    if preferred is not None and preferred.name not in blocked:
        pk = assess_pitch_keypoints(
            preferred, device=pitch_device, confidence=pitch_confidence
        )
        if pk.homography_usable:
            base = score_sequence(preferred)
            return ClipScore(
                name=base.name,
                path=base.path,
                frames=base.frames,
                ball_frames=base.ball_frames,
                carrier_frames=base.carrier_frames,
                avg_players=base.avg_players,
                both_teams=base.both_teams,
                score=base.score,
                pitch_kp=pk,
            )
    ranking = rank_clips(
        seq_dirs,
        assess_pitch=True,
        pitch_device=pitch_device,
        pitch_confidence=pitch_confidence,
        homography_demo=True,
    )
    if ranking:
        return ranking[0]
    ranking = rank_clips(
        seq_dirs,
        assess_pitch=True,
        pitch_device=pitch_device,
        pitch_confidence=pitch_confidence,
    )
    if not ranking:
        raise ValueError("No sequences with assessable pitch keypoints")
    return ranking[0]
