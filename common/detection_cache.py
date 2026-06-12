"""Disk cache for model detection + tracking (avoids re-running YOLO each render)."""

from __future__ import annotations

import pickle
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import supervision as sv

_CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache" / "detections"

DetectionSource = Callable[..., Iterator[tuple[int, sv.Detections]]]


def _sequence_fingerprint(sequence: Any) -> str:
    video_path = getattr(sequence, "video_path", None)
    if video_path is not None:
        st = Path(video_path).stat()
        return f"{sequence.name}_{st.st_size}_{int(st.st_mtime)}_{sequence.length}"
    return f"{sequence.name}_{sequence.length}"


def cache_path(sequence: Any, source: str, **params: Any) -> Path:
    parts = [_sequence_fingerprint(sequence), source]
    for key in sorted(params):
        val = params[key]
        if val is None:
            continue
        parts.append(f"{key}={val}")
    name = "__".join(str(p).replace("/", "_") for p in parts) + ".pkl"
    return _CACHE_DIR / name


def _detections_to_record(dets: sv.Detections) -> dict[str, Any]:
    n = len(dets)
    team = dets.data.get("team") if dets.data else None
    jersey = dets.data.get("jersey") if dets.data else None
    kf_vx = dets.data.get("kf_vx") if dets.data else None
    kf_vy = dets.data.get("kf_vy") if dets.data else None
    if team is None:
        team = np.full(n, -1, dtype=int)
    if jersey is None:
        jersey = np.asarray([""] * n, dtype=object)
    if kf_vx is None:
        kf_vx = np.full(n, np.nan, dtype=np.float32)
    if kf_vy is None:
        kf_vy = np.full(n, np.nan, dtype=np.float32)
    tid = dets.tracker_id
    if tid is None:
        tid = np.full(n, -1, dtype=int)
    return {
        "xyxy": dets.xyxy.astype(np.float32),
        "class_id": dets.class_id.astype(int),
        "tracker_id": tid.astype(int),
        "team": np.asarray(team, dtype=int),
        "jersey": np.asarray(jersey, dtype=object),
        "kf_vx": np.asarray(kf_vx, dtype=np.float32),
        "kf_vy": np.asarray(kf_vy, dtype=np.float32),
    }


def _record_to_detections(rec: dict[str, Any]) -> sv.Detections:
    data = {"team": rec["team"], "jersey": rec["jersey"]}
    if "kf_vx" in rec:
        data["kf_vx"] = rec["kf_vx"]
        data["kf_vy"] = rec["kf_vy"]
    return sv.Detections(
        xyxy=rec["xyxy"],
        class_id=rec["class_id"],
        tracker_id=rec["tracker_id"],
        data=data,
    )


def enrich_cached_kalman_velocity(
    sequence: Any,
    frames: list[tuple[int, sv.Detections]],
    *,
    tracker: str = "bytetrack",
) -> list[tuple[int, sv.Detections]]:
    """Replay tracker on cached boxes to attach ``kf_vx`` / ``kf_vy`` (no YOLO)."""
    from world_cup_projects.common.player_tracker import TrackerKind, create_player_tracker
    from world_cup_projects.common.soccernet import ROLE_GOALKEEPER, ROLE_PLAYER
    from world_cup_projects.common.tracking_facing import (
        detections_have_kalman_velocity,
        kalman_velocity_arrays,
    )
    from world_cup_projects.common.video import read_sequence_frame

    if not frames or detections_have_kalman_velocity(frames[0][1]):
        return frames

    kind: TrackerKind = tracker if tracker in ("bytetrack", "botsort", "botsort_nocmc") else "bytetrack"
    player_tracker = create_player_tracker(sequence.frame_rate, kind=kind)
    needs_frame = kind == "botsort"
    enriched: list[tuple[int, sv.Detections]] = []

    for frame_idx, dets in frames:
        pmask = np.isin(dets.class_id, (ROLE_PLAYER, ROLE_GOALKEEPER))
        if pmask.any():
            trackable = dets[pmask]
            image = read_sequence_frame(sequence, frame_idx) if needs_frame else None
            player_tracker.update(trackable, frame=image if needs_frame else None)
            kf_vx_sub, kf_vy_sub = kalman_velocity_arrays(trackable, player_tracker)
        else:
            kf_vx_sub = kf_vy_sub = np.array([], dtype=np.float32)

        n = len(dets)
        kf_vx = np.full(n, np.nan, dtype=np.float32)
        kf_vy = np.full(n, np.nan, dtype=np.float32)
        if pmask.any():
            kf_vx[pmask] = kf_vx_sub
            kf_vy[pmask] = kf_vy_sub
        data = dict(dets.data) if dets.data else {}
        data["kf_vx"] = kf_vx
        data["kf_vy"] = kf_vy
        enriched.append(
            (
                frame_idx,
                sv.Detections(
                    xyxy=dets.xyxy,
                    class_id=dets.class_id,
                    tracker_id=dets.tracker_id,
                    data=data,
                ),
            )
        )
    return enriched


def load_cached_detections(path: Path) -> tuple[dict[str, Any], list[tuple[int, sv.Detections]]] | None:
    if not path.is_file():
        return None
    with path.open("rb") as f:
        payload = pickle.load(f)
    frames = [(int(fi), _record_to_detections(rec)) for fi, rec in payload["frames"]]
    return payload.get("meta", {}), frames


def save_cached_detections(
    path: Path,
    frames: list[tuple[int, sv.Detections]],
    *,
    meta: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": meta,
        "frames": [(fi, _detections_to_record(d)) for fi, d in frames],
    }
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)


def wrap_detections_cache(
    source: DetectionSource,
    *,
    source_name: str,
    refresh: bool = False,
    **cache_params: Any,
) -> DetectionSource:
    """Persist detection iterator results; reload on subsequent renders."""

    def _cached(
        sequence: Any,
        *,
        start: int = 1,
        end: int | None = None,
        **kwargs: Any,
    ) -> Iterator[tuple[int, sv.Detections]]:
        last = sequence.length if end is None else min(end, sequence.length)
        key_params = {**cache_params, **kwargs}
        path = cache_path(sequence, source_name, start=start, end=last, **key_params)

        if not refresh:
            loaded = load_cached_detections(path)
            if loaded is None and "ball_threshold" in key_params:
                legacy_params = {
                    k: v for k, v in key_params.items() if k != "ball_threshold"
                }
                legacy_path = cache_path(
                    sequence, source_name, start=start, end=last, **legacy_params
                )
                loaded = load_cached_detections(legacy_path)
                if loaded is not None:
                    path = legacy_path
            if loaded is not None:
                meta, frames = loaded
                if (
                    meta.get("sequence") == sequence.name
                    and int(meta.get("end", 0)) >= last
                    and int(meta.get("start", 1)) <= start
                ):
                    tracker = str(meta.get("tracker", "bytetrack"))
                    from world_cup_projects.common.tracking_facing import (
                        detections_have_kalman_velocity,
                    )

                    if frames and not detections_have_kalman_velocity(frames[0][1]):
                        print(
                            f"Enriching cache with Kalman velocities ({tracker} replay)..."
                        )
                        frames = enrich_cached_kalman_velocity(
                            sequence, frames, tracker=tracker
                        )
                        save_cached_detections(path, frames, meta=meta)
                    print(f"Loaded cached detections: {path.name} ({len(frames)} frames)")
                    for frame_idx, dets in frames:
                        if start <= frame_idx <= last:
                            yield frame_idx, dets
                    return

        print(f"Running detector (will cache to {path.name})...")
        frames = list(
            source(sequence, start=start, end=last, **kwargs, **cache_params)
        )
        save_cached_detections(
            path,
            frames,
            meta={
                "sequence": sequence.name,
                "start": start,
                "end": last,
                "source": source_name,
                **key_params,
            },
        )
        print(f"Wrote detection cache: {path} ({len(frames)} frames)")
        yield from frames

    return _cached
