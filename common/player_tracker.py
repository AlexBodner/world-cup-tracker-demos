"""Shared player tracker factory (ByteTrack / BoT-SORT + optional CMC)."""

from __future__ import annotations

from typing import Literal

from trackers import BoTSORTTracker, ByteTrackTracker
from trackers.core.base import BaseTracker

TrackerKind = Literal["bytetrack", "botsort", "botsort_nocmc"]


def create_player_tracker(
    frame_rate: float,
    *,
    kind: TrackerKind = "bytetrack",
    track_activation_threshold: float = 0.4,
    high_conf_det_threshold: float = 0.5,
) -> BaseTracker:
    """Build a multi-object tracker for football player boxes."""
    if kind == "bytetrack":
        return ByteTrackTracker(
            frame_rate=frame_rate,
            track_activation_threshold=track_activation_threshold,
        )
    return BoTSORTTracker(
        frame_rate=frame_rate,
        track_activation_threshold=track_activation_threshold,
        high_conf_det_threshold=high_conf_det_threshold,
        enable_cmc=kind == "botsort",
        cmc_method="sparseOptFlow",
    )
