"""Shared player tracker factory (ByteTrack / BoT-SORT + optional CMC)."""

from __future__ import annotations

from typing import Literal

from trackers import BoTSORTTracker, ByteTrackTracker
from trackers.core.base import BaseTracker

TrackerKind = Literal["bytetrack", "botsort", "botsort_nocmc"]

# Tuned for football: fewer spurious track births, stronger high/low-conf split.
DEFAULT_TRACK_ACTIVATION_THRESHOLD = 0.55
DEFAULT_HIGH_CONF_DET_THRESHOLD = 0.6
DEFAULT_MINIMUM_IOU_THRESHOLD_FIRST_ASSOC = 0.15


def tracker_cache_key_params() -> dict[str, float]:
    """Cache-key fragment so detection caches invalidate when tracker tuning changes."""
    return {
        "track_activation_threshold": DEFAULT_TRACK_ACTIVATION_THRESHOLD,
        "high_conf_det_threshold": DEFAULT_HIGH_CONF_DET_THRESHOLD,
        "minimum_iou_threshold_first_assoc": DEFAULT_MINIMUM_IOU_THRESHOLD_FIRST_ASSOC,
    }


def create_player_tracker(
    frame_rate: float,
    *,
    kind: TrackerKind = "bytetrack",
    track_activation_threshold: float = DEFAULT_TRACK_ACTIVATION_THRESHOLD,
    high_conf_det_threshold: float = DEFAULT_HIGH_CONF_DET_THRESHOLD,
    minimum_iou_threshold_first_assoc: float = DEFAULT_MINIMUM_IOU_THRESHOLD_FIRST_ASSOC,
) -> BaseTracker:
    """Build a multi-object tracker for football player boxes."""
    if kind == "bytetrack":
        return ByteTrackTracker(
            frame_rate=frame_rate,
            track_activation_threshold=track_activation_threshold,
            high_conf_det_threshold=high_conf_det_threshold,
        )
    return BoTSORTTracker(
        frame_rate=frame_rate,
        track_activation_threshold=track_activation_threshold,
        high_conf_det_threshold=high_conf_det_threshold,
        minimum_iou_threshold_first_assoc=minimum_iou_threshold_first_assoc,
        enable_cmc=kind == "botsort",
        cmc_method="sparseOptFlow",
    )
