"""Shared utilities for the World Cup tracker demos."""

from world_cup_projects.common.clips import (  # noqa: F401
    ClipScore,
    PITCH_GAMEPLAY_AVOID,
    PITCH_HOMOGRAPHY_DEMO_CLIP,
    PITCH_KEYPOINT_AVOID,
    PITCH_KEYPOINT_AVOID_NOTES,
    pick_homography_demo_clip,
    pitch_keypoints_unreliable,
    PitchKeypointSummary,
    assess_pitch_keypoints,
    rank_clips,
    score_sequence,
)
from world_cup_projects.common.possession import (  # noqa: F401
    Carrier,
    ball_xy,
    feet_xy,
    find_ball_carrier,
    player_mask,
)
from world_cup_projects.common.soccernet import (  # noqa: F401
    ROLE_BALL,
    ROLE_GOALKEEPER,
    ROLE_PLAYER,
    ROLE_REFEREE,
    TEAM_LEFT,
    TEAM_NONE,
    TEAM_RIGHT,
    SoccerNetSequence,
    TrackletMeta,
    find_sequences,
    iter_gt_detections,
    load_sequence,
)
