# Changelog

Historical export of the internal `world_cup_projects` package (Roboflow monorepo).

## [0.2.0] — 2026-06

### Pass Network & Visuals
- Implemented `pass_network_run.py` to analyze passing events and build collaboration links.
- Added dynamic pass highlights: glowing, perspective-aware ellipses at players' feet and an arrow that interpolates to follow the ball's flight path.
- Added a "twinkle" pulse effect upon successful pass reception.
- Updated the stats end-card to use clear, self-explanatory labels ("Passes", "Avg Quality", "Made", "Received") instead of technical abbreviations.

### Pitch & Team Logic
- **Defensive Block Goal Inference**: Overhauled `infer_goal_defenders` to use the 3 most defensive players per team and a margin-of-advantage check, fixing issues where offensive presses caused inverted goal assignments.
- **Goalkeeper Stabilization**: Introduced `stabilize_goalkeeper_teams` to use tracklet history to reliably map Goalkeepers to their defending goal and prevent team/role flickering.

## [0.1.0] — 2025-06 snapshot

### Demos

- Pass alternatives with metric homography, teammate lane blocking, pitch keypoint debug overlay.
- Player speed & distance with multi-lag mean speed (m/s), radar minimap, homography orientation lock.

### Speed / homography

- Multi-scale speed: `median(v_{j,j-1} … v_{j,j-K})` with per-step `H_{i-1}`, `H_i` warps.
- No speed caps; distance sums all homography steps.
- Pitch keypoint skeleton fix (1-based edge indices).
- Radar: sports-style ``H(feet)`` per frame (removed mirror lock + temporal smoother).

### Packaging

- Single tree: `world_cup_projects/` only (removed duplicate `world-cup-tracker-demos/` export).
- `SOCCERNET_TRACKING_ROOT` env var; auto-detects monorepo tracking path when present.
- `pyproject.toml` + `[full]` extras for RF-DETR, Ultralytics pitch model, torch.

### Not included in git

- Rendered MP4s (regenerate locally).
- SoccerNet dataset (license / size).
- Internal Roboflow monorepo paths.
