# Changelog

Historical export of the internal `world_cup_projects` package (Roboflow monorepo).

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
