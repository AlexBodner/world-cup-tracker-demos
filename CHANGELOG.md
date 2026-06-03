# Changelog

Historical export of the internal `world_cup_projects` package (Roboflow monorepo).

## [0.1.0] — 2025-06 snapshot

### Demos

- Pass alternatives with metric homography, teammate lane blocking, pitch keypoint debug overlay.
- Player speed & distance with multi-lag mean speed (m/s), radar minimap, homography orientation lock.

### Speed / homography

- Multi-scale speed: `mean(v_{j,j-1} … v_{j,j-K})` with per-step `H_{i-1}`, `H_i` warps.
- Distance from per-step pitch steps; outlier rejection on distance only (not label zeroing).
- Pitch keypoint skeleton fix (1-based edge indices); radar EMA jump reset on H flips.

### Packaging

- Single tree: `world_cup_projects/` only (removed duplicate `world-cup-tracker-demos/` export).
- `SOCCERNET_TRACKING_ROOT` env var; auto-detects monorepo tracking path when present.
- `pyproject.toml` + `[full]` extras for RF-DETR, Ultralytics pitch model, torch.

### Not included in git

- Rendered MP4s (regenerate locally).
- SoccerNet dataset (license / size).
- Internal Roboflow monorepo paths.
