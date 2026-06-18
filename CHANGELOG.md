# Changelog

Historical export of the internal `world_cup_projects` package (Roboflow monorepo).

## [0.2.0] — 2026-06

### Model ids
- Centralized pinned Universe ids in `common/model_ids.py` (players v11, ball v4, pitch f07vi/15).
- CLI help and README aligned with defaults; removed stale `rejhg/1` and `/20` default references.

### Pass Detection Robustness (ball dynamics)
- **Fly-by vs touch via ball motion**: a candidate touch must now change the ball's motion to
  count. Added transit fly-by (speed), release-inbound fly-by (far-but-slow gravity drop),
  and redirect-vs-gravity-arc (inbound/outbound angle + speed ratio) gates in
  `common/possession_touch.py`. Motivated by long aerial passes (e.g. `#31 → #1`) being
  scored as chains of ground passes (`#31 → #5 → #1`) through players merely standing under
  the ball's arc.
- **Redirect override**: a real redirect can rescue a touch the transit gate would veto
  (`redirect_overrides_transit_flyby`), with a stricter angle+ratio test during long
  in-flight releases, a `speed_ratio > 4` noise cap, and an aerial veto. Fixes a fly-by being
  credited as a one-touch return when the ball merely passed by a player.
- **Ball teleport filter**: single-frame ball-detection jumps `> 180 px/frame` are dropped
  before fitting redirect metrics (`_ball_path_samples`, `max_ball_teleport_px_per_frame`),
  removing false redirects from outlier ball boxes.
- **Intermediate hops**: `_try_emit_intermediate_hop` credits a genuine two-link relay (and
  re-anchors the release) only when a different teammate gains real control mid-flight,
  instead of splitting one long pass into bogus short ones.

### Turnover attribution (possession epochs)
- **Possession epochs** (`_should_credit_team_possession`): opponent reception-only touches
  during your in-flight release no longer update `last_possession` / `turnover_snapshot` —
  only redirects or real control count. Stops fly-bys (e.g. `#13` under `#23`'s long ball on
  `08fd33_8`) from anchoring false losing passers.
- **Same-team arrival guard** (`_same_team_pass_arrival_in_progress`): while a teammate
  arrival streak is building, skip deferred turnover queue and snapshot refresh on the
  receiver's control — one physical reception is either a pass or a turnover, not both.
  Fixes `#23 → #12` also appearing as `#13 → #12`.

### Pass Network & Visuals
- Implemented `pass_network_run.py` to analyze passing events and build collaboration links.
- Added dynamic pass highlights: glowing, perspective-aware ellipses at players' feet and an arrow that interpolates to follow the ball's flight path.
- Added a "twinkle" pulse effect upon successful pass reception.
- Removed the per-pass `#passer → #receiver` text chip (it appeared only for long passes and
  read as a flicker); pass arrows + feet highlights now stand alone. Turnover chips remain.
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
