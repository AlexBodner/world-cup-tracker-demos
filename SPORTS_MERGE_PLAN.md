# Merging World Cup Tracker Demos into `roboflow/sports`

Design doc / implementation plan. **Plan-first: nothing is implemented until this is reviewed.**

## 0. Locked decisions

| Decision | Choice |
|---|---|
| Detection backend | **Roboflow Inference** (`inference.get_model`) — promote Roboflow libraries |
| Tracking | **Roboflow `trackers`** (ByteTrack/BoTSORT). Also **migrate the example's existing modes** (`PLAYER_TRACKING`, `TEAM_CLASSIFICATION`, `RADAR`) off `sv.ByteTrack` onto `trackers` so the whole example is consistent. Bonus: Kalman state comes free → enables carrier-motion + speed + kalman demos |
| Where logic lives | **Local modules under `examples/soccer/`** (not the installable `sports/` package) |
| Pass network | **Included** (collaboration web is the hero demo) |
| Carrier-motion penalty | **Kept** (Kalman velocity available via `trackers`; nothing downgraded) |
| Pass-network render | **Single inference pass + in-memory per-frame detection cache**, reused by the render pass (no double inference, no disk cache) |
| Homography | **Port our gated `common/pitch.py`** (`PitchHomographyTracker`, RANSAC, reprojection checks, orientation lock, goal-defender warmup) — not the sports example's naive per-frame `ViewTransformer`. Thresholds were tuned against this; keep it. |
| Scope | Three pass modes **+ port the user-facing demos**: **speed & distance** and **kalman motion**. Leave pure dev/debug tooling behind (benchmarks, ground-truth, freeze-debug, ball-speed-debug, velocity analysis, SoccerNet GT loaders) |
| Status | **Still scoping** — design doc under active revision; implement only after sign-off |

---

## 1. Goal

Contribute to `examples/soccer/` in `roboflow/sports`:

Pass analytics (new):
1. A **pass detection** module (carrier handoff state machine + turnovers).
2. A **pass alternatives** module (lane scoring — "what else could he have played?").
3. A **pass network** module (collaboration web).

Ported demos (existing world-cup demos, now as sports modes):
4. **Speed & distance** (km/h + meters run per player, radar minimap).
5. **Kalman motion** (velocity joystick dots / facing + speed badges).

Wire all of them into `examples/soccer/main.py` as new `--mode` values, and put the heavier
**explanation video generators** in a separate `examples/soccer/explain/` folder.

Everything runs on top of what the `sports` example provides, using Roboflow **Inference** for
the models and Roboflow **trackers** for tracking.

---

## 2. The core insight (why this is tractable)

The pass logic is almost fully decoupled from our heavy infra. Verified facts:

- `image_to_pitch_m(pts, T)` ≡ `T.transform_points(pts) / 100` — the math is the same as
  `sports.common.view.ViewTransformer`, but **we port the full gated homography stack**
  (`PitchHomographyTracker`, confidence filtering, RANSAC, reprojection checks, orientation lock)
  from `common/pitch.py`, not the sports example's naive "build H from raw keypoints every frame."
  Pass/possession thresholds were tuned against the gated version.
- `pass_alternatives/pass_options.py` only needs `common.geometry`, `common/possession`,
  `common/possession_config` — pure `numpy` / `supervision`.
- `player_stats/pass_events.py`'s only `trackers`-coupled dependency is `carrier_kalman_direction`
  (from `tracking_facing`, needs Kalman state) for the `use_carrier_motion` backward-run penalty.
  Because we standardize on `trackers`, the Kalman state is present, so this is **kept as-is** —
  we port a trimmed `tracking_facing` providing the Kalman velocity/facing helpers.

**Net:** the pass core consumes plain `sv.Detections` (numpy + supervision), fed by Inference
detections + `trackers` + `TeamClassifier` + **gated per-frame homography from `pitch.py`**.
The kept demos (speed, kalman) additionally use the trimmed `tracking_facing` Kalman helpers.

---

## 3. Target file layout (in `roboflow/sports`)

```
examples/soccer/
  main.py                       # + 5 new Mode values (3 pass + speed + kalman); existing modes -> trackers
  requirements.txt              # + inference, trackers
  soccer_analytics/             # NEW local package: the ported logic
    __init__.py
    detection.py                # Inference adapter -> per-frame sv.Detections contract (§5)
    tracking.py                 # trackers (ByteTrack/BoTSORT) factory  (from common/player_tracker.py)
    teams.py                    # tracker-id team stabilizer + GK-by-centroid
    pitch.py                    # gated homography from common/pitch.py (PitchHomographyTracker, etc.)
    tracking_facing.py          # trimmed Kalman velocity / facing helpers (for carrier motion + demos)
    possession.py               # feet/ball geometry + find_ball_carrier + touch validation
    pass_options.py             # PassWeights + score_pass_options + top_pass_options
    passes.py                   # PassDetectionConfig + scan_possession_events (+ turnovers)
    pass_network.py             # collaboration links + player summaries
    speed_distance.py           # per-player kinematics (km/h, meters)         [demo #4]
    annotations.py              # annotators (extends sports.annotators.soccer): ellipses, glow,
                                #   pass arrows, lanes, radar, speed/kalman badges, joystick dots
  explain/                      # NEW: presentation/social video generators (slimmed)
    __init__.py
    pass_detection_explain.py
    pass_alternatives_explain.py
```

`soccer_analytics/` imports from the installed `sports` package for shared primitives:
`sports.configs.soccer.SoccerPitchConfiguration`, `sports.common.view.ViewTransformer`,
`sports.common.team.TeamClassifier`, `sports.annotators.soccer.*`.

---

## 4. Source → target mapping

| Target module | Ported from | Action |
|---|---|---|
| `soccer_analytics/possession.py` | `common/possession.py` + `common/possession_config.py` + `common/possession_touch.py` + `common/geometry.py` | Merge into one lean module; keep `feet_xy`, `ball_xy`, `player_mask`, `Carrier`, `find_ball_carrier`, touch validation, lane geometry helpers. Drop SoccerNet role constants → use sports class IDs. |
| `soccer_analytics/pass_options.py` | `pass_alternatives/pass_options.py` | Near 1:1. Keep `score_pass_options`, `top_pass_options`, `PassWeights`, `attack_direction`; use `pitch.image_to_pitch_m` / `pitch_attack_direction`. |
| `soccer_analytics/passes.py` | `player_stats/pass_events.py` + `player_stats/carrier_tracking.py` | Port state machine. Import from `pitch.py` + `tracking_facing.py`. Keep `PassDetectionConfig`, `InferredPass`, `InferredTurnover`, `scan_possession_events`, `PassQualityScorer`. |
| `soccer_analytics/pass_network.py` | `player_stats/pass_network.py` | 1:1 (only depends on pass events). |
| `soccer_analytics/pitch.py` | `common/pitch.py` | **Port substantially intact.** Keep `PitchHomographyTracker`, Inference pitch model, RANSAC/reprojection gating, orientation lock, goal-defender warmup, `image_to_pitch_m`, attack direction, radar helpers. Trim: SoccerNet-only paths, break `visual.py` import cycle (move debug draw helpers to `annotations.py` or inline). Optional: drop disk pickle cache (use in-memory per clip instead). |
| `soccer_analytics/tracking.py` | `common/player_tracker.py` | `trackers` ByteTrack/BoTSORT factory (also used to migrate the example's existing modes). |
| `soccer_analytics/tracking_facing.py` | trimmed `common/tracking_facing.py` | Keep Kalman velocity / facing / carrier-direction helpers needed by carrier-motion penalty + speed/kalman demos. Drop SoccerNet-only paths. |
| `soccer_analytics/teams.py` | slice of `common/teams.py` | `TrackletTeamStabilizer` (majority vote per `tracker_id`, flip after streak ≥ 8) + GK-by-centroid. ~40 lines. |
| `soccer_analytics/speed_distance.py` | `player_stats/speed_distance.py` | Per-player kinematics (km/h, meters). Swap SoccerNet source for the Inference detection stream. |
| `soccer_analytics/detection.py` | slice of `common/detect.py` | Inference wrapper: player+ball+pitch model IDs → per-frame `sv.Detections` + `sv.KeyPoints`. Referee suppression optional. |
| `soccer_analytics/annotations.py` | slice of `common/visual.py` | Re-implement the pieces the kept modes need: ellipses, carrier glow, pass arrow, pulsing receiver, lane corridors, radar overlay, **speed badge/legend + kalman joystick dots + facing arrows**. Trim, don't dump the whole 1226-line file. |
| `examples/soccer/main.py` `run_speed_distance` | `player_stats/run.py` + `player_stats/render.py` | Speed & distance demo as a mode. |
| `examples/soccer/main.py` `run_kalman_motion` | `player_stats/kalman_motion_run.py` + `kalman_motion_render.py` | Kalman motion demo as a mode. |
| `explain/pass_detection_explain.py` | `player_stats/pass_explain_*`, `pass_turnover_explain_*`, `pass_lane_detect_*` | Slim filmstrip/timeline/video generators reusing `annotations.py`. |
| `explain/pass_alternatives_explain.py` | `pass_alternatives/explain_*` | Slim 4-step explain reusing `annotations.py`. |

### Left behind in this repo (pure dev/debug tooling — NOT ported)
`freeze_debug*`, `compare_ball_detection`, `benchmark_ball_detection`, `compare_player_detectors`,
`pass_alternatives/compare_thresholds`, `analyze_velocities`, `ball_speed_debug_*`, `ground_truth/`
(+ `compare_passes`), `debug_*.py`, the notebook.

### Replaced infra (functionality kept, implementation swapped)
- `common/detect.py` RF-DETR/local-YOLO paths → Inference-only `detection.py` (kept: player/ball/ref handling, team fit).
- `common/detection_cache.py` (disk cache) → in-memory per-clip list for network render pass (pitch homography may optionally keep in-memory per-clip maps; disk pickle cache not required for sports example).
- `common/soccernet.py` + `common/video.py` GT loaders → Inference stream + `sv.VideoSink`/ffmpeg. SoccerNet GT path dropped (demos run on broadcast video).
- `common/clips.py`, `common/carrier_motion.py` → not needed for the ported modes.

**Not replaced:** `common/pitch.py` homography logic — ported as `soccer_analytics/pitch.py`.

---

## 5. The per-frame data contract (the adapter — most important piece)

The core consumes a stream of `(frame_idx, sv.Detections, transformer)`, where `Detections`
carries the fields below. `detection.py` + `tracking.py` + `teams.py` + `pitch.py` produce it from Inference.

Per `sv.Detections` (one entry per detected object in the frame):
- `xyxy` — boxes
- `class_id` — `0=ball, 1=gk, 2=player, 3=referee` (Inference `football-players-detection-3zvbc`; matches sports)
- `tracker_id` — from `trackers` ByteTrack/BoTSORT (players/gk only; ball untracked)
- `data["team"]` — `0` / `1` stabilized per `tracker_id` (refs excluded); GK resolved by centroid

`transformer: ViewTransformer | None` per frame:
- from **`PitchHomographyTracker`** (`pitch.py`): confidence-filtered keypoints, RANSAC fit,
  reprojection check, orientation lock — same gated H used for speed, radar, and pass metric gates
- `None` when the tracker rejects the frame's fit → core falls back to **pixel gates** (OR-gate)

Helper contract the core expects (provided by `possession.py` + `pitch.py`):
- `feet_xy(dets)` = bbox BOTTOM_CENTER, `ball_xy(dets)` = ball box bottom-center
- `image_to_pitch_m(pts, transformer)` = gated transformer's `transform_points(pts)/100` or `None`

> Model IDs (defaults in `common/model_ids.py`, overridable via CLI):
> - players+ball: `football-players-detection-3zvbc/11`
> - dedicated ball (optional, higher recall): `football-ball-detection-rejhg/4`
> - pitch keypoints: `football-field-detection-f07vi/15`
> Requires `ROBOFLOW_API_KEY`.

---

## 6. `main.py` wiring (sketch)

Add five values to the existing `Mode` enum, each with a `run_*` generator mirroring the
existing `run_radar` shape (fit `TeamClassifier` on sampled crops, then stream frames):

```python
class Mode(Enum):
    ...                              # existing modes, retargeted to trackers
    PASS_DETECTION = 'PASS_DETECTION'
    PASS_ALTERNATIVES = 'PASS_ALTERNATIVES'
    PASS_NETWORK = 'PASS_NETWORK'
    SPEED_DISTANCE = 'SPEED_DISTANCE'
    KALMAN_MOTION = 'KALMAN_MOTION'

def run_pass_detection(source_video_path, device):
    # 1. detection.py: Inference player/ball/pitch models
    # 2. fit TeamClassifier on sampled player crops (as run_radar does)
    # 3. tracking.py: trackers ByteTrack/BoTSORT
    # 4. per frame: detections + tracker_id + team + transformer  (the §5 contract)
    # 5. scan_possession_events(...) -> passes/turnovers
    # 6. annotations.py: carrier glow + pass arrows + radar
    yield annotated_frame
```

`main()` gets five new `elif` branches, and the existing `PLAYER_TRACKING` /
`TEAM_CLASSIFICATION` / `RADAR` branches switch from `sv.ByteTrack` to `trackers`. CLI gains
optional `--api-key` (falls back to `ROBOFLOW_API_KEY`); existing args unchanged.

**Pass-network rendering (the §9.3 "two-pass" item):** pass detection is causal — it discovers
passes as the clip plays — but the network end-card needs *every* pass before drawing. To avoid
running the models twice over the clip, `run_pass_network` does **one inference pass that stores
each frame's detections in an in-memory list**, computes the full event set, then renders from
the cached frames/detections. No disk cache, no double inference.

---

## 7. Explain folder

`examples/soccer/explain/` holds the talk/social generators. They reuse `soccer_analytics`
for detection + events and `annotations.py` for drawing, then build filmstrips / timelines /
slow-mo MP4s. These are slimmed versions of `pass_explain_visual.py` (2101 lines) and
`pass_alternatives/explain_*` (~3500 lines) with the `visual.py`/`pitch.py` dependencies
removed. Output via `cv2` + optional `ffmpeg` (h264 / GIF), no `common/video.py`.

Lower priority than the three core modes; can land in a follow-up PR.

---

## 8. Dependencies

`examples/soccer/requirements.txt` becomes:
```
inference          # Roboflow models (player/ball/pitch)
supervision        # detections, ByteTrack, annotators, VideoSink
ultralytics        # (kept for the existing YOLO-based modes)
gdown              # (kept for existing setup.sh sample assets)
```
`transformers`, `umap-learn`, `scikit-learn`, `tqdm` come transitively via the installed
`sports` package (`sports.common.team`). The pass core itself adds **no** new dependency
beyond `inference`. `ffmpeg` only needed for explain-video export.

---

## 9. Risks / resolved questions

1. **Tracking lib.** ✅ Resolved → Roboflow `trackers` (ByteTrack/BoTSORT) everywhere, including
   migrating the example's existing `sv.ByteTrack` modes. Gives Kalman state for free.
2. **Homography quality.** ✅ Resolved → **port our gated `common/pitch.py`**, not the sports
   example's naive per-frame transformer. Keeps RANSAC, reprojection checks, orientation lock, and
   goal-defender warmup — the same ruler the pass thresholds were tuned against.
3. **Carrier-motion penalty.** ✅ Resolved → **kept**. `trackers` provides Kalman velocity, so
   `carrier_kalman_direction` works unchanged (trimmed `tracking_facing` ported).
4. **Pass-network render cost.** ✅ Resolved → single inference pass + in-memory per-frame
   detection cache, reused by the render pass. No disk cache, no double inference.
5. **Referees.** Inference returns referees (class 3); we exclude them from teams/possession
   (optional IoU suppression ported from `detect.py` if needed).
6. **Licensing.** Confirm the ported pieces are clean for the sports repo (Apache-2.0 origin
   noted in our README); core is our own logic + supervision.
7. **`trackers` migration of existing modes.** Swapping `sv.ByteTrack` → `trackers` in the
   stock modes changes their tracking behavior slightly; verify the existing demos still look
   correct after the swap. *(open verification item)*

---

## 10. Phased checklist

- [ ] **P1 — Core port (no rendering).** `possession.py`, `pass_options.py`, `passes.py`,
  `pass_network.py`, **`pitch.py` (gated homography)**, trimmed `tracking_facing.py`. Validate event
  output matches current repo on `08fd33_0` against `ground_truth/passes/08fd33_0.json` (dev check only).
- [ ] **P2 — Adapter + tracking.** `detection.py` (Inference) + `tracking.py` (`trackers`) +
  `teams.py` stabilizer producing the §5 contract.
- [ ] **P3 — Annotators.** `annotations.py` (carrier glow, pass arrow, receiver pulse, lanes,
  radar, speed badge, kalman joystick/facing).
- [ ] **P4 — Pass modes in main.py.** `PASS_DETECTION`, `PASS_ALTERNATIVES`, `PASS_NETWORK` + CLI.
- [ ] **P5 — Ported demo modes.** `speed_distance.py` → `SPEED_DISTANCE`; kalman render →
  `KALMAN_MOTION`. Migrate existing `PLAYER_TRACKING`/`TEAM_CLASSIFICATION`/`RADAR` to `trackers`.
- [ ] **P6 — Docs/deps.** `examples/soccer/README.md` modes + `requirements.txt` (`inference`, `trackers`).
- [ ] **P7 — Explain folder** (follow-up PR): slimmed explanation video generators.

---

## 11. Reference: validated dependency facts

- `pass_events.py` repo imports: `common.pitch` (only `image_to_pitch_m`), `common.possession`,
  `common.possession_config`, `common.possession_touch`, `common.soccernet` (only `ROLE_GOALKEEPER`),
  `common.tracking_facing` (only `carrier_kalman_direction`, optional), `pass_alternatives.pass_options`,
  `player_stats.carrier_tracking`.
- `pass_options.py` repo imports: `common.geometry`, `common.possession`, `common.possession_config`. Pure numpy/sv.
- `image_to_pitch_m` = `ViewTransformer.transform_points(pts) / 100` (cm → m).
- Class IDs identical across both repos: `ball=0, gk=1, player=2, ref=3`.
