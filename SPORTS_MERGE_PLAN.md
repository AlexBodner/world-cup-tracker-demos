# Merging World Cup Tracker Demos into `roboflow/sports`

Design doc / implementation plan. **Plan-first: nothing is implemented in `roboflow/sports` until this is reviewed.**

**Last audited:** June 2026  
**Repo snapshot:** commits `7321966` (HEAD) / `0e47a48` — package version **0.2.0**

---

## Repo snapshot (current `world_cup_projects/` layout)

Audited against the tree at `7321966`. Counts are Python modules unless noted.

```
world_cup_projects/
  __init__.py                 # __version__ = "0.2.0"
  pyproject.toml              # setuptools packages: common, pass_alternatives, player_stats only
  README.md, CHANGELOG.md, SPORTS_MERGE_PLAN.md
  world_cup_demos.ipynb       # walkthrough (not ported)

  common/                     # 19 modules — shared infra
    cli.py                    # FootballDetectionDefaults, shared argparse flags
    pipeline.py               # load_detections_source, load_metric_context, prepare_model_frames
    model_ids.py              # pinned Universe ids (players v11, ball v4, pitch f07vi/15)
    detect.py, detection_cache.py, player_tracker.py, device.py
    pitch.py, geometry.py, possession.py, possession_config.py, possession_touch.py
    teams.py, tracking_facing.py, visual.py, carrier_motion.py
    clips.py, video.py, soccernet.py

  player_stats/               # 13 files — production demos + pass core
    pass_events.py            # pass/turnover state machine (~3500 lines, v0.2 ball-dynamics)
    carrier_tracking.py
    pass_network.py, pass_network_run.py, pass_network_render.py
    pass_alternatives integration via pass_network_run --show-predictions
    run.py, render.py, speed_distance.py          # speed & distance demo
    kalman_motion_run.py, kalman_motion_render.py # kalman joystick demo
    tracking_run.py, tracking_render.py         # tracking-only overlay (see §4)

  pass_alternatives/          # 5 files — lane scoring + freeze render
    pass_options.py, run.py, render.py, lane_visual.py

  explain/                    # 10 modules (+ __init__) — talk/social video generators
    pass_explain_run.py, pass_explain_visual.py
    pass_turnover_explain_run.py, pass_turnover_explain_visual.py
    pass_lane_detect_run.py, pass_lane_detect_visual.py
    pass_alternatives_run.py, pass_alternatives_visual.py
    pass_alternatives_conference.py, pass_alternatives_conference_metrics.py

  dev/                        # 13 scripts (+ __init__) — debug/analysis (not ported)
    freeze_debug.py, freeze_debug_run.py
    ball_speed_debug_run.py, ball_speed_debug_render.py
    compare_ball_detection.py, benchmark_ball_detection.py
    compare_player_detectors.py, compare_thresholds.py, analyze_velocities.py
    debug_pass_window.py, debug_state_trace.py
    find_cross_team_pass_overlaps.py, _pitch_hosted_shim.py

  ground_truth/               # dev validation only
    passes/*.json             # 08fd33_0, 08fd33_4, 08fd33_8, 0bfacc_1
    compare_passes.py, scan_gt_with_ball_ensemble.py

  tests/                      # 3 test modules + conftest
    test_08fd33_8_passes.py, test_0bfacc_1_passes.py, test_one_touch_reception.py

  scripts/
    render_explain_demo_passes.sh

  assets/                     # outputs (gitignored)
  bundesliga_videos/          # test MP4s (not in git)
```

**Recent structural changes (0.2.0):**
- Explain scripts moved from `player_stats/` and `pass_alternatives/` → `explain/` (`a3ba9d1`); backward-compat shims removed (`b00c152`).
- Debug scripts moved from `player_stats/` → `dev/` (`0e47a48`); `player_stats/` is production-only.
- Shared `common/pipeline.py` + `common/cli.py` dedupe detection/homography setup across demos (`8405fd8`, `d4be498`).
- Model ids centralized in `common/model_ids.py` (`8405fd8`).
- Pass alternatives default overlay switched from facing arrows → Kalman **joystick dots** (`7007a85`); `--facing-mode motion|kalman|both` kept as deprecated.

**Behavioral changes since original plan (still in scope to port):**
- **Ball dynamics** (`common/possession_touch.py`): transit fly-by, release-inbound fly-by, redirect-vs-gravity-arc gates, redirect override with aerial veto, ball teleport filter (`>180 px/frame`), intermediate-hop relay logic.
- **Turnover attribution** (`pass_events.py`): possession epochs, same-team arrival guard, fly-by no longer anchors false losing passers.
- **Pass network visuals**: dynamic pass highlights, twinkle on reception; long-pass label chip removed.

---

## 0. Locked decisions

| Decision | Choice |
|---|---|
| Detection backend | **Roboflow Inference** (`inference.get_model`) — promote Roboflow libraries. *Re-validate:* this repo still runs **local YOLO `.pt` (v11)** by default for player/ball; README notes RF-DETR Inference had worse pass-network results on `08fd33_0`. Confirm Inference-only is acceptable for the sports example. |
| Tracking | **Roboflow `trackers`** (ByteTrack/BoTSORT). Also **migrate the example's existing modes** (`PLAYER_TRACKING`, `TEAM_CLASSIFICATION`, `RADAR`) off `sv.ByteTrack` onto `trackers` so the whole example is consistent. Bonus: Kalman state comes free → enables carrier-motion penalty + speed + kalman demos |
| Where logic lives | **Local modules under `examples/soccer/`** (not the installable `sports/` package) |
| Pass network | **Included** (collaboration web is the hero demo) |
| Carrier-motion penalty | **Kept** — primary direction from `carrier_kalman_direction` (`tracking_facing.py`, Kalman velocity on detections). `PassWeights.use_carrier_motion` defaults `True`. |
| Pass-network render | **Single inference pass + in-memory per-frame detection cache**, reused by the render pass (no double inference, no disk cache). *Re-validate:* this repo still uses **disk** caches (`.cache/detections/`, `.cache/pitch/`) for dev iteration; sports port drops disk cache. |
| Homography | **Port our gated `common/pitch.py`** (`PitchHomographyTracker`, RANSAC, reprojection checks, orientation lock, goal-defender warmup) — not the sports example's naive per-frame `ViewTransformer`. Thresholds were tuned against this; keep it. |
| Scope | Three pass modes **+ port the user-facing demos**: **speed & distance** and **kalman motion**. Leave pure dev/debug tooling behind (see §4). **Tracking-only overlay** (`tracking_run.py`) left behind — sports example already has `PLAYER_TRACKING`. |
| Status | **Pre-port snapshot at v0.2.0** — design doc audited against current repo; implement in `roboflow/sports` only after sign-off |

---

## 1. Goal

Contribute to `examples/soccer/` in `roboflow/sports`:

Pass analytics (new):
1. A **pass detection** module (carrier handoff state machine + turnovers).
2. A **pass alternatives** module (lane scoring — "what else could he have played?").
3. A **pass network** module (collaboration web).

Ported demos (existing world-cup demos, now as sports modes):
4. **Speed & distance** (km/h + meters run per player, radar minimap).
5. **Kalman motion** (velocity joystick dots + optional speed badges; facing arrows deprecated).

Wire all of them into `examples/soccer/main.py` as new `--mode` values, and put the heavier
**explanation video generators** in a separate `examples/soccer/explain/` folder.

Everything runs on top of what the `sports` example provides, using Roboflow **Inference** for
the models and Roboflow **trackers** for tracking.

---

## 2. The core insight (why this is tractable)

The pass logic is almost fully decoupled from our heavy infra. Verified facts (re-checked at `7321966`):

- `image_to_pitch_m(pts, T)` ≡ `T.transform_points(pts) / 100` — the math is the same as
  `sports.common.view.ViewTransformer`, but **we port the full gated homography stack**
  (`PitchHomographyTracker`, confidence filtering, RANSAC, reprojection checks, orientation lock)
  from `common/pitch.py`, not the sports example's naive "build H from raw keypoints every frame."
  Pass/possession thresholds were tuned against the gated version.
- `pass_alternatives/pass_options.py` only needs `common.geometry`, `common/possession`,
  `common/possession_config` — pure `numpy` / `supervision`.
- `player_stats/pass_events.py` uses `carrier_kalman_direction` from `tracking_facing.py`
  (Kalman velocity on `sv.Detections`) for the `use_carrier_motion` backward-run penalty when
  scoring pass quality. Because we standardize on `trackers`, Kalman state is present.
- **Ball-dynamics gates** live in `common/possession_touch.py` (not in the original plan) —
  port this module intact; pass/turnover logic depends on it.
- **`common/pipeline.py`** centralizes detection loading + metric homography warmup — fold
  its responsibilities into the sports adapter rather than porting the module verbatim.

**Net:** the pass core consumes plain `sv.Detections` (numpy + supervision), fed by Inference
detections + `trackers` + `TeamClassifier` + **gated per-frame homography from `pitch.py`**.
The kept demos (speed, kalman) additionally use `tracking_facing` Kalman helpers and
`common/visual.py` joystick drawing.

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
    teams.py                    # tracker-id team stabilizer + GK-by-centroid + goal-defender warmup hooks
    pitch.py                    # gated homography from common/pitch.py (PitchHomographyTracker, etc.)
    tracking_facing.py          # Kalman velocity / facing / carrier_kalman_direction helpers
    possession.py               # feet/ball geometry + find_ball_carrier + touch validation
                                #   (merge possession.py + possession_config.py + possession_touch.py + geometry.py)
    pass_options.py             # PassWeights + score_pass_options + top_pass_options
    passes.py                   # PassDetectionConfig + scan_possession_events (+ turnovers, ball dynamics)
    pass_network.py             # collaboration links + player summaries
    speed_distance.py           # per-player kinematics (km/h, meters)         [demo #4]
    carrier_motion.py           # BallPositionHistory only (freeze-moment ball speed); optional TrackPositionHistory if keeping deprecated arrows
    annotations.py              # annotators (extends sports.annotators.soccer): ellipses, glow,
                                #   pass arrows, lanes, radar, speed/kalman badges, joystick dots
  explain/                      # NEW: presentation/social video generators (slimmed)
    __init__.py
    pass_detection_explain.py   # merge pass_explain + pass_turnover + pass_lane_detect run/visual pairs
    pass_alternatives_explain.py
```

`soccer_analytics/` imports from the installed `sports` package for shared primitives:
`sports.configs.soccer.SoccerPitchConfiguration`, `sports.common.view.ViewTransformer`,
`sports.common.team.TeamClassifier`, `sports.annotators.soccer.*`.

---

## 4. Source → target mapping

| Target module | Ported from | Action |
|---|---|---|
| `soccer_analytics/possession.py` | `common/possession.py` + `common/possession_config.py` + `common/possession_touch.py` + `common/geometry.py` | Merge into one lean module; keep `feet_xy`, `ball_xy`, `player_mask`, `Carrier`, `find_ball_carrier`, touch validation (incl. v0.2 ball-dynamics gates), lane geometry helpers. Drop SoccerNet role constants → use sports class IDs. |
| `soccer_analytics/pass_options.py` | `pass_alternatives/pass_options.py` | Near 1:1. Keep `score_pass_options`, `top_pass_options`, `PassWeights`, `attack_direction`; use `pitch.image_to_pitch_m` / `pitch_attack_direction`. |
| `soccer_analytics/passes.py` | `player_stats/pass_events.py` + `player_stats/carrier_tracking.py` | Port state machine. Import from `pitch.py` + `tracking_facing.py`. Keep `PassDetectionConfig`, `InferredPass`, `InferredTurnover`, `scan_possession_events`, `PassQualityScorer`, turnover epoch logic. |
| `soccer_analytics/pass_network.py` | `player_stats/pass_network.py` | 1:1 (only depends on pass events). |
| `soccer_analytics/pitch.py` | `common/pitch.py` | **Port substantially intact.** Keep `PitchHomographyTracker`, Inference pitch model, RANSAC/reprojection gating, orientation lock, goal-defender warmup, `image_to_pitch_m`, attack direction, radar helpers. Trim: SoccerNet-only paths, break `visual.py` import cycle (move debug draw helpers to `annotations.py` or inline). Drop disk pickle cache → in-memory per clip. |
| `soccer_analytics/tracking.py` | `common/player_tracker.py` | `trackers` ByteTrack/BoTSORT factory (also used to migrate the example's existing modes). |
| `soccer_analytics/tracking_facing.py` | `common/tracking_facing.py` | Keep `carrier_kalman_direction`, `kalman_velocity_arrays`, `JoystickDotSmoother`, speed helpers. Drop SoccerNet-only paths. |
| `soccer_analytics/teams.py` | slice of `common/teams.py` | `TrackletTeamStabilizer` (majority vote per `tracker_id`, flip after streak ≥ 8) + GK-by-centroid + `stabilize_goalkeeper_teams`. ~40–80 lines. |
| `soccer_analytics/speed_distance.py` | `player_stats/speed_distance.py` | Per-player kinematics (km/h, meters). Swap SoccerNet source for the Inference detection stream. |
| `soccer_analytics/carrier_motion.py` | slice of `common/carrier_motion.py` | **Port `BallPositionHistory`** (used by pass-alternatives freeze selection for ball-speed gates). `TrackPositionHistory` only needed if keeping deprecated `--facing-mode motion\|kalman\|both` arrows — default is joystick; can drop arrows in sports port. |
| `soccer_analytics/detection.py` | slice of `common/detect.py` + `common/model_ids.py` | Inference wrapper: player+ball+pitch model IDs → per-frame `sv.Detections` + `sv.KeyPoints`. Referee suppression optional. Pin defaults from `model_ids.py`. |
| `soccer_analytics/annotations.py` | slice of `common/visual.py` + `pass_alternatives/lane_visual.py` | Re-implement the pieces the kept modes need: ellipses, carrier glow, pass arrow, pulsing receiver, lane corridors, radar overlay, **speed badge/legend + kalman joystick dots** (primary). Deprecated facing arrows optional. Trim, don't dump the whole ~1200-line `visual.py`. |
| `examples/soccer/main.py` `run_speed_distance` | `player_stats/run.py` + `player_stats/render.py` | Speed & distance demo as a mode. |
| `examples/soccer/main.py` `run_kalman_motion` | `player_stats/kalman_motion_run.py` + `kalman_motion_render.py` | Kalman motion demo as a mode. |
| `examples/soccer/main.py` pass modes | `player_stats/pass_network_run.py` + `pass_network_render.py`, `pass_alternatives/run.py` + `render.py` | Three pass modes + shared pipeline logic (from `common/pipeline.py`). |
| `explain/pass_detection_explain.py` | `explain/pass_explain_*`, `pass_turnover_explain_*`, `pass_lane_detect_*` | Slim filmstrip/timeline/video generators reusing `annotations.py`. |
| `explain/pass_alternatives_explain.py` | `explain/pass_alternatives_*` (+ optional conference variants) | Slim 4-step explain reusing `annotations.py`. |

### Production (port to sports)

| Area | Modules |
|---|---|
| Pass core | `player_stats/pass_events.py`, `carrier_tracking.py`, `pass_network.py` |
| Pass demos | `pass_network_run/render`, `pass_alternatives/run`, `render`, `lane_visual`, `pass_options` |
| User demos | `run.py`, `render.py`, `speed_distance.py`, `kalman_motion_run/render` |
| Shared infra | `common/pitch`, `possession*`, `teams`, `tracking_facing`, `visual` (trimmed), `player_tracker`, `geometry`, `model_ids`, `pipeline` (logic only), `cli` (flags only), `carrier_motion` (`BallPositionHistory`) |

### Left behind in this repo (pure dev/debug — NOT ported)

| Area | Contents |
|---|---|
| `dev/` | All 13 scripts: `freeze_debug*`, `ball_speed_debug_*`, `compare_ball_detection`, `benchmark_ball_detection`, `compare_player_detectors`, `compare_thresholds`, `analyze_velocities`, `debug_pass_window`, `debug_state_trace`, `find_cross_team_pass_overlaps`, `_pitch_hosted_shim` |
| `ground_truth/` | `passes/*.json`, `compare_passes.py`, `scan_gt_with_ball_ensemble.py` — use for **pre-port validation only** |
| `player_stats/tracking_run.py`, `tracking_render.py` | Tracking-only overlay — superseded by sports `PLAYER_TRACKING` mode |
| Notebook | `world_cup_demos.ipynb` |
| SoccerNet GT path | `common/soccernet.py`, `common/video.py` GT loaders, `--source gt` — demos run on broadcast video in sports example |
| Deprecated overlays | `--facing-mode motion\|kalman\|both` arrow paths (optional to omit in sports port) |

### Replaced infra (functionality kept, implementation swapped)

- `common/detect.py` RF-DETR/local-YOLO paths → Inference-only `detection.py` (kept: player/ball/ref handling, team fit, Kalman attachment).
- `common/detection_cache.py` (disk cache) → in-memory per-clip list for network render pass (pitch homography in-memory per clip; no `.cache/` dirs).
- `common/soccernet.py` + GT `--source gt` → Inference stream + `sv.VideoSink`/ffmpeg.
- `common/clips.py` → not needed (clip metadata inlined in run scripts).

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
- Kalman velocity arrays (via `tracking_facing.attach_kalman_velocity` or cached on detections) — required for joystick dots + carrier-motion penalty

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
> Requires `ROBOFLOW_API_KEY` for pitch (and Inference player/ball if not using local weights).

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

**Pass-network rendering:** pass detection is causal — it discovers passes as the clip plays —
but the network end-card needs *every* pass before drawing. To avoid running the models twice,
`run_pass_network` does **one inference pass that stores each frame's detections in an in-memory
list**, computes the full event set, then renders from the cached frames/detections. No disk
cache, no double inference.

---

## 7. Explain folder

**Current repo:** `explain/` has **10 modules** (not the 2-file sketch in the original plan).
Each feature is split into `*_run.py` (CLI + pipeline) and `*_visual.py` (drawing/filmstrip).
Conference variants (`pass_alternatives_conference*.py`) are optional extras.

**Sports target:** consolidate into two slim modules under `examples/soccer/explain/`:
- `pass_detection_explain.py` ← `pass_explain_*`, `pass_turnover_explain_*`, `pass_lane_detect_*`
- `pass_alternatives_explain.py` ← `pass_alternatives_*` (drop or stub conference metrics unless needed)

They reuse `soccer_analytics` for detection + events and `annotations.py` for drawing, then build
filmstrips / timelines / slow-mo MP4s. Output via `cv2` + optional `ffmpeg` (h264 / GIF), no
`common/video.py`.

Lower priority than the three core modes; can land in a follow-up PR (P7).

---

## 8. Dependencies

`examples/soccer/requirements.txt` becomes:
```
inference          # Roboflow models (player/ball/pitch)
trackers           # ByteTrack/BoTSORT + Kalman state
supervision        # detections, annotators, VideoSink
ultralytics        # (kept for the existing YOLO-based modes until fully on Inference)
gdown              # (kept for existing setup.sh sample assets)
```
`transformers`, `umap-learn`, `scikit-learn`, `tqdm` come transitively via the installed
`sports` package (`sports.common.team`). The pass core itself adds **no** new dependency
beyond `inference` + `trackers`. `ffmpeg` only needed for explain-video export.

---

## 9. Risks / resolved questions

1. **Tracking lib.** ✅ Resolved → Roboflow `trackers` (ByteTrack/BoTSORT) everywhere, including
   migrating the example's existing `sv.ByteTrack` modes. Gives Kalman state for free.
2. **Homography quality.** ✅ Resolved → **port our gated `common/pitch.py`**, not the sports
   example's naive per-frame transformer. Keeps RANSAC, reprojection checks, orientation lock, and
   goal-defender warmup — the same ruler the pass thresholds were tuned against.
3. **Carrier-motion penalty.** ✅ Resolved → **kept**. `carrier_kalman_direction` from
   `tracking_facing` (Kalman velocity on detections). `BallPositionHistory` in `carrier_motion.py`
   is separate — used for freeze-moment ball-speed scoring, not the backward-run penalty.
4. **Pass-network render cost.** ✅ Resolved → single inference pass + in-memory per-frame
   detection cache, reused by the render pass. No disk cache, no double inference.
5. **Referees.** Inference returns referees (class 3); we exclude them from teams/possession
   (optional IoU suppression ported from `detect.py` if needed).
6. **Ball dynamics / turnovers (v0.2).** ✅ Implemented in source repo — port `possession_touch.py`
   gates and turnover epoch logic together with `pass_events.py`; validate against
   `ground_truth/passes/08fd33_0.json` and `08fd33_8.json` before sports PR.
7. **Licensing.** Confirm the ported pieces are clean for the sports repo (Apache-2.0 origin
   noted in our README); core is our own logic + supervision.
8. **`trackers` migration of existing modes.** Swapping `sv.ByteTrack` → `trackers` in the
   stock modes changes their tracking behavior slightly; verify the existing demos still look
   correct after the swap. *(open verification item)*
9. **Inference vs local YOLO.** ⚠️ **Re-validate during port.** This repo defaults to local YOLO
   v11 for player/ball; README recommends staying on YOLO for demo quality. Sports plan assumes
   Inference-only — confirm quality on hero clip `08fd33_0` before dropping YOLO path.

---

## 10. Phased checklist

- [ ] **P1 — Core port (no rendering).** `possession.py` (incl. `possession_touch` ball-dynamics),
  `pass_options.py`, `passes.py`, `pass_network.py`, **`pitch.py` (gated homography)**,
  `tracking_facing.py`, `BallPositionHistory` slice. Validate event output matches current repo
  on `08fd33_0` against `ground_truth/passes/08fd33_0.json` and turnover cases on `08fd33_8.json`
  (dev check only — GT stays behind).
- [ ] **P2 — Adapter + tracking.** `detection.py` (Inference) + `tracking.py` (`trackers`) +
  `teams.py` stabilizer producing the §5 contract (incl. Kalman velocity attachment).
- [ ] **P3 — Annotators.** `annotations.py` (carrier glow, pass arrow, receiver pulse, lanes,
  radar, speed badge, **kalman joystick dots** as default overlay).
- [ ] **P4 — Pass modes in main.py.** `PASS_DETECTION`, `PASS_ALTERNATIVES`, `PASS_NETWORK` + CLI.
- [ ] **P5 — Ported demo modes.** `speed_distance.py` → `SPEED_DISTANCE`; kalman render →
  `KALMAN_MOTION`. Migrate existing `PLAYER_TRACKING`/`TEAM_CLASSIFICATION`/`RADAR` to `trackers`.
  *(Do not port `tracking_run.py` — redundant with sports `PLAYER_TRACKING`.)*
- [ ] **P6 — Docs/deps.** `examples/soccer/README.md` modes + `requirements.txt` (`inference`, `trackers`).
- [ ] **P7 — Explain folder** (follow-up PR): slimmed explanation video generators from the
  10-module `explain/` package.

---

## 11. Assumptions to re-validate during port

| Topic | Current repo behavior | Sports port assumption | Action |
|---|---|---|---|
| Detection backend | Local YOLO v11 default; Inference optional | Inference-only | Run pass-network on `08fd33_0` with Inference `/11`; compare to YOLO baseline |
| Disk detection cache | `.cache/detections/` + `.cache/pitch/` pickle maps | In-memory per clip | Accept slower re-runs in sports example; no cache dirs |
| `--source gt` | SoccerNet GT detections for dev/tests | Dropped | Tests use saved JSON fixtures or mock detections in sports repo |
| Facing overlays | Joystick default; arrows deprecated | Joystick only in sports | Drop `TrackPositionHistory` / `--facing-mode` unless explicitly requested |
| RF-DETR player model | Available via `--detector-backend inference` | Not in scope | Stick to YOLO-family Inference model unless quality gap closes |
| Ball ensemble | `--ball-ensemble-mode fallback` in CLI | TBD | Decide whether sports example needs dedicated ball model stacking |
| Class IDs | `0=ball, 1=gk, 2=player, 3=ref` | Same | Verified identical to sports example |
| Package layout | `explain/` + `dev/` not in setuptools `packages` | N/A | Sports example uses flat `examples/soccer/` tree |

---

## 12. Reference: validated dependency facts (snapshot `7321966`)

- `pass_events.py` imports: `common.pitch` (`image_to_pitch_m`, `image_to_pitch_cm`,
  `pitch_attack_direction`), `common.possession`, `common.possession_config`,
  `common.possession_touch`, `common.soccernet` (only `ROLE_GOALKEEPER`),
  `common.tracking_facing` (`carrier_kalman_direction`), `pass_alternatives.pass_options`,
  `player_stats.carrier_tracking`.
- `pass_options.py` imports: `common.geometry`, `common.possession`, `common.possession_config`.
  Pure numpy/sv.
- `pass_alternatives/render.py` imports: `common.carrier_motion` (`BallPositionHistory`,
  `TrackPositionHistory`), `common.tracking_facing` (`carrier_kalman_direction`), `common.visual`
  (joystick drawing).
- `common/pipeline.py` shared by: `pass_network_run`, `kalman_motion_run`, `tracking_run`, explain
  `*_run.py` scripts.
- `image_to_pitch_m` = `ViewTransformer.transform_points(pts) / 100` (cm → m).
- Class IDs identical across both repos: `ball=0, gk=1, player=2, ref=3`.
- Version **0.2.0** (`pyproject.toml`, `__init__.py`, `CHANGELOG.md`).
