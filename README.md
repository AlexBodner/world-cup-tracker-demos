# sWorld Cup Tracker Demos

Football analytics demos built on [trackers](https://github.com/roboflow/trackers),
[supervision](https://github.com/roboflow/supervision), and pieces from
[roboflow/sports](https://github.com/roboflow/sports).

We run them on short **Bundesliga broadcast clips** in `bundesliga_videos/` — the same
footage domain our player detector and pitch keypoint models were trained on.

### The three demos

1. **Pass network** — detect passes and turnovers across a clip, score each pass, draw a
  collaboration web and optional “what else could he have played?” freeze frames.
2. **Pass alternatives** — pick strong possession moments and show the top 3 passing lanes.
3. **Speed & distance** — km/h and meters run per player, plus a radar minimap.

All three produce MP4s (and JSON sidecars) under `assets/`.

---

## Run it

From the monorepo root (or `cd world_cup_projects` and drop the `world_cup_projects.` prefix):

```bash
pip install -r world_cup_projects/requirements.txt
pip install -e ".[full]"          # standalone
pip install -e trackers             # monorepo only

export ROBOFLOW_API_KEY=your_key    # pitch keypoints via Inference

# Main demo — pass network
PYTHONPATH=. python -m world_cup_projects.player_stats.pass_network_run \
    --video world_cup_projects/bundesliga_videos/08fd33_0.mp4 \
    --metric --render --show-predictions

# Pass alternatives only
PYTHONPATH=. python -m world_cup_projects.pass_alternatives.run \
    --video world_cup_projects/bundesliga_videos/08fd33_0.mp4 --metric
```

**Test clips**


| File           | Good for                     |
| -------------- | ---------------------------- |
| `08fd33_0.mp4` | Default hero clip            |
| `08fd33_8.mp4` | Missing ball, long air balls |
| `c01561_3.mp4` | Tricky camera / team colors  |


**Useful flags**


| Flag                                                    | What it does                                                            |
| ------------------------------------------------------- | ----------------------------------------------------------------------- |
| `--metric`                                              | Real meters via pitch homography (needed for radar + lane scoring)      |
| `--render`                                              | Write MP4                                                               |
| `--show-predictions`                                    | Insert pass-alternative freezes on pass release frames                  |
| `--refresh-detections-cache`                            | Re-run player/ball detector (ignores `.cache/detections/`)              |
| `--debug-pitch-keypoints`                               | Keypoints on main video (indices); radar shows kp whenever `--metric`   |
| `--device mps` / `cpu`                                  | Torch device for the football player YOLO                               |
| `--detector-backend inference`                          | Run player/ball model via Roboflow Inference (needs `ROBOFLOW_API_KEY`) |
| `--player-model-id football-players-detection-3zvbc/20` | Universe model version (Inference or future export)                     |
| `--detection-threshold 0.5`                             | Detection confidence threshold                                          |


**Caches**

- Player detections: `.cache/detections/`
- Pitch homography per frame: `.cache/pitch/` — delete `08fd33_0_*.pkl` to re-run keypoints
without re-detecting players

Walkthrough notebook: `[world_cup_demos.ipynb](world_cup_demos.ipynb)`.

---

## How it works

Each section starts with the design idea in plain language, then the exact thresholds we
code. Defaults are from `PassDetectionConfig` / `PassWeights.metric()` at 25 fps.

### Two layers on the video

The demos stack two related ideas on top of the same detections:


| Layer                 | What you see                                                 | What it answers                          |
| --------------------- | ------------------------------------------------------------ | ---------------------------------------- |
| **Pass detection**    | Small arrows + pulsing ellipses at players' feet             | *Did a pass happen?* Who passed to whom? |
| **Pass alternatives** | Dimmed freeze + top 3 ranked lanes (green / yellow / orange) | *What else could they have played?*      |


Both start from the same pipeline: detect players and ball → track IDs → assign teams →
(optionally) map feet to pitch meters. After that the logic splits.

**Pass detection** watches carrier handoffs frame by frame: someone had the ball, released
it, a teammate received it. When a pass is in flight we draw a **low-alpha arrow** from the
passer toward the ball/receiver and highlight both players' feet. On reception the arrow
drops and the receiver twinkles.

**Pass alternatives** scores **every visible teammate** as a possible receiver, ranks the
lanes, and (when the freeze gates pass) shows the **top 3**. The same scorer grades real
passes as `quality_score` on the release frame.

---

### Detection and tracking

We need to know *who* each blob is across frames — not just “a player in this box right
now.” Every later step (possession, passes, team colors) assumes the same person keeps the
same `tracker_id` for the whole clip.

**In code:** YOLO **football-players-detection** per frame → ByteTrack/BoTSORT → stable
`tracker_id` per box (`common/detect.py`).

### Team colors

Passes are only credited inside a team, so we paint everyone red or blue from jersey color.
Raw per-frame clustering flickers under broadcast lighting; we also need to know which goal
each team defends for the radar and attack direction.

**In code:**

- Jersey crops → KMeans on SigLIP features → raw team 0/1 per frame.
- **Stabilizer:** `stable[tracker_id]` only flips after `streak >= 8` disagreeing frames.
- **Goal lock:** `mean(pitch_x | team)` lower → defends left goal; GK pinned to that team.

### Pitch homography

To measure meters, draw the radar, and score “forward” passes we map image pixels onto a
standard pitch. Keypoints come from a Bundesliga-trained model; bad frames are rejected
rather than poisoning the whole clip.

**In code:** Inference `football-field-detection-f07vi/15` → homography to
`SoccerPitchConfiguration` vertices. Radar uses a simple per-frame H; pass length and speed
use a gated `PitchHomographyTracker` (`common/pitch.py`).

---

### Who has the ball?

**Intuition:** possession means the ball is at someone’s feet on the ground — not “the ball
happened to be near them in the air” or “closest in a flat 2D projection.” Each frame we
take the **closest player to the ball** and accept them only if they fall inside a radius
gate. In the UI the carrier gets a **soft white glow at their feet** during live play; freeze
frames use a **spotlight** on a dimmed background.

We use a **tight gate for dribbling (control)** and a **looser gate for first touch
(reception)**. Per-frame carrier assignment is immediate — we do **not** require consecutive
frames just to mark someone “on the ball.” Consecutive-frame **streaks** come later, when we
confirm a **passer** or **receiver** (see Pass detection).

**Problems we patched:**

- **Missing ball boxes** — short gaps do not reset pass state (see missing-ball bridge).
- **Aerial fly-bys** — when the ball passes in the air near a player, 2D “closest to ball”
lies. We veto metric distance when `|ball.y − feet.y| > 20px` and stretch vertical offset
in pixel space so fly-bys do not look like control at the feet.

**Limits:** carrier assignment only runs when the detector returns a ball box (and players to
compare against). Broadcast occlusion — the carrier’s body between ball and camera — motion
blur, and long air balls often mean **no ball detection** or a box offset from the true
ball. A missed player box can also leave the real carrier out of the race. When that happens,
possession drops or jumps even if the on-pitch action is obvious.

**In code** (`find_ball_carrier`):

Ball ground point: `ball = (cx, bottom_of_bbox)`.

For each player `i`, feet = bbox bottom-center:

```
dx = feet_x - ball_x
dy = feet_y - ball_y
dy_eff = dy * 2.5   if |dy| > 10px   else dy      # stretch vertical in px space
dist_px[i] = hypot(dx, dy_eff)
```

**Control** (tight dribble) — player `i` is a control carrier if:

```
dist_px[i] ≤ 55        (GK: ≤ 137)
  OR
dist_m[i] ≤ 0.8 m      (GK: ≤ 2.8 m)   AND  |dy| ≤ 20px
```

**Reception** (first touch) — tried only if no control carrier; same logic with looser gates:

```
dist_px[i] ≤ 120
  OR
dist_m[i] ≤ 1.8 m     AND  |dy| ≤ 20px
```

Pick `argmin dist` among players passing the gate. Metric and pixel gates are **OR**’d so a
brief bad homography doesn’t drop possession — but **metric is vetoed when `|dy| > 20px`**
(ball treated as aerial in 2D).

### Valid touch (pass detection gate)

**Intuition:** before we credit a touch for pass logic, we double-check the tracker didn’t
assign the ball to the wrong player for one frame, and we throw out fly-bys where the ball
skims under someone’s feet or floats above them.

**In code** (`common.possession_touch.is_valid_possession_touch`):

A touch by player `tid` counts only if **all** of:

```
nearest_player(ball) == tid
  → argmin_i hypot(feet_i, ball) == tid
```

Plus touch-kind vetoes:


| Kind          | Reject when                                                   |
| ------------- | ------------------------------------------------------------- |
| **control**   | `|ball.y - feet.y| > 20px`                                    |
| **reception** | `ball.y - feet.y > 40px` (fly-by under feet; chest height OK) |


---

### Pass detection

**Intuition:** a pass is a story in three beats — someone *had* the ball, *released* it, and a
teammate *received* it — with no opponent taking control in between. We don’t look for ball
speed or trajectory directly; we watch carrier handoffs frame by frame. Each team keeps its
own memory of who last released the ball so one team’s possession doesn’t block the other’s.

Rough flow:

1. **Find the carrier** each frame (closest player within the control/reception radius).
2. **Confirm the passer** — they had control at the feet for a short streak, then released.
3. **Ball in flight** — no carrier, or brief ball-detection dropouts bridged (≤10 frames).
4. **Find the receiver** — teammate with a valid touch streak after the release.
5. **Emit pass** — same team, plausible gap, min travel, no opponent control in between.

Because broadcast tracking is noisy, we require short *streaks* of evidence (not one lucky
frame) and special cases for goalkeepers and one-touch plays. Pass logic inherits the same
detector limits as carrier assignment — see **Who has the ball?** — plus the missing-ball
bridge below for short in-flight gaps.

**On screen:** from the release frame through reception, a **small arrow** follows the pass
(path from passer feet toward ball/receiver, ~35% opacity). Ellipses pulse at passer and
receiver feet; on reception the arrow is removed and the receiver **twinkles** twice.

#### Passer (who released the ball)

**Intuition:** the passer is whoever had real control before the ball left — usually three
frames glued to the feet. Goalkeepers and quick releases get shorter proof because punts and
one-touch passes don’t sit at the feet for long.

**In code** (`_TeamPossessionState.release`):

Per team, track `control_streak` for the current `tid`.

**Become passer (outfield):**

```
touch_kind == "control"
AND valid_touch
AND control_streak >= 3
  → release = (frame, dets, carrier, tid)
```

**Goalkeeper shortcut:**

```
is_gk(carrier) AND (control OR reception) AND valid_touch
  → release immediately (min_gk_control_frames = 1)
```

**Pre-flight release** (one-touch / punt — ball leaves player range before 3 control frames):

```
ball in-flight
AND last_touch within 10 frames
AND valid_touch at that last_touch
  → promote last_touch to release
```

#### Receiver (who got the ball)

**Intuition:** the receiver must *keep* the ball for a few frames — not just appear nearest
for one frame when the ball deflects off a shin. Quick one-twos need stricter proof (real
control at the receiver); long switches can confirm with a looser reception streak.

**In code:**

After `release` by `passer_tid`, watch teammates `tid ≠ passer_tid`:

```
each frame with valid_touch by tid:
  arrival_streak++   (reset if tid changes)
  if touch_kind == "control": arrival_control_streak++
```

**Arrival ready** when:

```
gap = frame - release_frame

arrival_streak >= 3
  OR  (touch_kind == "reception" AND gap >= 15 AND arrival_streak >= 2)

AND if gap < 15:
      arrival_control_streak >= 3    # quick combo: need real control, not fly-by reception
AND if gap >= 15:
      arrival_control_streak >= 1    # long pass: ball at feet once, not just approaching
```

Reception radius tightened to **100 px / 1.5 m** (was 120 / 1.8) so “nearest player” while
the ball is still in flight is less likely to start the arrival streak early.

Do **not** move `release` to receiver until a pass is emitted or opponent blocks.

#### Emit pass

**Intuition:** once passer and receiver are both credible, we check the pass “makes sense” —
same team, reasonable time gap, ball actually traveled, nobody intercepted, not a duplicate
count of the same link. We also bridge short stretches where the ball detector blinks out
mid-pass.

**In code** (`_try_emit_pass`):

All must be true:

```
receiver.team == passer.team
1 ≤ gap ≤ 75 frames                    # ~3 s at 25 fps
ball_travel ≥ 1.0 m   (or ≥ 25 px)    # warp release→arrival ball positions
NOT duplicate(passer, receiver) within 12 frames
NOT opponent_touch_between(release_frame, arrival_frame)
  → ∃ frame in (release, arrival) where opponent has touch_kind == "control"
```

**Missing ball bridge:** no ball detection for `≤ 10` consecutive frames → keep `release` +
`arrival_streak` (same as in-flight). Beyond 10 → clear bridge.

On success → append pass; score `quality_score` at **release frame** via lane model.

---

### Turnover detection

**Intuition:** a turnover is “you tried to play forward and they got it first.” We snapshot
the moment an opponent first touches after your release, then attribute the steal when that
opponent later completes a pass — not when they eventually shoot or dribble in isolation.

**In code** (`_try_emit_turnover`):

When team A has `release` and an opponent gets `valid_touch`:

```
turnover_snapshot = (release_frame, passer_tid)   # first opponent touch only
```

When that opponent later **emits a pass** to a teammate:

```
intercept_frame = first valid_touch frame by interceptor after snapshot
AND opponent_control_between(A.release_frame, intercept_frame)
  → emit turnover(passer=A, interceptor=B, frame=intercept_frame)
```

If opponent had control between A’s release and a would-be teammate arrival, **skip** A’s
arrival streak (no fake pass credit).

---

### Pass alternatives — scoring each lane

**Intuition:** imagine the carrier could play any teammate. We **score a lane to every
visible teammate** (same team, excluding the carrier), then **drop** lanes outside length
bounds, **sort** by score, and show the **top 3** on freeze frames. The same function grades
real passes: on emit we look up the lane to the **actual receiver** and store its score as
`quality_score` (or `null` if that receiver was out of range).

**Scoring flow** (`score_pass_options`):

```
all teammates in frame
  → score lane C → R for each
  → skip if L < 2 m or L > 45 m
  → sort by score
  → top 3 for display / freeze
  → single-lane lookup for quality_score on real passes
```

For each candidate pass we ask three football-sense questions:

1. **Openness** — is the pass corridor clear of rivals?
2. **Forward** — does the pass advance toward the opponent goal?
3. **Space** — does the receiver have room around them?

We lightly penalize teammates standing in a narrow lane, passes that go backward while the
player is running forward, and absurd lengths.

#### 1. Openness — rivals in the corridor

For carrier **C** → teammate **R**, draw the pass segment and a **corridor band** around it
(the shaded overlay on frame/radar). **Longer passes use a wider band** (more time/angle for
a defender to step in); short passes use the tightest width.

**Rival corridor width** (full width = 2× half-width):

```
L ≤ 18 m  → 2.5 m
L ≤ 28 m  → 3.25 m
L > 28 m  → 4.0 m  (cap 4.5 m)
```

A rival only counts if they are **inside** the band (on the segment between C and R, within
`width/2` perpendicular to the line). We use `min(feet, body center)` per player.

**Openness** = minimum perpendicular distance from the pass line to the **nearest rival
inside the corridor** (rivals outside the band are ignored). If nobody is in the band →
openness is effectively infinite → full openness term after normalization.

Teammates are checked in a **separate, narrower band** (1.2 m) — they do **not** reduce
openness; they feed a small penalty below.

#### 2. Forward progress

**Forward gain** = `δ · attack_dir` where `δ = R − C` and `attack_dir` points toward the
goal that team is attacking (from homography + goal-defender lock). Only positive forward
progress counts in the score.

#### 3. Receiver space

**Space** = distance from the receiver to the **nearest opponent** anywhere on the pitch
(not limited to the corridor).

#### Final score

**In code** (`score_pass_options`, metric), for `L = |δ|`:

```
n_open    = min(openness / 8 m, 1)
n_forward = min(max(forward_gain / 25 m, 0), 1)
n_space   = min(receiver_space / 8 m, 1)

score = 0.45 × n_open
      + 0.30 × n_forward
      + 0.25 × n_space
      − teammate_penalty              # up to 0.10 if teammate in 1.2 m corridor
      − backward_run_penalty          # up to 0.22 if pass·run < −0.15
      − backward_attack_penalty       # up to 0.18 if forward < 0 AND run·attack ≥ 0.15
```

`quality_score = null` if the actual receiver was not in the scored lane list (e.g. length
out of range).

---

### When we freeze the video

We have **two freeze triggers** — same lane scorer, different moment pickers:


| Demo                                               | When we freeze                               | Question                                                |
| -------------------------------------------------- | -------------------------------------------- | ------------------------------------------------------- |
| **Pass alternatives** (`pass_alternatives/run.py`) | Scans the clip for strong possession moments | “This is a good decision point — what are the options?” |
| **Pass network** (`--show-predictions`)            | On a **real inferred pass release**          | “He just played it — what else could he have played?”   |


**Pass alternatives demo — intuition:** don’t freeze every touch. Scan the clip, score lanes
whenever someone has the ball, and pick moments that are locally “good decisions” — high lane
score, calm possession (slow ball, tight feet), spaced apart in time.

**In code** (`plan_events`):

```
pick_score = top_lane_score
           + bonus if ball_speed < 1.2 m/s and feet_dist < 0.4 m
           - penalty if ball fast

keep if pick_score ≥ 0.66
   AND top_lane ≥ 0.45
   AND local peak in ±12 frames
   AND ≥ 90 frames from previous pick
```

**Pass network (`--show-predictions`) — intuition:** here the freeze is tied to a *real*
inferred pass. On the release frame we ask: “what else could he have played?” — only if we
could score that pass and find alternatives.

**In code:**

```
quality_score != null
AND quality_score ≥ --freeze-quality-threshold   (default 0)
AND passer in detections at release frame
AND top_options(k=3) non-empty
```

---

### Why passing-lane overlays are sometimes missing

Passes can be detected and drawn (glow + arrow) **without** showing the top-3 lane freeze.
Overlays are gated separately in each demo:


| Demo                                               | Lane overlay shown when                                                                                                                                                                        |
| -------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Pass alternatives** (`pass_alternatives/run.py`) | `plan_events` keeps the frame: carrier in control, ≥2 scorable teammates, top lane score ≥ thresholds, local score peak, ≥90 frames since last freeze, ball not too fast.                      |
| **Pass network** (`--show-predictions`)            | Real pass emitted at this frame **and** `quality_score` is not null **and** ≥ `--freeze-quality-threshold` **and** inferred passer visible in detections **and** `top_options(k=3)` non-empty. |


Common reasons a pass happens but **no lane viz** appears:

- `**quality_score` is null** — actual receiver was outside the scored lane list (pass length < 2 m or > 45 m in metric mode).
- **Passer not in detections at release** — inferred passer box missing on the release frame (we no longer require tight feet control there; the ball has just left).
- **Pass alternatives only** — frame failed freeze heuristics (low score, not a local peak, ball moving fast, too soon after previous freeze).
- `**--show-predictions` off** — pass network still highlights passes; lane freeze is optional.
- **No metric homography** — corridor shading and radar lanes need `ROBOFLOW_API_KEY` + `--metric`; pass arrows may still render in pixel mode.

---

### Freeze debug (why no lane overlay on a detected pass?)

Per-pass gate report + hold video for `--show-predictions` freezes:

```bash
PYTHONPATH=. python -m world_cup_projects.player_stats.freeze_debug_run \
    --video world_cup_projects/bundesliga_videos/08fd33_0.mp4 \
    --metric --device cpu
```

Writes `assets/freeze_debug/freeze_debug_metric_<clip>.mp4` (holds ~3.5 s on each detected
pass with a checklist) and `.json` with `blockers` / `gates` per pass.

**Compare player detectors** (YOLO v11 vs Inference v11/v20):

```bash
export ROBOFLOW_API_KEY=your_key

PYTHONPATH=. python -m world_cup_projects.player_stats.compare_player_detectors \\
    --video world_cup_projects/bundesliga_videos/08fd33_0.mp4 \\
    --metric --configs yolo:11,inference:11,inference:20
```

Writes `assets/detector_compare/compare_<clip>.json` with ball-detection rate, pass count,
`quality_score` coverage, and would-freeze counts per config.

---

### Explain frames (talk / social)

Presentation-oriented PNGs (large step titles, minimal on-image copy). Step 1 includes a
radar with candidate pass lines (no keypoints). Step 2 shows shaded corridors and red
blocker markers on both main view and radar. Step 3 decomposes score into OPEN / FWD /
SPACE plus TM / RUN / BACK penalties.

```bash
export ROBOFLOW_API_KEY=your_key   # recommended: shaded pitch corridors + radar

PYTHONPATH=. python -m world_cup_projects.pass_alternatives.explain_run \
    --video world_cup_projects/bundesliga_videos/08fd33_0.mp4 \
    --metric --layout talk --auto-frame

# Square 1080×1080 crops for X / Instagram
PYTHONPATH=. python -m world_cup_projects.pass_alternatives.explain_run \
    --video world_cup_projects/bundesliga_videos/08fd33_0.mp4 \
    --metric --layout social --frame 418 \
    --out-dir world_cup_projects/assets/explain_frames
```

Outputs under `assets/explain_frames/` (flat folder, shared with pass-detection explain):


| File                          | Content                                     |
| ----------------------------- | ------------------------------------------- |
| `pass_lane_01_candidates.png` | Carrier + teammates; radar pass lines       |
| `pass_lane_02_corridors.png`  | Corridors + rival ! markers (video + radar) |
| `pass_lane_03_scoring.png`    | Score bars + penalty breakdown              |
| `pass_lane_04_ranking.png`    | Top 3 ranked lanes                          |
| `pass_lane_grid_2x2.png`      | All four on one slide                       |


---

### Pass detection explain frames (talk / social)

**How we illustrate it:** one real pass as **vertical filmstrips** — consecutive frames
stacked top-to-bottom with a left gutter showing frame number + state (`CONTROL 1/3`,
`PASSER LOCKED`, `IN FLIGHT`, …). The scene is heavily dimmed; only the focus player
(team-colored ring + spotlight) stays bright. Same crop across panels so the audience
sees *time*, not unrelated freeze frames.


| Strip           | Frames                              | What builds                              |
| --------------- | ----------------------------------- | ---------------------------------------- |
| Lock passer     | first `N` control frames of the run | touch `1/N … N/N` → **PASSER LOCKED**    |
| Ball travelling | one in-flight frame                 | passer faint in dark; **ball spotlight** |
| Lock receiver   | first touch … arrival               | `arrival_streak` → **RECEIVER LOCKED**   |
| Summary         | arrival                             | credited pass overlay + checklist        |


```bash
PYTHONPATH=. python -m world_cup_projects.player_stats.pass_explain_run \
    --video world_cup_projects/bundesliga_videos/08fd33_0.mp4 \
    --metric --layout talk

# Pick a specific pass (0 = best quality, gap ≥ 10 frames)
PYTHONPATH=. python -m world_cup_projects.player_stats.pass_explain_run \
    --video world_cup_projects/bundesliga_videos/08fd33_0.mp4 \
    --metric --pass-index 1
```

Uses cached pitch homography when available (`--metric`); radar appears on the summary panel only.

Outputs in `assets/explain_frames/`:


| File                             | Content                                                         |
| -------------------------------- | --------------------------------------------------------------- |
| `pass_detect_strip_passer.png`   | 3-frame filmstrip — control streak                              |
| `pass_detect_strip_flight.png`   | Single frame — ball travelling, passer dim                      |
| `pass_detect_strip_receiver.png` | 3-frame filmstrip — arrival streak                              |
| `pass_detect_summary.png`        | Pass credited + quality checklist                               |
| `pass_detect_timeline.png`       | All strips stacked (one poster slide)                           |
| `pass_detect_explain.mp4`        | Annotated slow-motion clip — real consecutive frames at low FPS |
| `pass_detect_explain.gif`        | Downscaled GIF export of the same walkthrough                   |


Prefer `--explain-video` for smooth slow motion (overlays on every source frame, default 8 fps).
MP4 is re-encoded to h264 for playback. `--gif` exports from that MP4 via ffmpeg palettegen
(default width 1280; `--gif-width 0` for full res). `--video-fps` / `--video-hold` tune pacing.

```bash
PYTHONPATH=. python -m world_cup_projects.player_stats.pass_explain_run \
    --video world_cup_projects/bundesliga_videos/08fd33_0.mp4 \
    --metric --layout talk --explain-video --video-fps 8

# GIF export (same annotated walkthrough, smaller)
PYTHONPATH=. python -m world_cup_projects.player_stats.pass_explain_run \
    --video world_cup_projects/bundesliga_videos/08fd33_0.mp4 \
    --metric --layout talk --explain-video --gif
```

---

### Speed and distance

**Intuition:** track each player’s feet every frame, measure how far they moved on the pitch
between frames, convert to km/h. Drop impossible jumps (tracker glitches). Without
homography we approximate scale from how tall they look in the image.

**In code:**

```
feet_m[t] - feet_m[t-1]  →  speed = dist / Δt
drop step if speed > 12.5 m/s
```

Without homography: meters/px from bbox height ≈ 1.8 m.

---

## Models


| Piece           | Source                                                                                                                                                                                                                                                                                                                                                                                                                |
| --------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Players + ball  | [football-players-detection](https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc) — **default: local YOLO `.pt` (v11)** with `--ball-threshold 0.20`. Optional Inference (`--detector-backend inference`) for Universe `/11` YOLO or `/18` `/20` RF-DETR — RF-DETR had higher ball recall on sampled frames but worse pass-network results on `08fd33_0`, so **stick with YOLO for demos**. |
| Pitch keypoints | Inference `football-field-detection-f07vi/15` — needs `ROBOFLOW_API_KEY`                                                                                                                                                                                                                                                                                                                                              |
| Tracking        | [trackers](https://github.com/roboflow/trackers) ByteTrack / BoTSORT                                                                                                                                                                                                                                                                                                                                                  |


Optional: `--source rfdetr` uses generic **COCO** RF-DETR (not soccer-trained; not recommended for Bundesliga clips).

**Env tip:** use a dedicated venv, not conda `base`. If `inference` crashes on import with
a NumPy / scikit-image error, run `pip install --force-reinstall scikit-image inference`.

---

## Repo layout

```
common/              possession, pitch, teams, detect, visual
pass_alternatives/   lane scoring, freeze render, explain frames
player_stats/        passes, network, speed/distance
bundesliga_videos/   test MP4s (not in git)
assets/              outputs (gitignored)
```

---

## Credits

[trackers](https://github.com/roboflow/trackers), [RF-DETR](https://github.com/roboflow/rf-detr),
`supervision`, and vendored [roboflow/sports](https://github.com/roboflow/sports) pitch/team utilities (Apache-2.0).