# World Cup Tracker Demos

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

| File | Good for |
|------|----------|
| `08fd33_0.mp4` | Default hero clip |
| `08fd33_8.mp4` | Missing ball, long air balls |
| `c01561_3.mp4` | Tricky camera / team colors |

**Useful flags**

| Flag | What it does |
|------|----------------|
| `--metric` | Real meters via pitch homography (needed for radar + lane scoring) |
| `--render` | Write MP4 |
| `--show-predictions` | Insert pass-alternative freezes on pass release frames |
| `--refresh-detections-cache` | Re-run player/ball detector (ignores `.cache/detections/`) |
| `--debug-pitch-keypoints` | Keypoints on main video (indices); radar shows kp whenever `--metric` |
| `--device mps` / `cpu` | Torch device for the football player YOLO |

**Caches**

- Player detections: `.cache/detections/`
- Pitch homography per frame: `.cache/pitch/` — delete `08fd33_0_*.pkl` to re-run keypoints
  without re-detecting players

Walkthrough notebook: [`world_cup_demos.ipynb`](world_cup_demos.ipynb).

---

## How it works

Each section starts with the design idea in plain language, then the exact thresholds we
code. Defaults are from `PassDetectionConfig` / `PassWeights.metric()` at 25 fps.

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
happened to be near them in the air” or “closest in a flat 2D projection.” We use a tight
gate for dribbling control and a looser one for first touch, then pick the closest valid
player.

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

| Kind | Reject when |
|------|-------------|
| **control** | `\|ball.y - feet.y\| > 20px` |
| **reception** | `ball.y - feet.y > 40px` (fly-by under feet; chest height OK) |

---

### Pass detection

**Intuition:** a pass is a story in three beats — someone *had* the ball, *released* it, and a
teammate *received* it — with no opponent taking control in between. We don’t look for ball
speed or trajectory directly; we watch carrier handoffs frame by frame. Each team keeps its
own memory of who last released the ball so one team’s possession doesn’t block the other’s.

Because broadcast tracking is noisy, we require short *streaks* of evidence (not one lucky
frame) and special cases for goalkeepers and one-touch plays. Pass logic inherits the same
detector limits as carrier assignment — see **Who has the ball?** — plus the missing-ball
bridge below for short in-flight gaps.

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
```

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
NOT opponent_control_between(release_frame, arrival_frame)
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

**Intuition:** imagine the carrier could play any teammate. For each candidate pass we ask:
how open is the *line* between them (are rivals blocking the corridor?), how *forward* is
it toward the opponent goal, and how much *space* does the receiver have around them? We
lightly penalize teammates standing in a narrow lane, passes that go backward while the
player is running forward, and absurd lengths. The top three lanes are what we draw on
freeze frames; the same score grades real passes after the fact.

**In code** (`score_pass_options`, metric):

For carrier C → teammate R, `δ = R - C`, `L = |δ|`:

**Skip** if `L < 2 m` or `L > 45 m`.

**Rival corridor half-width** (full width = 2×):

```
L ≤ 18 m  → 2.5 m
L ≤ 28 m  → 3.25 m
L > 28 m  → 4.0 m  (cap 4.5 m)
```

Teammate corridor: **1.2 m** (separate penalty, not openness).

```
openness   = min distance rival → segment(C,R) inside corridor
space      = min distance any rival → R
forward    = δ · attack_dir          # meters toward opponent goal

score = 0.45·min(openness/8, 1)
      + 0.30·min(max(forward/25, 0), 1)
      + 0.25·min(space/8, 1)
      - teammate_penalty              # up to 0.10 if teammate in 1.2 m corridor
      - backward_run_penalty          # up to 0.22 if pass·run < -0.15
      - backward_attack_penalty       # up to 0.18 if forward < 0 AND run·attack ≥ 0.15
```

`quality_score = null` if receiver not in scored lane list (e.g. length out of range).

---

### When we freeze the video

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
AND carrier found
AND top_options(k=3) non-empty
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

| Piece | Source |
|-------|--------|
| Players + ball | [football-players-detection](https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc) YOLO — auto-downloads to `.cache/models/` |
| Pitch keypoints | Inference `football-field-detection-f07vi/15` — needs `ROBOFLOW_API_KEY` |
| Tracking | [trackers](https://github.com/roboflow/trackers) ByteTrack / BoTSORT |

Optional: RF-DETR via `--source rfdetr` (generic COCO; weaker team split on video).

**Env tip:** use a dedicated venv, not conda `base`. If `inference` crashes on import with
a NumPy / scikit-image error, run `pip install --force-reinstall scikit-image inference`.

---

## Repo layout

```
common/              possession, pitch, teams, detect, visual
pass_alternatives/   lane scoring + freeze render
player_stats/        passes, network, speed/distance
bundesliga_videos/   test MP4s (not in git)
assets/              outputs (gitignored)
```

---

## Credits

[trackers](https://github.com/roboflow/trackers), [RF-DETR](https://github.com/roboflow/rf-detr),
`supervision`, and vendored [roboflow/sports](https://github.com/roboflow/sports) pitch/team utilities (Apache-2.0).
