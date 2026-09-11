# Walker-Gait Repo

Clinical gait-asymmetry analysis for post-stroke rehabilitation, from a rollator-mounted
Orbbec Femto Bolt RGB-D camera. The camera looks down at the walker user's lower body;
the system extracts 12 lower-body keypoints per frame, backprojects them to 3D using the
depth stream, detects heel strikes and toe-offs, and computes 7 clinical gait metrics.

All processing is **offline** on SHARCNET. There is no real-time requirement anywhere in
this project — do not optimize for speed.

---

## Handoff status

**Read this section first.** It is the honest state of the project, not the plan.

| Stage | Status |
|---|---|
| Data collection | ✅ **Done.** 120 trials across 3 sessions recorded and verified |
| Pose inference on the recorded data | ✅ **Done.** All 120 trials have `keypoints.json` — but see blocker #1: inference used COCO-17 body-only (6 keypoints, `dataset_index=0`), not wholebody. Files will need to be regenerated after fixing the model. |
| Keypoint annotation | ❌ **Not started.** `data/annotations/` is empty |
| Fine-tuning | ❌ **Not done.** `checkpoints/` is empty; no trained weights exist |
| Gait module (backproject → smooth → events → metrics) | ✅ **Written and unit-tested.** 18/18 tests pass |
| Gait module validated on real data | ⚠️ **Partially.** `process_session.py` runs end to end on a real trial and writes `gait_metrics.json`, but all metrics return NaN — see blockers #1 and #3 |
| Camera extrinsic calibration (tilt, floor offset) | ❌ **Not measured.** Current values are guesses |

The pipeline has been run end to end on one trial. It does not crash. It produces NaN
metrics. The reasons are known and documented in the blockers below.

---

## Known blockers

These are real defects, found by reading the code. Fix them in roughly this order.

### 1a. The inference pipeline only produces 6 keypoints, but the gait code needs 12

[src/walker_gait/pipeline/inference.py](src/walker_gait/pipeline/inference.py) was
temporarily switched to plain COCO-17 body keypoints:

```python
COCO_WB_INDICES = [11, 12, 13, 14, 15, 16]   # hips, knees, ankles only
COCO_WHOLEBODY_DATASET_INDEX = 0             # 0 = COCO body, not 5 = WholeBody
KEYPOINT_NAMES = [...]                       # length 6, no heels, no toes
```

Everything downstream — `event_detector.py`, `metrics.py`, `process_session.py` — expects
the 12-name list in
[event_detector.py:14](src/walker_gait/gait/event_detector.py#L14), including
`left_heel`, `right_heel`, `left_big_toe`, `right_big_toe`.

**What was actually run:** `batch_inference.py` was run on all 120 trials on SHARCNET
(Nibi cluster, H100 GPU) using `dataset_index=0` and the 6-name `KEYPOINT_NAMES` list.
All 120 trial folders now contain `keypoints.json` with hips, knees, and ankles only.
These files are usable for cadence and step-time metrics but not for heel/toe-dependent
metrics. They will need to be regenerated after the fix below is applied.

**Consequence if you ignore this:** the pipeline will not crash. `process_session.py`
fills missing keypoint names with NaN, so heel and toe tracks are entirely NaN, no gait
events are detected, and every metric comes out `NaN`. This failure is silent. Do not
mistake it for a tuning problem.

**Fix:** restore `COCO_WB_INDICES = [11..22]` and `dataset_index = 5`, and restore the
12-name `KEYPOINT_NAMES`. The correct values are in [configs/model.yaml](configs/model.yaml)
and [src/walker_gait/models/vitpose_finetune.py](src/walker_gait/models/vitpose_finetune.py),
which were never downgraded. Then verify on one frame that the returned array is
`(N, 12, 2)` and that heel/toe points land on the foot — `dataset_index` errors are silent
and produce plausible-looking but wrong keypoints.

Two smaller things in the same file, worth understanding before you trust its output:

- The person detector is bypassed; the full frame is used as the bounding box (see
  `WalkerGaitPipeline.__call__`). Reasonable for this rig — exactly one user, always in
  frame — but it means the pose crop has whatever aspect ratio the camera gives, not a
  tight person box.
- Confidence is `sigmoid(max_heatmap_value)`. Heatmap peaks are not logits, so these scores
  are not calibrated probabilities. `backproject_sequence` gates on `score_threshold=0.3`
  against them — re-check that threshold once you have seen real score distributions.

### 1b. Ankle depth is consistently invalid at floor level

Even after restoring 12 keypoints, heel-strike detection via 3D world-frame Z-height
will likely fail. The Femto Bolt's iToF sensor returns 0 (invalid) at ankle and foot
pixel locations because these fall in the near-floor region where oblique-angle IR
reflectance is poor. In testing, hip and knee depth was reliable (1,100–1,500 mm from
camera) but ankle pixels consistently returned 0 mm.

**Consequence:** `backproject_sequence` leaves ankle/heel/toe Z-coordinates as NaN.
`event_detector.py` requires non-NaN Z values to detect heel strikes. Zero events →
all metrics NaN.

**Workaround already implemented:** `detect_gait_events_2d()` was added to
`event_detector.py`. It uses the vertical pixel Y coordinate of the ankle keypoints
as a proxy for foot height — when ankle Y reaches a local maximum (foot lowest in
frame), that is a heel strike. This bypasses depth entirely and was confirmed to
detect events on a test trial.

**To use it**, change `process_session.py` to call `detect_gait_events_2d(keypoints_2d)`
instead of `detect_gait_events(keypoints_3d)`. The 2D version is exported from
`walker_gait.gait`.

**Longer-term fix:** the depth limitation is a sensor physics problem at this camera
angle. Options include a different mounting geometry, a structured-light sensor instead
of iToF, or accepting that 3D metrics (step length) require a different approach.

### 2. `process_all_sessions.sh` passes a flag `process_session.py` does not accept

[scripts/process_all_sessions.sh](scripts/process_all_sessions.sh) calls
`process_session.py --keypoints-json "$KEYPOINTS_JSON"`, but
[scripts/process_session.py](scripts/process_session.py) has no such argument — it hardcodes
`session_dir / "keypoints.json"`. Every SLURM array task dies immediately on
"unrecognized arguments".

The two scripts also disagree about where keypoints live: `batch_inference.py` writes
`keypoints.json` **inside each trial folder**, while `process_all_sessions.sh` expects a
mirrored `results/han/<relative path>.json` tree.

**Fix:** pick one convention. Simplest is to drop `--keypoints-json` from the shell script
and let both use the in-trial location.

### 3. Camera extrinsics are uncalibrated

`--tilt-deg` (camera's downward tilt from horizontal) and `--floor-offset-mm` (vertical
distance from camera mount to floor) have never been measured. The defaults disagree between
the two entry points — `30 / 900` in [process_session.py](scripts/process_session.py),
`35 / 950` in [process_all_sessions.sh](scripts/process_all_sessions.sh).

Every 3D coordinate depends on these. `floor_offset_mm` sets where `z = 0` is, and heel
strikes are detected as heel-height minima below `floor_tol_mm = 50` mm — so if the floor
offset is wrong by more than ~5 cm, **zero events are detected and all metrics are NaN**.
Measure both with a tape measure before trusting any output, and record them in the data
card.

### 4. `extract_frames.py` does not match how the data is actually stored

[scripts/extract_frames.py](scripts/extract_frames.py) opens video files with
`cv2.VideoCapture`. The recorder saves **folders of JPEGs**, not video files. This script is
only needed for the annotation workflow (choosing frames to label); it needs rewriting to
walk `color/*.jpg`. Not a blocker for the metrics pipeline.

### 5. Cosmetic

- There is a stray directory literally named `{configs,data/{raw_video,frames,...}` in the
  repo root — debris from a Bash brace-expansion `mkdir` run under Windows. It holds only
  empty directories and is untracked by git. Safe to delete.
- Docstring paths in `process_session.py` and `batch_inference.py` say `data/han/...`; the
  real path is `data/raw_video/han/...`.

---

## Start here

Goal: get one trial all the way through, then eyeball the numbers.

```bash
conda activate walker-gait
pip install -e .              # src/ layout — tests import walker_gait directly
pytest tests/ -q              # expect 18 passed (~100 s; the smoothing tests are slow)
```

Then:

1. **Fix blocker #1** (12 keypoints, `dataset_index=5`). Nothing below works without it.
2. **Measure the camera tilt and floor offset** on the rig (blocker #3).
3. **Run inference on one trial:**
   ```bash
   python scripts/batch_inference.py \
       --input data/raw_video/han/session_01/block1_healthy_baseline/comfortable_right/trial_01 \
       --model usyd-community/vitpose-plus-large \
       --device cuda
   ```
   Uses pretrained weights — no fine-tuning needed for a first result. Writes
   `keypoints.json` into the trial folder, and **skips any trial that already has one**, so
   delete the file to re-run.
4. **Overlay the keypoints on a few frames and look at them.** Use `draw_skeleton` in
   [src/walker_gait/utils/visualization.py](src/walker_gait/utils/visualization.py) or
   [notebooks/04_inference_demo.ipynb](notebooks/04_inference_demo.ipynb). Pretrained
   ViTPose++ has never seen this steep top-down walker viewpoint — expect the feet to be the
   weak point. How bad they look determines whether fine-tuning is actually necessary.
5. **Run the gait pipeline on that trial:**
   ```bash
   python scripts/process_session.py \
       --session data/raw_video/han/session_01/block1_healthy_baseline/comfortable_right/trial_01 \
       --tilt-deg <measured> \
       --floor-offset-mm <measured> \
       --fps 30
   ```
   Writes `gait_metrics.json` into the trial folder and prints it.
6. **Sanity-check against physiology, not against the code.** For a healthy adult at a
   comfortable pace: stride time ≈ 1.0–1.2 s, cadence ≈ 100–120 steps/min, double support
   ≈ 0.1–0.2 s, and the block-1 healthy trials should score near 0 on every asymmetry index.
   Block 3 and 4 trials were walked with *deliberate* asymmetry and should score high. That
   contrast is the cheapest validation available. Use it.

If step 5 returns all `NaN`, the cause is almost always one of two things: no heel/toe
keypoints (blocker #1), or a floor offset that puts the feet nowhere near `z = 0`
(blocker #3).

---

## Pipeline

```
color/*.jpg  ──►  batch_inference.py        ──►  keypoints.json   (2D px + scores per frame)
                    ViTPose++, full-frame box
                              │
depth/*.npy  ──►  gait/backproject.py       ──►  (T, 12, 3) world mm, NaN where invalid
                    pixel+depth → camera 3D → world 3D
                              │
                  gait/smooth.py            ──►  gaps ≤10 frames filled, 6 Hz low-pass
                    constant-velocity Kalman, then zero-phase Butterworth
                              │
                  gait/event_detector.py    ──►  heel strikes + toe-offs (frame indices)
                    heel-height minima near floor; toe-off = last near-floor toe sample
                              │
                  gait/metrics.py           ──►  gait_metrics.json (7 metrics)
```

`scripts/process_session.py` runs everything from `keypoints.json` onward for one trial.

**Coordinate frames.** Camera frame is OpenCV convention: +x right, +y down, +z forward out
of the lens. World frame: +x lateral, +y forward (direction of travel), **+z up with z = 0
at the floor**. All distances in millimetres. The event detector depends on the world frame
being right — see [src/walker_gait/gait/backproject.py](src/walker_gait/gait/backproject.py).

**NaN is the missing-data convention throughout.** Low-confidence keypoints, invalid depth
(0 mm), out-of-frame pixels, and gaps longer than `max_gap_frames` all stay NaN rather than
being interpolated. Downstream code skips NaN rather than crashing on it — which is exactly
why bad input yields NaN metrics instead of an error.

---

## The 7 gait metrics

Computed deterministically in [src/walker_gait/gait/metrics.py](src/walker_gait/gait/metrics.py).
There is no learned metrics head, by design — every number traces back to a detected event.

| # | Metric | Field in `gait_metrics.json` |
|---|---|---|
| a | Stride time — mean interval between same-foot heel strikes | `stride_time_s` |
| b | Step counts per side | `left_step_count`, `right_step_count` |
| c | Cadence — total steps per minute | `cadence_steps_per_min` |
| d | Step time asymmetry | `step_time_asymmetry_pct` |
| e | Step length asymmetry (requires depth) | `step_length_asymmetry_pct` |
| f | Double support time — mean bilateral floor contact | `double_support_time_s` |
| g | Asymmetry score — mean of (d), (e), and double-support asymmetry | `asymmetry_score` |

Symmetry index (Robinson 1987): `|L − R| / (0.5 × (L + R)) × 100%`; 0% is perfectly
symmetric. Step length uses forward-axis (world y) separation between the two heels at the
instant of heel strike.

Thresholds you will likely need to revisit against real data:

| Parameter | Default | Where |
|---|---|---|
| `score_threshold` | 0.3 | `backproject_sequence` |
| `max_gap_frames` | 10 (≈0.33 s) | `smooth_keypoint_sequence` |
| `cutoff_hz` | 6.0 | `smooth_keypoint_sequence` |
| `min_step_interval_s` | 0.3 | `detect_gait_events` |
| `floor_tol_mm` | 50.0 | `detect_gait_events` |

---

## Data

**The recordings are gitignored and do not travel with this repo.** 114 trials at roughly
340 MB each — about 38 GB total — living at
`C:\Users\hanna\Documents\USRA\walker-gait\data\raw_video\han\`. Arrange a physical copy
before the outgoing student's machine is wiped; no other copy is referenced anywhere in the
repo.

Both sessions are marked `"status": "complete"` in their `session_info.json`. Session 01 has
the **right** side as affected, session 02 the **left**.

### Layout

```
data/raw_video/han/session_01/
├── session_info.json          # start time, affected_side, status
├── completion_log.json        # per-trial log written by the collection script
└── block1_healthy_baseline/
    └── comfortable_right/
        └── trial_01/
            ├── color/          000000.jpg …  (JPEG, BGR, 1280×960 @ 30 fps)
            ├── depth/          000000.npy …  (uint16, millimetres, 640×576 @ 30 fps)
            ├── timestamps.csv  frame_idx, color_ts_ms, depth_ts_ms
            ├── intrinsics.json fx, fy, cx, cy, depth_scale, width, height
            ├── metadata.json   subject, condition, notes, bpm, affected_side
            └── timing_marks_manual.json   ← analysis window; pipeline refuses to run without it
```

### Analysis window

Only the middle of each walk is analyzed. The operator pressed Q after passing the 12 m
mark, and the recorder discarded the first and last 3 seconds. The surviving range lives in
`timing_marks_manual.json`:

```json
{"analysis_window": {"start_frame": 90, "end_frame": 992,
                     "discard_start_s": 3.0, "discard_end_s": 3.0}}
```

`process_session.py` reads this and raises if it is missing. Frames outside it are never
touched.

### The 6-block protocol

3 trials per condition, 18 conditions, 54+ recordings per session. 14 m corridor: 2 m run-in
+ 10 m analysis + 2 m run-out.

| Block | Name | What it gives you |
|---|---|---|
| 1 | Healthy baseline | Ground truth for symmetric gait, 3 speeds |
| 2 | Speed reduction | Slow symmetric gait across the stroke speed range |
| 3 | Temporal asymmetry | Delayed push-off on the affected side — timing only |
| 4 | Spatial asymmetry | Shorter step on the affected side — length only, plus combined |
| 5 | Compensatory patterns | Foot drop, circumduction, shuffling |
| 6 | 10MWT validation | Ground-truth walking speed for cross-validation |

Blocks 3 and 4 are deliberately one-dimensional: block 3 perturbs timing without length,
block 4 length without timing. That separation is what lets you confirm
`step_time_asymmetry_pct` and `step_length_asymmetry_pct` really are measuring different
things.

### Recording new data

Collection lives in a **separate repo**: `C:\Users\hanna\Documents\USRA\data-collection\`
(`scripts/run_protocol.py`, `scripts/record_session.py`, `scripts/verify_session.py`,
`configs/protocol.yaml`). See [data/DATA_CARD.md](data/DATA_CARD.md) for the ethics checklist
and annotation guidelines — **no recording without confirmed REB approval.**

---

## Pose model

`usyd-community/vitpose-plus-large` via HuggingFace Transformers.

- **Why ViTPose++ and not ViTPose:** only the `++` MoE variants expose COCO-WholeBody
  (133 keypoints) through `dataset_index=5`. Plain ViTPose has 17 body keypoints and **no
  feet** — and this project lives or dies on heel and toe positions.
- **`dataset_index=5` is critical.** Pass the wrong expert index and you get wrong keypoints
  with no error and no warning.
- **Why not RTMPose:** foot AP is ~8 points lower than ViTPose+-L on COCO-WholeBody, and its
  real-time capability buys us nothing here.
- **The HF port does not compute a loss.** `VitPoseForPoseEstimation.forward()` ignores
  labels. Fine-tuning therefore uses the hand-written `masked_heatmap_loss()` in
  [src/walker_gait/models/vitpose_finetune.py](src/walker_gait/models/vitpose_finetune.py),
  which supervises only the 12 relevant heatmap channels.
- **Fine-tuning strategy:** freeze the ViT backbone, train the decoder head only. With fewer
  than ~500 annotated frames, do not unfreeze transformer blocks.

### The 12-keypoint subset

| Ours | Name | COCO-WholeBody |
|---|---|---|
| 0 | left_hip | 11 |
| 1 | right_hip | 12 |
| 2 | left_knee | 13 |
| 3 | right_knee | 14 |
| 4 | left_ankle | 15 |
| 5 | right_ankle | 16 |
| 6 | left_big_toe | 17 |
| 7 | left_small_toe | 18 |
| 8 | left_heel | 19 |
| 9 | right_big_toe | 20 |
| 10 | right_small_toe | 21 |
| 11 | right_heel | 22 |

This ordering is load-bearing — `event_detector.KEYPOINT_NAMES`, the `flip_pairs` in
`configs/train.yaml`, and the `skeleton` in `configs/data.yaml` all assume it.

### If you decide to fine-tune

Only worth doing if step 4 of "Start here" shows the pretrained model failing on feet. It
requires annotation work that has not been started.

1. Extract candidate frames (after fixing blocker #4), ~500 across conditions.
2. Annotate in CVAT or Label Studio with the 12-point skeleton from `configs/data.yaml`.
   Prioritize frames where the pretrained model is least confident. Guidelines and
   visibility-flag conventions are in [data/DATA_CARD.md](data/DATA_CARD.md).
3. Export COCO keypoint JSON to `data/annotations/{train,val}.json`.
   **Split by subject, never by random frame** — consecutive frames are near-duplicates, and
   random splitting leaks the validation set into training.
4. Train: `python scripts/train.py --config configs/train.yaml`, or
   `sbatch scripts/sharcnet_job.sh`.
5. Pass the resulting checkpoint to `batch_inference.py --checkpoint`.

---

## Full-dataset processing on SHARCNET

Once one trial is verified end to end:

```bash
# 1. Pose inference over every trial (GPU; skips trials already done)
python scripts/batch_inference.py --input data/raw_video/han/ --device cuda

# 2. Gait pipeline, one SLURM array task per trial (CPU)
#    NOTE: fix blocker #2 first, or every task fails immediately.
N=$(bash scripts/process_all_sessions.sh --count)
sbatch --array=0-$((N - 1)) scripts/process_all_sessions.sh
```

`process_all_sessions.sh` reads `TILT_DEG`, `FLOOR_OFFSET_MM`, and `FPS` from the
environment. `batch_inference.py` is resumable by design — it skips any trial that already
has a `keypoints.json`.

---

## Environments

Two different environments, deliberately:

- **Local (Windows, no GPU)** — `gait-pose` conda env: `pyorbbecsdk`,
  `opencv-contrib-python`, `numpy`, `scipy`, `pyyaml`. Used for recording and for
  running the tests. The gait module is pure numpy/scipy and runs fine here.
- **SHARCNET (Nibi cluster)** — `virtualenv` at
  `~/projects/def-pviswana/hcolquho/venv/`, not a conda env. Created with
  `python -m venv` under `module load StdEnv/2023 python/3.11 cuda/12.2 gcc opencv/4.14.0`.
  [requirements.txt](requirements.txt) pins the Compute Canada wheel builds
  (`+computecanada`, `transformers==5.15.1`). Activate with:
```bash
  module load StdEnv/2023 python/3.11 cuda/12.2 gcc opencv/4.14.0
  source ~/projects/def-pviswana/hcolquho/venv/bin/activate
```
  The opencv module **must be loaded before activating the venv** or `pip install -e .`
  will fail. The package is installed in editable mode (`pip install -e .`).
  Account: `def-pviswana`. Data lives at:
  `~/projects/def-pviswana/hcolquho/data/han/` (120 trial folders, 85 GB).

---

## Hardware

- **Walker:** Evolution Mini Trillium rollator
- **Camera:** Orbbec Femto Bolt — WFOV 120°×120°, iToF depth, built-in 6-DoF IMU
  (serial CL8S16100V3, firmware 1.1.3)
  - Color 1280×960 @ 30 fps RGB; depth 640×576 @ 30 fps Y16 (millimetres)
  - Intrinsics: fx=1004.1, fy=1003.7, cx=640.1, cy=487.6
  - Hardware frame sync on; color/depth gap ≈ 1.1 ms
- **Mount:** SmallRig 4862 super clamp + magic arm, clamped to the bar below the seat,
  angled down at the user's lower body
- **Sensors:** the Femto Bolt and its built-in IMU, and nothing else. No external IMU, no
  force-sensitive resistors. (Earlier drafts referenced a BNO085 and FSRs — both dropped.)

### pyorbbecsdk gotchas (v1.x, relevant to the collection repo)

```python
cp.get_video_stream_profile(1280, 0, OBFormat.RGB, 30)   # height 0 avoids OBError
dp.get_video_stream_profile(640,  0, OBFormat.Y16, 30)
pipeline.wait_for_frames(200)   # positional only — timeout_ms=200 raises
pipeline.enable_frame_sync()    # must precede pipeline.start(config)
```

---

## Quirks worth knowing

- **`timestamps.csv` sometimes has fewer rows than there are frames** (~86 short). The
  missing rows always fall inside the 3-second discard zones, so the analysis window is
  unaffected. `process_session.py` does not read the CSV at all — it derives timestamps as
  `frame_index / fps` at a fixed 30 fps. If you ever need true inter-frame timing, read the
  CSV instead.
- **ArUco timing-mark detection was abandoned.** `detect_timing_marks.py` in the collection
  repo never worked reliably — the camera moves with the walker and motion blur destroys the
  markers. The manual Q-press replaced it. Do not revive the ArUco approach without solving
  the blur problem first.
- **The notebooks have no saved outputs** and mirror the scripts rather than extending them.
  `scripts/train.py` is the maintained version of `02_finetune_vitpose.ipynb`.

---

## Repo map

```
configs/          data.yaml (paths, keypoint subset, skeleton), model.yaml (backbone,
                  dataset_index, freeze), train.yaml (lr, epochs, augmentation, SLURM)
data/
  raw_video/      gitignored — 120 recorded trials (session_01: 57 right-affected,
                  session_02: 57 left-affected, session_03: 6 healthy controls).
                  All 120 have keypoints.json from SHARCNET inference (COCO-17, 6 kp).
  frames/         gitignored — extracted frames for annotation (empty)
  annotations/    COCO keypoint JSONs (empty)
  DATA_CARD.md    ethics checklist, capture protocol, annotation guidelines
notebooks/        01 frame selection · 02 fine-tune · 03 eval/visualize · 04 inference demo
src/walker_gait/
  data/           GaitKeypointDataset, affine transforms, heatmap targets, augmentation
  models/         vitpose_finetune.py — load/freeze, masked_heatmap_loss
  pipeline/       inference.py — WalkerGaitPipeline  ← see blocker #1
  training/       loop.py (train/eval), metrics.py (PCK, heatmap decoding)
  utils/          visualization.py — draw_skeleton
  gait/           backproject.py · smooth.py · event_detector.py · metrics.py
scripts/          extract_frames.py · train.py · batch_inference.py · process_session.py
                  sharcnet_job.sh (training) · process_all_sessions.sh (SLURM array)
tests/            test_data_and_loss.py · test_gait.py   — 18 tests, all passing
CLAUDE.md         context file for Claude Code; mirrors much of this README
```

## Tests

```bash
pytest tests/ -q
```

18 tests, all passing, ~100 seconds. They cover affine transforms, heatmap target
generation, masked loss, PCK, and the whole gait chain — but **only on synthetic signals**.
A green suite means the maths is self-consistent, not that the system measures gait
correctly. Real-data validation is entirely ahead of you.
