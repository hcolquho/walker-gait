# Walker-Gait Project — Claude Code Context

## Project Overview

Clinical gait analysis system for post-stroke patients using a rollator-mounted
Orbbec Femto Bolt RGB-D camera. The system records synchronized colour + depth
video of a walker user's lower body, runs pose estimation to extract 12 lower-body
keypoints, backprojects to 3D using depth, and computes 7 clinical gait metrics.

All processing is **offline** on SHARCNET (Compute Canada HPC cluster).
There is no real-time processing requirement.

---

## Hardware

- **Walker**: Evolution Mini Trillium rollator
- **Camera**: Orbbec Femto Bolt (WFOV 120°×120°, iToF depth, built-in 6-DoF IMU)
  - Serial: CL8S16100V3, Firmware: 1.1.3
  - Color: 1280×960 @ 30fps, RGB format
  - Depth: 640×576 @ 30fps, Y16 format (values in mm)
  - Intrinsics: fx=1004.1, fy=1003.7, cx=640.1, cy=487.6
  - Sync gap: ~1.1ms (hardware frame sync enabled)
- **Mount**: SmallRig 4862 super clamp + magic arm, clamped to bar below walker seat
- **Camera view**: top-down angled, captures hips to floor of walker user
- **Sensor suite**: ONLY the Femto Bolt and its built-in IMU. No external IMU, no FSR sensors.

---

## Developer Environment

- **Local machine**: Windows, Anaconda, cmd.exe, no GPU, 32GB RAM
- **Conda env**: `gait-pose` (pyorbbecsdk, opencv-contrib-python, numpy, scipy, pyyaml)
- **SHARCNET env**: `walker-gait` conda env (torch, transformers, scipy, pycocotools)
- **Repo root**: `C:\Users\hanna\Documents\USRA\walker-gait`
- **Data collection scripts**: separate folder `C:\Users\hanna\Documents\USRA\data-collection`

---

## Repo Structure

```
walker-gait/
├── configs/
│   ├── data.yaml               # paths, keypoint subset, split ratios
│   ├── model.yaml              # backbone name, dataset_index, freeze strategy
│   └── train.yaml              # lr, epochs, batch size
├── data/
│   ├── raw_video/              # gitignored — recording sessions from data-collection repo
│   ├── frames/                 # extracted candidate frames for annotation
│   ├── annotations/            # COCO-format keypoint JSONs (train/val splits)
│   └── DATA_CARD.md            # capture protocol, ethics checklist
├── notebooks/
│   ├── 01_frame_selection.ipynb
│   ├── 02_finetune_vitpose.ipynb   # main training loop
│   ├── 03_eval_visualize.ipynb
│   └── 04_inference_demo.ipynb
├── src/walker_gait/
│   ├── data/                   # GaitKeypointDataset, affine, heatmap targets, augmentation
│   ├── models/                 # vitpose_finetune.py — load, freeze, masked loss
│   ├── pipeline/               # inference.py — WalkerGaitPipeline (detect → pose → filter)
│   ├── training/               # loop.py, metrics.py (PCK, heatmap decoding)
│   ├── utils/                  # visualization.py — draw_skeleton
│   └── gait/                   # deterministic gait analysis module
│       ├── __init__.py
│       ├── backproject.py      # (u,v) + depth → 3D camera coords → world coords
│       ├── smooth.py           # gap filling + Butterworth filter  ← TODO: build this
│       ├── event_detector.py   # heel strikes + toe-offs from 3D keypoint tracks
│       └── metrics.py          # 7 clinical gait metrics
├── scripts/
│   ├── extract_frames.py       # video → PNG frames for annotation
│   ├── train.py                # SLURM mirror of notebook 02
│   ├── batch_inference.py      # offline pose inference on session folders
│   └── process_session.py      # full gait pipeline on one trial  ← TODO: build this
├── checkpoints/                # gitignored
├── environment/
│   └── environment.yml
└── tests/
    ├── test_data_and_loss.py
    └── test_gait.py
```

---

## Pose Model

- **Model**: `usyd-community/vitpose-plus-large` (HuggingFace Transformers)
- **Why ViTPose++**: MoE head supports COCO-WholeBody (133 keypoints) via `dataset_index=5`
  - dataset_index=5 is CRITICAL — without it you get wrong keypoints silently
  - ViTPose (non-++) only has 17 body keypoints, no feet
- **Why not RTMPose**: foot AP is ~8 points lower than ViTPose+-L on COCO-WholeBody
- **Why not real-time**: all processing is offline on SHARCNET, speed irrelevant
- **Fine-tuning strategy**: freeze ViT backbone, train decoder head only
  - HF port does NOT compute loss natively — must manually compute MSE on heatmaps
  - Use `masked_heatmap_loss()` in `src/walker_gait/models/vitpose_finetune.py`

---

## Keypoint Subset (12 of 133 COCO-WholeBody)

| Our index | Name | COCO-WB index |
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

```python
COCO_WB_INDICES = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
COCO_WHOLEBODY_DATASET_INDEX = 5
```

---

## Gait Metrics (7 total)

All computed deterministically from 3D keypoint tracks — no learned metrics head.

| Metric | Description |
|---|---|
| (a) Stride time | Mean time between consecutive same-foot heel strikes |
| (b) Left/right step counts | Heel strike counts per side |
| (c) Cadence | Total steps per minute |
| (d) Step time asymmetry | Symmetry index of left vs right step times |
| (e) Step length asymmetry | Symmetry index of left vs right step lengths (needs depth) |
| (f) Double support time | Mean bilateral floor-contact duration |
| (g) Asymmetry score | Weighted aggregate of (d), (e), DS-time asymmetry |

Symmetry index formula (Robinson 1987): `|L - R| / (0.5 * (L + R)) * 100%`

---

## Processing Pipeline Order

```
batch_inference.py          → 2D keypoints + scores per frame (JSON)
    ↓
gait/backproject.py         → 2D + depth → 3D world coords (mm)
    ↓
gait/smooth.py              → fill NaN gaps (linear interp) + Butterworth 6Hz
    ↓
gait/event_detector.py      → heel strikes + toe-offs (frame indices + timestamps)
    ↓
gait/metrics.py             → GaitMetrics dataclass → .to_dict()
```

---

## Data Collection Setup (separate repo)

Data collection lives in `C:\Users\hanna\Documents\USRA\data-collection\` with:

- `scripts/run_protocol.py` — interactive 6-block protocol runner
- `scripts/record_session.py` — records one trial with pyorbbecsdk
- `scripts/detect_timing_marks.py` — ArUco-based 10MWT crossing detection
- `scripts/verify_session.py` — sanity checks a completed trial
- `configs/protocol.yaml` — full 6-block protocol definition

### Data folder structure per trial

```
data/raw_video/han/session_01/block1_healthy_baseline/comfortable_right/trial_01/
├── color/          # 000000.jpg ... (JPEG uint8, BGR)
├── depth/          # 000000.npy ... (uint16, values in mm)
├── timestamps.csv  # frame_idx, color_ts_ms, depth_ts_ms
├── intrinsics.json # fx, fy, cx, cy, depth_scale, width, height
├── metadata.json   # subject, condition, notes, bpm, affected_side
└── timing_marks_manual.json  # analysis_window: start_frame, end_frame
```

### Analysis window

User presses Q in preview window after passing the 12m mark.
Script discards first 3s and last 3s of frames automatically.
Analysis window stored in `timing_marks_manual.json`:

```json
{
  "analysis_window": {
    "start_frame": 90,
    "end_frame": 992,
    "discard_start_s": 3.0,
    "discard_end_s": 3.0
  }
}
```

---

## 6-Block Data Collection Protocol

| Block | Name | Purpose |
|---|---|---|
| 1 | Healthy baseline | Ground truth — symmetric normal gait at 3 speeds |
| 2 | Speed reduction | Slow symmetric gait across stroke speed range |
| 3 | Temporal asymmetry | Delayed push-off on affected side (timing only) |
| 4 | Spatial asymmetry | Shorter step on affected side (length only) + combined |
| 5 | Compensatory patterns | Foot drop, circumduction, shuffling |
| 6 | 10MWT validation | Ground-truth walking speed cross-validation |

- 3 trials per condition, 18 conditions = 54 recordings per session
- Affected side kept consistent within a session, swapped between sessions
- All trials use 14m corridor: 2m run-in + 10m analysis + 2m run-out
- Participant: `han`, sessions in `data/raw_video/han/session_01/` (right affected) and `session_02/` (left affected)

---

## Key SDK Notes (pyorbbecsdk v1.x)

```python
# Correct profile API — use 0 for height to avoid OBError
cp.get_video_stream_profile(1280, 0, OBFormat.RGB, 30)
dp.get_video_stream_profile(640,  0, OBFormat.Y16, 30)

# wait_for_frames takes positional arg only — no keyword
pipeline.wait_for_frames(200)   # NOT timeout_ms=200

# Always enable frame sync before start
pipeline.enable_frame_sync()
pipeline.start(config)
```

---

## Known Issues / TODOs

- `smooth.py` not yet built — needed before gait event detection can run
- `process_session.py` not yet built — needed to chain full pipeline on SHARCNET
- Camera tilt angle and floor offset not yet calibrated — needed for `world_from_camera()`
- Fine-tuning not yet done — first processing pass will use pretrained ViTPose++ weights
- Timestamp count occasionally fewer than frame count (~86 row gap) — harmless, missing rows are always in the discard zone
- ArUco timing mark detection unreliable with moving camera (motion blur) — replaced by manual Q-press method
