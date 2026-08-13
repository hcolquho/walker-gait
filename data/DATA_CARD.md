# Data Card — Walker-Gait Capture Protocol

## Ethics & Consent

- [ ] REB/IRB approval obtained (protocol #: _______)
- [ ] Informed consent collected from all participants
- [ ] Video data de-identified (face blurring applied where needed)

**Do not record any participant without confirmed ethics approval.**

## Hardware Setup

- **Walker**: Evolution Mini Trillium rollator
- **Camera**: Orbbec Femto Bolt, WFOV unbinned mode (120°×120°)
- **Mount**: SmallRig 4862 super clamp + magic arm, clamped to bar below seat
- **Camera angle**: Angled downward for centre lower-body view (~1200mm floor coverage)
- **Recording software**: Orbbec SDK / pyorbbecsdk
- **Color stream**: 1280×720 or 640×576, 30 fps
- **Depth stream**: 640×576, 30 fps (saved as .bag or synced frames)

## Capture Protocol

### Session Checklist

1. Confirm walker height is adjusted for participant
2. Verify camera is securely mounted, lens unobstructed
3. Record participant ID (anonymized), date, footwear type
4. Capture ≥ 3 straight-line walks (~10m each, turn-around excluded from annotation)
5. Include at least 1 walk at a deliberately slower pace
6. If possible, include a few steps with the walker partially blocking foot view

### Diversity Requirements (across all sessions)

- [ ] ≥ 5 distinct participants varying in height/build
- [ ] ≥ 2 footwear types (shoes, socks/barefoot if safe)
- [ ] Both directions of travel if camera view is asymmetric
- [ ] Include any atypical gait patterns available (shuffling, leaning on walker)

## Frame Extraction

Run `scripts/extract_frames.py` to pull frames from raw video:

```bash
python scripts/extract_frames.py \
    --input data/raw_video/ \
    --output data/frames/ \
    --fps 5 \
    --blur-threshold 100
```

## Annotation

### Tool

CVAT (self-hosted) or Label Studio, configured with:
- 12-keypoint skeleton (see `configs/data.yaml`)
- Bounding box per person (COCO format: x, y, w, h)
- Keypoint visibility flags: 0 = not labeled, 1 = labeled but occluded, 2 = visible

### Annotation Guidelines

1. Annotate only the walker user (ignore bystanders)
2. Bounding box should tightly enclose the person from hip to feet
3. Mark keypoints as **occluded (v=1)** when hidden behind walker frame but position
   can be reasonably estimated
4. Mark keypoints as **not labeled (v=0)** only when position is truly unknowable
5. Feet keypoints: place big_toe at the tip of the big toe (or shoe tip), small_toe
   at the outer edge, heel at the rearmost point of the heel

### Export Format

Export as COCO Keypoint JSON. Place files at:
- `data/annotations/train.json`
- `data/annotations/val.json`

Split by **subject** (all frames from one subject go into the same split).

## Pilot Batch Strategy

1. Extract ~500 candidate frames across all subjects
2. Run pretrained ViTPose++ to get initial predictions
3. Prioritize annotating frames where model confidence is LOW (active learning)
4. Annotate ~200-300 frames for first fine-tuning round
5. Evaluate, then decide whether to annotate more
