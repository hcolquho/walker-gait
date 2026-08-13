# Walker-Gait: Fine-Tuned ViTPose++ for Walker-Mounted Gait Analysis

Clinical gait asymmetry analysis from a rollator-mounted Orbbec Femto Bolt RGB-D camera,
using a ViTPose++ (HuggingFace Transformers) backbone fine-tuned on walker-perspective
lower-body keypoints.

## Project Structure

```
walker-gait/
├── configs/                    # YAML configs for data, model, training
├── data/
│   ├── raw_video/              # Untouched capture sessions (gitignored)
│   ├── frames/                 # Extracted candidate frames for annotation
│   ├── annotations/            # COCO-format keypoint JSONs (train/val splits)
│   └── DATA_CARD.md            # Provenance, consent, device settings
├── notebooks/
│   ├── 01_frame_selection.ipynb
│   ├── 02_finetune_vitpose.ipynb   # Main training loop
│   ├── 03_eval_visualize.ipynb
│   └── 04_inference_demo.ipynb
├── src/walker_gait/
│   ├── data/                   # Dataset, augmentation, target generation
│   ├── models/                 # ViTPose wrapper, freeze logic, loss
│   ├── pipeline/               # Inference pipeline (detector → pose → filter)
│   ├── training/               # Train/eval loops, metrics
│   └── utils/                  # I/O, visualization helpers
├── scripts/
│   ├── extract_frames.py       # Video → frame extraction
│   ├── train.py                # SLURM-compatible mirror of notebook 02
│   └── batch_inference.py      # Offline batch processing
├── checkpoints/                # Saved weights (gitignored)
├── environment/
│   └── environment.yml         # Conda env spec
└── tests/
```

## Quickstart

### 1. Create the environment

```bash
conda env create -f environment/environment.yml
conda activate walker-gait
```

### 2. Collect & annotate data

See `data/DATA_CARD.md` for the capture protocol. Use CVAT or Label Studio
with the 12-keypoint skeleton defined in `configs/data.yaml`.

### 3. Fine-tune

Open `notebooks/02_finetune_vitpose.ipynb` or run the script:

```bash
python scripts/train.py --config configs/train.yaml
```

### 4. Batch inference on SHARCNET

```bash
sbatch scripts/sharcnet_job.sh
```

## Keypoint Definition

We use 12 keypoints extracted from the 133-point COCO-WholeBody skeleton:

| Index (in our 12) | Name            | COCO-WB Index |
|--------------------|-----------------|---------------|
| 0                  | left_hip        | 11            |
| 1                  | right_hip       | 12            |
| 2                  | left_knee       | 13            |
| 3                  | right_knee      | 14            |
| 4                  | left_ankle      | 15            |
| 5                  | right_ankle     | 16            |
| 6                  | left_big_toe    | 17            |
| 7                  | left_small_toe  | 18            |
| 8                  | left_heel       | 19            |
| 9                  | right_big_toe   | 20            |
| 10                 | right_small_toe | 21            |
| 11                 | right_heel      | 22            |

## Hardware

- **Walker**: Evolution Mini Trillium rollator
- **Camera**: Orbbec Femto Bolt (WFOV 120°×120°, iToF depth)
- **Mount**: SmallRig 4862 super clamp + magic arm, bar below seat
