#!/usr/bin/env python3
"""
Batch inference on all recorded trials.

Recursively finds all trial folders under --input, runs pose inference
on each trial's color/ frames, and saves keypoints.json inside each trial folder.

Usage:
    python scripts/batch_inference.py \
        --input data/han/ \
        --model usyd-community/vitpose-plus-large
"""

import argparse
import json
import sys
from pathlib import Path

from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from walker_gait.pipeline import WalkerGaitPipeline


def find_trial_dirs(root: Path) -> list[Path]:
    """Recursively find all trial folders (contain a color/ subfolder)."""
    return sorted(p.parent for p in root.rglob("color") if p.is_dir())


def process_trial(trial_dir: Path, pipeline: WalkerGaitPipeline) -> int:
    """Run inference on one trial folder, save keypoints.json. Returns frame count."""
    color_dir = trial_dir / "color"
    output_path = trial_dir / "keypoints.json"

    # Skip if already done
    if output_path.exists():
        print(f"  [SKIP] {trial_dir.name} — keypoints.json already exists")
        return 0

    image_extensions = {".jpg", ".jpeg", ".png"}
    image_paths = sorted(
        p for p in color_dir.iterdir()
        if p.suffix.lower() in image_extensions
    )

    if not image_paths:
        print(f"  [WARN] No images in {color_dir}")
        return 0

    results = []
    for img_path in tqdm(image_paths, desc=f"  {trial_dir.name}", leave=False):
        image = Image.open(img_path).convert("RGB")
        pose = pipeline(image)

        frame_result = {
            "file_name": img_path.name,
            "persons": [],
        }

        for p_idx in range(len(pose.keypoints)):
            person = {
                "keypoints": {
                    name: {
                        "x": float(pose.keypoints[p_idx, k, 0]),
                        "y": float(pose.keypoints[p_idx, k, 1]),
                        "score": float(pose.scores[p_idx, k]),
                    }
                    for k, name in enumerate(pose.keypoint_names)
                },
                "bbox": pose.boxes[p_idx].tolist(),
            }
            frame_result["persons"].append(person)

        results.append(frame_result)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    return len(results)


def main():
    parser = argparse.ArgumentParser(description="Batch pose inference on all trials")
    parser.add_argument("--input",      type=str, required=True,
                        help="Root data directory (e.g. data/han/)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Fine-tuned checkpoint dir (or None for pretrained)")
    parser.add_argument("--model",      type=str,
                        default="usyd-community/vitpose-plus-large")
    parser.add_argument("--device",     type=str, default="cuda")
    args = parser.parse_args()

    input_dir = Path(args.input)

    # Find all trial folders
    trial_dirs = find_trial_dirs(input_dir)
    print(f"Found {len(trial_dirs)} trial folders under {input_dir}\n")

    if not trial_dirs:
        print("No trial folders found. Check that color/ subfolders exist.")
        return

    # Initialize pipeline once
    pipeline = WalkerGaitPipeline(
        pose_model_name=args.model,
        pose_checkpoint=args.checkpoint,
        device=args.device,
    )

    # Process each trial
    total_frames = 0
    for i, trial_dir in enumerate(trial_dirs, 1):
        print(f"[{i}/{len(trial_dirs)}] {trial_dir.relative_to(input_dir)}")
        n = process_trial(trial_dir, pipeline)
        total_frames += n
        print(f"  → {n} frames saved to keypoints.json")

    print(f"\nDone. {total_frames} total frames processed across {len(trial_dirs)} trials.")


if __name__ == "__main__":
    main()
