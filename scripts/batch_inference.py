#!/usr/bin/env python3
"""
Batch inference on recorded frames/video.

Runs the full pipeline (detect → pose → extract 12 keypoints) on all
frames in an input directory and saves results as a JSON file.

Usage:
    python scripts/batch_inference.py \
        --input data/frames/ \
        --output results/session_001.json \
        --checkpoint checkpoints/best
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from walker_gait.pipeline import WalkerGaitPipeline


def main():
    parser = argparse.ArgumentParser(description="Batch pose inference")
    parser.add_argument("--input", type=str, required=True,
                        help="Directory of frame images")
    parser.add_argument("--output", type=str, required=True,
                        help="Output JSON path")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Fine-tuned checkpoint dir (or None for pretrained)")
    parser.add_argument("--model", type=str,
                        default="usyd-community/vitpose-plus-large")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--threshold", type=float, default=0.3,
                        help="Detection confidence threshold")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Initialize pipeline
    pipeline = WalkerGaitPipeline(
        pose_model_name=args.model,
        pose_checkpoint=args.checkpoint,
        device=args.device,
        person_threshold=args.threshold,
    )

    # Find images
    image_extensions = {".png", ".jpg", ".jpeg", ".bmp"}
    image_paths = sorted(
        p for p in input_dir.iterdir()
        if p.suffix.lower() in image_extensions
    )
    print(f"Found {len(image_paths)} images in {input_dir}")

    # Process
    results = []
    for img_path in tqdm(image_paths, desc="Inference"):
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

    # Save
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved {len(results)} frame results to {output_path}")


if __name__ == "__main__":
    main()
