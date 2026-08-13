#!/usr/bin/env python3
"""
Run the full offline gait pipeline on one recorded trial:

    2D keypoints (batch_inference.py output) + depth frames
        -> backproject to 3D world coords
        -> Kalman gap-fill + Butterworth smoothing
        -> heel-strike / toe-off event detection
        -> 7 clinical gait metrics

Saves `gait_metrics.json` inside the trial folder.

Usage:
    python scripts/process_session.py \
        --session data/raw_video/han/session_01/block1_healthy_baseline/comfortable_right/trial_01 \
        --keypoints-json results/session_01_trial_01.json \
        --tilt-deg 35 \
        --floor-offset-mm 950 \
        --fps 30
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from walker_gait.gait import (
    backproject_sequence,
    compute_gait_metrics,
    detect_gait_events,
    smooth_keypoint_sequence,
    KEYPOINT_NAMES,
)


def load_keypoints_2d(keypoints_json_path: Path, frame_indices: list[int]) -> np.ndarray:
    """Load per-frame 2D keypoints for our 12-keypoint subset from a
    batch_inference.py output JSON, indexed by frame position (row order)."""
    with open(keypoints_json_path) as f:
        frame_results = json.load(f)

    T = len(frame_indices)
    K = len(KEYPOINT_NAMES)
    keypoints_2d = np.full((T, K, 3), np.nan, dtype=np.float64)

    for t, frame_idx in enumerate(frame_indices):
        if frame_idx >= len(frame_results):
            continue
        persons = frame_results[frame_idx]["persons"]
        if not persons:
            continue
        # Single walker user expected in frame; take the highest-confidence person.
        person = max(
            persons,
            key=lambda p: np.mean([p["keypoints"][name]["score"] for name in KEYPOINT_NAMES]),
        )
        for k, name in enumerate(KEYPOINT_NAMES):
            kp = person["keypoints"][name]
            keypoints_2d[t, k] = [kp["x"], kp["y"], kp["score"]]

    return keypoints_2d


def load_depth_frames(depth_dir: Path, frame_indices: list[int]) -> np.ndarray:
    frames = [np.load(depth_dir / f"{frame_idx:06d}.npy") for frame_idx in frame_indices]
    return np.stack(frames, axis=0)


def main():
    parser = argparse.ArgumentParser(description="Run the full gait pipeline on one trial")
    parser.add_argument("--session", type=str, required=True,
                        help="Trial folder path (contains color/, depth/, intrinsics.json, "
                             "timing_marks_manual.json)")
    parser.add_argument("--keypoints-json", type=str, required=True,
                        help="batch_inference.py output JSON for this trial")
    parser.add_argument("--tilt-deg", type=float, required=True,
                        help="Camera downward tilt from horizontal, degrees")
    parser.add_argument("--floor-offset-mm", type=float, required=True,
                        help="Vertical distance from camera mount to floor, mm")
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()

    session_dir = Path(args.session)

    with open(session_dir / "timing_marks_manual.json") as f:
        timing_marks = json.load(f)
    window = timing_marks["analysis_window"]
    frame_indices = list(range(window["start_frame"], window["end_frame"] + 1))

    with open(session_dir / "intrinsics.json") as f:
        intrinsics = json.load(f)

    print(f"Loading {len(frame_indices)} frames from {session_dir}")
    keypoints_2d = load_keypoints_2d(Path(args.keypoints_json), frame_indices)
    depth_frames = load_depth_frames(session_dir / "depth", frame_indices)

    print("Backprojecting to 3D world coordinates...")
    keypoints_3d = backproject_sequence(
        keypoints_2d, depth_frames, intrinsics,
        tilt_deg=args.tilt_deg, floor_offset_mm=args.floor_offset_mm,
    )

    print("Smoothing (Kalman gap-fill + Butterworth)...")
    keypoints_3d = smooth_keypoint_sequence(keypoints_3d, fps=args.fps)

    print("Detecting gait events...")
    events = detect_gait_events(keypoints_3d, fps=args.fps)

    print("Computing gait metrics...")
    metrics = compute_gait_metrics(events, keypoints_3d, fps=args.fps)

    output_path = session_dir / "gait_metrics.json"
    with open(output_path, "w") as f:
        json.dump(metrics.to_dict(), f, indent=2)

    print(f"Saved gait metrics to {output_path}")


if __name__ == "__main__":
    main()
