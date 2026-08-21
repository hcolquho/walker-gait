#!/usr/bin/env python3
"""
Run the full offline gait pipeline on one recorded trial:

    2D keypoints (keypoints.json from batch_inference.py)
        -> backproject to 3D world coords using depth frames
        -> Kalman gap-fill + Butterworth smoothing
        -> heel-strike / toe-off event detection
        -> 7 clinical gait metrics

Saves gait_metrics.json inside the trial folder.

Usage:
    python scripts/process_session.py \
        --session data/han/session_01/block1_healthy_baseline/comfortable_right/trial_01 \
        --tilt-deg 35 \
        --floor-offset-mm 950
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
    world_from_camera,
    KEYPOINT_NAMES,
)
from walker_gait.gait.backproject import load_intrinsics


def load_keypoints_2d(keypoints_json_path: Path) -> tuple[np.ndarray, list[str]]:
    """Load per-frame 2D keypoints from batch_inference.py output JSON.

    Returns:
        keypoints_2d: (T, K, 2) array of (x, y) pixel coords
        file_names:   list of frame filenames in order
    """
    with open(keypoints_json_path) as f:
        frame_results = json.load(f)

    T = len(frame_results)
    K = len(KEYPOINT_NAMES)
    keypoints_2d = np.full((T, K, 2), np.nan, dtype=np.float32)
    file_names = []

    for t, frame in enumerate(frame_results):
        file_names.append(frame["file_name"])
        persons = frame.get("persons", [])
        if not persons:
            continue
        # Single walker user — take highest mean confidence person
        person = max(
            persons,
            key=lambda p: np.mean([
                p["keypoints"][name]["score"]
                for name in KEYPOINT_NAMES
                if name in p["keypoints"]
            ]),
        )
        for k, name in enumerate(KEYPOINT_NAMES):
            if name in person["keypoints"]:
                kp = person["keypoints"][name]
                keypoints_2d[t, k] = [kp["x"], kp["y"]]

    return keypoints_2d, file_names


def load_depth_frames(depth_dir: Path, file_names: list[str]) -> np.ndarray:
    """Load depth .npy files matching the colour frame filenames."""
    frames = []
    for fname in file_names:
        npy_path = depth_dir / (Path(fname).stem + ".npy")
        if npy_path.exists():
            frames.append(np.load(npy_path))
        else:
            # If missing, fill with zeros (will produce NaN after backproject)
            frames.append(np.zeros((576, 640), dtype=np.uint16))
    return np.stack(frames, axis=0)


def get_analysis_frames(session_dir: Path) -> tuple[list[int], float]:
    """Read start/end frame indices from timing_marks_manual.json.

    Returns:
        (frame_indices, fps)
    """
    tm_path = session_dir / "timing_marks_manual.json"
    if not tm_path.exists():
        raise FileNotFoundError(
            f"timing_marks_manual.json not found in {session_dir}. "
            "Record a trial with Q pressed at the end of the walk."
        )
    with open(tm_path) as f:
        timing = json.load(f)

    window = timing.get("analysis_window", {})
    start = window.get("start_frame", 0)
    end   = window.get("end_frame")

    if end is None:
        raise ValueError("analysis_window.end_frame missing from timing_marks_manual.json")

    return list(range(start, end + 1)), 30.0


def main():
    parser = argparse.ArgumentParser(
        description="Run the full gait pipeline on one trial"
    )
    parser.add_argument("--session", type=str, required=True,
                        help="Trial folder path")
    parser.add_argument("--tilt-deg", type=float, default=30.0,
                        help="Camera downward tilt from horizontal (degrees)")
    parser.add_argument("--floor-offset-mm", type=float, default=900.0,
                        help="Vertical distance from camera to floor (mm)")
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()

    session_dir = Path(args.session)
    print(f"\nProcessing: {session_dir}\n{'─'*50}")

    # ── Load analysis window ──────────────────────────────────────────────────
    frame_indices, fps = get_analysis_frames(session_dir)
    fps = args.fps
    print(f"Analysis window: frames {frame_indices[0]} → {frame_indices[-1]} "
          f"({len(frame_indices)} frames)")

    # ── Load 2D keypoints from inference output ───────────────────────────────
    keypoints_json = session_dir / "keypoints.json"
    if not keypoints_json.exists():
        raise FileNotFoundError(
            f"keypoints.json not found in {session_dir}. "
            "Run batch_inference.py first."
        )

    all_keypoints_2d, all_file_names = load_keypoints_2d(keypoints_json)

    # Slice to analysis window
    keypoints_2d = all_keypoints_2d[frame_indices]    # (T, K, 2)
    file_names   = [all_file_names[i] for i in frame_indices]
    timestamps   = np.array(frame_indices, dtype=np.float32) / fps

    print(f"Keypoints loaded: {keypoints_2d.shape}")

    # ── Load depth frames ─────────────────────────────────────────────────────
    print("Loading depth frames...")
    depth_frames = load_depth_frames(session_dir / "depth", file_names)
    print(f"Depth frames loaded: {depth_frames.shape}")

    # ── Load intrinsics ───────────────────────────────────────────────────────
    intrinsics = load_intrinsics(session_dir / "intrinsics.json")

    # ── Backproject to 3D camera coordinates ─────────────────────────────────
    print("Backprojecting to 3D...")
    keypoints_3d_cam = backproject_sequence(keypoints_2d, depth_frames, intrinsics)

    # ── Convert to world frame ────────────────────────────────────────────────
    keypoints_3d = world_from_camera(
        keypoints_3d_cam,
        camera_tilt_deg=args.tilt_deg,
        floor_offset_mm=args.floor_offset_mm,
    )
    print(f"3D keypoints shape: {keypoints_3d.shape}")

    # ── Smooth ────────────────────────────────────────────────────────────────
    print("Smoothing (Kalman gap-fill + Butterworth)...")
    keypoints_3d = smooth_keypoint_sequence(keypoints_3d, fps=fps)

    # ── Event detection ───────────────────────────────────────────────────────
    print("Detecting gait events...")
    events = detect_gait_events(
        keypoints_3d,
        fps=fps,
        timestamps=timestamps,
    )
    print(f"  Left heel strikes : {len(events.left_heel_strikes)}")
    print(f"  Right heel strikes: {len(events.right_heel_strikes)}")
    print(f"  Left toe-offs     : {len(events.left_toe_offs)}")
    print(f"  Right toe-offs    : {len(events.right_toe_offs)}")

    # ── Gait metrics ──────────────────────────────────────────────────────────
    print("Computing gait metrics...")
    metrics = compute_gait_metrics(
        events,
        keypoints_3d,
        bout_duration_s=float(timestamps[-1] - timestamps[0]),
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    output_path = session_dir / "gait_metrics.json"
    with open(output_path, "w") as f:
        json.dump(metrics.to_dict(), f, indent=2)

    print(f"\nGait metrics saved to {output_path}")
    print(json.dumps(metrics.to_dict(), indent=2))


if __name__ == "__main__":
    main()
