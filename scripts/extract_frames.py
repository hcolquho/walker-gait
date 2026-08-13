#!/usr/bin/env python3
"""
Extract candidate frames from raw capture videos.

Pulls frames at a target FPS and filters out blurry ones (below a
Laplacian-variance threshold). Outputs numbered PNGs into the frames
directory for downstream annotation.

Usage:
    python scripts/extract_frames.py \
        --input data/raw_video/ \
        --output data/frames/ \
        --fps 5 \
        --blur-threshold 100
"""

import argparse
from pathlib import Path

import cv2


def laplacian_variance(image_gray: "np.ndarray") -> float:
    """Compute Laplacian variance as a blur metric (higher = sharper)."""
    return cv2.Laplacian(image_gray, cv2.CV_64F).var()


def extract_frames(
    video_path: Path,
    output_dir: Path,
    target_fps: float = 5.0,
    blur_threshold: float = 100.0,
    prefix: str = "",
) -> int:
    """Extract frames from a single video file.

    Returns:
        Number of frames saved.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [SKIP] Cannot open {video_path}")
        return 0

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0:
        video_fps = 30.0  # fallback

    frame_interval = max(1, int(round(video_fps / target_fps)))
    frame_idx = 0
    saved = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur_score = laplacian_variance(gray)

            if blur_score >= blur_threshold:
                fname = f"{prefix}frame_{frame_idx:06d}.png"
                cv2.imwrite(str(output_dir / fname), frame)
                saved += 1

        frame_idx += 1

    cap.release()
    return saved


def main():
    parser = argparse.ArgumentParser(description="Extract frames from videos")
    parser.add_argument("--input", type=str, required=True,
                        help="Directory containing raw video files")
    parser.add_argument("--output", type=str, required=True,
                        help="Directory to save extracted frames")
    parser.add_argument("--fps", type=float, default=5.0,
                        help="Target extraction FPS (default: 5)")
    parser.add_argument("--blur-threshold", type=float, default=100.0,
                        help="Laplacian variance threshold (default: 100)")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_extensions = {".mp4", ".avi", ".mkv", ".mov", ".bag"}
    videos = sorted(
        p for p in input_dir.iterdir()
        if p.suffix.lower() in video_extensions
    )

    if not videos:
        print(f"No video files found in {input_dir}")
        return

    total_saved = 0
    for video in videos:
        prefix = f"{video.stem}_"
        print(f"Processing {video.name}...")
        n = extract_frames(
            video, output_dir, args.fps, args.blur_threshold, prefix
        )
        print(f"  Saved {n} frames")
        total_saved += n

    print(f"\nTotal: {total_saved} frames saved to {output_dir}")


if __name__ == "__main__":
    main()
