"""2D pixel + depth -> 3D world coordinates.

Camera frame (OpenCV convention): +x right, +y down, +z forward (out of lens).
World frame: +x lateral (right), +y forward (direction of travel), +z up
(height above floor, 0 at the floor).
"""

from __future__ import annotations

import numpy as np


def pixel_to_camera(u, v, depth_mm, fx, fy, cx, cy):
    """Backproject pixel coords + depth (mm) to camera-frame 3D coords (mm)."""
    z = depth_mm
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return np.stack([x, y, z], axis=-1)


def world_from_camera(points_camera, tilt_deg, floor_offset_mm):
    """Rotate camera-frame points into the world frame (z=0 at the floor).

    The camera is mounted looking down at the walker user's lower body,
    tilted `tilt_deg` below horizontal. `floor_offset_mm` is the vertical
    distance from the camera mount down to the floor, measured along the
    untilted (horizontal-camera) vertical axis.

    points_camera: (..., 3) in mm
    """
    theta = np.deg2rad(tilt_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    x = points_camera[..., 0]
    y = points_camera[..., 1]
    z = points_camera[..., 2]

    world_x = x
    world_y = y * cos_t + z * sin_t
    # Distance below the camera mount along the untilted vertical axis.
    below_mount_mm = -y * sin_t + z * cos_t
    world_z = floor_offset_mm - below_mount_mm

    return np.stack([world_x, world_y, world_z], axis=-1)


def backproject_sequence(
    keypoints_2d: np.ndarray,
    depth_frames: np.ndarray,
    intrinsics: dict,
    tilt_deg: float,
    floor_offset_mm: float,
    score_threshold: float = 0.3,
) -> np.ndarray:
    """Backproject a 2D keypoint sequence to 3D world coordinates.

    Args:
        keypoints_2d: (T, K, 3) array of (x_px, y_px, score) per frame/keypoint.
        depth_frames: (T, H, W) array, depth values in mm (0 = invalid).
        intrinsics: dict with fx, fy, cx, cy.
        tilt_deg: camera downward tilt from horizontal, degrees.
        floor_offset_mm: vertical distance from camera mount to floor, mm.
        score_threshold: keypoints below this confidence are left as NaN.

    Returns:
        (T, K, 3) world coords in mm; NaN where the keypoint/depth was invalid.
    """
    T, K, _ = keypoints_2d.shape
    fx, fy, cx, cy = intrinsics["fx"], intrinsics["fy"], intrinsics["cx"], intrinsics["cy"]
    H, W = depth_frames.shape[1], depth_frames.shape[2]

    world = np.full((T, K, 3), np.nan, dtype=np.float64)

    # Loop over frames and keypoints, backprojecting valid ones to 3D world coordinates.
    for t in range(T):
        depth_frame = depth_frames[t]
        for k in range(K):
            u, v, score = keypoints_2d[t, k]
            if np.isnan(score) or score < score_threshold:
                continue
            if np.isnan(u) or np.isnan(v):
                continue
            ui, vi = int(round(u)), int(round(v)))
            if not (0 <= ui < W and 0 <= vi < H):
                continue
            depth_mm = float(depth_frame[vi, ui])
            if depth_mm <= 0:
                continue
            cam_pt = pixel_to_camera(u, v, depth_mm, fx, fy, cx, cy)
            world[t, k] = world_from_camera(cam_pt, tilt_deg, floor_offset_mm)

    return world
