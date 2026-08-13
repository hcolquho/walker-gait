"""Heel-strike and toe-off event detection from smoothed 3D keypoint tracks.

Assumes world-frame keypoints where z = height above the floor (mm), as
produced by `backproject.world_from_camera`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks

KEYPOINT_NAMES = [
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "left_big_toe", "left_small_toe", "left_heel",
    "right_big_toe", "right_small_toe", "right_heel",
]
KP = {name: i for i, name in enumerate(KEYPOINT_NAMES)}


@dataclass
class SideEvents:
    heel_strikes: np.ndarray  # frame indices
    toe_offs: np.ndarray      # frame indices, one per heel strike (paired stance)


@dataclass
class GaitEvents:
    left: SideEvents
    right: SideEvents
    fps: float


def _detect_side_events(
    heel_z: np.ndarray,
    toe_z: np.ndarray,
    fps: float,
    min_step_interval_s: float,
    floor_tol_mm: float,
) -> SideEvents:
    """
    Heel strike: local minimum of heel height near the floor (foot
    decelerating into ground contact).
    Toe-off: the last near-floor sample of the toe within a foot's stance
    window (heel strike to next heel strike) -- i.e. push-off just before
    the toe leaves the ground for swing.
    """
    min_distance = max(1, int(round(min_step_interval_s * fps)))

    valid = ~(np.isnan(heel_z) | np.isnan(toe_z))
    if valid.sum() < min_distance * 2:
        return SideEvents(np.array([], dtype=int), np.array([], dtype=int))

    minima_idx, _ = find_peaks(-np.nan_to_num(heel_z, nan=np.inf), distance=min_distance)
    heel_strikes = np.array(
        [i for i in minima_idx if valid[i] and heel_z[i] < floor_tol_mm],
        dtype=int,
    )

    toe_offs = []
    kept_strikes = []
    for i, start in enumerate(heel_strikes):
        end = heel_strikes[i + 1] if i + 1 < len(heel_strikes) else len(toe_z)
        window_valid = valid[start:end]
        window_toe = toe_z[start:end]
        near_floor = (window_toe < floor_tol_mm) & window_valid
        if not near_floor.any():
            continue
        last_near_floor = np.where(near_floor)[0][-1]
        toe_offs.append(start + last_near_floor)
        kept_strikes.append(start)

    return SideEvents(
        heel_strikes=np.array(kept_strikes, dtype=int),
        toe_offs=np.array(toe_offs, dtype=int),
    )


def detect_gait_events(
    keypoints_3d: np.ndarray,
    fps: float,
    min_step_interval_s: float = 0.3,
    floor_tol_mm: float = 50.0,
) -> GaitEvents:
    """
    keypoints_3d: (T, K, 3) smoothed world coords, mm, z = height above floor.
    Returns GaitEvents with per-side heel-strike and toe-off frame indices.
    """
    left = _detect_side_events(
        keypoints_3d[:, KP["left_heel"], 2],
        keypoints_3d[:, KP["left_big_toe"], 2],
        fps, min_step_interval_s, floor_tol_mm,
    )
    right = _detect_side_events(
        keypoints_3d[:, KP["right_heel"], 2],
        keypoints_3d[:, KP["right_big_toe"], 2],
        fps, min_step_interval_s, floor_tol_mm,
    )
    return GaitEvents(left=left, right=right, fps=fps)
