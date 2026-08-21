"""Deterministic 3D gait analysis: backprojection, smoothing, event detection, metrics."""

from .backproject import backproject_sequence, pixel_to_camera, world_from_camera
from .smooth import butterworth_smooth, kalman_fill_gaps, smooth_keypoint_sequence
from .event_detector import KEYPOINT_NAMES, GaitEvents, SideEvents, detect_gait_events
from .metrics import GaitMetrics, compute_gait_metrics

__all__ = [
    "backproject_sequence",
    "pixel_to_camera",
    "kalman_fill_gaps",
    "butterworth_smooth",
    "smooth_keypoint_sequence",
    "KEYPOINT_NAMES",
    "SideEvents",
    "GaitEvents",
    "detect_gait_events",
    "GaitMetrics",
    "compute_gait_metrics",
]
