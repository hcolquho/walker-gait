"""Deterministic clinical gait metrics computed from detected gait events."""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from .event_detector import GaitEvents, KP


def _symmetry_index(left: float, right: float) -> float:
    """Robinson (1987) symmetry index, as a percentage."""
    if np.isnan(left) or np.isnan(right) or (left + right) == 0:
        return float("nan")
    return abs(left - right) / (0.5 * (left + right)) * 100.0


@dataclass
class GaitMetrics:
    stride_time_s: float
    left_step_count: int
    right_step_count: int
    cadence_steps_per_min: float
    step_time_asymmetry_pct: float
    step_length_asymmetry_pct: float
    double_support_time_s: float
    asymmetry_score: float

    def to_dict(self) -> dict:
        return asdict(self)


def _mean_interval_s(frame_idx: np.ndarray, fps: float) -> float:
    if len(frame_idx) < 2:
        return float("nan")
    return float(np.mean(np.diff(frame_idx)) / fps)


def _step_times_by_landing_side(left_strikes, right_strikes, fps):
    """Step time = interval ending in a heel strike, bucketed by which foot lands."""
    tagged = sorted(
        [(int(f), "L") for f in left_strikes] + [(int(f), "R") for f in right_strikes]
    )
    left_times, right_times = [], []
    for (f0, s0), (f1, s1) in zip(tagged, tagged[1:]):
        if s0 == s1:
            continue
        dt = (f1 - f0) / fps
        (left_times if s1 == "L" else right_times).append(dt)
    return left_times, right_times


def _mean_step_length_mm(landing_strikes, keypoints_3d, landing_kp, trailing_kp) -> float:
    """Mean forward-axis (y) distance between the landing foot and the
    trailing foot's heel position at each landing-foot heel strike."""
    lengths = []
    for f in landing_strikes:
        pt_land = keypoints_3d[f, landing_kp]
        pt_trail = keypoints_3d[f, trailing_kp]
        if np.isnan(pt_land).any() or np.isnan(pt_trail).any():
            continue
        lengths.append(abs(pt_land[1] - pt_trail[1]))
    return float(np.mean(lengths)) if lengths else float("nan")


def _double_support(events: GaitEvents, fps: float):
    """Mean duration + asymmetry of bilateral floor contact (both feet in stance)."""
    left_intervals = list(zip(events.left.heel_strikes, events.left.toe_offs))
    right_intervals = list(zip(events.right.heel_strikes, events.right.toe_offs))

    left_led, right_led = [], []
    for l_start, l_end in left_intervals:
        for r_start, r_end in right_intervals:
            overlap = min(l_end, r_end) - max(l_start, r_start)
            if overlap > 0:
                dt = overlap / fps
                (left_led if l_start <= r_start else right_led).append(dt)

    all_overlaps = left_led + right_led
    ds_time = float(np.mean(all_overlaps)) if all_overlaps else float("nan")
    mean_left = float(np.mean(left_led)) if left_led else float("nan")
    mean_right = float(np.mean(right_led)) if right_led else float("nan")
    return ds_time, _symmetry_index(mean_left, mean_right)


def compute_gait_metrics(events: GaitEvents, keypoints_3d: np.ndarray, fps: float) -> GaitMetrics:
    left_strikes = events.left.heel_strikes
    right_strikes = events.right.heel_strikes

    left_stride_time = _mean_interval_s(left_strikes, fps)
    right_stride_time = _mean_interval_s(right_strikes, fps)
    stride_times = [t for t in (left_stride_time, right_stride_time) if not np.isnan(t)]
    stride_time_s = float(np.mean(stride_times)) if stride_times else float("nan")

    left_step_count = int(len(left_strikes))
    right_step_count = int(len(right_strikes))
    total_steps = left_step_count + right_step_count

    duration_min = keypoints_3d.shape[0] / fps / 60.0
    cadence = total_steps / duration_min if duration_min > 0 else float("nan")

    left_step_times, right_step_times = _step_times_by_landing_side(left_strikes, right_strikes, fps)
    mean_left_step_time = float(np.mean(left_step_times)) if left_step_times else float("nan")
    mean_right_step_time = float(np.mean(right_step_times)) if right_step_times else float("nan")
    step_time_asymmetry_pct = _symmetry_index(mean_left_step_time, mean_right_step_time)

    left_step_length = _mean_step_length_mm(left_strikes, keypoints_3d, KP["left_heel"], KP["right_heel"])
    right_step_length = _mean_step_length_mm(right_strikes, keypoints_3d, KP["right_heel"], KP["left_heel"])
    step_length_asymmetry_pct = _symmetry_index(left_step_length, right_step_length)

    double_support_time_s, ds_asymmetry_pct = _double_support(events, fps)

    components = [
        v for v in (step_time_asymmetry_pct, step_length_asymmetry_pct, ds_asymmetry_pct)
        if not np.isnan(v)
    ]
    asymmetry_score = float(np.mean(components)) if components else float("nan")

    return GaitMetrics(
        stride_time_s=stride_time_s,
        left_step_count=left_step_count,
        right_step_count=right_step_count,
        cadence_steps_per_min=float(cadence),
        step_time_asymmetry_pct=step_time_asymmetry_pct,
        step_length_asymmetry_pct=step_length_asymmetry_pct,
        double_support_time_s=double_support_time_s,
        asymmetry_score=asymmetry_score,
    )
