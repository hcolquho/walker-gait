import numpy as np
import pytest

from walker_gait.gait import (
    backproject_sequence,
    butterworth_smooth,
    compute_gait_metrics,
    detect_gait_events,
    kalman_fill_gaps,
    smooth_keypoint_sequence,
    world_from_camera,
    KEYPOINT_NAMES,
)

KP = {name: i for i, name in enumerate(KEYPOINT_NAMES)}


# --- backproject ---------------------------------------------------------

def test_world_from_camera_no_tilt():
    # Point straight ahead of an untilted camera, 2000mm out: at zero tilt,
    # camera z (forward) maps to "below the mount", not world-forward.
    point_camera = np.array([0.0, 0.0, 2000.0])
    world = world_from_camera(point_camera, tilt_deg=0.0, floor_offset_mm=900.0)
    assert world == pytest.approx([0.0, 0.0, 900.0 - 2000.0])


def test_world_from_camera_90deg_tilt():
    # At 90deg tilt (camera facing straight down), camera z becomes
    # world-forward and camera y becomes "below the mount".
    point_camera = np.array([0.0, 0.0, 2000.0])
    world = world_from_camera(point_camera, tilt_deg=90.0, floor_offset_mm=900.0)
    assert world == pytest.approx([0.0, 2000.0, 900.0], abs=1e-9)


def test_backproject_sequence_shape_and_nan_masking():
    T, K, H, W = 5, len(KEYPOINT_NAMES), 100, 100
    keypoints_2d = np.zeros((T, K, 3))
    keypoints_2d[..., 0] = 50  # u
    keypoints_2d[..., 1] = 50  # v
    keypoints_2d[..., 2] = 0.9  # score, above threshold

    # Low-confidence keypoint should be dropped.
    keypoints_2d[0, 0, 2] = 0.1

    depth_frames = np.full((T, H, W), 1000, dtype=np.uint16)
    intrinsics = {"fx": 500.0, "fy": 500.0, "cx": 50.0, "cy": 50.0}

    world = backproject_sequence(
        keypoints_2d, depth_frames, intrinsics, tilt_deg=0.0, floor_offset_mm=900.0
    )

    assert world.shape == (T, K, 3)
    assert np.isnan(world[0, 0]).all()
    assert not np.isnan(world[0, 1]).any()


# --- smooth ---------------------------------------------------------------

def test_butterworth_smooth_reduces_noise():
    fps = 30.0
    t = np.arange(200) / fps
    clean = np.sin(2 * np.pi * 0.5 * t)  # slow 0.5Hz signal, well under 6Hz cutoff
    rng = np.random.default_rng(0)
    noisy = clean + rng.normal(scale=0.3, size=t.shape)

    keypoints_3d = np.zeros((len(t), 1, 3))
    keypoints_3d[:, 0, 0] = noisy

    smoothed = butterworth_smooth(keypoints_3d, fps=fps)

    error_before = np.mean((noisy - clean) ** 2)
    error_after = np.mean((smoothed[:, 0, 0] - clean) ** 2)
    assert error_after < error_before


def test_butterworth_smooth_skips_nan_and_short_series():
    fps = 30.0
    T = 200
    keypoints_3d = np.zeros((T, 2, 3))
    keypoints_3d[:, 0, 0] = np.sin(np.linspace(0, 10, T))
    keypoints_3d[5, 0, 1] = np.nan  # has NaNs -> skipped
    keypoints_3d[:5, 1, 2] = 1.0  # < 10 valid samples -> skipped (rest is 0, not nan)
    keypoints_3d[5:, 1, 2] = np.nan

    smoothed = butterworth_smooth(keypoints_3d, fps=fps)

    assert np.isnan(smoothed[5, 0, 1])
    np.testing.assert_array_equal(smoothed[:, 1, 2], keypoints_3d[:, 1, 2])


def test_smooth_keypoint_sequence_fills_and_smooths():
    fps = 30.0
    T = 200
    keypoints_3d = np.zeros((T, 1, 3))
    keypoints_3d[:, 0, 0] = np.sin(np.linspace(0, 10, T))
    keypoints_3d[20:25, 0, 0] = np.nan  # small gap

    out = smooth_keypoint_sequence(keypoints_3d, fps=fps)

    assert out.shape == keypoints_3d.shape
    assert not np.isnan(out[:, 0, 0]).any()


# --- event detection + metrics --------------------------------------------

def _make_synthetic_walk(fps=30.0, n_cycles=6, period_frames=30, speed_mm_per_frame=20.0):
    """Two feet touching down in alternating phase, walking forward at constant speed."""
    T = n_cycles * period_frames
    keypoints_3d = np.full((T, len(KEYPOINT_NAMES), 3), np.nan)
    t = np.arange(T)

    def foot_height(phase_frames):
        return 100.0 * (1 - np.cos(2 * np.pi * (t - phase_frames) / period_frames))

    forward = speed_mm_per_frame * t

    left_z = foot_height(0)
    right_z = foot_height(period_frames / 2)

    for name in ("left_heel", "left_big_toe"):
        keypoints_3d[:, KP[name], 1] = forward
        keypoints_3d[:, KP[name], 2] = left_z
    for name in ("right_heel", "right_big_toe"):
        keypoints_3d[:, KP[name], 1] = forward
        keypoints_3d[:, KP[name], 2] = right_z

    return keypoints_3d, fps


def test_detect_gait_events_finds_alternating_strikes():
    keypoints_3d, fps = _make_synthetic_walk()
    events = detect_gait_events(keypoints_3d, fps=fps)

    assert len(events.left.heel_strikes) >= 4
    assert len(events.right.heel_strikes) >= 4
    assert len(events.left.toe_offs) == len(events.left.heel_strikes)
    assert len(events.right.toe_offs) == len(events.right.heel_strikes)


def test_compute_gait_metrics_on_symmetric_walk():
    keypoints_3d, fps = _make_synthetic_walk()
    events = detect_gait_events(keypoints_3d, fps=fps)
    metrics = compute_gait_metrics(events, keypoints_3d, fps=fps)

    assert metrics.left_step_count > 0
    assert metrics.right_step_count > 0
    assert abs(metrics.left_step_count - metrics.right_step_count) <= 1
    assert metrics.cadence_steps_per_min > 0
    assert not np.isnan(metrics.stride_time_s)

    d = metrics.to_dict()
    assert d["left_step_count"] == metrics.left_step_count
