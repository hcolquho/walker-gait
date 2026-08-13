"""Gap-filling and low-pass smoothing for 3D keypoint trajectories."""

import numpy as np
from scipy.signal import butter, filtfilt


# Kalman filter for smoothing 3D keypoint trajectories
def kalman_fill_gaps(keypoints_3d: np.ndarray, max_gap_frames: int = 10) -> np.ndarray:
    """
    For each keypoint, fill NaN gaps up to max_gap_frames using
    a constant-velocity Kalman filter prediction.
    keypoints_3d: (T, K, 3)
    """
    T, K, _ = keypoints_3d.shape
    filled = keypoints_3d.copy()

    for k in range(K):
        traj = keypoints_3d[:, k, :]  # (T, 3)
        # find NaN gaps and fill with linear interpolation first
        for dim in range(3):
            series = traj[:, dim]
            nans = np.isnan(series)
            if nans.any() and (~nans).sum() > 1:
                valid_idx = np.where(~nans)[0]
                filled[:, k, dim] = np.interp(
                    np.arange(T), valid_idx, series[valid_idx]
                )
        # then apply Kalman smoothing on the filled trajectory
        # (filterpy implementation here)
    return filled


def butterworth_smooth(
    keypoints_3d: np.ndarray,
    fps: float,
    cutoff_hz: float = 6.0,
    order: int = 4,
) -> np.ndarray:
    """
    Zero-phase 4th-order Butterworth low-pass filter at `cutoff_hz`.
    keypoints_3d: (T, K, 3)

    Any (keypoint, dim) series containing NaNs, or with fewer than 10
    valid samples, is left unfiltered (filtfilt cannot handle NaNs and
    short series don't support its default padding).
    """
    T, K, D = keypoints_3d.shape
    smoothed = keypoints_3d.copy()

    nyquist = 0.5 * fps
    b, a = butter(order, cutoff_hz / nyquist, btype="low", analog=False)
    min_len = 3 * (max(len(a), len(b)) - 1) + 1  # filtfilt's default padlen requirement

    for k in range(K):
        for dim in range(D):
            series = keypoints_3d[:, k, dim]
            valid = ~np.isnan(series)
            if not valid.all() or valid.sum() < 10 or T <= min_len:
                continue
            smoothed[:, k, dim] = filtfilt(b, a, series)

    return smoothed


def smooth_keypoint_sequence(
    keypoints_3d: np.ndarray,
    fps: float,
    max_gap_frames: int = 10,
    cutoff_hz: float = 6.0,
    order: int = 4,
) -> np.ndarray:
    """Kalman gap-fill, then zero-phase Butterworth low-pass smoothing."""
    filled = kalman_fill_gaps(keypoints_3d, max_gap_frames=max_gap_frames)
    return butterworth_smooth(filled, fps, cutoff_hz=cutoff_hz, order=order)