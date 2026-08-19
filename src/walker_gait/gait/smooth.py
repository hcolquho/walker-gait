"""Gap-filling and low-pass smoothing for 3D keypoint trajectories."""

import numpy as np
from scipy.signal import butter, filtfilt


def _kalman_fill_1d(
    series: np.ndarray,
    max_gap_frames: int,
    process_var: float = 1e-2,
    measurement_var: float = 1.0,
) -> np.ndarray:
    """Constant-velocity Kalman filter over a single 1D series (one keypoint,
    one coordinate axis). Predicts through NaN runs up to `max_gap_frames`
    long; longer runs, and any leading NaNs before the first observation,
    are left as NaN."""
    T = len(series)
    filled = series.copy()
    observed = ~np.isnan(series)
    if observed.sum() < 2:
        return filled

    dt = 1.0
    F = np.array([[1.0, dt], [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    Q = np.eye(2) * process_var
    R = np.array([[measurement_var]])
    I = np.eye(2)

    start = int(np.argmax(observed))
    x = np.array([series[start], 0.0])
    P = np.eye(2)

    gap_len = 0
    for t in range(start, T):
        x = F @ x
        P = F @ P @ F.T + Q

        if observed[t]:
            gap_len = 0
            innovation = series[t] - (H @ x)[0]
            S = (H @ P @ H.T + R)[0, 0]
            K = (P @ H.T).flatten() / S
            x = x + K * innovation
            P = (I - np.outer(K, H)) @ P
            filled[t] = x[0]
        else:
            gap_len += 1
            filled[t] = x[0] if gap_len <= max_gap_frames else np.nan

    return filled


def kalman_fill_gaps(keypoints_3d: np.ndarray, max_gap_frames: int = 10) -> np.ndarray:
    """
    For each keypoint, fill NaN gaps up to max_gap_frames using
    a constant-velocity Kalman filter prediction. Gaps longer than
    max_gap_frames (and leading NaNs before the first observation)
    are left as NaN.
    keypoints_3d: (T, K, 3)
    """
    _, K, D = keypoints_3d.shape
    filled = keypoints_3d.copy()

    for k in range(K):
        for dim in range(D):
            filled[:, k, dim] = _kalman_fill_1d(
                keypoints_3d[:, k, dim], max_gap_frames=max_gap_frames
            )

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