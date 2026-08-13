"""
Visualization utilities for keypoint overlay on images.
"""

import numpy as np
import cv2


# Skeleton edges in our 12-keypoint space
SKELETON_EDGES = [
    (0, 2),   # left_hip → left_knee
    (2, 4),   # left_knee → left_ankle
    (4, 6),   # left_ankle → left_big_toe
    (4, 7),   # left_ankle → left_small_toe
    (4, 8),   # left_ankle → left_heel
    (1, 3),   # right_hip → right_knee
    (3, 5),   # right_knee → right_ankle
    (5, 9),   # right_ankle → right_big_toe
    (5, 10),  # right_ankle → right_small_toe
    (5, 11),  # right_ankle → right_heel
    (0, 1),   # left_hip → right_hip
]

# Colors: left side = blue, right side = red, hip bridge = green
EDGE_COLORS = [
    (255, 128, 0),    # left_hip → left_knee
    (255, 128, 0),    # left_knee → left_ankle
    (255, 200, 0),    # left_ankle → left_big_toe
    (255, 200, 0),    # left_ankle → left_small_toe
    (255, 200, 0),    # left_ankle → left_heel
    (0, 128, 255),    # right_hip → right_knee
    (0, 128, 255),    # right_knee → right_ankle
    (0, 200, 255),    # right_ankle → right_big_toe
    (0, 200, 255),    # right_ankle → right_small_toe
    (0, 200, 255),    # right_ankle → right_heel
    (0, 255, 0),      # left_hip → right_hip
]

KEYPOINT_NAMES = [
    "L_Hip", "R_Hip", "L_Knee", "R_Knee",
    "L_Ankle", "R_Ankle",
    "L_BigToe", "L_SmToe", "L_Heel",
    "R_BigToe", "R_SmToe", "R_Heel",
]


def draw_skeleton(
    image: np.ndarray,
    keypoints: np.ndarray,
    scores: np.ndarray | None = None,
    score_threshold: float = 0.3,
    radius: int = 4,
    thickness: int = 2,
    show_labels: bool = False,
) -> np.ndarray:
    """Draw skeleton overlay on an image.

    Args:
        image: (H, W, 3) BGR or RGB image (will be drawn on in-place).
        keypoints: (12, 2) keypoint coordinates.
        scores: (12,) confidence scores, or None to draw all.
        score_threshold: Minimum score to draw a keypoint.
        radius: Keypoint circle radius.
        thickness: Edge line thickness.
        show_labels: If True, draw keypoint name labels.

    Returns:
        Image with skeleton overlay.
    """
    vis = image.copy()

    if scores is None:
        scores = np.ones(len(keypoints))

    # Draw edges first (under points)
    for (i, j), color in zip(SKELETON_EDGES, EDGE_COLORS):
        if scores[i] > score_threshold and scores[j] > score_threshold:
            pt1 = tuple(keypoints[i].astype(int))
            pt2 = tuple(keypoints[j].astype(int))
            cv2.line(vis, pt1, pt2, color, thickness)

    # Draw keypoints
    for k in range(len(keypoints)):
        if scores[k] > score_threshold:
            pt = tuple(keypoints[k].astype(int))
            cv2.circle(vis, pt, radius, (255, 255, 255), -1)
            cv2.circle(vis, pt, radius, (0, 0, 0), 1)

            if show_labels:
                cv2.putText(
                    vis, KEYPOINT_NAMES[k],
                    (pt[0] + 5, pt[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1,
                )

    return vis


def draw_keypoints_on_image(
    image: np.ndarray,
    all_keypoints: np.ndarray,
    all_scores: np.ndarray | None = None,
    score_threshold: float = 0.3,
) -> np.ndarray:
    """Draw skeletons for all detected persons.

    Args:
        image: (H, W, 3) image.
        all_keypoints: (N, 12, 2) keypoints for N persons.
        all_scores: (N, 12) scores, or None.

    Returns:
        Image with all skeletons drawn.
    """
    vis = image.copy()
    n_persons = len(all_keypoints)

    for i in range(n_persons):
        scores_i = all_scores[i] if all_scores is not None else None
        vis = draw_skeleton(
            vis, all_keypoints[i], scores_i,
            score_threshold=score_threshold,
        )

    return vis
