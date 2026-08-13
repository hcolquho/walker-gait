"""
Data augmentation for top-down keypoint training.

All transforms operate on (image, keypoints, visibility) tuples
in the original image coordinate space, BEFORE the affine crop.
"""

import numpy as np
import cv2


def random_horizontal_flip(
    image: np.ndarray,
    keypoints_xy: np.ndarray,
    visibility: np.ndarray,
    flip_pairs: list[tuple[int, int]],
    prob: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Randomly flip image and swap left/right keypoints.

    Args:
        image: (H, W, 3) image.
        keypoints_xy: (K, 2) keypoint coordinates.
        visibility: (K,) visibility flags.
        flip_pairs: List of (left_idx, right_idx) to swap.
        prob: Flip probability.

    Returns:
        (image, keypoints, visibility) — possibly flipped.
    """
    if np.random.random() > prob:
        return image, keypoints_xy, visibility

    h, w = image.shape[:2]
    image = cv2.flip(image, 1)  # horizontal flip

    kp = keypoints_xy.copy()
    vis = visibility.copy()
    kp[:, 0] = w - 1 - kp[:, 0]

    # Swap left ↔ right
    for l_idx, r_idx in flip_pairs:
        kp[[l_idx, r_idx]] = kp[[r_idx, l_idx]]
        vis[[l_idx, r_idx]] = vis[[r_idx, l_idx]]

    return image, kp, vis


def random_scale_rotation(
    center: np.ndarray,
    scale: np.ndarray,
    scale_range: tuple[float, float] = (0.75, 1.25),
    rotation_range: float = 30.0,
) -> tuple[np.ndarray, float]:
    """Randomly perturb scale and rotation for data augmentation.

    Args:
        center: (2,) bounding box center (not modified).
        scale: (2,) normalized scale — will be multiplied.
        scale_range: (min, max) scale factor.
        rotation_range: Max rotation in degrees (±).

    Returns:
        (new_scale, rotation_angle_deg).
    """
    sf = np.random.uniform(*scale_range)
    new_scale = scale * sf
    rot = np.random.uniform(-rotation_range, rotation_range)
    return new_scale, rot


def apply_rotation_to_affine(
    center: np.ndarray,
    scale: np.ndarray,
    rotation_deg: float,
    output_size: tuple[int, int],
    pixel_std: float = 200.0,
) -> np.ndarray:
    """Build affine transform with rotation.

    Like get_affine_transform but with an added rotation around center.
    """
    src_w = scale[0] * pixel_std
    dst_w, dst_h = output_size

    rot_rad = np.deg2rad(rotation_deg)
    cos_r, sin_r = np.cos(rot_rad), np.sin(rot_rad)

    src_dir = np.array([0, -src_w * 0.5], dtype=np.float32)
    rotated_dir = np.array([
        src_dir[0] * cos_r - src_dir[1] * sin_r,
        src_dir[0] * sin_r + src_dir[1] * cos_r,
    ], dtype=np.float32)
    rotated_perp = np.array([-rotated_dir[1], rotated_dir[0]], dtype=np.float32)

    src = np.float32([
        center,
        center + rotated_dir,
        center + rotated_perp,
    ])
    dst = np.float32([
        [dst_w / 2.0, dst_h / 2.0],
        [dst_w / 2.0, 0],
        [0, dst_h / 2.0],
    ])

    return cv2.getAffineTransform(src, dst)
