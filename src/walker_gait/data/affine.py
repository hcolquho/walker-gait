"""
Affine transform utilities for top-down pose estimation.

Replicates the same crop/warp the HuggingFace VitPoseImageProcessor applies
at inference time, so that ground-truth keypoints land in the same coordinate
frame as the model's heatmap output during training.
"""

import numpy as np
import cv2


def box_to_center_scale(
    box_xywh: np.ndarray,
    input_size: tuple[int, int],
    pixel_std: float = 200.0,
    padding: float = 1.25,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a COCO-format bounding box to (center, scale).

    Args:
        box_xywh: (4,) array [x, y, w, h].
        input_size: (H, W) of model input (e.g. (256, 192)).
        pixel_std: Normalization constant (matches HF processor default).
        padding: Scale padding factor.

    Returns:
        center: (2,) float32 array.
        scale: (2,) float32 array.
    """
    x, y, w, h = box_xywh
    center = np.array([x + w / 2.0, y + h / 2.0], dtype=np.float32)

    aspect_ratio = input_size[1] / input_size[0]  # W / H
    if w > aspect_ratio * h:
        h = w / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio

    scale = np.array([w / pixel_std, h / pixel_std], dtype=np.float32) * padding
    return center, scale


def get_affine_transform(
    center: np.ndarray,
    scale: np.ndarray,
    output_size: tuple[int, int],
    pixel_std: float = 200.0,
    inv: bool = False,
) -> np.ndarray:
    """Compute affine transform matrix from center/scale to output_size.

    Args:
        center: (2,) center of the bounding box.
        scale: (2,) normalized scale.
        output_size: (W, H) of the target image.
        pixel_std: Normalization constant.
        inv: If True, compute inverse transform (output→input coords).

    Returns:
        (2, 3) affine transformation matrix.
    """
    src_w = scale[0] * pixel_std
    dst_w, dst_h = output_size

    src = np.float32([
        center,
        center + [0, -src_w * 0.5],
        center + [-src_w * 0.5, 0],
    ])
    dst = np.float32([
        [dst_w / 2.0, dst_h / 2.0],
        [dst_w / 2.0, 0],
        [0, dst_h / 2.0],
    ])

    if inv:
        return cv2.getAffineTransform(dst, src)
    return cv2.getAffineTransform(src, dst)


def apply_affine_to_image(
    image: np.ndarray,
    trans: np.ndarray,
    output_size: tuple[int, int],
) -> np.ndarray:
    """Warp image using affine transform.

    Args:
        image: (H, W, 3) uint8 image.
        trans: (2, 3) affine matrix.
        output_size: (W, H) target size.

    Returns:
        Warped image of shape (output_size[1], output_size[0], 3).
    """
    return cv2.warpAffine(
        image, trans, output_size, flags=cv2.INTER_LINEAR
    )


def warp_keypoints(
    keypoints_xy: np.ndarray,
    trans: np.ndarray,
) -> np.ndarray:
    """Transform keypoint coordinates using an affine matrix.

    Args:
        keypoints_xy: (N, 2) array of (x, y) coordinates.
        trans: (2, 3) affine transformation matrix.

    Returns:
        (N, 2) transformed coordinates.
    """
    ones = np.ones((keypoints_xy.shape[0], 1), dtype=np.float32)
    pts = np.concatenate([keypoints_xy.astype(np.float32), ones], axis=1)  # (N, 3)
    return (trans @ pts.T).T  # (N, 2)
