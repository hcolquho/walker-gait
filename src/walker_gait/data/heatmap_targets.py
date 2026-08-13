"""
Gaussian heatmap target generation for keypoint supervision.

Generates 2D Gaussian blobs centered on each visible keypoint, at the
spatial resolution of the model's heatmap output (typically input_size / 4).
"""

import numpy as np


def generate_target_heatmaps(
    keypoints_xy: np.ndarray,
    visibility: np.ndarray,
    heatmap_size: tuple[int, int],
    sigma: float = 2.0,
) -> np.ndarray:
    """Generate Gaussian heatmap targets for supervised keypoints.

    Args:
        keypoints_xy: (K, 2) keypoint coordinates in heatmap-resolution space.
        visibility: (K,) visibility flag per keypoint.
            0 = not labeled (target is zero, masked out of loss).
            1 = labeled but occluded (still supervised).
            2 = labeled and visible (still supervised).
        heatmap_size: (W, H) spatial dimensions of the heatmap.
        sigma: Standard deviation of the Gaussian blob.

    Returns:
        (K, H, W) float32 array of target heatmaps.
    """
    W, H = heatmap_size
    K = len(keypoints_xy)
    targets = np.zeros((K, H, W), dtype=np.float32)

    # Pre-compute coordinate grids once
    xx, yy = np.meshgrid(np.arange(W, dtype=np.float32),
                         np.arange(H, dtype=np.float32))

    two_sigma_sq = 2.0 * sigma * sigma

    for i in range(K):
        if visibility[i] == 0:
            continue  # not labeled → zero heatmap, masked in loss

        x, y = keypoints_xy[i]

        # Skip keypoints that fall outside the heatmap
        if x < 0 or x >= W or y < 0 or y >= H:
            continue

        targets[i] = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / two_sigma_sq)

    return targets


def keypoints_image_to_heatmap(
    keypoints_xy: np.ndarray,
    input_size: tuple[int, int],
    heatmap_size: tuple[int, int],
) -> np.ndarray:
    """Rescale keypoint coordinates from model input space to heatmap space.

    Args:
        keypoints_xy: (K, 2) in input-image pixel coordinates.
        input_size: (H, W) of the model input image.
        heatmap_size: (W, H) of the heatmap output.

    Returns:
        (K, 2) rescaled coordinates in heatmap pixel space.
    """
    input_h, input_w = input_size
    hm_w, hm_h = heatmap_size

    scale_x = hm_w / input_w
    scale_y = hm_h / input_h

    scaled = keypoints_xy.copy().astype(np.float32)
    scaled[:, 0] *= scale_x
    scaled[:, 1] *= scale_y
    return scaled
