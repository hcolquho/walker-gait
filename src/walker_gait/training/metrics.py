"""
Evaluation metrics for keypoint estimation.

PCK (Percentage of Correct Keypoints) — the primary metric for evaluating
fine-tuned lower-body keypoint accuracy.
"""

import numpy as np
import torch


def compute_pck(
    pred_keypoints: np.ndarray,
    gt_keypoints: np.ndarray,
    visibility: np.ndarray,
    bbox_size: float,
    threshold: float = 0.05,
) -> float:
    """Compute PCK for a single sample.

    A predicted keypoint is "correct" if its Euclidean distance to the
    ground truth is within `threshold * bbox_size`.

    Args:
        pred_keypoints: (K, 2) predicted keypoint coordinates.
        gt_keypoints: (K, 2) ground-truth keypoint coordinates.
        visibility: (K,) mask — 1.0 for evaluated keypoints, 0.0 to skip.
        bbox_size: Normalization factor (e.g. bbox diagonal or max dim).
        threshold: Fraction of bbox_size considered "correct".

    Returns:
        Fraction of visible keypoints within the threshold.
    """
    vis_mask = visibility > 0
    if vis_mask.sum() == 0:
        return 0.0

    dist = np.linalg.norm(
        pred_keypoints[vis_mask] - gt_keypoints[vis_mask], axis=1
    )
    correct = (dist < threshold * bbox_size).sum()
    return float(correct / vis_mask.sum())


def compute_pck_per_keypoint(
    pred_keypoints: np.ndarray,
    gt_keypoints: np.ndarray,
    visibility: np.ndarray,
    bbox_size: float,
    threshold: float = 0.05,
    keypoint_names: list[str] | None = None,
) -> dict[str, float]:
    """Compute PCK for each keypoint individually.

    Returns:
        Dict mapping keypoint name/index to PCK value.
    """
    K = pred_keypoints.shape[0]
    if keypoint_names is None:
        keypoint_names = [str(i) for i in range(K)]

    results = {}
    for i in range(K):
        if visibility[i] == 0:
            results[keypoint_names[i]] = float("nan")
            continue
        dist = np.linalg.norm(pred_keypoints[i] - gt_keypoints[i])
        results[keypoint_names[i]] = float(dist < threshold * bbox_size)
    return results


def heatmaps_to_keypoints(
    heatmaps: torch.Tensor,
    heatmap_size: tuple[int, int],
    input_size: tuple[int, int],
) -> torch.Tensor:
    """Decode argmax keypoint locations from heatmaps.

    Simple argmax decoding (no sub-pixel refinement). Sufficient for
    evaluation; the HF post-processor handles inference with Gaussian
    modulation.

    Args:
        heatmaps: (N, K, H, W) predicted heatmaps.
        heatmap_size: (W, H) of the heatmap.
        input_size: (H, W) of the model input image.

    Returns:
        (N, K, 2) keypoint coordinates in input-image space.
    """
    N, K, H, W = heatmaps.shape
    flat = heatmaps.view(N, K, -1)
    max_idx = flat.argmax(dim=2)  # (N, K)

    # Convert flat index to (x, y) in heatmap space
    hm_x = (max_idx % W).float()
    hm_y = (max_idx // W).float()

    # Scale to input image space
    input_h, input_w = input_size
    hm_w, hm_h = heatmap_size
    scale_x = input_w / hm_w
    scale_y = input_h / hm_h

    kp_x = hm_x * scale_x
    kp_y = hm_y * scale_y

    return torch.stack([kp_x, kp_y], dim=2)  # (N, K, 2)
