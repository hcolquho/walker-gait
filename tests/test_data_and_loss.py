"""
Unit tests for affine transforms, heatmap targets, and masked loss.

Run with:  pytest tests/ -v
"""

import numpy as np
import torch
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


class TestAffine:
    def test_box_to_center_scale(self):
        from walker_gait.data.affine import box_to_center_scale
        box = np.array([100, 200, 80, 160], dtype=np.float32)
        center, scale = box_to_center_scale(box, input_size=(256, 192))
        assert center.shape == (2,)
        assert scale.shape == (2,)
        np.testing.assert_allclose(center, [140, 280], atol=1)
        assert scale[0] > 0 and scale[1] > 0

    def test_affine_transform_shape(self):
        from walker_gait.data.affine import (
            box_to_center_scale, get_affine_transform, apply_affine_to_image
        )
        box = np.array([50, 50, 100, 200], dtype=np.float32)
        center, scale = box_to_center_scale(box, input_size=(256, 192))
        trans = get_affine_transform(center, scale, output_size=(192, 256))
        assert trans.shape == (2, 3)

        # Warp a dummy image
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        warped = apply_affine_to_image(image, trans, (192, 256))
        assert warped.shape == (256, 192, 3)

    def test_warp_keypoints_roundtrip(self):
        from walker_gait.data.affine import (
            box_to_center_scale, get_affine_transform, warp_keypoints
        )
        box = np.array([100, 100, 100, 200], dtype=np.float32)
        center, scale = box_to_center_scale(box, input_size=(256, 192))
        trans_fwd = get_affine_transform(center, scale, (192, 256), inv=False)
        trans_inv = get_affine_transform(center, scale, (192, 256), inv=True)

        pts = np.array([[150, 200], [120, 250]], dtype=np.float32)
        warped = warp_keypoints(pts, trans_fwd)
        recovered = warp_keypoints(warped, trans_inv)
        np.testing.assert_allclose(recovered, pts, atol=1.0)


class TestHeatmapTargets:
    def test_shape_and_peak(self):
        from walker_gait.data.heatmap_targets import generate_target_heatmaps
        kp = np.array([[24.0, 32.0], [10.0, 10.0]], dtype=np.float32)
        vis = np.array([2, 0], dtype=np.float32)  # only first is visible
        targets = generate_target_heatmaps(kp, vis, heatmap_size=(48, 64), sigma=2.0)
        assert targets.shape == (2, 64, 48)
        # First keypoint should have peak near 1.0
        assert targets[0].max() > 0.99
        # Second keypoint (not visible) should be all zeros
        assert targets[1].max() == 0.0

    def test_rescale(self):
        from walker_gait.data.heatmap_targets import keypoints_image_to_heatmap
        kp = np.array([[96.0, 128.0]], dtype=np.float32)
        scaled = keypoints_image_to_heatmap(kp, input_size=(256, 192), heatmap_size=(48, 64))
        # 96 * (48/192) = 24, 128 * (64/256) = 32
        np.testing.assert_allclose(scaled, [[24.0, 32.0]], atol=0.01)


class TestMaskedLoss:
    def test_loss_zero_when_target_matches(self):
        from walker_gait.models.vitpose_finetune import masked_heatmap_loss
        # Fake 133-channel prediction: set our 12 channels to match target
        pred = torch.zeros(2, 133, 8, 8)
        target = torch.zeros(2, 12, 8, 8)
        visibility = torch.ones(2, 12)

        # Make them identical
        target[:, 0, 4, 4] = 1.0
        pred[:, 11, 4, 4] = 1.0  # index 11 = our index 0 (left_hip)

        loss = masked_heatmap_loss(pred, target, visibility)
        # Won't be exactly zero (other channels contribute) but should be small
        assert loss.item() < 0.1

    def test_masked_channels_excluded(self):
        from walker_gait.models.vitpose_finetune import masked_heatmap_loss
        pred = torch.randn(1, 133, 8, 8)
        target = torch.zeros(1, 12, 8, 8)
        visibility = torch.zeros(1, 12)  # all masked out

        loss = masked_heatmap_loss(pred, target, visibility)
        assert loss.item() == 0.0


class TestMetrics:
    def test_pck_perfect(self):
        from walker_gait.training.metrics import compute_pck
        pred = np.array([[10, 20], [30, 40]], dtype=np.float32)
        gt = pred.copy()
        vis = np.array([1, 1], dtype=np.float32)
        pck = compute_pck(pred, gt, vis, bbox_size=100, threshold=0.05)
        assert pck == 1.0

    def test_pck_all_wrong(self):
        from walker_gait.training.metrics import compute_pck
        pred = np.array([[10, 20], [30, 40]], dtype=np.float32)
        gt = np.array([[100, 200], [300, 400]], dtype=np.float32)
        vis = np.array([1, 1], dtype=np.float32)
        pck = compute_pck(pred, gt, vis, bbox_size=100, threshold=0.05)
        assert pck == 0.0

    def test_heatmaps_to_keypoints(self):
        from walker_gait.training.metrics import heatmaps_to_keypoints
        hm = torch.zeros(1, 2, 64, 48)
        hm[0, 0, 32, 24] = 1.0  # peak at (24, 32) in heatmap space
        kps = heatmaps_to_keypoints(hm, heatmap_size=(48, 64), input_size=(256, 192))
        assert kps.shape == (1, 2, 2)
        # 24 * (192/48) = 96, 32 * (256/64) = 128
        np.testing.assert_allclose(kps[0, 0].numpy(), [96, 128], atol=1)
