"""
COCO-format keypoint dataset for walker-gait fine-tuning.

Loads a COCO-WholeBody-style annotation file, extracts the 12 lower-body
keypoints, applies top-down affine crop + augmentation, and generates
Gaussian heatmap targets for supervision.
"""

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .affine import box_to_center_scale, get_affine_transform, apply_affine_to_image, warp_keypoints
from .augmentation import random_horizontal_flip, random_scale_rotation, apply_rotation_to_affine
from .heatmap_targets import generate_target_heatmaps, keypoints_image_to_heatmap

# COCO-WholeBody indices for our 12 lower-body keypoints
COCO_WB_INDICES = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]

# Left↔right flip pairs (in our 12-keypoint space, 0-indexed)
FLIP_PAIRS = [(0, 1), (2, 3), (4, 5), (6, 9), (7, 10), (8, 11)]

# ImageNet normalization (matches HF VitPoseImageProcessor defaults)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class GaitKeypointDataset(Dataset):
    """Top-down keypoint dataset for walker-gait fine-tuning.

    Each sample is one person crop (defined by a bounding box) from an image,
    with 12 ground-truth keypoints.

    Args:
        ann_file: Path to COCO-format annotation JSON.
        img_dir: Path to directory containing images.
        input_size: (H, W) model input resolution.
        heatmap_size: (W, H) model heatmap output resolution.
        sigma: Gaussian sigma for heatmap targets.
        augment: Whether to apply data augmentation.
        scale_range: (min, max) random scale factor.
        rotation_range: Max random rotation in degrees.
    """

    def __init__(
        self,
        ann_file: str,
        img_dir: str,
        input_size: tuple[int, int] = (256, 192),
        heatmap_size: tuple[int, int] = (48, 64),
        sigma: float = 2.0,
        augment: bool = False,
        scale_range: tuple[float, float] = (0.75, 1.25),
        rotation_range: float = 30.0,
    ):
        self.img_dir = Path(img_dir)
        self.input_size = input_size      # (H, W)
        self.heatmap_size = heatmap_size  # (W, H) — note: width first
        self.sigma = sigma
        self.augment = augment
        self.scale_range = scale_range
        self.rotation_range = rotation_range

        # Load COCO annotations
        with open(ann_file, "r") as f:
            coco = json.load(f)

        # Build image lookup
        self.images = {img["id"]: img for img in coco["images"]}

        # Build samples: one entry per annotated person instance
        self.samples = []
        for ann in coco["annotations"]:
            if ann.get("num_keypoints", 0) == 0 and not ann.get("keypoints"):
                continue

            # Parse keypoints: flat list [x1,y1,v1, x2,y2,v2, ...]
            kp_flat = ann["keypoints"]
            num_total = len(kp_flat) // 3

            # Extract our 12-keypoint subset
            all_kp = np.array(kp_flat, dtype=np.float32).reshape(num_total, 3)
            subset_kp = all_kp[COCO_WB_INDICES]  # (12, 3)

            # Skip if no lower-body keypoints are visible at all
            if subset_kp[:, 2].sum() == 0:
                continue

            self.samples.append({
                "image_id": ann["image_id"],
                "bbox": np.array(ann["bbox"], dtype=np.float32),  # [x, y, w, h]
                "keypoints": subset_kp[:, :2],   # (12, 2) xy
                "visibility": subset_kp[:, 2],   # (12,) 0/1/2
            })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]

        # Load image
        img_info = self.images[sample["image_id"]]
        img_path = self.img_dir / img_info["file_name"]
        image = cv2.imread(str(img_path))
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        keypoints = sample["keypoints"].copy()   # (12, 2)
        visibility = sample["visibility"].copy()  # (12,)

        # ── Augmentation (before affine crop) ──
        if self.augment:
            image, keypoints, visibility = random_horizontal_flip(
                image, keypoints, visibility, FLIP_PAIRS
            )

        # ── Top-down affine crop ──
        center, scale = box_to_center_scale(
            sample["bbox"], self.input_size
        )

        if self.augment:
            scale, rot = random_scale_rotation(
                center, scale, self.scale_range, self.rotation_range
            )
            trans = apply_rotation_to_affine(
                center, scale, rot,
                output_size=(self.input_size[1], self.input_size[0]),  # (W, H)
            )
        else:
            trans = get_affine_transform(
                center, scale,
                output_size=(self.input_size[1], self.input_size[0]),
            )

        # Warp image
        cropped = apply_affine_to_image(
            image, trans,
            output_size=(self.input_size[1], self.input_size[0]),  # (W, H)
        )

        # Warp keypoints to cropped image space
        kp_warped = warp_keypoints(keypoints, trans)  # (12, 2)

        # ── Generate heatmap targets ──
        kp_heatmap = keypoints_image_to_heatmap(
            kp_warped, self.input_size, self.heatmap_size
        )
        target_heatmaps = generate_target_heatmaps(
            kp_heatmap, visibility, self.heatmap_size, self.sigma
        )  # (12, H_hm, W_hm)

        # ── Normalize image ──
        image_tensor = cropped.astype(np.float32) / 255.0
        image_tensor = (image_tensor - IMAGENET_MEAN) / IMAGENET_STD
        image_tensor = np.transpose(image_tensor, (2, 0, 1))  # (3, H, W)

        # Visibility mask: 1 for labeled keypoints (v=1 or v=2), 0 for v=0
        vis_mask = (visibility > 0).astype(np.float32)

        return {
            "pixel_values": torch.from_numpy(image_tensor),
            "target_heatmaps": torch.from_numpy(target_heatmaps),
            "visibility": torch.from_numpy(vis_mask),
            "keypoints_raw": torch.from_numpy(kp_warped),  # for PCK eval
        }
