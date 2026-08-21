"""
End-to-end inference pipeline for walker-mounted gait analysis.

Detect person → ViTPose++ wholebody → extract 12 lower-body keypoints.
Designed for offline batch processing on SHARCNET.
"""

from pathlib import Path
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    RTDetrForObjectDetection,
    VitPoseForPoseEstimation,
)

# Our 12-keypoint subset
COCO_WB_INDICES = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
COCO_WHOLEBODY_DATASET_INDEX = 5

KEYPOINT_NAMES = [
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "left_big_toe", "left_small_toe", "left_heel",
    "right_big_toe", "right_small_toe", "right_heel",
]


@dataclass
class PoseResult:
    """Result for one frame."""
    keypoints: np.ndarray       # (N_persons, 12, 2) in original image coords
    scores: np.ndarray          # (N_persons, 12) confidence per keypoint
    boxes: np.ndarray           # (N_persons, 4) bounding boxes [x1,y1,x2,y2]
    keypoint_names: list[str]   # length-12 list of names


class WalkerGaitPipeline:
    """End-to-end person detection → ViTPose++ → 12-keypoint extraction.

    Args:
        pose_model_name: HuggingFace model ID for ViTPose++.
        detector_model_name: HuggingFace model ID for person detector.
        pose_checkpoint: Optional local checkpoint path (fine-tuned weights).
        device: Device string.
        person_threshold: Detection confidence threshold.
        max_persons: Max persons to process per frame.
    """

    def __init__(
        self,
        pose_model_name: str = "usyd-community/vitpose-plus-large",
        detector_model_name: str = "PekingU/rtdetr_r50vd_coco_o365",
        pose_checkpoint: str | None = None,
        device: str = "cuda",
        person_threshold: float = 0.3,
        max_persons: int = 1,
    ):
        self.device = device
        self.person_threshold = person_threshold
        self.max_persons = max_persons

        # --- Person detector ---
        self.det_processor = AutoProcessor.from_pretrained(detector_model_name)
        self.det_model = RTDetrForObjectDetection.from_pretrained(
            detector_model_name, device_map=device
        ).eval()

        # --- Pose estimator ---
        self.pose_processor = AutoProcessor.from_pretrained(pose_model_name)
        if pose_checkpoint:
            self.pose_model = VitPoseForPoseEstimation.from_pretrained(
                pose_checkpoint, device_map=device
            ).eval()
        else:
            self.pose_model = VitPoseForPoseEstimation.from_pretrained(
                pose_model_name, device_map=device
            ).eval()

    @torch.no_grad()
    def detect_persons(self, image: Image.Image) -> np.ndarray:
        """Detect persons and return bounding boxes in COCO [x,y,w,h] format.

        Returns:
            (N, 4) array of boxes, sorted by confidence descending.
        """
        inputs = self.det_processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.det_model(**inputs)
        results = self.det_processor.post_process_object_detection(
            outputs,
            target_sizes=torch.tensor([(image.height, image.width)]),
            threshold=self.person_threshold,
        )[0]

        # Filter to person class (index 0 in COCO)
        person_mask = results["labels"] == 0
        boxes_xyxy = results["boxes"][person_mask].cpu().numpy()
        scores = results["scores"][person_mask].cpu().numpy()

        if len(boxes_xyxy) == 0:
            return np.zeros((0, 4), dtype=np.float32)

        # Sort by score, keep top N
        order = np.argsort(-scores)[: self.max_persons]
        boxes_xyxy = boxes_xyxy[order]

        # Convert to COCO format [x, y, w, h]
        boxes_xywh = boxes_xyxy.copy()
        boxes_xywh[:, 2] -= boxes_xyxy[:, 0]
        boxes_xywh[:, 3] -= boxes_xyxy[:, 1]

        return boxes_xywh

    @torch.no_grad()
    def estimate_pose(
        self, image: Image.Image, boxes_xywh: np.ndarray
    ) -> PoseResult:
        """Run ViTPose++ on detected person crops and extract 12 keypoints.

        Args:
            image: PIL Image.
            boxes_xywh: (N, 4) bounding boxes in COCO format.

        Returns:
            PoseResult with keypoints, scores, and boxes.
        """
        if len(boxes_xywh) == 0:
            return PoseResult(
                keypoints=np.zeros((0, 12, 2)),
                scores=np.zeros((0, 12)),
                boxes=np.zeros((0, 4)),
                keypoint_names=KEYPOINT_NAMES,
            )

        inputs = self.pose_processor(
            image, boxes=[boxes_xywh], return_tensors="pt"
        ).to(self.device)

        n_crops = inputs["pixel_values"].shape[0]
        dataset_index = torch.full(
            (n_crops,), COCO_WHOLEBODY_DATASET_INDEX,
            dtype=torch.long, device=self.device,
        )

        outputs = self.pose_model(**inputs, dataset_index=dataset_index)

        # Decode directly from heatmaps to get all 133 keypoints
        # post_process_pose_estimation defaults to 17 (COCO body) — bypass it
        heatmaps = outputs.heatmaps  # (N, 133, H, W)
        N, K, H, W = heatmaps.shape

        all_kpts = []
        all_scores = []
        for i in range(N):
            hm = heatmaps[i]  # (133, H, W)

            # Argmax decoding
            flat = hm.view(K, -1)
            max_vals, max_idx = flat.max(dim=1)
            x = (max_idx % W).float() / W
            y = (max_idx // W).float() / H

            # Scale back to input image coordinates
            box = torch.tensor(boxes_xywh[i], dtype=torch.float32, device=self.device)
            x_img = x * box[2] + box[0]
            y_img = y * box[3] + box[1]

            kpts = torch.stack([x_img, y_img], dim=1).cpu().numpy()  # (133, 2)
            scores = torch.sigmoid(max_vals).cpu().numpy()            # (133,)

            all_kpts.append(kpts[COCO_WB_INDICES])
            all_scores.append(scores[COCO_WB_INDICES])

        return PoseResult(
            keypoints=np.stack(all_kpts) if all_kpts else np.zeros((0, 12, 2)),
            scores=np.stack(all_scores) if all_scores else np.zeros((0, 12)),
            boxes=boxes_xywh,
            keypoint_names=KEYPOINT_NAMES,
        )

    def __call__(self, image: Image.Image) -> PoseResult:
        """Full pipeline: use full frame as bbox → pose → filter."""
        # Skip detector — camera always shows one person hips-to-floor
        w, h = image.size
        boxes = np.array([[0, 0, w, h]], dtype=np.float32)  # full frame [x,y,w,h]
        return self.estimate_pose(image, boxes)
