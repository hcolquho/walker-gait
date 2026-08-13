"""
ViTPose++ fine-tuning wrapper.

The HuggingFace transformers VitPoseForPoseEstimation port is inference-only —
its forward() does not compute loss even when labels are provided. This module
provides:
  - Freeze-strategy control (backbone frozen, decoder trainable).
  - A masked MSE loss that supervises only the 12 lower-body channels.
  - A thin forward helper that routes dataset_index correctly for MoE.
"""

import torch
import torch.nn as nn
from transformers import VitPoseForPoseEstimation

# Indices into the 133-channel COCO-WholeBody heatmap output
LEG_ANKLE_FOOT_INDICES = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]

# MoE expert index for COCO-WholeBody in ViTPose++ models
COCO_WHOLEBODY_DATASET_INDEX = 5


def load_model_for_finetune(
    model_name: str = "usyd-community/vitpose-plus-large",
    freeze_backbone: bool = True,
    unfreeze_last_n_blocks: int = 0,
    device: str = "cuda",
) -> VitPoseForPoseEstimation:
    """Load a pretrained ViTPose++ model and configure for fine-tuning.

    Args:
        model_name: HuggingFace model ID.
        freeze_backbone: If True, freeze the entire ViT backbone.
        unfreeze_last_n_blocks: If >0 and freeze_backbone is True, unfreeze
            the last N transformer blocks (useful with 500+ annotated frames).
        device: Target device.

    Returns:
        Model with appropriate parameters frozen.
    """
    model = VitPoseForPoseEstimation.from_pretrained(model_name)
    model = model.to(device)

    if freeze_backbone:
        # Freeze everything in backbone
        for param in model.backbone.parameters():
            param.requires_grad = False

        # Optionally unfreeze the last N transformer blocks
        if unfreeze_last_n_blocks > 0 and hasattr(model.backbone, "encoder"):
            encoder = model.backbone.encoder
            if hasattr(encoder, "layer"):
                layers = encoder.layer
                for layer in layers[-unfreeze_last_n_blocks:]:
                    for param in layer.parameters():
                        param.requires_grad = True

    # Ensure decoder head is trainable
    for name, param in model.named_parameters():
        if "head" in name or "decoder" in name:
            param.requires_grad = True

    # Log parameter counts
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model loaded: {total:,} params total, {trainable:,} trainable "
          f"({100 * trainable / total:.1f}%)")

    return model


def masked_heatmap_loss(
    pred_heatmaps: torch.Tensor,
    target_heatmaps: torch.Tensor,
    visibility: torch.Tensor,
) -> torch.Tensor:
    """Compute MSE loss on the 12 lower-body heatmap channels only.

    Args:
        pred_heatmaps: (N, 133, H, W) full model output.
        target_heatmaps: (N, 12, H, W) targets for our subset.
        visibility: (N, 12) binary mask — 1 = supervised, 0 = ignore.

    Returns:
        Scalar loss.
    """
    # Select our 12 channels from the 133-channel prediction
    pred_subset = pred_heatmaps[:, LEG_ANKLE_FOOT_INDICES]  # (N, 12, H, W)

    # Expand visibility to spatial dims: (N, 12) → (N, 12, 1, 1)
    mask = visibility[:, :, None, None]

    # Masked MSE
    sq_diff = (pred_subset - target_heatmaps) ** 2
    loss = (sq_diff * mask).sum() / mask.sum().clamp(min=1)
    return loss


def get_dataset_index_tensor(batch_size: int, device: str = "cuda") -> torch.Tensor:
    """Create dataset_index tensor for COCO-WholeBody MoE expert selection."""
    return torch.full(
        (batch_size,),
        COCO_WHOLEBODY_DATASET_INDEX,
        dtype=torch.long,
        device=device,
    )
