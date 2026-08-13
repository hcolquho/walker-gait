"""
Training and evaluation loops.

Designed to be called from both the Jupyter notebook (interactive) and
scripts/train.py (SLURM batch job on SHARCNET). All state lives in the
arguments — no global variables.
"""

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from walker_gait.models.vitpose_finetune import (
    masked_heatmap_loss,
    get_dataset_index_tensor,
    LEG_ANKLE_FOOT_INDICES,
)
from .metrics import compute_pck, heatmaps_to_keypoints


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str = "cuda",
    epoch: int = 0,
) -> float:
    """Run one training epoch.

    Args:
        model: ViTPose model.
        dataloader: Training DataLoader.
        optimizer: Optimizer.
        device: Device string.
        epoch: Current epoch number (for logging).

    Returns:
        Mean loss over the epoch.
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}", leave=False)
    for batch in pbar:
        pixel_values = batch["pixel_values"].to(device)
        target_heatmaps = batch["target_heatmaps"].to(device)
        visibility = batch["visibility"].to(device)

        bs = pixel_values.shape[0]
        dataset_index = get_dataset_index_tensor(bs, device)

        # Forward
        outputs = model(pixel_values=pixel_values, dataset_index=dataset_index)

        # Loss on 12 lower-body channels only
        loss = masked_heatmap_loss(outputs.heatmaps, target_heatmaps, visibility)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    heatmap_size: tuple[int, int],
    input_size: tuple[int, int],
    pck_threshold: float = 0.05,
    device: str = "cuda",
) -> dict:
    """Run evaluation and compute PCK + mean loss.

    Args:
        model: ViTPose model.
        dataloader: Validation DataLoader.
        heatmap_size: (W, H) of heatmap output.
        input_size: (H, W) of model input.
        pck_threshold: PCK threshold as fraction of bbox diagonal.
        device: Device string.

    Returns:
        Dict with keys "loss", "pck".
    """
    model.eval()
    total_loss = 0.0
    all_pck = []
    n_batches = 0

    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        pixel_values = batch["pixel_values"].to(device)
        target_heatmaps = batch["target_heatmaps"].to(device)
        visibility = batch["visibility"].to(device)
        gt_keypoints = batch["keypoints_raw"].to(device)  # (N, 12, 2) in input space

        bs = pixel_values.shape[0]
        dataset_index = get_dataset_index_tensor(bs, device)

        outputs = model(pixel_values=pixel_values, dataset_index=dataset_index)

        loss = masked_heatmap_loss(outputs.heatmaps, target_heatmaps, visibility)
        total_loss += loss.item()
        n_batches += 1

        # Decode predicted keypoints from heatmaps
        pred_hm = outputs.heatmaps[:, LEG_ANKLE_FOOT_INDICES]  # (N, 12, H, W)
        pred_kps = heatmaps_to_keypoints(pred_hm, heatmap_size, input_size)

        # PCK per sample
        for i in range(bs):
            vis = visibility[i].cpu().numpy()
            if vis.sum() == 0:
                continue
            pck = compute_pck(
                pred_kps[i].cpu().numpy(),
                gt_keypoints[i].cpu().numpy(),
                vis,
                bbox_size=max(input_size),  # use input size as normalization
                threshold=pck_threshold,
            )
            all_pck.append(pck)

    mean_loss = total_loss / max(n_batches, 1)
    mean_pck = float(sum(all_pck) / max(len(all_pck), 1))

    return {"loss": mean_loss, "pck": mean_pck}
