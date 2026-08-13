#!/usr/bin/env python3
"""
Training script — mirrors notebooks/02_finetune_vitpose.ipynb.

For SLURM submission on SHARCNET:
    sbatch scripts/sharcnet_job.sh

Or run directly:
    python scripts/train.py --config configs/train.yaml
"""

import argparse
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

# Add src/ to path so walker_gait is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from walker_gait.data import GaitKeypointDataset
from walker_gait.models.vitpose_finetune import load_model_for_finetune
from walker_gait.training import train_one_epoch, evaluate


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train.yaml")
    parser.add_argument("--data-config", type=str, default="configs/data.yaml")
    parser.add_argument("--model-config", type=str, default="configs/model.yaml")
    args = parser.parse_args()

    train_cfg = load_config(args.config)
    data_cfg = load_config(args.data_config)
    model_cfg = load_config(args.model_config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── Model ──
    model = load_model_for_finetune(
        model_name=model_cfg["backbone"]["name"],
        freeze_backbone=model_cfg["freeze"]["backbone"],
        unfreeze_last_n_blocks=model_cfg["freeze"]["unfreeze_last_n_blocks"],
        device=device,
    )

    # ── Data ──
    input_size = tuple(model_cfg["backbone"]["input_size"])
    # Heatmap size = input_size / 4 (for simple decoder with scale_factor=4)
    heatmap_size = (input_size[1] // 4, input_size[0] // 4)  # (W, H)

    train_ds = GaitKeypointDataset(
        ann_file=data_cfg["paths"]["train_ann"],
        img_dir=data_cfg["paths"]["frames_dir"],
        input_size=input_size,
        heatmap_size=heatmap_size,
        sigma=model_cfg["heatmap"]["sigma"],
        augment=True,
        scale_range=tuple(train_cfg["augmentation"]["random_scale"]),
        rotation_range=train_cfg["augmentation"]["random_rotation"],
    )
    val_ds = GaitKeypointDataset(
        ann_file=data_cfg["paths"]["val_ann"],
        img_dir=data_cfg["paths"]["frames_dir"],
        input_size=input_size,
        heatmap_size=heatmap_size,
        sigma=model_cfg["heatmap"]["sigma"],
        augment=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=train_cfg["training"]["num_workers"],
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=train_cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=train_cfg["training"]["num_workers"],
        pin_memory=True,
    )

    # ── Optimizer + scheduler ──
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=train_cfg["optimizer"]["lr"],
        weight_decay=train_cfg["optimizer"]["weight_decay"],
    )

    total_epochs = train_cfg["training"]["epochs"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_epochs - train_cfg["scheduler"]["warmup_epochs"],
        eta_min=train_cfg["scheduler"]["min_lr"],
    )

    # ── Training loop ──
    ckpt_dir = Path(train_cfg["checkpoint"]["dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_pck = 0.0

    for epoch in range(total_epochs):
        # Warmup: linear ramp for first N epochs
        if epoch < train_cfg["scheduler"]["warmup_epochs"]:
            warmup_lr = train_cfg["optimizer"]["lr"] * (epoch + 1) / train_cfg["scheduler"]["warmup_epochs"]
            for pg in optimizer.param_groups:
                pg["lr"] = warmup_lr

        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)

        # Step scheduler after warmup
        if epoch >= train_cfg["scheduler"]["warmup_epochs"]:
            scheduler.step()

        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, lr={optimizer.param_groups[0]['lr']:.2e}")

        # Validate
        if (epoch + 1) % train_cfg["training"]["val_interval"] == 0:
            val_metrics = evaluate(
                model, val_loader, heatmap_size, input_size,
                pck_threshold=train_cfg["evaluation"]["pck_threshold"],
                device=device,
            )
            print(f"  val_loss={val_metrics['loss']:.4f}, val_pck={val_metrics['pck']:.4f}")

            # Save best
            if val_metrics["pck"] > best_pck:
                best_pck = val_metrics["pck"]
                model.save_pretrained(str(ckpt_dir / "best"))
                print(f"  ✓ New best PCK: {best_pck:.4f}")

    # Save final
    model.save_pretrained(str(ckpt_dir / "last"))
    print(f"\nTraining complete. Best PCK: {best_pck:.4f}")


if __name__ == "__main__":
    main()
