#!/bin/bash
#SBATCH --job-name=walker-gait-train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

# ── Load modules (adjust to your SHARCNET cluster) ──
# module load python/3.10
# module load cuda/12.1

# ── Activate conda env ──
source activate walker-gait

# ── Create log directory ──
mkdir -p logs

# ── Run training ──
python scripts/train.py \
    --config configs/train.yaml \
    --data-config configs/data.yaml \
    --model-config configs/model.yaml

echo "Training complete at $(date)"
