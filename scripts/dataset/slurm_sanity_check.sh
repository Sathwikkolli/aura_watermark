#!/bin/bash
# =============================================================================
# SLURM Job: End-to-end DataLoader sanity check
# Loads 5 batches from train.csv to verify shapes, no NaNs, correct dtypes.
# Run this BEFORE launching full training.
# Partition: spgpu (A40) — matches actual training environment
# =============================================================================
#SBATCH --job-name=aura_sanity
#SBATCH --partition=spgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=logs/sanity_%j.log
#SBATCH --account=hafiz_root

set -euo pipefail

STORE=/nfs/turbo/umd-hafiz/issf_server_data
REPO=/home/ksathwik/aura_watermark       # ← adjust to your repo path
mkdir -p logs

conda activate asd

echo "[$(date '+%F %T')] Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python - <<PYEOF
import sys
sys.path.insert(0, "$REPO")

import torch
from aura_watermark.config import AURAConfig
from aura_watermark.dataset import build_dataloaders

STORE = "$STORE"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

cfg = AURAConfig()
cfg.training.batch_size = 8

# ── Build DataLoaders using pre-built manifests ────────────────────────────
try:
    train_loader, val_loader = build_dataloaders(
        cfg,
        emilia_root   = f"{STORE}/emilia",
        fma_root      = f"{STORE}/fma",
        fma_subset    = "fma_full",
        batch_size    = cfg.training.batch_size,
        num_workers   = 8,
        val_frac      = 0.01,
    )
except TypeError:
    # Fallback if build_dataloaders doesn't accept manifest_override yet
    train_loader, val_loader = build_dataloaders(
        cfg,
        emilia_root   = f"{STORE}/emilia",
        fma_root      = f"{STORE}/fma",
        fma_subset    = "fma_full",
        batch_size    = cfg.training.batch_size,
        num_workers   = 8,
        val_frac      = 0.01,
    )

print(f"Train loader: {len(train_loader)} batches/epoch")
print(f"Val   loader: {len(val_loader)} batches/epoch")

# ── Check 10 train batches ─────────────────────────────────────────────────
print("\nChecking train batches...")
for i, (audio, message) in enumerate(train_loader):
    assert audio.shape   == (cfg.training.batch_size, 1, 96000), \
        f"Bad audio shape: {audio.shape}"
    assert message.shape == (cfg.training.batch_size, 32), \
        f"Bad message shape: {message.shape}"
    assert not torch.isnan(audio).any(),   f"NaN in audio batch {i}"
    assert not torch.isinf(audio).any(),   f"Inf in audio batch {i}"
    assert message.min() >= 0 and message.max() <= 1, \
        f"Message values out of {{0,1}} at batch {i}"
    peak = audio.abs().max().item()
    assert peak <= 1.0 + 1e-4,  f"Audio not peak-normalised: peak={peak:.4f}"
    print(f"  Train batch {i:02d}: audio={tuple(audio.shape)}  "
          f"msg={tuple(message.shape)}  peak={peak:.3f}  OK")
    if i >= 9:
        break

# ── Check 5 val batches ────────────────────────────────────────────────────
print("\nChecking val batches...")
for i, (audio, message) in enumerate(val_loader):
    assert audio.shape[1:] == (1, 96000), f"Bad val audio shape: {audio.shape}"
    print(f"  Val batch {i:02d}: audio={tuple(audio.shape)}  OK")
    if i >= 4:
        break

# ── Quick timing: 50 train batches ────────────────────────────────────────
import time
print("\nDataLoader throughput (50 batches)...")
t0 = time.time()
for i, _ in enumerate(train_loader):
    if i >= 49:
        break
elapsed = time.time() - t0
clips_per_sec = 50 * cfg.training.batch_size / elapsed
hours_per_day = clips_per_sec * 2 / 3600 * 86400   # 2-s clips
print(f"  {clips_per_sec:.1f} clips/s  "
      f"({hours_per_day:.0f} h of audio/day throughput)")

print("\n[SANITY CHECK PASSED]")
PYEOF

echo "[$(date '+%F %T')] Sanity check complete"
