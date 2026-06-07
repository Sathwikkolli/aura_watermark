#!/bin/bash
# =============================================================================
# SLURM Job: AURA full training — 200,000 steps, two-stage curriculum
#
# Node spec (spgpu):  32 CPUs · 381 GB RAM · 8× A40 (48 GB each)
# We use:            28 CPUs · 360 GB RAM · 1× A40
#   - 28 workers feed the GPU continuously (no starvation)
#   - 360 GB RAM allows OS to cache audio files in page cache
#   - Gradient checkpointing DISABLED — A40 has enough VRAM
# =============================================================================
#SBATCH --job-name=aura_train
#SBATCH --partition=spgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=360G
#SBATCH --gres=gpu:1
#SBATCH --time=14-00:00:00
#SBATCH --output=logs/train_%j.log
#SBATCH --account=hafiz1

set -euo pipefail

STORE=/nfs/turbo/umd-hafiz/issf_server_data
REPO="${SLURM_SUBMIT_DIR}"
PYTHON=/home/ksathwik/.conda/envs/aura/bin/python
# Fresh restart with convergence + crash fixes (run_001 learned a degenerate
# loud-but-unreadable code; kept for comparison). New runs land in run_002.
CKPT_DIR="$REPO/checkpoints/run_002"

mkdir -p "$REPO/logs" "$CKPT_DIR"

echo "[$(date '+%F %T')] Node     : $(hostname)"
echo "[$(date '+%F %T')] GPU      : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "[$(date '+%F %T')] CPUs     : $(nproc)"
echo "[$(date '+%F %T')] RAM      : $(free -h | awk '/^Mem:/{print $2}')"
echo "[$(date '+%F %T')] Emilia   : $STORE/emilia"
echo "[$(date '+%F %T')] FMA      : $STORE/fma"
echo "[$(date '+%F %T')] Checkpoints: $CKPT_DIR"

# ── Auto-detect latest checkpoint for resume ──────────────────────────────────
RESUME_FLAG=""
LATEST=$(ls -t "$CKPT_DIR"/step_*.pt 2>/dev/null | grep -v final | head -1)
if [ -n "$LATEST" ]; then
    echo "[$(date '+%F %T')] Resuming from: $LATEST"
    RESUME_FLAG="--resume $LATEST"
else
    echo "[$(date '+%F %T')] No checkpoint found — starting from scratch"
fi

# ── Train ─────────────────────────────────────────────────────────────────────
$PYTHON "$REPO/train.py" \
    --emilia-root    "$STORE/emilia" \
    --fma-root       "$STORE/fma" \
    --checkpoint-dir "$CKPT_DIR" \
    --num-workers    24 \
    --log-every      50 \
    --val-every      1000 \
    --save-every     5000 \
    $RESUME_FLAG

echo "[$(date '+%F %T')] Training complete"
