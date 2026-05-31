#!/bin/bash
# =============================================================================
# SLURM Job: FMA selective download via HuggingFace streaming
#
# Source: benjamin-paine/free-music-archive-full (ungated, no login needed)
# Downloads only 2,500 h of filtered audio — no 879 GB zip required.
#
# This is a NETWORK I/O job — GPU sits idle but required for SPANK plugin.
# =============================================================================
#SBATCH --job-name=fma_dl
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=14-00:00:00
#SBATCH --output=logs/fma_dl_%j.log
#SBATCH --account=hafiz1

set -euo pipefail

STORE=/nfs/turbo/umd-hafiz/issf_server_data
REPO="${SLURM_SUBMIT_DIR}"
SCRIPTS="$REPO/scripts/dataset"
FMA_DIR="$STORE/fma"
MANIFEST_DIR="$FMA_DIR/manifests"
PYTHON=/home/ksathwik/.conda/envs/aura/bin/python

mkdir -p "$FMA_DIR/fma_full" "$MANIFEST_DIR" "$REPO/logs"

echo "[$(date '+%F %T')] Node: $(hostname)"
echo "[$(date '+%F %T')] Python: $($PYTHON --version)"
echo "[$(date '+%F %T')] Output: $FMA_DIR/fma_full"
echo "[$(date '+%F %T')] Manifest: $MANIFEST_DIR/fma_raw.csv"

# ── Auto-detect resume ────────────────────────────────────────────────────────
RESUME_FLAG=""
if [ -f "$MANIFEST_DIR/fma_raw.csv" ]; then
    N=$(wc -l < "$MANIFEST_DIR/fma_raw.csv")
    echo "[$(date '+%F %T')] Existing manifest found ($N lines) — resuming"
    RESUME_FLAG="--resume"
fi

# ── Stream + filter + download ────────────────────────────────────────────────
$PYTHON "$SCRIPTS/download_fma.py" \
    --output-dir "$FMA_DIR" \
    --manifest   "$MANIFEST_DIR/fma_raw.csv" \
    --target-h   2500 \
    $RESUME_FLAG

echo "[$(date '+%F %T')] FMA download complete"
find "$FMA_DIR/fma_full" -name "*.mp3" -o -name "*.wav" | wc -l
du -sh "$FMA_DIR/fma_full"
