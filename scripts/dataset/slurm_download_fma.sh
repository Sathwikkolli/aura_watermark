#!/bin/bash
# =============================================================================
# SLURM Job: FMA selective download — metadata → curate → per-track MP3s
#
# This is a NETWORK I/O job — no GPU needed.
# See partition guide in slurm_download_emilia.sh for details.
# Default: gpu partition + 1 GPU (works universally on Great Lakes).
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
#SBATCH --account=hafiz_root

set -euo pipefail

STORE=/nfs/turbo/umd-hafiz/issf_server_data
REPO="${SLURM_SUBMIT_DIR}"
SCRIPTS="$REPO/scripts/dataset"
FMA_DIR="$STORE/fma"
MANIFEST_DIR="$FMA_DIR/manifests"
TRACK_LIST="$MANIFEST_DIR/fma_selected_ids.csv"

mkdir -p "$FMA_DIR/fma_full" "$MANIFEST_DIR" "$REPO/logs"

conda activate aura

echo "[$(date '+%F %T')] Node: $(hostname)"

# ── Stage 1+2: metadata download + curation (idempotent) ─────────────────────
python "$SCRIPTS/download_fma.py" curate \
    --fma-dir     "$FMA_DIR" \
    --track-list  "$TRACK_LIST" \
    --seed        42

echo "[$(date '+%F %T')] Track list ready — starting audio download"

# ── Stage 3: parallel MP3 download (32 threads) ────────────────────────────
python "$SCRIPTS/download_fma.py" audio \
    --fma-dir      "$FMA_DIR" \
    --track-list   "$TRACK_LIST" \
    --connections  32 \
    --resume

echo "[$(date '+%F %T')] FMA download complete"
find "$FMA_DIR/fma_full" -name "*.mp3" | wc -l
du -sh "$FMA_DIR/fma_full"
