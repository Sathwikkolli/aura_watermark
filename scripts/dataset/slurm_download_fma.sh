#!/bin/bash
# =============================================================================
# SLURM Job: FMA selective download — metadata → curate → per-track MP3s
#
# Output layout:
#   /nfs/turbo/umd-hafiz/issf_server_data/
#   └── fma/
#       ├── fma_metadata/           ← tracks.csv, genres.csv (342 MB)
#       ├── fma_full/
#       │   └── 000/ … 106/         ← only selected MP3s
#       └── manifests/
#           └── fma_selected_ids.csv
#
# Resume: resubmit this script — already-downloaded MP3s are skipped.
# =============================================================================
#SBATCH --job-name=fma_dl
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=14-00:00:00
#SBATCH --output=logs/fma_dl_%j.log
#SBATCH --account=hafiz_root

set -euo pipefail

STORE=/nfs/turbo/umd-hafiz/issf_server_data
REPO=/home/ksathwik/aura_watermark
SCRIPTS="$REPO/scripts/dataset"
FMA_DIR="$STORE/fma"
MANIFEST_DIR="$FMA_DIR/manifests"
TRACK_LIST="$MANIFEST_DIR/fma_selected_ids.csv"

mkdir -p "$FMA_DIR/fma_full" "$MANIFEST_DIR" logs

conda activate asd

echo "[$(date '+%F %T')] Node: $(hostname)"
echo "[$(date '+%F %T')] FMA dir:    $FMA_DIR"
echo "[$(date '+%F %T')] Track list: $TRACK_LIST"

# ── Stage 1+2: metadata download + curation (fast, idempotent) ────────────
python "$SCRIPTS/download_fma.py" curate \
    --fma-dir     "$FMA_DIR" \
    --track-list  "$TRACK_LIST" \
    --seed        42

echo "[$(date '+%F %T')] Track list ready. Starting audio download..."

# ── Stage 3: parallel MP3 download (32 threads) ────────────────────────────
python "$SCRIPTS/download_fma.py" audio \
    --fma-dir      "$FMA_DIR" \
    --track-list   "$TRACK_LIST" \
    --connections  32 \
    --resume

echo "[$(date '+%F %T')] FMA download complete"
echo "Track count:"
find "$FMA_DIR/fma_full" -name "*.mp3" | wc -l
echo "Disk usage:"
du -sh "$FMA_DIR/fma_full"
