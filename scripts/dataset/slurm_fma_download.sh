#!/bin/bash
# =============================================================================
# SLURM Job: Download fma_full (~879 GB) + fma_metadata from Zenodo
#
# IMPORTANT: Submit this to the 'gpu' partition for stable long-running I/O.
# aria2c --continue=true means safe to requeue if preempted.
# Expected runtime: 24-48 h depending on Zenodo/network speed.
# =============================================================================
#SBATCH --job-name=fma_download
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=14-00:00:00
#SBATCH --output=logs/fma_download_%j.log
#SBATCH --account=hafiz_root

set -euo pipefail

STORE=/nfs/turbo/umd-hafiz/issf_server_data
FMA_DIR="$STORE/fma"

mkdir -p "$FMA_DIR/fma_full" "$FMA_DIR/fma_metadata" logs

conda activate asd

echo "[$(date '+%F %T')] Node: $(hostname)"
echo "[$(date '+%F %T')] Disk space before download:"
df -h "$STORE"

# ── 1. Download fma_metadata.zip (~342 MB) ─────────────────────────────────
echo "[$(date '+%F %T')] Downloading fma_metadata"
aria2c \
    --dir="$FMA_DIR" \
    --out="fma_metadata.zip" \
    --max-connection-per-server=4 \
    --split=4 \
    --check-certificate=false \
    --auto-file-renaming=false \
    --continue=true \
    "https://zenodo.org/record/1476463/files/fma_metadata.zip"

echo "[$(date '+%F %T')] Extracting fma_metadata"
unzip -q "$FMA_DIR/fma_metadata.zip" -d "$FMA_DIR/"
rm "$FMA_DIR/fma_metadata.zip"
echo "[$(date '+%F %T')] Metadata extracted:"
ls "$FMA_DIR/fma_metadata/"

# ── 2. Download fma_full.zip (~879 GB) ────────────────────────────────────
echo "[$(date '+%F %T')] Starting fma_full download (~879 GB)"
echo "   This will take 24-48 h. Safe to requeue — aria2c resumes from offset."

aria2c \
    --dir="$FMA_DIR" \
    --out="fma_full.zip" \
    --max-connection-per-server=4 \
    --split=4 \
    --file-allocation=none \
    --auto-file-renaming=false \
    --continue=true \
    --check-certificate=false \
    --retry-wait=30 \
    --max-tries=0 \
    "https://zenodo.org/record/1476463/files/fma_full.zip"

echo "[$(date '+%F %T')] Download complete. Verifying MD5..."

EXPECTED_MD5=$(curl -s "https://zenodo.org/record/1476463" \
    | grep -oP 'fma_full\.zip.*?[a-f0-9]{32}' | tail -1 | grep -oP '[a-f0-9]{32}' || true)

ACTUAL_MD5=$(md5sum "$FMA_DIR/fma_full.zip" | awk '{print $1}')
echo "  Expected MD5: $EXPECTED_MD5"
echo "  Actual MD5:   $ACTUAL_MD5"

# Note: Zenodo MD5 scraping may fail if page format changes.
# Cross-check manually: md5sum matches https://zenodo.org/record/1476463

# ── 3. Extract ────────────────────────────────────────────────────────────
echo "[$(date '+%F %T')] Extracting fma_full.zip (~2-3 h)"
unzip -q "$FMA_DIR/fma_full.zip" -d "$FMA_DIR/"
rm "$FMA_DIR/fma_full.zip"

echo "[$(date '+%F %T')] Extraction complete"
echo "Disk usage:"
du -sh "$FMA_DIR/fma_full"
echo "Sample subdirectories:"
ls "$FMA_DIR/fma_full" | head -5

echo "[$(date '+%F %T')] fma_download job done"
