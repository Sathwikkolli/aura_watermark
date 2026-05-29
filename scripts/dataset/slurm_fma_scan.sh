#!/bin/bash
# =============================================================================
# SLURM Job: FMA audio quality scan + genre-stratified curation
# Partition: gpu  (40 CPUs per node — all used for parallel soundfile reads)
# Expected runtime: 3-4 h scan  +  < 5 min curation
#
# Output:
#   fma/manifests/fma_raw.csv       ← quality scan results
#   fma/manifests/fma_curated.csv   ← genre-stratified final selection
# =============================================================================
#SBATCH --job-name=fma_scan
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=logs/fma_scan_%j.log
#SBATCH --account=hafiz_root

set -euo pipefail

STORE=/nfs/turbo/umd-hafiz/issf_server_data
SCRIPTS="${SLURM_SUBMIT_DIR}/scripts/dataset"
FMA_DIR="$STORE/fma"
MANIFEST_DIR="$FMA_DIR/manifests"
mkdir -p "$MANIFEST_DIR" logs

conda activate aura

echo "[$(date '+%F %T')] Node: $(hostname)"
echo "[$(date '+%F %T')] Checking FMA directories"
ls "$FMA_DIR/fma_full" | head -5
ls "$FMA_DIR/fma_metadata/"

# ── Quality scan (40 parallel workers) ────────────────────────────────────
echo "[$(date '+%F %T')] Starting FMA quality scan"
python "$SCRIPTS/scan_fma.py" \
    --fma-root      "$FMA_DIR/fma_full" \
    --metadata-dir  "$FMA_DIR/fma_metadata" \
    --out           "$MANIFEST_DIR/fma_raw.csv" \
    --workers       40

echo "[$(date '+%F %T')] Scan done."
wc -l "$MANIFEST_DIR/fma_raw.csv"

# ── Genre-stratified curation ─────────────────────────────────────────────
echo "[$(date '+%F %T')] Running curation"
python "$SCRIPTS/curate_fma.py" \
    --raw   "$MANIFEST_DIR/fma_raw.csv" \
    --out   "$MANIFEST_DIR/fma_curated.csv" \
    --seed  42

echo "[$(date '+%F %T')] Done."
wc -l "$MANIFEST_DIR/fma_curated.csv"
