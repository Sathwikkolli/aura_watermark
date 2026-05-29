#!/bin/bash
# =============================================================================
# SLURM Job: Emilia EN selective download via HuggingFace streaming
#
# This is a NETWORK I/O job — no GPU needed.
#
# PARTITION GUIDE (Great Lakes):
#   If you have 'standard' partition access:
#     #SBATCH --partition=standard
#     (remove the --gres line entirely)
#
#   If you only have 'gpu' / 'spgpu' access:
#     #SBATCH --partition=gpu
#     #SBATCH --gres=gpu:1          ← one V100 allocated but unused
#
#   The DEFAULT below uses 'gpu' + 1 GPU which works universally.
#   Switch to 'standard' if your sinfo shows it (faster queue time).
#
# Check your partitions: sinfo -a -o "%-20P %-10a" | sort -u
# =============================================================================
#SBATCH --job-name=emilia_dl
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=14-00:00:00
#SBATCH --output=logs/emilia_dl_%j.log
#SBATCH --account=hafiz_root         # fix: sacctmgr show user $USER

set -euo pipefail

STORE=/nfs/turbo/umd-hafiz/issf_server_data
REPO="${SLURM_SUBMIT_DIR}"          # = wherever you ran sbatch from (repo root)
SCRIPTS="$REPO/scripts/dataset"
EMILIA_DIR="$STORE/emilia"
MANIFEST_DIR="$EMILIA_DIR/manifests"

mkdir -p "$EMILIA_DIR/EN" "$MANIFEST_DIR" "$REPO/logs"

conda activate aura

# Uncomment if not using huggingface-cli login:
# export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx

echo "[$(date '+%F %T')] Node: $(hostname)"
echo "[$(date '+%F %T')] Output:   $EMILIA_DIR"
echo "[$(date '+%F %T')] Manifest: $MANIFEST_DIR/emilia_raw.csv"

# ── Auto-detect resume ────────────────────────────────────────────────────────
RESUME_FLAG=""
if [ -f "$MANIFEST_DIR/emilia_raw.csv" ]; then
    N=$(wc -l < "$MANIFEST_DIR/emilia_raw.csv")
    echo "[$(date '+%F %T')] Existing manifest found ($N lines) — resuming"
    RESUME_FLAG="--resume"
fi

# ── Run download ───────────────────────────────────────────────────────────────
python "$SCRIPTS/download_emilia.py" \
    --output-dir "$EMILIA_DIR" \
    --manifest   "$MANIFEST_DIR/emilia_raw.csv" \
    --workers    8 \
    $RESUME_FLAG

echo "[$(date '+%F %T')] Download complete"
wc -l "$MANIFEST_DIR/emilia_raw.csv"
du -sh "$EMILIA_DIR/EN"
