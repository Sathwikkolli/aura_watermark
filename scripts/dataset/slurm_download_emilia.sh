#!/bin/bash
# =============================================================================
# SLURM Job: Emilia EN selective download via HuggingFace streaming
#
# Downloads only English utterances that pass DNSMOS >= 3.2, duration 3-30 s,
# speaker cap 1 h. Target: ~2,500 h.
#
# Output layout:
#   /nfs/turbo/umd-hafiz/issf_server_data/
#   └── emilia/
#       ├── EN/
#       │   └── {speaker}/
#       │       └── {utt_id}.wav    ← only passing utterances
#       └── manifests/
#           └── emilia_raw.csv      ← written live (append mode, safe to resume)
#
# Prerequisites (once, on login node):
#   conda activate asd
#   pip install datasets huggingface_hub soundfile tqdm pandas
#   huggingface-cli login            ← accept terms at HF first
#   # OR: export HF_TOKEN=hf_xxx    ← uncomment line below
#
# Resume: just resubmit this script — already-saved utterances are skipped.
# =============================================================================
#SBATCH --job-name=emilia_dl
#SBATCH --partition=spgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:0
#SBATCH --time=14-00:00:00
#SBATCH --output=logs/emilia_dl_%j.log
#SBATCH --account=hafiz_root         # confirm: sacctmgr show user $USER

set -euo pipefail

STORE=/nfs/turbo/umd-hafiz/issf_server_data
REPO=/home/ksathwik/aura_watermark        # ← adjust to your repo path
SCRIPTS="$REPO/scripts/dataset"
EMILIA_DIR="$STORE/emilia"
MANIFEST_DIR="$EMILIA_DIR/manifests"

mkdir -p "$EMILIA_DIR/EN" "$MANIFEST_DIR" logs

conda activate asd

# Uncomment if not using huggingface-cli login:
# export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx

echo "[$(date '+%F %T')] Node: $(hostname)"
echo "[$(date '+%F %T')] Output:   $EMILIA_DIR"
echo "[$(date '+%F %T')] Manifest: $MANIFEST_DIR/emilia_raw.csv"

# ── Auto-detect resume ────────────────────────────────────────────────────
RESUME_FLAG=""
if [ -f "$MANIFEST_DIR/emilia_raw.csv" ]; then
    N=$(wc -l < "$MANIFEST_DIR/emilia_raw.csv")
    echo "[$(date '+%F %T')] Existing manifest found ($N lines) — resuming"
    RESUME_FLAG="--resume"
fi

# ── Download ──────────────────────────────────────────────────────────────
python "$SCRIPTS/download_emilia.py" \
    --output-dir "$EMILIA_DIR" \
    --manifest   "$MANIFEST_DIR/emilia_raw.csv" \
    --workers    8 \
    $RESUME_FLAG

echo "[$(date '+%F %T')] Download complete"
echo "Utterances saved:"
wc -l "$MANIFEST_DIR/emilia_raw.csv"
echo "Disk usage:"
du -sh "$EMILIA_DIR/EN"
