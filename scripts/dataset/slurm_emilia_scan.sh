#!/bin/bash
# =============================================================================
# SLURM Job: Emilia JSON scan + curation
# Partition: gpu  (40 CPUs, 184 GB RAM per node)
# Expected runtime: 2-3 h scan  +  < 5 min curation
# =============================================================================
#SBATCH --job-name=emilia_scan
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=05:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=logs/emilia_scan_%j.log
#SBATCH --account=hafiz_root          # ← confirm with: sacctmgr show user $USER

set -euo pipefail

STORE=/nfs/turbo/umd-hafiz/issf_server_data
SCRIPTS="${SLURM_SUBMIT_DIR}/scripts/dataset"
mkdir -p "$STORE/emilia/manifests" logs

echo "[$(date '+%F %T')] Node: $(hostname)"
echo "[$(date '+%F %T')] Starting Emilia scan (32 workers)"

conda activate aura

# ── Phase 1a: Scan all JSON files ─────────────────────────────────────────────
python "$SCRIPTS/scan_emilia.py" \
    --emilia-root "$STORE/emilia" \
    --out         "$STORE/emilia/manifests/emilia_raw.csv" \
    --workers     32

echo "[$(date '+%F %T')] Scan complete. Lines in CSV:"
wc -l "$STORE/emilia/manifests/emilia_raw.csv"

# ── Phase 1b: DNSMOS-tiered curation ──────────────────────────────────────────
echo "[$(date '+%F %T')] Running curation"

python "$SCRIPTS/curate_emilia.py" \
    --raw   "$STORE/emilia/manifests/emilia_raw.csv" \
    --out   "$STORE/emilia/manifests/emilia_curated.csv" \
    --seed  42

echo "[$(date '+%F %T')] Curation complete. Lines in curated CSV:"
wc -l "$STORE/emilia/manifests/emilia_curated.csv"

echo "[$(date '+%F %T')] Job finished successfully"
