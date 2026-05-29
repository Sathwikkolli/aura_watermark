#!/bin/bash
# =============================================================================
# AURA — Great Lakes environment setup
# Run once on a LOGIN NODE after cloning the repo.
#
# Usage:
#   bash scripts/setup_greatlakes.sh
# =============================================================================
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
echo "Repo: $REPO_DIR"

# ── 1. Create / update conda environment ─────────────────────────────────────
ENV_NAME="aura"

if conda env list | grep -q "^${ENV_NAME} "; then
    echo "[1/5] Conda env '${ENV_NAME}' already exists — updating"
    conda activate ${ENV_NAME}
else
    echo "[1/5] Creating conda env '${ENV_NAME}' (Python 3.11)"
    conda create -y -n ${ENV_NAME} python=3.11
    conda activate ${ENV_NAME}
fi

# ── 2. Install PyTorch for CUDA 12.4 ─────────────────────────────────────────
# Great Lakes A40 GPUs: driver 580 supports CUDA up to 13.0 runtime.
# PyTorch cu124 binaries run correctly on any driver >= CUDA 12.4.
echo "[2/5] Installing PyTorch (cu124 — compatible with driver 580 / CUDA 13.0)"
pip install torch torchaudio \
    --index-url https://download.pytorch.org/whl/cu124 \
    --upgrade

# Verify GPU is visible
python -c "
import torch
print(f'  PyTorch:  {torch.__version__}')
print(f'  CUDA ok:  {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU:      {torch.cuda.get_device_name(0)}')
    print(f'  VRAM:     {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')
"

# ── 3. Install project + all extras ──────────────────────────────────────────
echo "[3/5] Installing aura_watermark package"
cd "$REPO_DIR"
pip install -e ".[data,logging]"

# ── 4. Install dataset utilities ─────────────────────────────────────────────
echo "[4/5] Installing dataset tools"
pip install soundfile datasets huggingface_hub tqdm pandas requests
conda install -c conda-forge aria2 -y   # resumable downloader

# ── 5. Quick smoke test ───────────────────────────────────────────────────────
echo "[5/5] Smoke test"
python -c "
from aura_watermark.config import AURAConfig
from aura_watermark.embedder import StegaformerEmbedder
import torch

cfg = AURAConfig()
cfg.conformer.n_blocks = 1
cfg.conformer.use_gradient_checkpointing = False
m = StegaformerEmbedder(cfg)
x = torch.randn(1, 1, 96000)
msg = torch.randint(0, 2, (1, 32))
out, _, _ = m(x, msg)
assert out.shape == (1, 1, 96000)
print('  Import + forward pass: OK')
"

echo ""
echo "============================================================"
echo " Setup complete. Activate with:  conda activate ${ENV_NAME}"
echo "============================================================"
