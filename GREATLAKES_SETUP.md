# Great Lakes Deployment Guide
## Push code → clone on HPC → set up environment → run dataset jobs

---

## Overview

```
Your Windows PC                    Great Lakes (UMich HPC)
──────────────────                 ──────────────────────────────
aura_watermark/          git push  github.com/you/aura_watermark
    └── all code      ──────────►  ↓  git clone
                                   ~/aura_watermark/
                                   ↓  bash setup_greatlakes.sh
                                   conda env aura  (PyTorch + deps)
                                   ↓  sbatch slurm_download_emilia.sh
                                   /nfs/turbo/.../emilia/EN/   ← 2,500 h audio
                                   /nfs/turbo/.../fma/fma_full/ ← 2,500 h audio
```

---

## Part 1 — Push Code to GitHub (Windows PC)

### Step 1.1 — Install Git (if not already installed)

Download from https://git-scm.com/download/win and install with default options.  
Open **Git Bash** (search in Start Menu) for all commands below.

### Step 1.2 — Configure Git identity (one-time)

```bash
git config --global user.name  "Sathwik Kolli"
git config --global user.email "gayathrikolli62@gmail.com"
```

### Step 1.3 — Create a GitHub repository

1. Go to https://github.com/new
2. Repository name: `aura_watermark`
3. Set to **Private** (research code)
4. **Do NOT** initialise with README / .gitignore — you already have them
5. Click **Create repository**
6. Copy the HTTPS URL shown: `https://github.com/Sathwikkolli/aura_watermark.git`

### Step 1.4 — Create a GitHub Personal Access Token

GitHub no longer accepts passwords over HTTPS. You need a token.

1. Go to https://github.com/settings/tokens
2. Click **Generate new token (classic)**
3. Name: `greatlakes-push`
4. Expiry: 90 days (or No expiration for research)
5. Scopes: check **repo** (full control of private repositories)
6. Click **Generate token**
7. **Copy the token immediately** — you will not see it again  
   It looks like: `ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`

### Step 1.5 — Initialise git and push

Open **Git Bash**, then:

```bash
cd /c/Users/Sathwik/aura_watermark

# Initialise (if not already a git repo)
git init
git branch -M main

# Stage everything
git add .

# Verify what will be committed (check nothing sensitive is included)
git status
# You should NOT see: .env, *.pt, checkpoints/, data/train.csv, data/val.csv

# First commit
git commit -m "Initial commit: AURA watermarking implementation (ICASSP 2026)"

# Add remote
git remote add origin https://github.com/Sathwikkolli/aura_watermark.git

# Push
git push -u origin main
# When prompted:
#   Username: Sathwikkolli
#   Password: paste your token  (ghp_xxxx...)
```

**Verify:** Go to `https://github.com/Sathwikkolli/aura_watermark` — all files should be there.

### Step 1.6 — Store the token in Git credential cache (optional but convenient)

```bash
git config --global credential.helper store
# Next push will save the token — no more prompts
```

---

## Part 2 — Set Up on Great Lakes

All commands below run in your **SSH session** on Great Lakes.

### Step 2.1 — SSH into Great Lakes

```bash
# From any terminal (Windows PowerShell, Git Bash, macOS Terminal)
ssh ksathwik@greatlakes.arc-ts.umich.edu
```

You land on a **login node** (gl-login1 or similar). This is where you run
`git`, `conda`, `sbatch`, and short commands. Never run training here.

### Step 2.2 — Configure Git on Great Lakes (one-time)

```bash
git config --global user.name  "Sathwik Kolli"
git config --global user.email "gayathrikolli62@gmail.com"
git config --global credential.helper store   # saves token after first pull
```

### Step 2.3 — Clone the repository

```bash
# Clone into your home directory
cd ~
git clone https://github.com/Sathwikkolli/aura_watermark.git
# Username: Sathwikkolli
# Password: your GitHub token (ghp_xxxx...)

cd aura_watermark
ls    # should see: aura_watermark/ train.py infer.py scripts/ tests/ ...
```

### Step 2.4 — Create the conda environment and install everything

```bash
# Run the setup script (takes ~10 min, run on login node — it's just pip installs)
bash scripts/setup_greatlakes.sh
```

This script:
- Creates conda env `aura` (Python 3.11)
- Installs PyTorch + torchaudio for CUDA 12.4 (compatible with your driver 580)
- Installs the project package in editable mode (`pip install -e .`)
- Installs dataset tools (datasets, huggingface_hub, soundfile, aria2c)
- Runs a quick smoke test (forward pass on CPU)

Expected output at the end:
```
  Import + forward pass: OK
============================================================
 Setup complete. Activate with:  conda activate aura
============================================================
```

### Step 2.5 — Verify GPU access on a compute node

```bash
# Request a 5-min interactive session on one A40 GPU
srun --partition=spgpu --nodes=1 --cpus-per-task=4 \
     --mem=16G --gres=gpu:1 --time=00:05:00 \
     --account=hafiz_root --pty bash

# Once inside the compute node:
conda activate aura
python -c "
import torch
print('CUDA available:', torch.cuda.is_available())
print('GPU:', torch.cuda.get_device_name(0))
print('VRAM:', torch.cuda.get_device_properties(0).total_memory / 1e9, 'GB')
"
# Expected:
# CUDA available: True
# GPU: NVIDIA A40
# VRAM: 46.068... GB
exit   # leave the interactive session
```

> **Note on `--account`:** If `hafiz_root` doesn't work, check your account with:
> ```bash
> sacctmgr show user $USER format=user,account,defaultaccount
> ```

### Step 2.6 — Set up HuggingFace token (for Emilia)

```bash
conda activate aura

# Accept dataset terms first (do this in your browser):
# https://huggingface.co/datasets/amphion/Emilia-Dataset
# Click "Access repository" and accept the terms.

# Then log in:
huggingface-cli login
# Paste your HF token (hf_xxxx...) when prompted
# Token is saved at ~/.cache/huggingface/token

# Verify:
python -c "from huggingface_hub import whoami; print('Logged in as:', whoami()['name'])"
```

### Step 2.7 — Create storage directories

```bash
STORE=/nfs/turbo/umd-hafiz/issf_server_data

mkdir -p $STORE/emilia/EN
mkdir -p $STORE/emilia/manifests
mkdir -p $STORE/fma/fma_full
mkdir -p $STORE/fma/manifests
mkdir -p ~/aura_watermark/data
mkdir -p ~/aura_watermark/logs

# Confirm
ls $STORE
# Should see: emilia/  fma/  (plus existing datasets like ADD, musan, etc.)
```

---

## Part 3 — Run the Dataset Download Jobs

Submit both jobs immediately — they run in parallel.

### Step 3.1 — Update the SLURM account in all scripts (one-time)

Check your actual Slurm account first:
```bash
sacctmgr show user $USER format=user,account,defaultaccount
```

If your account is NOT `hafiz_root`, update all scripts:
```bash
cd ~/aura_watermark

# Replace hafiz_root with your actual account everywhere
YOUR_ACCOUNT=hafiz_root   # ← change this if needed

# Check what's set
grep "account" scripts/dataset/slurm_*.sh

# Update if different:
# sed -i "s/hafiz_root/$YOUR_ACCOUNT/g" scripts/dataset/slurm_*.sh
```

### Step 3.2 — Submit Emilia download

```bash
conda activate aura
cd ~/aura_watermark

sbatch scripts/dataset/slurm_download_emilia.sh

# Check it started
squeue --me
# Expected output:
#  JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST
# 123456     spgpu emilia_dl ksathwik  R       0:05      1 gl1507
```

### Step 3.3 — Submit FMA download (same time)

```bash
sbatch scripts/dataset/slurm_download_fma.sh

squeue --me
# Should now see 2 jobs running
```

### Step 3.4 — Monitor progress

```bash
# Watch job status (updates every 30 s)
watch -n 30 squeue --me

# Follow Emilia download log live
tail -f logs/emilia_dl_*.log

# Follow FMA download log live (in another terminal)
tail -f logs/fma_dl_*.log

# Check how much has been saved so far
STORE=/nfs/turbo/umd-hafiz/issf_server_data
du -sh $STORE/emilia/EN
find $STORE/fma/fma_full -name "*.mp3" | wc -l
wc -l $STORE/emilia/manifests/emilia_raw.csv
```

### Step 3.5 — After downloads finish

```bash
STORE=/nfs/turbo/umd-hafiz/issf_server_data

# Curate Emilia (< 5 min, login node)
python scripts/dataset/curate_emilia.py \
    --raw  $STORE/emilia/manifests/emilia_raw.csv \
    --out  $STORE/emilia/manifests/emilia_curated.csv \
    --seed 42

# Scan FMA audio quality (3-4 h, SLURM job)
sbatch scripts/dataset/slurm_fma_scan.sh

# After fma_scan finishes — build combined manifest (login node)
python scripts/dataset/build_combined_manifest.py \
    --emilia  $STORE/emilia/manifests/emilia_curated.csv \
    --fma     $STORE/fma/manifests/fma_curated.csv \
    --out-dir ~/aura_watermark/data \
    --val-h   50 \
    --seed    42

# Sanity check before training
sbatch scripts/dataset/slurm_sanity_check.sh
tail -f logs/sanity_*.log
```

---

## Part 4 — Keeping Code in Sync

After making changes on your **Windows PC**:

```bash
# On Windows (Git Bash):
cd /c/Users/Sathwik/aura_watermark
git add .
git commit -m "describe your change"
git push
```

On **Great Lakes** to pull the latest code:

```bash
# On Great Lakes login node:
cd ~/aura_watermark
git pull
```

If you edited files **on Great Lakes** and want them back locally:

```bash
# On Great Lakes:
git add scripts/dataset/slurm_emilia_scan.sh   # stage specific files
git commit -m "fix SLURM account name"
git push

# On Windows:
git pull
```

---

## Part 5 — Run Training (After Sanity Check Passes)

```bash
cd ~/aura_watermark
conda activate aura

# Dry run first (2 steps, verifies end-to-end on real data)
python train.py \
    --emilia-root /nfs/turbo/umd-hafiz/issf_server_data/emilia \
    --fma-root    /nfs/turbo/umd-hafiz/issf_server_data/fma \
    --fma-subset  fma_full \
    --dry-run \
    --device cuda

# Full training (submit as SLURM job — see below)
sbatch scripts/train/slurm_train.sh   # ← to be created
```

---

## Quick Reference

```bash
# ── SSH ────────────────────────────────────────────────────────────────────
ssh ksathwik@greatlakes.arc-ts.umich.edu

# ── Conda ──────────────────────────────────────────────────────────────────
conda activate aura
conda deactivate

# ── Git ────────────────────────────────────────────────────────────────────
git status
git pull
git add . && git commit -m "msg" && git push

# ── SLURM ──────────────────────────────────────────────────────────────────
sbatch <script.sh>           # submit job
squeue --me                  # list your jobs
scancel <jobid>              # cancel job
seff <jobid>                 # efficiency report after job ends
tail -f logs/<name>_*.log    # follow live log

# ── Storage ────────────────────────────────────────────────────────────────
STORE=/nfs/turbo/umd-hafiz/issf_server_data
du -sh $STORE/*              # disk usage by dataset
df -h /nfs/turbo/umd-hafiz/  # total quota

# ── Interactive GPU session (debugging) ────────────────────────────────────
srun --partition=spgpu --nodes=1 --cpus-per-task=8 \
     --mem=32G --gres=gpu:1 --time=01:00:00 \
     --account=hafiz_root --pty bash
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `Permission denied (publickey)` on `git push` | Using SSH URL instead of HTTPS | Use `https://github.com/...` not `git@github.com:...` |
| `remote: Repository not found` | Wrong URL or no access | Check URL: `git remote -v` |
| `conda: command not found` on GL | Module not loaded | `module load python3.10-anaconda` or check with `which conda` |
| `sbatch: error: Invalid account` | Wrong `--account` | Run `sacctmgr show user $USER` to find correct account |
| `CUDA out of memory` | Batch size too large | Reduce `--batch-size` or increase `--grad-accum` |
| `huggingface_hub.errors.RepositoryNotFoundError` | Not logged in / not accepted terms | `huggingface-cli login` + accept terms at HF |
| Job immediately fails (`(F)` in squeue) | Script error | `tail logs/<name>_*.log` for the actual error |
| `ModuleNotFoundError: No module named 'aura_watermark'` | Not installed in env | `cd ~/aura_watermark && pip install -e .` |
