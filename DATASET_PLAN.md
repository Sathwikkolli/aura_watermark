# AURA Dataset Preparation Plan
## Great Lakes HPC (UMich) — `/nfs/turbo/umd-hafiz/issf_server_data`

**Speech:** Emilia — English only — 2,500 h  
**Music:**  FMA full-length — genre-stratified — 2,500 h  
**Cluster:** `spgpu` (A40 × 8) and `gpu` (V100, 40 CPUs)

---

## Directory Layout

Each dataset owns its own subfolder **and** its own `manifests/` subfolder.  
The combined `train.csv` / `val.csv` live in the repo, not on the storage mount,
because they contain absolute HPC paths that are project-specific.

```
/nfs/turbo/umd-hafiz/issf_server_data/
│
├── emilia/
│   ├── EN/                              ← English audio only
│   │   └── {speaker_id}/
│   │       └── {utt_id}.wav            ← only DNSMOS-passing utterances
│   └── manifests/
│       ├── emilia_raw.csv              ← written live during streaming
│       └── emilia_curated.csv          ← DNSMOS tiers + speaker cap applied
│
└── fma/
    ├── fma_metadata/                    ← tracks.csv, genres.csv (342 MB)
    ├── fma_full/
    │   └── 000/  001/  … 106/          ← only selected MP3s
    └── manifests/
        ├── fma_selected_ids.csv         ← pre-download track list
        ├── fma_raw.csv                  ← quality scan results
        └── fma_curated.csv             ← genre-stratified final selection

~/aura_watermark/
└── data/
    ├── train.csv                        ← combined manifest (gitignored)
    └── val.csv
```

---

## Step 0 — One-Time Setup (Login Node, 15 min)

```bash
ssh ksathwik@greatlakes.arc-ts.umich.edu
conda activate asd

# Install packages
pip install datasets huggingface_hub soundfile tqdm pandas requests
conda install -c conda-forge aria2 -y

# Create storage directories
STORE=/nfs/turbo/umd-hafiz/issf_server_data
mkdir -p $STORE/emilia/EN $STORE/emilia/manifests
mkdir -p $STORE/fma/fma_full $STORE/fma/manifests
mkdir -p ~/aura_watermark/data
mkdir -p ~/aura_watermark/logs

# HuggingFace login (Emilia is a gated dataset)
# 1. Go to https://huggingface.co/datasets/amphion/Emilia-Dataset
# 2. Accept terms of use (click "Access repository")
# 3. Go to https://huggingface.co/settings/tokens → New token (Read)
huggingface-cli login    # paste token when prompted

# Verify
python -c "from huggingface_hub import whoami; print(whoami()['name'])"
```

---

## Step 1 — Download Emilia EN (~2,500 h)

### How it works

Streams one HuggingFace shard at a time. For each utterance the DNSMOS score
and duration are checked **before** the audio bytes are saved — only passing
utterances are written to disk. The manifest CSV is written in append mode so
the job is safe to resume after preemption.

**Filter:** DNSMOS ≥ 3.2 · duration 3–30 s · speaker cap 1 h · EN only  
**Output size:** ~200 GB (2,500 h × ~22 kbps average WAV → re-encode MP3 if needed)

```bash
# Submit (runs 12-24 h, resumable)
cd ~/aura_watermark
sbatch scripts/dataset/slurm_download_emilia.sh

# Monitor
tail -f logs/emilia_dl_*.log
squeue --me

# After job finishes — run curation tiers (login node, < 5 min)
STORE=/nfs/turbo/umd-hafiz/issf_server_data
python scripts/dataset/curate_emilia.py \
    --raw  $STORE/emilia/manifests/emilia_raw.csv \
    --out  $STORE/emilia/manifests/emilia_curated.csv \
    --seed 42
```

### DNSMOS tiers applied during curation

| Tier | DNSMOS | Target |
|------|--------|--------|
| 1 — High quality | ≥ 3.90 | 1,400 h |
| 2 — Good quality | 3.50 – 3.90 | 700 h |
| 3 — Acceptable | 3.20 – 3.50 | 300 h |
| Speaker cap | — | 1 h / speaker |

Tier 3 is intentionally included: it exposes the NMR loss to audio where
background noise already occupies psychoacoustic masking bands — improving
robustness to the `noise` and `pink_noise` attacks.

---

## Step 2 — Download FMA (~2,500 h)

### How it works

```
fma_metadata.zip (342 MB) → curate_fma.py → fma_selected_ids.csv
  → 32 parallel wget threads → individual MP3s in fma_full/
```

FMA distributes individual track MP3s at:
`https://files.freemusicarchive.org/storage-freemusicarchive-org/tracks/012345.mp3`

Per-track download means:
- Skip tracks that fail metadata quality pre-filter (saves bandwidth)
- Parallelize with 32 threads (much faster than one 879 GB zip)
- Resume trivially on preemption — already-downloaded files are skipped

```bash
# Submit (runs 12-24 h, resumable)
sbatch scripts/dataset/slurm_download_fma.sh

# After download — run quality scan then curation
sbatch scripts/dataset/slurm_fma_scan.sh
```

### Genre cap applied during curation

No single genre contributes more than 300 h. Without this, Electronic and
Rock would dominate (~45% combined) and the model would over-optimise the
watermark for those spectral characteristics.

---

## Step 3 — Build Combined Manifests (Login Node, < 1 min)

Run after both curated CSVs exist:

```bash
STORE=/nfs/turbo/umd-hafiz/issf_server_data
conda activate asd

python scripts/dataset/build_combined_manifest.py \
    --emilia  $STORE/emilia/manifests/emilia_curated.csv \
    --fma     $STORE/fma/manifests/fma_curated.csv \
    --out-dir ~/aura_watermark/data \
    --val-h   50 \
    --seed    42
```

Expected output:
```
train.csv:  ~1.8 M clips   4,950 h
  Speech:   ~940,000 utts   (2,475 h)
  Music:    ~860,000 clips  (2,475 h)
val.csv:    ~22,000 clips   50 h
  Emilia val: speaker-disjoint (held-out speakers not in train)
  FMA val:    random 25 h holdout
```

---

## Step 4 — Sanity Check

```bash
sbatch scripts/dataset/slurm_sanity_check.sh
tail -f logs/sanity_*.log
# Must end with: [SANITY CHECK PASSED]
```

---

## Full Execution Order

Steps 1 and 2 can run **in parallel** — submit both immediately after Step 0.

```
Step 0  Login node   Setup + HF login                    15 min
Step 1  spgpu job    Emilia EN download (resumable)       12-24 h  ┐ parallel
Step 2  gpu job      FMA download + scan (resumable)      12-24 h  ┘
Step 2b gpu job      FMA quality scan (after download)    3-4 h
Step 3  Login node   Build combined manifest              < 1 min
Step 4  spgpu job    Sanity check                         30 min
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `401 Unauthorized` from HF | `huggingface-cli login` again, or `export HF_TOKEN=hf_xxx` in SLURM script |
| FMA tracks returning 404 | Normal — ~2-5% of FMA tracks were deleted by creators. Rerun with `--resume` for transient network failures. |
| Job preempted | Resubmit same `sbatch` command — both scripts auto-resume |
| Disk quota | `df -h /nfs/turbo/umd-hafiz/` — contact ARC-TS if needed |

```bash
# Check download progress at any time
STORE=/nfs/turbo/umd-hafiz/issf_server_data

echo "=== Emilia ==="
du -sh $STORE/emilia/EN
wc -l $STORE/emilia/manifests/emilia_raw.csv

echo "=== FMA ==="
find $STORE/fma/fma_full -name "*.mp3" | wc -l
du -sh $STORE/fma/fma_full
```

---

## Manifest Schema

Both `train.csv` and `val.csv` share this schema:

```
path        — absolute path to audio file (.wav for Emilia, .mp3 for FMA)
duration_s  — clip duration in seconds
speaker     — speaker ID (Emilia) or "music" (FMA)
language    — "EN" (Emilia) or "music" (FMA)
dnsmos      — DNSMOS OVRL score (Emilia only; -1.0 for FMA)
rms_db      — RMS energy dB (FMA only; -1.0 for Emilia)
clip_frac   — fraction of clipped samples (FMA only)
genre_top   — top-level genre (FMA only; "speech" for Emilia)
dataset     — "emilia" or "fma"
```

`dataset.py` reads only `path` and `duration_s`.
All other columns are metadata for post-hoc analysis.
