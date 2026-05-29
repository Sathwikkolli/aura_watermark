#!/usr/bin/env python3
"""
Emilia selective download — stream HuggingFace parquet shards, apply
DNSMOS + duration + speaker-cap filters, save only passing audio as MP3.

WHY THIS APPROACH:
  Emilia is ~46,000 hours. We need ~2,500 hours.
  Downloading everything wastes 18× the disk space we actually need.
  Instead: pull one parquet shard at a time (~200-500 MB each),
  extract passing utterances, delete the shard, move to the next.
  Net disk usage: ~200 GB audio (2,500 h × ~22 kbps MP3) + ~500 MB scratch.

REQUIREMENTS:
  pip install huggingface_hub datasets soundfile tqdm pandas

  You MUST have HuggingFace access to amphion/Emilia-Dataset:
    1. Go to https://huggingface.co/datasets/amphion/Emilia-Dataset
    2. Accept the terms of use
    3. Run: huggingface-cli login   (paste your token)

Usage (on Great Lakes):
  python download_emilia.py \\
      --output-dir /nfs/turbo/umd-hafiz/issf_server_data/emilia \\
      --manifest   /nfs/turbo/umd-hafiz/issf_server_data/emilia/manifests/emilia_raw.csv \\
      --workers    8 \\
      --hf-token   hf_xxxxxxxxxxxx

  # Resume an interrupted run (already-saved utterances are skipped):
  python download_emilia.py \\
      --output-dir /nfs/turbo/umd-hafiz/issf_server_data/emilia \\
      --manifest   /nfs/turbo/umd-hafiz/issf_server_data/emilia/manifests/emilia_raw.csv \\
      --resume
"""

import argparse
import csv
import io
import os
import tempfile
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm

# ── Filter constants — mirror curate_emilia.py ────────────────────────────────
DNSMOS_MIN     = 3.20    # absolute floor; we keep tiers 1-3
DUR_MIN_S      = 3.0
DUR_MAX_S      = 30.0
TARGET_H       = 2500.0  # stop after this many hours are saved
LANGUAGES      = {"EN"}             # English only

# ── Per-shard HF config ───────────────────────────────────────────────────────
HF_REPO        = "amphion/Emilia-Dataset"
HF_CONFIGS     = ["EN"]             # English only — no ZH/DE/FR/JA/KO


# ─────────────────────────────────────────────────────────────────────────────
# Audio save helper
# ─────────────────────────────────────────────────────────────────────────────

def save_audio_bytes(
    audio_bytes: bytes,
    out_path: Path,
    sample_rate: int = 48_000,
) -> bool:
    """Write audio bytes (raw PCM float32 array or bytes) to an MP3 file."""
    try:
        import soundfile as sf
        import numpy as np

        # audio_bytes could be raw float32 array bytes OR a pre-encoded audio file
        # HF Emilia typically provides {'array': np.ndarray, 'sampling_rate': int}
        # but some shards store raw bytes
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(audio_bytes, bytes):
            # Try writing as raw WAV, then let soundfile figure it out
            with io.BytesIO(audio_bytes) as buf:
                data, sr = sf.read(buf, dtype="float32", always_2d=False)
        elif isinstance(audio_bytes, np.ndarray):
            data, sr = audio_bytes, sample_rate
        else:
            return False

        # Write as 16-bit PCM WAV (smallest lossless format; re-encode to MP3 later if needed)
        sf.write(str(out_path), data, sr, subtype="PCM_16")
        return True

    except Exception:
        return False


def save_audio_array(
    array,       # np.ndarray
    sr: int,
    out_path: Path,
) -> bool:
    try:
        import soundfile as sf
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_path), array, sr, subtype="PCM_16")
        return True
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# DNSMOS extractor — handles all known Emilia schema variants
# ─────────────────────────────────────────────────────────────────────────────

def get_dnsmos(sample: dict) -> float:
    """Extract OVRL DNSMOS score from any known Emilia schema variant."""
    # Variant 1: flat float
    for key in ("dnsmos", "DNSMOS", "dns_mos", "mos"):
        v = sample.get(key)
        if v is not None and not isinstance(v, dict):
            return float(v)

    # Variant 2: nested dict {"OVRL": x, "SIG": y, "BAK": z}
    for key in ("dnsmos", "DNSMOS"):
        v = sample.get(key)
        if isinstance(v, dict):
            ovrl = v.get("OVRL") or v.get("ovrl") or v.get("overall")
            if ovrl is not None:
                return float(ovrl)

    # Variant 3: separate columns dnsmos_ovrl / dnsmos_sig / dnsmos_bak
    ovrl = sample.get("dnsmos_ovrl") or sample.get("ovrl")
    if ovrl is not None:
        return float(ovrl)

    return -1.0   # unknown — will be excluded


def get_duration(sample: dict) -> float:
    for key in ("duration", "dur", "length"):
        v = sample.get(key)
        if v is not None:
            return float(v)
    # Compute from audio array if available
    audio = sample.get("audio") or {}
    if isinstance(audio, dict):
        arr = audio.get("array")
        sr  = audio.get("sampling_rate", 48000)
        if arr is not None and hasattr(arr, "__len__"):
            return len(arr) / sr
    return 0.0


def get_speaker(sample: dict) -> str:
    for key in ("speaker", "spk", "speaker_id", "spkid"):
        v = sample.get(key)
        if v is not None:
            return str(v)
    return "UNK"


def get_language(sample: dict) -> str:
    for key in ("language", "lang", "locale"):
        v = sample.get(key)
        if v is not None:
            code = str(v).split("-")[0].upper()[:2]
            return code
    return "UNK"


# ─────────────────────────────────────────────────────────────────────────────
# Main download loop
# ─────────────────────────────────────────────────────────────────────────────

def download_emilia(
    output_dir: Path,
    manifest_path: Path,
    hf_token: str | None,
    workers: int,
    resume: bool,
):
    try:
        from datasets import load_dataset, DownloadConfig
    except ImportError:
        raise ImportError("Run: pip install datasets")

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Load existing manifest for resume ──────────────────────────────────
    saved_paths: set = set()
    total_saved_h = 0.0

    if resume and manifest_path.exists():
        df_existing = pd.read_csv(manifest_path)
        saved_paths = set(df_existing["path"].tolist())
        total_saved_h = df_existing["duration_s"].sum() / 3600
        print(f"Resuming: {len(saved_paths):,} utterances already saved "
              f"({total_saved_h:.1f} h)")

    # ── Open manifest CSV in append mode ───────────────────────────────────
    manifest_file = open(manifest_path, "a", newline="", encoding="utf-8")
    manifest_writer = csv.writer(manifest_file)
    if not resume or not manifest_path.exists():
        manifest_writer.writerow(["path", "duration_s", "speaker", "language", "dnsmos"])

    # ── Speaker hour tracking ───────────────────────────────────────────────
    speaker_hours: dict = {}
    SPEAKER_CAP_H  = 1.0

    if resume and manifest_path.exists():
        df_ex = pd.read_csv(manifest_path)
        speaker_hours = (df_ex.groupby("speaker")["duration_s"].sum() / 3600).to_dict()

    print(f"\nStreaming Emilia EN from HuggingFace (target: {TARGET_H:.0f} h)")
    print(f"DNSMOS filter: >= {DNSMOS_MIN}  |  Duration: {DUR_MIN_S}-{DUR_MAX_S} s")
    print(f"Speaker cap: {SPEAKER_CAP_H} h  |  Language: EN only\n")

    try:
        for lang_config in HF_CONFIGS:
            if total_saved_h >= TARGET_H:
                break

            print(f"\n[{lang_config}] Streaming shards …")
            try:
                ds = load_dataset(
                    HF_REPO,
                    name         = lang_config,
                    split        = "train",
                    streaming    = True,
                    token        = hf_token,
                    trust_remote_code = True,
                )
            except Exception as e:
                print(f"  Could not load {lang_config}: {e} — skipping")
                continue

            lang_saved   = 0
            lang_skipped = 0
            pbar = tqdm(desc=f"{lang_config}", unit="utt")

            for sample in ds:
                if total_saved_h >= TARGET_H:
                    break

                pbar.update(1)

                # ── Extract metadata ──────────────────────────────────────
                dnsmos   = get_dnsmos(sample)
                duration = get_duration(sample)
                speaker  = get_speaker(sample)
                language = get_language(sample)

                # ── Apply filters ─────────────────────────────────────────
                if dnsmos < DNSMOS_MIN:
                    lang_skipped += 1
                    continue
                if not (DUR_MIN_S <= duration <= DUR_MAX_S):
                    lang_skipped += 1
                    continue
                if speaker_hours.get(speaker, 0.0) >= SPEAKER_CAP_H:
                    lang_skipped += 1
                    continue

                # ── Build output path ─────────────────────────────────────
                # Use a content-addressed path: lang/spk/utt_id.wav
                utt_id   = str(sample.get("id") or sample.get("utt_id") or
                               f"{speaker}_{lang_saved:08d}")
                rel_path = Path(language) / speaker[:8] / f"{utt_id}.wav"
                out_path = output_dir / rel_path

                if str(out_path) in saved_paths:
                    continue   # resume: already saved

                # ── Save audio ────────────────────────────────────────────
                audio_data = sample.get("audio") or sample.get("wav")
                saved = False

                if isinstance(audio_data, dict):
                    arr = audio_data.get("array")
                    sr  = int(audio_data.get("sampling_rate", 48000))
                    if arr is not None:
                        saved = save_audio_array(arr, sr, out_path)
                elif isinstance(audio_data, bytes):
                    saved = save_audio_bytes(audio_data, out_path)

                if not saved:
                    lang_skipped += 1
                    continue

                # ── Update state ──────────────────────────────────────────
                dur_h = duration / 3600
                speaker_hours[speaker]  = speaker_hours.get(speaker, 0.0) + dur_h
                total_saved_h          += dur_h
                lang_saved             += 1
                saved_paths.add(str(out_path))

                manifest_writer.writerow([str(out_path), duration, speaker,
                                          language, dnsmos])
                manifest_file.flush()

                pbar.set_postfix({
                    "saved": f"{lang_saved}",
                    "total_h": f"{total_saved_h:.1f}",
                    "skip": f"{lang_skipped}",
                })

            pbar.close()
            lang_h = lang_hours.get(lang_config, 0.0)
            print(f"  {lang_config}: saved {lang_saved:,} utts  "
                  f"({lang_h:.1f} h)  |  skipped {lang_skipped:,}")

    finally:
        manifest_file.close()

    print(f"\n{'='*55}")
    print(f"EMILIA DOWNLOAD COMPLETE")
    print(f"  Total saved: {total_saved_h:.1f} h")
    print(f"  Utterances:  {len(saved_paths):,}")
    print(f"  Manifest:    {manifest_path}")
    print(f"{'='*55}")


# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Selective Emilia download via HuggingFace streaming",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--output-dir", required=True,
                    help="Local root to save audio (e.g. …/emilia)")
    ap.add_argument("--manifest",   required=True,
                    help="CSV manifest to write (e.g. …/emilia/manifests/emilia_raw.csv)")
    ap.add_argument("--hf-token",   default=None,
                    help="HuggingFace token (or set HF_TOKEN env var). "
                         "Required: dataset is gated.")
    ap.add_argument("--workers",    type=int, default=8,
                    help="Parallel IO threads for saving audio")
    ap.add_argument("--resume",     action="store_true",
                    help="Skip already-saved utterances (safe to rerun)")
    args = ap.parse_args()

    token = args.hf_token or os.environ.get("HF_TOKEN")
    if not token:
        print("WARNING: No HuggingFace token provided.")
        print("  If the dataset is gated you will get a 401 error.")
        print("  Run: huggingface-cli login")
        print("  Or:  export HF_TOKEN=hf_xxxxxxxxxxxx")

    download_emilia(
        output_dir    = Path(args.output_dir),
        manifest_path = Path(args.manifest),
        hf_token      = token,
        workers       = args.workers,
        resume        = args.resume,
    )


if __name__ == "__main__":
    main()
