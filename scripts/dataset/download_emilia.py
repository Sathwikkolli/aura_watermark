#!/usr/bin/env python3
"""
Emilia selective download — stream HuggingFace shards, apply DNSMOS +
duration + speaker-cap filters, save only English utterances.

KEY FACTS about amphion/Emilia-Dataset on HuggingFace:
  - Only ONE config exists: 'default'  (not separate 'EN', 'ZH' etc.)
  - Language is a FIELD inside each sample, not a separate config
  - We stream 'default' and filter samples where language == 'EN'
  - Dataset is gated: must accept terms and login with huggingface-cli

REQUIREMENTS:
  pip install datasets huggingface_hub soundfile tqdm pandas requests
  huggingface-cli login            # one-time, saves token to ~/.cache/

Usage (on Great Lakes login node):
  python download_emilia.py \\
      --output-dir /nfs/turbo/umd-hafiz/issf_server_data/emilia \\
      --manifest   /nfs/turbo/umd-hafiz/issf_server_data/emilia/manifests/emilia_raw.csv

  # Resume after interruption:
  python download_emilia.py \\
      --output-dir /nfs/turbo/umd-hafiz/issf_server_data/emilia \\
      --manifest   /nfs/turbo/umd-hafiz/issf_server_data/emilia/manifests/emilia_raw.csv \\
      --resume
"""

from __future__ import annotations

import argparse
import csv
import io
import os
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# ── Filter constants ──────────────────────────────────────────────────────────
DNSMOS_MIN    = 3.20
DUR_MIN_S     = 3.0
DUR_MAX_S     = 30.0
TARGET_H      = 2500.0
SPEAKER_CAP_H = 1.0
TARGET_LANG   = "EN"           # English utterances only

# ── HuggingFace config ────────────────────────────────────────────────────────
HF_REPO   = "amphion/Emilia-Dataset"
HF_CONFIG = "default"          # Only config available — language filtered by field


# ─────────────────────────────────────────────────────────────────────────────
# Token helpers
# ─────────────────────────────────────────────────────────────────────────────

def resolve_token(cli_token: str | None) -> str | None:
    """
    Token resolution priority:
      1. --hf-token argument
      2. HF_TOKEN environment variable
      3. Token saved by `huggingface-cli login` (~/.cache/huggingface/token)
    """
    if cli_token:
        return cli_token
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    try:
        from huggingface_hub import HfFolder
        cached = HfFolder.get_token()
        if cached:
            return cached
    except Exception:
        pass
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Metadata extractors — handle all known Emilia schema variants
# ─────────────────────────────────────────────────────────────────────────────

def get_dnsmos(sample: dict) -> float:
    for key in ("dnsmos", "DNSMOS", "dns_mos", "mos"):
        v = sample.get(key)
        if v is not None and not isinstance(v, dict):
            return float(v)
    for key in ("dnsmos", "DNSMOS"):
        v = sample.get(key)
        if isinstance(v, dict):
            ovrl = v.get("OVRL") or v.get("ovrl") or v.get("overall")
            if ovrl is not None:
                return float(ovrl)
    ovrl = sample.get("dnsmos_ovrl") or sample.get("ovrl")
    if ovrl is not None:
        return float(ovrl)
    return -1.0


def get_duration(sample: dict) -> float:
    for key in ("duration", "dur", "length"):
        v = sample.get(key)
        if v is not None:
            return float(v)
    audio = sample.get("audio") or {}
    if isinstance(audio, dict):
        arr = audio.get("array")
        sr  = audio.get("sampling_rate", 48_000)
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
    """Return normalised 2-letter language code: 'en-US' → 'EN'."""
    for key in ("language", "lang", "locale"):
        v = sample.get(key)
        if v is not None:
            return str(v).split("-")[0].upper()[:2]
    return "UNK"


# ─────────────────────────────────────────────────────────────────────────────
# Audio save
# ─────────────────────────────────────────────────────────────────────────────

def save_audio_array(array, sr: int, out_path: Path) -> bool:
    try:
        import soundfile as sf
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_path), array, sr, subtype="PCM_16")
        return True
    except Exception:
        return False


def save_audio_bytes(audio_bytes: bytes, out_path: Path) -> bool:
    try:
        import soundfile as sf
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with io.BytesIO(audio_bytes) as buf:
            data, sr = sf.read(buf, dtype="float32", always_2d=False)
        sf.write(str(out_path), data, sr, subtype="PCM_16")
        return True
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Main download
# ─────────────────────────────────────────────────────────────────────────────

def download_emilia(
    output_dir:    Path,
    manifest_path: Path,
    hf_token:      str | None,
    resume:        bool,
) -> None:
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Run: pip install datasets")

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Resume state ──────────────────────────────────────────────────────────
    saved_paths:   set  = set()
    speaker_hours: dict = {}
    total_saved_h       = 0.0

    if resume and manifest_path.exists():
        df_ex = pd.read_csv(manifest_path)
        saved_paths   = set(df_ex["path"].tolist())
        speaker_hours = (df_ex.groupby("speaker")["duration_s"].sum() / 3600).to_dict()
        total_saved_h = df_ex["duration_s"].sum() / 3600
        print(f"Resuming: {len(saved_paths):,} utterances already saved "
              f"({total_saved_h:.1f} h)")

    # ── Open manifest ─────────────────────────────────────────────────────────
    write_header = not (resume and manifest_path.exists())
    manifest_file   = open(manifest_path, "a", newline="", encoding="utf-8")
    manifest_writer = csv.writer(manifest_file)
    if write_header:
        manifest_writer.writerow(["path", "duration_s", "speaker", "language", "dnsmos"])

    print(f"\nStreaming {HF_REPO} config='{HF_CONFIG}' (target: {TARGET_H:.0f} h EN)")
    print(f"Filters: DNSMOS>={DNSMOS_MIN}  dur {DUR_MIN_S}-{DUR_MAX_S}s  "
          f"speaker<={SPEAKER_CAP_H}h  lang={TARGET_LANG}")

    saved   = 0
    skipped_lang  = 0
    skipped_other = 0

    try:
        ds = load_dataset(
            HF_REPO,
            name              = HF_CONFIG,
            split             = "train",
            streaming         = True,
            token             = hf_token,
            trust_remote_code = True,
        )

        pbar = tqdm(desc="Streaming Emilia", unit="utt")

        for sample in ds:
            if total_saved_h >= TARGET_H:
                break

            pbar.update(1)

            # ── Language filter (must be EN) ──────────────────────────────────
            language = get_language(sample)
            if language != TARGET_LANG:
                skipped_lang += 1
                continue

            # ── Extract metadata ──────────────────────────────────────────────
            dnsmos   = get_dnsmos(sample)
            duration = get_duration(sample)
            speaker  = get_speaker(sample)

            # ── Quality / duration / DNSMOS filters ───────────────────────────
            if dnsmos < DNSMOS_MIN:
                skipped_other += 1
                continue
            if not (DUR_MIN_S <= duration <= DUR_MAX_S):
                skipped_other += 1
                continue
            if speaker_hours.get(speaker, 0.0) >= SPEAKER_CAP_H:
                skipped_other += 1
                continue

            # ── Build output path ─────────────────────────────────────────────
            utt_id   = str(sample.get("id") or sample.get("utt_id") or
                           f"{speaker}_{saved:09d}")
            rel_path = Path("EN") / speaker[:12] / f"{utt_id}.wav"
            out_path = output_dir / rel_path

            if str(out_path) in saved_paths:
                continue  # already saved in previous run

            # ── Save audio ────────────────────────────────────────────────────
            audio_data = sample.get("audio") or sample.get("wav")
            ok = False
            if isinstance(audio_data, dict):
                arr = audio_data.get("array")
                sr  = int(audio_data.get("sampling_rate", 48_000))
                if arr is not None:
                    ok = save_audio_array(arr, sr, out_path)
            elif isinstance(audio_data, bytes):
                ok = save_audio_bytes(audio_data, out_path)

            if not ok:
                skipped_other += 1
                continue

            # ── Update state ──────────────────────────────────────────────────
            dur_h                    = duration / 3600
            speaker_hours[speaker]   = speaker_hours.get(speaker, 0.0) + dur_h
            total_saved_h           += dur_h
            saved                   += 1
            saved_paths.add(str(out_path))

            manifest_writer.writerow([str(out_path), round(duration, 3),
                                      speaker, language, round(dnsmos, 4)])
            manifest_file.flush()

            if saved % 500 == 0:
                pbar.set_postfix({
                    "EN_saved": f"{saved}",
                    "total_h":  f"{total_saved_h:.1f}",
                    "non_EN_skip": f"{skipped_lang}",
                })

        pbar.close()

    finally:
        manifest_file.close()

    print(f"\n{'='*55}")
    print(f"EMILIA DOWNLOAD COMPLETE")
    print(f"  EN utterances saved: {saved:,}  ({total_saved_h:.1f} h)")
    print(f"  Skipped (non-EN):    {skipped_lang:,}")
    print(f"  Skipped (filter):    {skipped_other:,}")
    print(f"  Manifest:            {manifest_path}")
    print(f"{'='*55}")


# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Selective Emilia EN download via HuggingFace streaming",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--manifest",   required=True)
    ap.add_argument("--hf-token",   default=None,
                    help="HF token (auto-loaded from cache if not set)")
    ap.add_argument("--resume",     action="store_true")
    args = ap.parse_args()

    token = resolve_token(args.hf_token)
    if not token:
        print("ERROR: No HuggingFace token found.")
        print("  Run: huggingface-cli login")
        print("  Or:  export HF_TOKEN=hf_xxxxxxxxxxxx")
        raise SystemExit(1)

    print(f"HF token: {'*' * 8}{token[-4:] if len(token) > 4 else '****'}")

    download_emilia(
        output_dir    = Path(args.output_dir),
        manifest_path = Path(args.manifest),
        hf_token      = token,
        resume        = args.resume,
    )


if __name__ == "__main__":
    main()
