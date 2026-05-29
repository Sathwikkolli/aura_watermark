#!/usr/bin/env python3
"""
Emilia selective download — reads parquet shards directly via huggingface_hub
+ pyarrow. NO datasets library, NO torchcodec, NO FFmpeg required.

Strategy:
  1. List all parquet shards in the HF repo using list_repo_files()
  2. Download one shard at a time (~200-800 MB each)
  3. Read with pyarrow — audio bytes are a plain binary column, no decoding
  4. Apply filters (language=EN, DNSMOS, duration, speaker cap)
  5. Save passing audio with soundfile, update manifest CSV
  6. Delete the shard file to reclaim space
  7. Repeat until 2,500 h collected

REQUIREMENTS:
  pip install huggingface_hub pyarrow soundfile pandas tqdm
  huggingface-cli login    (accept dataset terms at HF first)

Usage:
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
import tempfile
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# ── Filter constants ──────────────────────────────────────────────────────────
DNSMOS_MIN    = 3.20
DUR_MIN_S     = 3.0
DUR_MAX_S     = 30.0
TARGET_H      = 2500.0
SPEAKER_CAP_H = 1.0
TARGET_LANG   = "EN"

HF_REPO = "amphion/Emilia-Dataset"


# ─────────────────────────────────────────────────────────────────────────────
# Token
# ─────────────────────────────────────────────────────────────────────────────

def resolve_token(cli_token: str | None) -> str | None:
    if cli_token:
        return cli_token
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    try:
        from huggingface_hub import get_token
        t = get_token()
        if t:
            return t
    except Exception:
        pass
    try:
        from huggingface_hub import HfFolder
        t = HfFolder.get_token()
        if t:
            return t
    except Exception:
        pass
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Metadata extractors — handle all known Emilia schema variants
# ─────────────────────────────────────────────────────────────────────────────

def _str(v) -> str:
    return str(v) if v is not None else ""


def get_language(row: dict) -> str:
    for key in ("language", "lang", "locale"):
        v = row.get(key)
        if v:
            return str(v).split("-")[0].upper()[:2]
    return "UNK"


def get_dnsmos(row: dict) -> float:
    for key in ("dnsmos", "DNSMOS", "dns_mos", "mos", "dnsmos_ovrl", "ovrl"):
        v = row.get(key)
        if v is None:
            continue
        if isinstance(v, dict):
            ovrl = v.get("OVRL") or v.get("ovrl") or v.get("overall")
            if ovrl is not None:
                return float(ovrl)
        else:
            try:
                return float(v)
            except (TypeError, ValueError):
                pass
    return -1.0


def get_duration(row: dict) -> float:
    for key in ("duration", "dur", "length"):
        v = row.get(key)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                pass
    return 0.0


def get_speaker(row: dict) -> str:
    for key in ("speaker", "spk", "speaker_id", "spkid"):
        v = row.get(key)
        if v:
            return str(v)
    return "UNK"


def get_utt_id(row: dict) -> str:
    for key in ("id", "utt_id", "utterance_id", "file"):
        v = row.get(key)
        if v:
            return str(v)
    return ""


def get_audio_bytes(row: dict) -> bytes | None:
    """
    Extract raw audio bytes from a parquet row.

    Emilia stores audio as:
      {"bytes": b"...", "path": "filename.mp3"}   ← struct column
    or sometimes as a plain bytes column.
    """
    audio = row.get("audio") or row.get("wav") or row.get("audio_bytes")
    if audio is None:
        return None
    if isinstance(audio, bytes):
        return audio
    if isinstance(audio, dict):
        b = audio.get("bytes") or audio.get("data")
        if isinstance(b, bytes):
            return b
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Audio save (soundfile only — no torchcodec needed)
# ─────────────────────────────────────────────────────────────────────────────

def save_audio_bytes(raw_bytes: bytes, out_path: Path) -> bool:
    """Decode audio bytes with soundfile and save as 16-bit PCM WAV."""
    try:
        import soundfile as sf
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with io.BytesIO(raw_bytes) as buf:
            data, sr = sf.read(buf, dtype="float32", always_2d=False)
        sf.write(str(out_path), data, sr, subtype="PCM_16")
        return True
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Parquet shard list
# ─────────────────────────────────────────────────────────────────────────────

def list_parquet_shards(token: str) -> list[str]:
    """Return all .parquet file paths in the HF repo, sorted."""
    from huggingface_hub import list_repo_files
    files = list(list_repo_files(HF_REPO, repo_type="dataset", token=token))
    shards = sorted(f for f in files if f.endswith(".parquet"))
    print(f"Found {len(shards)} parquet shards in {HF_REPO}")
    return shards


# ─────────────────────────────────────────────────────────────────────────────
# Main download
# ─────────────────────────────────────────────────────────────────────────────

def download_emilia(
    output_dir:    Path,
    manifest_path: Path,
    hf_token:      str,
    resume:        bool,
    tmp_dir:       Path,
) -> None:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # ── Resume state ──────────────────────────────────────────────────────────
    saved_paths:      set  = set()
    speaker_hours:    dict = {}
    processed_shards: set  = set()
    total_saved_h          = 0.0

    progress_path = manifest_path.parent / "emilia_progress.txt"

    if resume and manifest_path.exists():
        df_ex         = pd.read_csv(manifest_path)
        saved_paths   = set(df_ex["path"].tolist())
        speaker_hours = (df_ex.groupby("speaker")["duration_s"].sum() / 3600).to_dict()
        total_saved_h = df_ex["duration_s"].sum() / 3600
        print(f"Resume: {len(saved_paths):,} utterances already saved "
              f"({total_saved_h:.1f} h)")

    if resume and progress_path.exists():
        processed_shards = set(progress_path.read_text().splitlines())
        print(f"Resume: {len(processed_shards)} shards already processed")

    # ── Manifest file ─────────────────────────────────────────────────────────
    write_header = not (resume and manifest_path.exists())
    mf   = open(manifest_path, "a", newline="", encoding="utf-8")
    mcsv = csv.writer(mf)
    if write_header:
        mcsv.writerow(["path", "duration_s", "speaker", "language", "dnsmos"])

    # ── Shard list ────────────────────────────────────────────────────────────
    shards = list_parquet_shards(hf_token)

    print(f"\nTarget: {TARGET_H:.0f} h EN  |  DNSMOS>={DNSMOS_MIN}  "
          f"|  dur {DUR_MIN_S}-{DUR_MAX_S}s  |  speaker<={SPEAKER_CAP_H}h\n")

    saved_total    = len(saved_paths)
    skipped_lang   = 0
    skipped_filter = 0
    skipped_audio  = 0

    try:
        for shard_idx, shard_path in enumerate(shards):
            if total_saved_h >= TARGET_H:
                print(f"Target {TARGET_H:.0f} h reached — stopping.")
                break

            if shard_path in processed_shards:
                continue

            shard_name = Path(shard_path).name
            print(f"[{shard_idx+1}/{len(shards)}] {shard_name}  "
                  f"(saved so far: {total_saved_h:.1f} h)")

            # ── Download shard ────────────────────────────────────────────────
            try:
                local_shard = hf_hub_download(
                    repo_id   = HF_REPO,
                    filename  = shard_path,
                    repo_type = "dataset",
                    token     = hf_token,
                    local_dir = str(tmp_dir),
                    local_dir_use_symlinks = False,
                )
            except Exception as e:
                print(f"  Download failed: {e} — skipping shard")
                continue

            # ── Read parquet with pyarrow ─────────────────────────────────────
            try:
                table = pq.read_table(local_shard)
            except Exception as e:
                print(f"  Parquet read failed: {e} — skipping")
                Path(local_shard).unlink(missing_ok=True)
                continue

            col_names = table.schema.names
            n_rows    = len(table)

            shard_saved   = 0
            shard_skipped = 0

            pbar = tqdm(total=n_rows, desc=f"  {shard_name}", unit="utt", leave=False)

            for i in range(n_rows):
                if total_saved_h >= TARGET_H:
                    break

                # Convert pyarrow row to plain Python dict
                row = {col: table[col][i].as_py() for col in col_names}

                pbar.update(1)

                # ── Language filter ───────────────────────────────────────────
                lang = get_language(row)
                if lang != TARGET_LANG:
                    skipped_lang += 1
                    continue

                # ── Quality filters ───────────────────────────────────────────
                dnsmos   = get_dnsmos(row)
                duration = get_duration(row)
                speaker  = get_speaker(row)

                if dnsmos < DNSMOS_MIN:
                    skipped_filter += 1
                    continue
                if not (DUR_MIN_S <= duration <= DUR_MAX_S):
                    skipped_filter += 1
                    continue
                if speaker_hours.get(speaker, 0.0) >= SPEAKER_CAP_H:
                    skipped_filter += 1
                    continue

                # ── Build output path ─────────────────────────────────────────
                utt_id   = get_utt_id(row) or f"{speaker}_{saved_total:09d}"
                out_path = output_dir / "EN" / speaker[:12] / f"{utt_id}.wav"

                if str(out_path) in saved_paths:
                    continue

                # ── Extract + save audio ──────────────────────────────────────
                audio_bytes = get_audio_bytes(row)
                if not audio_bytes:
                    skipped_audio += 1
                    continue

                if not save_audio_bytes(audio_bytes, out_path):
                    skipped_audio += 1
                    continue

                # ── Update state ──────────────────────────────────────────────
                dur_h                  = duration / 3600
                speaker_hours[speaker] = speaker_hours.get(speaker, 0.0) + dur_h
                total_saved_h         += dur_h
                saved_total           += 1
                saved_paths.add(str(out_path))
                shard_saved           += 1

                mcsv.writerow([str(out_path), round(duration, 3),
                               speaker, lang, round(dnsmos, 4)])
                mf.flush()

                pbar.set_postfix({
                    "saved_h": f"{total_saved_h:.1f}",
                    "shard":   f"{shard_saved}",
                })

            pbar.close()
            print(f"  Shard done: saved={shard_saved}  skipped_lang={shard_skipped}")

            # ── Delete shard to free space ────────────────────────────────────
            try:
                Path(local_shard).unlink()
            except Exception:
                pass

            # ── Mark shard as done ────────────────────────────────────────────
            processed_shards.add(shard_path)
            with open(progress_path, "a") as pf:
                pf.write(shard_path + "\n")

    finally:
        mf.close()

    print(f"\n{'='*55}")
    print(f"EMILIA DOWNLOAD COMPLETE")
    print(f"  EN utterances saved: {saved_total:,}  ({total_saved_h:.1f} h)")
    print(f"  Skipped non-EN:      {skipped_lang:,}")
    print(f"  Skipped by filter:   {skipped_filter:,}")
    print(f"  Skipped (no audio):  {skipped_audio:,}")
    print(f"  Manifest:            {manifest_path}")
    print(f"{'='*55}")


# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Selective Emilia EN download via pyarrow (no torchcodec)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--output-dir", required=True,
                    help="Root directory to save audio files")
    ap.add_argument("--manifest",   required=True,
                    help="CSV manifest path to write")
    ap.add_argument("--hf-token",   default=None,
                    help="HuggingFace token (auto-loaded from cache if omitted)")
    ap.add_argument("--tmp-dir",    default="/tmp/emilia_shards",
                    help="Temp directory for downloading parquet shards")
    ap.add_argument("--resume",     action="store_true",
                    help="Skip already-processed shards and saved utterances")
    args = ap.parse_args()

    token = resolve_token(args.hf_token)
    if not token:
        print("ERROR: No HuggingFace token found.")
        print("  Run:    huggingface-cli login")
        print("  Or set: export HF_TOKEN=hf_xxxx")
        raise SystemExit(1)

    print(f"HF token: {'*'*8}{token[-4:]}")

    download_emilia(
        output_dir    = Path(args.output_dir),
        manifest_path = Path(args.manifest),
        hf_token      = token,
        resume        = args.resume,
        tmp_dir       = Path(args.tmp_dir),
    )


if __name__ == "__main__":
    main()
