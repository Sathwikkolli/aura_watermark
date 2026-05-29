#!/usr/bin/env python3
"""
Emilia selective download — WebDataset tar format.

Dataset structure (confirmed):
  amphion/Emilia-Dataset on HuggingFace = 4,343 tar files (WebDataset)
  Path pattern: Emilia-YODAS/{LANG}/{LANG}-B{shard:06d}.tar
  We download ONLY English: Emilia-YODAS/EN/EN-B*.tar

Each tar contains paired files per utterance:
  EN_B000000_S000000_W000000.mp3   ← audio
  EN_B000000_S000000_W000000.json  ← metadata (dnsmos, duration, speaker, …)

Strategy:
  1. List all EN tar shards
  2. Download one shard at a time (~50-200 MB each)
  3. Stream through tar members — read JSON, apply filters, save audio
  4. Delete shard after processing (space efficient)
  5. Record processed shards → safe to resume

REQUIREMENTS:
  pip install huggingface_hub soundfile pandas tqdm
  huggingface-cli login   (accept dataset terms at HF first)

Usage:
  python download_emilia.py \\
      --output-dir /nfs/turbo/umd-hafiz/issf_server_data/emilia \\
      --manifest   /nfs/turbo/umd-hafiz/issf_server_data/emilia/manifests/emilia_raw.csv

  # Resume:
  python download_emilia.py ... --resume
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import tarfile
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

HF_REPO       = "amphion/Emilia-Dataset"
EN_TAR_PREFIX = "Emilia-YODAS/EN/"


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
# Metadata extractors — robust to all Emilia JSON variants
# ─────────────────────────────────────────────────────────────────────────────

def get_dnsmos(meta: dict) -> float:
    for key in ("dnsmos", "DNSMOS", "dns_mos", "mos", "dnsmos_ovrl", "ovrl"):
        v = meta.get(key)
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


def get_duration(meta: dict) -> float:
    for key in ("duration", "dur", "length"):
        v = meta.get(key)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                pass
    return 0.0


def get_speaker(meta: dict) -> str:
    for key in ("speaker", "spk", "speaker_id", "spkid"):
        v = meta.get(key)
        if v:
            return str(v)
    return "UNK"


# ─────────────────────────────────────────────────────────────────────────────
# Audio save — soundfile only, no torchcodec needed
# ─────────────────────────────────────────────────────────────────────────────

def save_audio_bytes(raw_bytes: bytes, out_path: Path) -> bool:
    """Decode MP3/WAV bytes with soundfile and write 16-bit PCM WAV."""
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
# Tar processing
# ─────────────────────────────────────────────────────────────────────────────

def process_tar(
    tar_path:      str,
    output_dir:    Path,
    mcsv:          csv.writer,
    mf,                           # open manifest file (for flush)
    speaker_hours: dict,
    saved_paths:   set,
    total_saved_h: float,
    saved_count:   int,
) -> tuple[float, int, int, int, int]:
    """
    Stream through one tar shard, apply filters, save EN audio.

    Returns:
        (total_saved_h, saved_count, skip_lang, skip_filter, skip_audio)
    """
    AUDIO_EXTS = {".mp3", ".wav", ".flac", ".ogg", ".opus"}
    skip_lang = skip_filter = skip_audio = 0

    try:
        tf = tarfile.open(tar_path, "r")
    except Exception as e:
        print(f"  Cannot open tar: {e}")
        return total_saved_h, saved_count, skip_lang, skip_filter, skip_audio

    # Build stem → {audio_member, json_member} mapping
    members: dict[str, dict] = {}
    try:
        for m in tf.getmembers():
            p    = Path(m.name)
            stem = p.stem
            ext  = p.suffix.lower()
            if ext in AUDIO_EXTS:
                members.setdefault(stem, {})["audio"]     = m
                members[stem]["audio_ext"] = ext
            elif ext == ".json":
                members.setdefault(stem, {})["json"] = m
    except Exception as e:
        print(f"  Error reading tar members: {e}")
        tf.close()
        return total_saved_h, saved_count, skip_lang, skip_filter, skip_audio

    for stem, files in members.items():
        if total_saved_h >= TARGET_H:
            break
        if "audio" not in files or "json" not in files:
            continue

        # ── Read JSON metadata ────────────────────────────────────────────────
        try:
            with tf.extractfile(files["json"]) as jf:
                meta = json.load(jf)
        except Exception:
            skip_audio += 1
            continue

        # ── Language check (should all be EN in EN tars, but verify) ─────────
        lang_raw = meta.get("language") or meta.get("lang") or "EN"
        lang     = str(lang_raw).split("-")[0].upper()[:2]
        if lang != TARGET_LANG:
            skip_lang += 1
            continue

        # ── Quality filters ───────────────────────────────────────────────────
        dnsmos   = get_dnsmos(meta)
        duration = get_duration(meta)
        speaker  = get_speaker(meta)

        if dnsmos < DNSMOS_MIN:
            skip_filter += 1
            continue
        if not (DUR_MIN_S <= duration <= DUR_MAX_S):
            skip_filter += 1
            continue
        if speaker_hours.get(speaker, 0.0) >= SPEAKER_CAP_H:
            skip_filter += 1
            continue

        # ── Output path ───────────────────────────────────────────────────────
        out_path = output_dir / "EN" / speaker[:12] / f"{stem}.wav"
        if str(out_path) in saved_paths:
            continue

        # ── Read + save audio ─────────────────────────────────────────────────
        try:
            with tf.extractfile(files["audio"]) as af:
                audio_bytes = af.read()
        except Exception:
            skip_audio += 1
            continue

        if not save_audio_bytes(audio_bytes, out_path):
            skip_audio += 1
            continue

        # ── Update state ──────────────────────────────────────────────────────
        dur_h                  = duration / 3600
        speaker_hours[speaker] = speaker_hours.get(speaker, 0.0) + dur_h
        total_saved_h         += dur_h
        saved_count           += 1
        saved_paths.add(str(out_path))

        mcsv.writerow([str(out_path), round(duration, 3),
                       speaker, lang, round(dnsmos, 4)])
        mf.flush()

    tf.close()
    return total_saved_h, saved_count, skip_lang, skip_filter, skip_audio


# ─────────────────────────────────────────────────────────────────────────────
# Main download loop
# ─────────────────────────────────────────────────────────────────────────────

def download_emilia(
    output_dir:    Path,
    manifest_path: Path,
    hf_token:      str,
    resume:        bool,
    tmp_dir:       Path,
) -> None:
    from huggingface_hub import list_repo_files, hf_hub_download

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # ── List EN shards ────────────────────────────────────────────────────────
    print("Listing EN tar shards …")
    all_files = list(list_repo_files(HF_REPO, repo_type="dataset", token=hf_token))
    en_shards = sorted(f for f in all_files
                       if f.startswith(EN_TAR_PREFIX) and f.endswith(".tar"))
    print(f"Found {len(en_shards)} EN shards")

    if not en_shards:
        print(f"ERROR: No shards found under '{EN_TAR_PREFIX}'")
        print("Available prefixes:", set(f.split("/")[1] for f in all_files if "/" in f))
        return

    # ── Resume state ──────────────────────────────────────────────────────────
    saved_paths:      set  = set()
    speaker_hours:    dict = {}
    processed_shards: set  = set()
    total_saved_h          = 0.0
    saved_count            = 0

    progress_path = manifest_path.parent / "emilia_progress.txt"

    if resume and manifest_path.exists():
        df_ex         = pd.read_csv(manifest_path)
        saved_paths   = set(df_ex["path"].tolist())
        speaker_hours = (df_ex.groupby("speaker")["duration_s"].sum() / 3600).to_dict()
        total_saved_h = df_ex["duration_s"].sum() / 3600
        saved_count   = len(saved_paths)
        print(f"Resume: {saved_count:,} utterances already saved ({total_saved_h:.1f} h)")

    if resume and progress_path.exists():
        processed_shards = set(progress_path.read_text().splitlines())
        print(f"Resume: {len(processed_shards)} shards already processed — skipping")

    # ── Manifest file ─────────────────────────────────────────────────────────
    write_header = not (resume and manifest_path.exists())
    mf   = open(manifest_path, "a", newline="", encoding="utf-8")
    mcsv = csv.writer(mf)
    if write_header:
        mcsv.writerow(["path", "duration_s", "speaker", "language", "dnsmos"])

    print(f"\nTarget: {TARGET_H:.0f} h  |  DNSMOS>={DNSMOS_MIN}  "
          f"|  dur {DUR_MIN_S}-{DUR_MAX_S}s  |  speaker<={SPEAKER_CAP_H}h")
    print(f"Processing {len(en_shards)} EN shards …\n")

    total_skip_lang = total_skip_filter = total_skip_audio = 0

    try:
        pbar = tqdm(en_shards, unit="shard")
        for shard_hf_path in pbar:
            if total_saved_h >= TARGET_H:
                print(f"\nTarget {TARGET_H:.0f} h reached — stopping.")
                break

            if shard_hf_path in processed_shards:
                continue

            shard_name = Path(shard_hf_path).name
            pbar.set_description(f"{shard_name}  saved={total_saved_h:.1f}h")

            # ── Download shard ────────────────────────────────────────────────
            try:
                local_path = hf_hub_download(
                    repo_id   = HF_REPO,
                    filename  = shard_hf_path,
                    repo_type = "dataset",
                    token     = hf_token,
                    local_dir = str(tmp_dir),
                    local_dir_use_symlinks = False,
                )
            except Exception as e:
                print(f"\n  Download failed: {e} — skipping {shard_name}")
                continue

            # ── Process shard ─────────────────────────────────────────────────
            (total_saved_h,
             saved_count,
             sl, sf_, sa) = process_tar(
                local_path, output_dir,
                mcsv, mf,
                speaker_hours, saved_paths,
                total_saved_h, saved_count,
            )
            total_skip_lang   += sl
            total_skip_filter += sf_
            total_skip_audio  += sa

            # ── Cleanup ───────────────────────────────────────────────────────
            try:
                Path(local_path).unlink()
            except Exception:
                pass

            processed_shards.add(shard_hf_path)
            with open(progress_path, "a") as pf:
                pf.write(shard_hf_path + "\n")

    finally:
        mf.close()

    print(f"\n{'='*55}")
    print(f"EMILIA DOWNLOAD COMPLETE")
    print(f"  EN utterances saved: {saved_count:,}  ({total_saved_h:.1f} h)")
    print(f"  Skipped non-EN:      {total_skip_lang:,}")
    print(f"  Skipped by filter:   {total_skip_filter:,}")
    print(f"  Skipped (no audio):  {total_skip_audio:,}")
    print(f"  Manifest:            {manifest_path}")
    print(f"{'='*55}")


# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Selective Emilia EN download (WebDataset tar format)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--manifest",   required=True)
    ap.add_argument("--hf-token",   default=None)
    ap.add_argument("--tmp-dir",    default="/tmp/emilia_shards",
                    help="Temp dir for tar downloads (deleted after each shard)")
    ap.add_argument("--resume",     action="store_true",
                    help="Skip already-processed shards")
    args = ap.parse_args()

    token = resolve_token(args.hf_token)
    if not token:
        print("ERROR: No HuggingFace token found.")
        print("  Run: huggingface-cli login")
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
