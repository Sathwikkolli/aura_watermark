#!/usr/bin/env python3
"""
FMA selective download via HuggingFace streaming.

Source  : benjamin-paine/free-music-archive-full  (ungated, no login needed)
Mirrors : same 106 k tracks as mdeff/fma fma_full.zip — full-length audio

Strategy:
  Stream one Parquet shard at a time from HuggingFace (no 879 GB zip).
  For every track inspect duration + genre BEFORE writing audio to disk.
  Stop once TARGET_H hours of passing audio are saved.
  Write a manifest CSV compatible with curate_fma.py and the AURA dataset loader.

Filters applied during streaming:
  • Duration   10 s – 1 800 s
  • Genre cap  ≤ 300 h per top-level genre  (prevents Rock/Electronic domination)
  • Global cap 2 500 h total

Output layout (identical to fma_full.zip extraction):
  <output_dir>/
    fma_full/
      000/  000002.mp3
      001/  001234.mp3
      …

Manifest CSV columns:
  path · duration_s · genre_top · track_id

Resume:
  Pass --resume to skip tracks already in the manifest.
  The HuggingFace dataset is ordered, so streaming restarts from shard 0
  but already-seen track IDs are skipped in O(1) via a set.

REQUIREMENTS (all in the aura conda env):
  pip install datasets soundfile pandas tqdm

Usage:
  python download_fma.py \\
      --output-dir /nfs/turbo/umd-hafiz/issf_server_data/fma \\
      --manifest   /nfs/turbo/umd-hafiz/issf_server_data/fma/manifests/fma_raw.csv

  # Resume after preemption:
  python download_fma.py ... --resume
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
from pathlib import Path

import pandas as pd
import soundfile as sf
from tqdm import tqdm

# ── Filter constants ──────────────────────────────────────────────────────────
DUR_MIN_S       = 10.0
DUR_MAX_S       = 1_800.0
MAX_H_PER_GENRE = 300.0
TARGET_H        = 2_500.0

HF_REPO = "benjamin-paine/free-music-archive-full"

# ── Candidate column names (the HF dataset uses FMA's original CSV headers) ───
# duration
_DUR_KEYS    = ("track_duration", "duration", "track.duration",
                "duration_s", "length")
# genre
_GENRE_KEYS  = ("track_genre_top", "genre_top", "genre", "track.genre_top",
                "top_genre")
# track id
_ID_KEYS     = ("track_id", "id", "track.id", "tid")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get(sample: dict, keys: tuple, default=None):
    """Return the first non-None value found among candidate keys."""
    for k in keys:
        v = sample.get(k)
        if v is not None:
            return v
    return default


def fma_path(output_dir: Path, track_id: int) -> Path:
    """
    Mirror FMA's 3-digit subdirectory structure.
      track_id=2      → fma_full/000/000002.mp3
      track_id=1234   → fma_full/001/001234.mp3
    """
    tid_str = f"{track_id:06d}"
    return output_dir / "fma_full" / tid_str[:3] / f"{tid_str}.mp3"


def save_audio(audio_data, out_path: Path) -> tuple[bool, float]:
    """
    Write audio from a HuggingFace Audio feature to disk.

    HF Audio columns arrive as one of two forms:
      A) {"bytes": b"<raw mp3/wav bytes>", "path": "..."}  — undecoded
      B) {"array": np.ndarray, "sampling_rate": int}       — decoded

    Returns (success: bool, duration_s: float).
    """
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(audio_data, dict):
            raw = audio_data.get("bytes")
            arr = audio_data.get("array")
            sr  = audio_data.get("sampling_rate", 44_100)

            if raw:
                # Write raw bytes directly (likely already MP3)
                out_path.write_bytes(raw)
                try:
                    with io.BytesIO(raw) as buf:
                        info = sf.info(buf)
                    return True, info.duration
                except Exception:
                    # Can't read duration from bytes — fall back to 0 (caller uses metadata)
                    return True, 0.0

            if arr is not None:
                # Decoded float array — save as 16-bit PCM WAV
                wav_path = out_path.with_suffix(".wav")
                sf.write(str(wav_path), arr, sr, subtype="PCM_16")
                return True, len(arr) / max(sr, 1)

        return False, 0.0

    except Exception as exc:
        # Silently skip corrupt/unreadable audio
        _ = exc
        return False, 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Main download loop
# ─────────────────────────────────────────────────────────────────────────────

def download_fma(
    output_dir:    Path,
    manifest_path: Path,
    resume:        bool,
    target_h:      float = TARGET_H,
) -> None:
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' library not found.")
        print("  Run: pip install datasets")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Resume state ──────────────────────────────────────────────────────────
    saved_ids:    set  = set()
    genre_hours:  dict = {}
    total_saved_h      = 0.0
    saved_count        = 0

    if resume and manifest_path.exists():
        try:
            df_ex = pd.read_csv(manifest_path)
            if df_ex.empty or "track_id" not in df_ex.columns:
                raise ValueError("empty or missing columns")
            df_ex["duration_s"] = pd.to_numeric(
                df_ex["duration_s"], errors="coerce"
            ).fillna(0.0)
            saved_ids     = set(df_ex["track_id"].astype(str).tolist())
            genre_hours   = (
                df_ex.groupby("genre_top")["duration_s"].sum() / 3600
            ).to_dict()
            total_saved_h = df_ex["duration_s"].sum() / 3600
            saved_count   = len(saved_ids)
            print(f"Resume: {saved_count:,} tracks already saved "
                  f"({total_saved_h:.1f} h)")
        except (pd.errors.EmptyDataError, ValueError):
            print("Manifest exists but is empty/corrupt — starting fresh.")

    # ── Manifest file ─────────────────────────────────────────────────────────
    write_header = not (resume and manifest_path.exists())
    mf   = open(manifest_path, "a", newline="", encoding="utf-8")
    mcsv = csv.writer(mf)
    if write_header:
        mcsv.writerow(["path", "duration_s", "genre_top", "track_id"])

    # ── Stream dataset ────────────────────────────────────────────────────────
    print(f"\nStreaming: {HF_REPO}")
    print(f"Target : {target_h:.0f} h | dur {DUR_MIN_S}–{DUR_MAX_S} s "
          f"| genre cap {MAX_H_PER_GENRE} h\n")

    ds = load_dataset(
        HF_REPO,
        split="train",
        streaming=True,
        trust_remote_code=True,
    )

    skip_resume = skip_dur = skip_genre = skip_audio = 0

    try:
        pbar = tqdm(ds, unit="track", dynamic_ncols=True)
        for sample in pbar:
            if total_saved_h >= target_h:
                print(f"\nTarget {target_h:.0f} h reached — stopping.")
                break

            # ── Extract metadata ──────────────────────────────────────────────
            raw_id   = _get(sample, _ID_KEYS, default="UNK")
            track_id = str(raw_id)
            duration = float(_get(sample, _DUR_KEYS, default=0.0) or 0.0)
            genre    = str(_get(sample, _GENRE_KEYS, default="Unknown") or "Unknown")

            # ── Resume skip ───────────────────────────────────────────────────
            if track_id in saved_ids:
                skip_resume += 1
                continue

            # ── Duration filter ───────────────────────────────────────────────
            if not (DUR_MIN_S <= duration <= DUR_MAX_S):
                skip_dur += 1
                continue

            # ── Genre cap ─────────────────────────────────────────────────────
            if genre_hours.get(genre, 0.0) >= MAX_H_PER_GENRE:
                skip_genre += 1
                continue

            # ── Audio ─────────────────────────────────────────────────────────
            audio_data = sample.get("audio")
            if audio_data is None:
                skip_audio += 1
                continue

            # Integer track_id for path construction
            try:
                tid_int = int(raw_id)
            except (ValueError, TypeError):
                tid_int = abs(hash(track_id)) % 999_999

            out_path = fma_path(output_dir, tid_int)
            ok, actual_dur = save_audio(audio_data, out_path)

            if not ok:
                skip_audio += 1
                continue

            # Prefer measured duration over metadata duration
            if actual_dur > 0.0:
                duration = actual_dur

            # ── Update state ──────────────────────────────────────────────────
            dur_h = duration / 3600
            genre_hours[genre]  = genre_hours.get(genre, 0.0) + dur_h
            total_saved_h      += dur_h
            saved_count        += 1
            saved_ids.add(track_id)

            mcsv.writerow([str(out_path), round(duration, 3), genre, track_id])
            mf.flush()

            pbar.set_postfix(
                saved=f"{total_saved_h:.1f}h",
                tracks=saved_count,
                skip_dur=skip_dur,
                skip_genre=skip_genre,
            )

    finally:
        mf.close()

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("FMA DOWNLOAD COMPLETE")
    print(f"  Tracks saved     : {saved_count:,}  ({total_saved_h:.1f} h)")
    print(f"  Skipped (resume) : {skip_resume:,}")
    print(f"  Skipped (dur)    : {skip_dur:,}")
    print(f"  Skipped (genre)  : {skip_genre:,}")
    print(f"  Skipped (audio)  : {skip_audio:,}")
    print(f"  Manifest         : {manifest_path}")
    print()

    # Genre distribution summary
    print("Genre distribution:")
    for g, h in sorted(genre_hours.items(), key=lambda x: -x[1]):
        print(f"  {g:<22s}  {h:6.1f} h")
    print(f"{'='*60}")


# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="FMA selective download via HuggingFace streaming",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--output-dir", required=True,
        help="FMA base directory — audio goes into <output-dir>/fma_full/",
    )
    ap.add_argument(
        "--manifest", required=True,
        help="Output manifest CSV (path, duration_s, genre_top, track_id)",
    )
    ap.add_argument(
        "--resume", action="store_true",
        help="Skip tracks already present in the manifest",
    )
    ap.add_argument(
        "--target-h", type=float, default=TARGET_H,
        help="Stop after this many hours of audio are saved",
    )
    args = ap.parse_args()

    download_fma(
        output_dir    = Path(args.output_dir),
        manifest_path = Path(args.manifest),
        resume        = args.resume,
        target_h      = args.target_h,
    )


if __name__ == "__main__":
    main()
