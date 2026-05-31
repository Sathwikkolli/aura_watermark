#!/usr/bin/env python3
"""
FMA selective download via HuggingFace streaming.

Source  : benjamin-paine/free-music-archive-full  (ungated, no login needed)
Mirrors : same 106 k tracks as mdeff/fma fma_full.zip — full-length audio

Confirmed dataset schema (from inspection):
  audio       : {"bytes": <mp3 bytes>, "path": "000002.mp3"}
  genres      : [<genre_id>, ...]   (list of integer genre IDs)
  url, title, artist, album_title, listens, language, ...
  NOTE: NO duration field — extracted from audio bytes via soundfile.

Strategy:
  Stream one Parquet shard at a time from HuggingFace (no 879 GB zip).
  For every track:
    1. Extract track_id from audio["path"] (e.g. "000002.mp3" → 2)
    2. Read duration from audio bytes via soundfile (no FFmpeg needed)
    3. Map genre IDs → top-level genre name via genres.csv
    4. Apply duration + genre cap filters
    5. Save passing audio to disk in FMA directory layout
  Stop once TARGET_H hours of qualifying audio are saved.

Filters:
  • Duration   10 s – 1 800 s
  • Genre cap  ≤ 300 h per top-level genre
  • Global cap 2 500 h total

Output layout (identical to fma_full.zip extraction):
  <output_dir>/
    fma_full/
      000/  000002.mp3
      001/  001234.mp3
      …

Manifest CSV: path · duration_s · genre_top · track_id

Resume:
  --resume skips track IDs already in the manifest (O(1) set lookup).
  Streaming restarts from shard 0 but skips already-saved tracks instantly.

REQUIREMENTS:
  pip install datasets soundfile pandas tqdm
  (No HF login — dataset is public and ungated)
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


# ─────────────────────────────────────────────────────────────────────────────
# Genre mapping  (genres.csv → integer ID → top-level genre name)
# ─────────────────────────────────────────────────────────────────────────────

def load_genre_map(fma_metadata_dir: Path) -> dict[int, str]:
    """
    Load FMA genres.csv and return {genre_id: top_level_genre_name}.

    genres.csv columns: #tracks, parent, title, top_level
    The "top_level" column is the genre_id of the root genre for each entry.
    """
    genres_csv = fma_metadata_dir / "genres.csv"
    if not genres_csv.exists():
        print(f"WARNING: genres.csv not found at {genres_csv}. "
              "Genre names will be 'Unknown'.")
        return {}
    try:
        df = pd.read_csv(genres_csv, index_col=0)
        id_to_title  = df["title"].to_dict()
        id_to_toplvl = df["top_level"].to_dict()
        return {
            int(gid): id_to_title.get(id_to_toplvl.get(gid, gid), "Unknown")
            for gid in df.index
        }
    except Exception as exc:
        print(f"WARNING: could not load genres.csv: {exc}")
        return {}


def genre_name(genre_ids: list, genre_map: dict[int, str]) -> str:
    """Return top-level genre name for the first genre ID in the list."""
    for gid in genre_ids or []:
        try:
            name = genre_map.get(int(gid))
            if name:
                return name
        except (TypeError, ValueError):
            pass
    return "Unknown"


# ─────────────────────────────────────────────────────────────────────────────
# Audio helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_duration(raw_bytes: bytes) -> float:
    """Read duration in seconds from raw MP3/WAV bytes via soundfile."""
    try:
        with io.BytesIO(raw_bytes) as buf:
            info = sf.info(buf)
        return info.duration
    except Exception:
        return 0.0


def fma_path(output_dir: Path, track_id: int) -> Path:
    """
    Mirror FMA's 3-digit subdirectory structure.
      track_id=2    → fma_full/000/000002.mp3
      track_id=1234 → fma_full/001/001234.mp3
    """
    tid_str = f"{track_id:06d}"
    return output_dir / "fma_full" / tid_str[:3] / f"{tid_str}.mp3"


def save_audio(raw_bytes: bytes, out_path: Path) -> bool:
    """Write raw MP3 bytes to disk. Returns True on success."""
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(raw_bytes)
        return True
    except Exception:
        return False


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
        from datasets import load_dataset, Audio
    except ImportError:
        print("ERROR: 'datasets' library not found. Run: pip install datasets")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Genre mapping ─────────────────────────────────────────────────────────
    fma_metadata_dir = output_dir / "fma_metadata"
    genre_map = load_genre_map(fma_metadata_dir)
    print(f"Loaded {len(genre_map)} genre mappings from genres.csv")

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
            print("Manifest empty/corrupt — starting fresh.")

    # ── Manifest file ─────────────────────────────────────────────────────────
    write_header = not (resume and manifest_path.exists()
                        and manifest_path.stat().st_size > 0)
    mf   = open(manifest_path, "a", newline="", encoding="utf-8")
    mcsv = csv.writer(mf)
    if write_header:
        mcsv.writerow(["path", "duration_s", "genre_top", "track_id"])

    # ── Stream dataset ────────────────────────────────────────────────────────
    print(f"\nStreaming: {HF_REPO}")
    print(f"Target : {target_h:.0f} h | dur {DUR_MIN_S}–{DUR_MAX_S} s "
          f"| genre cap {MAX_H_PER_GENRE} h\n")

    ds = load_dataset(HF_REPO, split="train", streaming=True)
    # Raw bytes — avoids torchcodec/FFmpeg dependency entirely
    ds = ds.cast_column("audio", Audio(decode=False))

    skip_resume = skip_dur = skip_genre = skip_audio = 0

    try:
        pbar = tqdm(ds, unit="track", dynamic_ncols=True)
        for sample in pbar:
            if total_saved_h >= target_h:
                print(f"\nTarget {target_h:.0f} h reached — stopping.")
                break

            # ── Extract audio field ───────────────────────────────────────────
            audio_field = sample.get("audio") or {}
            raw_bytes   = audio_field.get("bytes", b"") if isinstance(audio_field, dict) else b""
            audio_path  = audio_field.get("path", "") if isinstance(audio_field, dict) else ""

            if not raw_bytes:
                skip_audio += 1
                continue

            # ── Track ID from filename (e.g. "000002.mp3" → 2) ───────────────
            try:
                track_id = int(Path(audio_path).stem)
            except (ValueError, TypeError):
                track_id = abs(hash(audio_path)) % 999_999
            track_id_str = str(track_id)

            # ── Resume skip ───────────────────────────────────────────────────
            if track_id_str in saved_ids:
                skip_resume += 1
                continue

            # ── Duration from bytes (soundfile, no FFmpeg) ────────────────────
            duration = get_duration(raw_bytes)
            if not (DUR_MIN_S <= duration <= DUR_MAX_S):
                skip_dur += 1
                continue

            # ── Genre ─────────────────────────────────────────────────────────
            genre = genre_name(sample.get("genres") or [], genre_map)
            if genre_hours.get(genre, 0.0) >= MAX_H_PER_GENRE:
                skip_genre += 1
                continue

            # ── Save audio ────────────────────────────────────────────────────
            out_path = fma_path(output_dir, track_id)
            if not save_audio(raw_bytes, out_path):
                skip_audio += 1
                continue

            # ── Update state ──────────────────────────────────────────────────
            dur_h = duration / 3600
            genre_hours[genre]  = genre_hours.get(genre, 0.0) + dur_h
            total_saved_h      += dur_h
            saved_count        += 1
            saved_ids.add(track_id_str)

            mcsv.writerow([str(out_path), round(duration, 3), genre, track_id])
            mf.flush()

            pbar.set_postfix(
                saved=f"{total_saved_h:.1f}h",
                tracks=saved_count,
                s_dur=skip_dur,
                s_genre=skip_genre,
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
    if genre_hours:
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
    ap.add_argument("--output-dir", required=True,
                    help="FMA base directory — audio saved to <output-dir>/fma_full/")
    ap.add_argument("--manifest",   required=True,
                    help="Output manifest CSV (path, duration_s, genre_top, track_id)")
    ap.add_argument("--resume",     action="store_true",
                    help="Skip tracks already in the manifest")
    ap.add_argument("--target-h",   type=float, default=TARGET_H,
                    help="Stop after this many hours of audio are saved")
    args = ap.parse_args()

    download_fma(
        output_dir    = Path(args.output_dir),
        manifest_path = Path(args.manifest),
        resume        = args.resume,
        target_h      = args.target_h,
    )


if __name__ == "__main__":
    main()
