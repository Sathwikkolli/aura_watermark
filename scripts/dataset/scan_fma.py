#!/usr/bin/env python3
"""
Scan all FMA full-length audio files and write fma_raw.csv.

For each track:
  - Reads duration and genre from fma_metadata/tracks.csv
  - Samples 10 s from the middle of the file to compute:
      rms_db     — RMS energy (reject near-silence: rms_db < -50)
      clip_frac  — fraction of samples above 0.99 (reject heavy clipping)
  - Checks file exists on disk

Usage (on Great Lakes):
    python scan_fma.py \\
        --fma-root      /nfs/turbo/umd-hafiz/issf_server_data/fma/fma_full \\
        --metadata-dir  /nfs/turbo/umd-hafiz/issf_server_data/fma/fma_metadata \\
        --out           /nfs/turbo/umd-hafiz/issf_server_data/fma/manifests/fma_raw.csv \\
        --workers       40
"""

import argparse
import csv
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

# ── Quality thresholds ────────────────────────────────────────────────────────
RMS_MIN_DB   = -50.0   # below this = near-silence
CLIP_MAX     =  0.005  # above this fraction of clipped samples = reject


def _fma_path(fma_root: Path, track_id: int) -> Path:
    """FMA stores track 12345 at fma_root/012/012345.mp3"""
    subdir = f"{track_id:06d}"[:3]
    return fma_root / subdir / f"{track_id:06d}.mp3"


def check_audio(args_tuple):
    """
    args_tuple: (track_id, path_str, duration_s, genre_top)
    Returns:    (track_id, path_str, duration_s, genre_top, rms_db, clip_frac, ok)
    """
    track_id, path_str, duration_s, genre_top = args_tuple
    path = Path(path_str)

    try:
        info   = sf.info(path_str)
        sr     = info.samplerate
        total  = info.frames

        # Sample 10 s from the middle — avoids silent intros/outros
        mid         = max(0, total // 2 - 5 * sr)
        n_read      = min(10 * sr, total)
        data, _     = sf.read(path_str, start=mid, stop=mid + n_read,
                              always_2d=True, dtype="float32")
        mono        = data.mean(axis=1)

        rms_db      = float(20 * np.log10(np.sqrt(np.mean(mono ** 2)) + 1e-10))
        clip_frac   = float(np.mean(np.abs(mono) > 0.99))

        ok = (rms_db > RMS_MIN_DB) and (clip_frac < CLIP_MAX)
        return (track_id, path_str, duration_s, genre_top,
                round(rms_db, 2), round(clip_frac, 5), ok)

    except Exception:
        return (track_id, path_str, duration_s, genre_top, -99.0, 1.0, False)


def main():
    ap = argparse.ArgumentParser(
        description="Scan FMA full-length audio -> fma_raw.csv",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--fma-root",     required=True,
                    help="Path to fma_full/ directory (contains 000/ 001/ …)")
    ap.add_argument("--metadata-dir", required=True,
                    help="Path to fma_metadata/ directory (contains tracks.csv)")
    ap.add_argument("--out",          required=True,
                    help="Output CSV path")
    ap.add_argument("--workers",      type=int, default=40,
                    help="Parallel worker processes")
    args = ap.parse_args()

    fma_root = Path(args.fma_root)
    tracks_csv = Path(args.metadata_dir) / "tracks.csv"

    if not tracks_csv.exists():
        raise FileNotFoundError(f"tracks.csv not found: {tracks_csv}")

    # ── Load tracks.csv ───────────────────────────────────────────────────────
    # FMA tracks.csv has a 2-level header — read with header=[0,1]
    print(f"Loading {tracks_csv} …")
    tracks = pd.read_csv(tracks_csv, index_col=0, header=[0, 1])

    # Extract columns we need
    durations  = tracks[("track", "duration")].fillna(0.0)
    genres     = tracks[("track", "genre_top")].fillna("Unknown")

    # ── Build work list (only tracks present on disk) ─────────────────────────
    work = []
    missing = 0
    for tid in tracks.index:
        path = _fma_path(fma_root, int(tid))
        dur  = float(durations[tid])
        genre = str(genres[tid])
        if path.exists() and dur > 0:
            work.append((int(tid), str(path), dur, genre))
        else:
            missing += 1

    print(f"Tracks on disk: {len(work):,}  |  Missing: {missing:,}")

    # ── Parallel quality scan ─────────────────────────────────────────────────
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(check_audio, item): item for item in work}
        for fut in tqdm(as_completed(futures), total=len(futures),
                        desc="Scanning FMA", unit="track"):
            results.append(fut.result())

    n_ok    = sum(1 for r in results if r[-1])
    total_h = sum(r[2] for r in results if r[-1]) / 3600
    print(f"\nQuality OK: {n_ok:,} tracks  ({total_h:.1f} h)")
    print(f"Rejected:   {len(results) - n_ok:,} tracks")

    # ── Write CSV ─────────────────────────────────────────────────────────────
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["track_id", "path", "duration_s", "genre_top",
                    "rms_db", "clip_frac", "quality_ok"])
        w.writerows(results)

    print(f"Wrote: {args.out}")

    # ── Duration distribution summary ─────────────────────────────────────────
    ok_durs = [r[2] for r in results if r[-1]]
    if ok_durs:
        print(f"\nDuration summary (quality-OK tracks):")
        print(f"  mean={np.mean(ok_durs)/60:.1f} min  "
              f"median={np.median(ok_durs)/60:.1f} min  "
              f"min={min(ok_durs):.0f} s  max={max(ok_durs)/60:.0f} min")

    # ── Genre summary ─────────────────────────────────────────────────────────
    df = pd.DataFrame(results,
                      columns=["track_id", "path", "duration_s", "genre_top",
                               "rms_db", "clip_frac", "quality_ok"])
    ok_df = df[df["quality_ok"]]
    genre_h = ok_df.groupby("genre_top")["duration_s"].sum().sort_values(ascending=False) / 3600
    print("\nTop genres (quality-OK hours):")
    print(genre_h.head(10).to_string())


if __name__ == "__main__":
    main()
