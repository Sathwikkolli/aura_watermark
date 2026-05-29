#!/usr/bin/env python3
"""
FMA selective download — three-stage pipeline:

  Stage 1: Download fma_metadata.zip (~342 MB, fast)
  Stage 2: Curate track list from metadata (CPU only, seconds)
  Stage 3: Download individual MP3s for selected tracks in parallel

WHY INDIVIDUAL TRACKS instead of fma_full.zip:
  fma_full.zip = 879 GB. We need ~2,500 h ≈ all of fma_full.
  But many tracks are corrupt / silent / too short to pass quality filters.
  Downloading per-track lets us:
    - Skip tracks we know are bad from metadata (saves bandwidth)
    - Parallelize with 16-32 connections (much faster than one serial download)
    - Resume trivially (re-run script, already-downloaded files are skipped)
    - Stay under 600 GB if only ~80% of tracks pass quality filter

TRACK DOWNLOAD URLS:
  FMA stores all tracks at:
    https://files.freemusicarchive.org/storage-freemusicarchive-org/tracks/{6-digit-id}.mp3
  This URL pattern is stable and confirmed from the FMA dataset GitHub.
  The local path mirrors FMA's own 3-digit subdirectory structure:
    fma_full/000/000002.mp3
    fma_full/001/001234.mp3
    ...

REQUIREMENTS:
  pip install pandas tqdm requests

Usage (on Great Lakes):
  # Stage 1+2: metadata download and curation (fast, login node ok)
  python download_fma.py metadata \\
      --fma-dir  /nfs/turbo/umd-hafiz/issf_server_data/fma

  # Stage 3: parallel audio download (submit as SLURM job)
  python download_fma.py audio \\
      --fma-dir      /nfs/turbo/umd-hafiz/issf_server_data/fma \\
      --track-list   /nfs/turbo/umd-hafiz/issf_server_data/fma/manifests/fma_selected_ids.txt \\
      --connections  32 \\
      --resume
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

METADATA_URL = "https://zenodo.org/record/1476463/files/fma_metadata.zip"
# Confirmed working base URL for FMA audio files
FMA_AUDIO_BASE = "https://files.freemusicarchive.org/storage-freemusicarchive-org/tracks"

# Curation thresholds (pre-filter from metadata — final quality filter is done
# after download in scan_fma.py / curate_fma.py)
DUR_MIN_S       = 10.0
DUR_MAX_S       = 1800.0
MAX_H_PER_GENRE = 300.0
TARGET_H        = 2500.0


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def fma_path(fma_root: Path, track_id: int) -> Path:
    """fma_root/000/000002.mp3 for track_id=2"""
    subdir = f"{track_id:06d}"[:3]
    return fma_root / subdir / f"{track_id:06d}.mp3"


def fma_url(track_id: int) -> str:
    return f"{FMA_AUDIO_BASE}/{track_id:06d}.mp3"


def download_track(track_id: int, out_path: Path, retries: int = 3) -> bool:
    """Download a single FMA track MP3. Returns True on success."""
    if out_path.exists() and out_path.stat().st_size > 10_000:
        return True   # already downloaded

    out_path.parent.mkdir(parents=True, exist_ok=True)
    url = fma_url(track_id)
    tmp = out_path.with_suffix(".tmp")

    for attempt in range(retries):
        try:
            import requests
            resp = requests.get(url, stream=True, timeout=60)
            if resp.status_code == 404:
                return False    # track genuinely doesn't exist
            resp.raise_for_status()

            with open(tmp, "wb") as f:
                for chunk in resp.iter_content(chunk_size=65536):
                    f.write(chunk)

            tmp.rename(out_path)
            return True

        except Exception:
            if tmp.exists():
                tmp.unlink()
            if attempt < retries - 1:
                time.sleep(2 ** attempt)

    return False


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: Download + extract metadata
# ─────────────────────────────────────────────────────────────────────────────

def stage_metadata(fma_dir: Path):
    """Download and extract fma_metadata.zip (~342 MB)."""
    fma_dir.mkdir(parents=True, exist_ok=True)
    meta_dir = fma_dir / "fma_metadata"

    if (meta_dir / "tracks.csv").exists():
        print(f"fma_metadata already present at {meta_dir}")
        return

    zip_path = fma_dir / "fma_metadata.zip"

    print(f"Downloading fma_metadata.zip (~342 MB) …")
    cmd = [
        "aria2c",
        f"--dir={fma_dir}",
        "--out=fma_metadata.zip",
        "--max-connection-per-server=4",
        "--split=4",
        "--check-certificate=false",
        "--auto-file-renaming=false",
        "--continue=true",
        METADATA_URL,
    ]
    # Fallback to wget if aria2c not available
    if subprocess.run(["which", "aria2c"],
                      capture_output=True).returncode != 0:
        print("aria2c not found — falling back to wget")
        cmd = ["wget", "-c", "-O", str(zip_path), METADATA_URL]

    ret = subprocess.run(cmd)
    if ret.returncode != 0:
        print("Download failed. Check your internet connection.")
        sys.exit(1)

    print("Extracting …")
    subprocess.run(["unzip", "-q", str(zip_path), "-d", str(fma_dir)],
                   check=True)
    zip_path.unlink()
    print(f"Metadata extracted to {meta_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: Curate track list from metadata
# ─────────────────────────────────────────────────────────────────────────────

def stage_curate(fma_dir: Path, track_list_path: Path, seed: int = 42):
    """
    Read tracks.csv, apply duration + genre cap filters,
    write selected track IDs to track_list_path.
    """
    tracks_csv = fma_dir / "fma_metadata" / "tracks.csv"
    if not tracks_csv.exists():
        print(f"tracks.csv not found. Run: python download_fma.py metadata first.")
        sys.exit(1)

    print(f"Loading {tracks_csv} …")
    tracks = pd.read_csv(tracks_csv, index_col=0, header=[0, 1])

    # Build flat dataframe
    df = pd.DataFrame({
        "track_id":   tracks.index.astype(int),
        "duration_s": tracks[("track", "duration")].fillna(0).astype(float).values,
        "genre_top":  tracks[("track", "genre_top")].fillna("Unknown").astype(str).values,
    })

    print(f"Total tracks in metadata: {len(df):,}  "
          f"({df['duration_s'].sum()/3600:.0f} h)")

    # ── Duration filter ───────────────────────────────────────────────────
    df = df[(df["duration_s"] >= DUR_MIN_S) & (df["duration_s"] <= DUR_MAX_S)]
    print(f"After duration filter [{DUR_MIN_S}s, {DUR_MAX_S}s]:  "
          f"{len(df):,}  ({df['duration_s'].sum()/3600:.0f} h)")

    # ── Genre cap ─────────────────────────────────────────────────────────
    def cap_genre(grp):
        grp = grp.sample(frac=1, random_state=seed)
        cum = grp["duration_s"].cumsum() / 3600
        return grp[cum <= MAX_H_PER_GENRE]

    df = df.groupby("genre_top", group_keys=False).apply(cap_genre)

    # ── Trim to target ────────────────────────────────────────────────────
    df = df.sample(frac=1, random_state=seed)
    cum = df["duration_s"].cumsum() / 3600
    df  = df[cum <= TARGET_H]

    total_h = df["duration_s"].sum() / 3600
    print(f"Selected for download: {len(df):,} tracks  ({total_h:.0f} h)")

    track_list_path.parent.mkdir(parents=True, exist_ok=True)
    df[["track_id", "duration_s", "genre_top"]].to_csv(
        track_list_path, index=False
    )
    print(f"Track list written to {track_list_path}")

    # Also print what we expect genre-wise
    genre_h = df.groupby("genre_top")["duration_s"].sum().sort_values(ascending=False) / 3600
    print("\nExpected genre distribution:")
    for g, h in genre_h.head(10).items():
        print(f"  {g:<22s}  {h:5.0f} h")


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3: Parallel audio download
# ─────────────────────────────────────────────────────────────────────────────

def stage_audio(
    fma_dir: Path,
    track_list_path: Path,
    connections: int,
    resume: bool,
):
    """
    Download MP3 files for all tracks in track_list_path using a thread pool.
    Each thread downloads one track at a time. With 32 threads this typically
    saturates a 1 Gbps NFS link (~50-100 MB/s, ~3-6 TB/day).
    """
    if not track_list_path.exists():
        print(f"Track list not found: {track_list_path}")
        print("Run: python download_fma.py curate --fma-dir … first")
        sys.exit(1)

    df = pd.read_csv(track_list_path)
    track_ids = df["track_id"].astype(int).tolist()
    fma_root  = fma_dir / "fma_full"
    fma_root.mkdir(parents=True, exist_ok=True)

    # ── Count already downloaded ───────────────────────────────────────────
    if resume:
        remaining = []
        for tid in track_ids:
            p = fma_path(fma_root, tid)
            if not (p.exists() and p.stat().st_size > 10_000):
                remaining.append(tid)
        print(f"Resuming: {len(track_ids) - len(remaining):,} already done, "
              f"{len(remaining):,} remaining")
        track_ids = remaining
    else:
        print(f"Downloading {len(track_ids):,} tracks with {connections} connections")

    if not track_ids:
        print("All tracks already downloaded.")
        return

    # ── Progress tracking ──────────────────────────────────────────────────
    success_count = 0
    fail_count    = 0
    bytes_saved   = 0

    pbar = tqdm(total=len(track_ids), unit="track")

    with ThreadPoolExecutor(max_workers=connections) as ex:
        futures = {
            ex.submit(download_track, tid, fma_path(fma_root, tid)): tid
            for tid in track_ids
        }
        for fut in as_completed(futures):
            tid    = futures[fut]
            ok     = fut.result()
            p      = fma_path(fma_root, tid)
            if ok and p.exists():
                success_count += 1
                bytes_saved   += p.stat().st_size
            else:
                fail_count += 1

            pbar.update(1)
            pbar.set_postfix({
                "ok": f"{success_count}",
                "fail": f"{fail_count}",
                "GB": f"{bytes_saved/1e9:.1f}",
            })

    pbar.close()

    print(f"\nDownload complete:")
    print(f"  Success:  {success_count:,} tracks  ({bytes_saved/1e9:.1f} GB)")
    print(f"  Failed:   {fail_count:,} tracks (not found / network error)")
    print(f"  Location: {fma_root}")

    if fail_count > 0:
        print(f"\nNote: {fail_count} tracks failed.")
        print("  Some FMA tracks were deleted by their creators — this is expected.")
        print("  Rerun with --resume to retry transient network failures.")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="FMA selective download pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = ap.add_subparsers(dest="stage", required=True)

    # ── metadata ──────────────────────────────────────────────────────────
    mp = sub.add_parser("metadata", help="Stage 1: download + extract fma_metadata.zip")
    mp.add_argument("--fma-dir", required=True,
                    help="FMA base directory (e.g. …/fma)")

    # ── curate ────────────────────────────────────────────────────────────
    cp = sub.add_parser("curate",
                        help="Stage 2: curate track list from metadata")
    cp.add_argument("--fma-dir",       required=True)
    cp.add_argument("--track-list",    required=True,
                    help="Output CSV of selected track IDs")
    cp.add_argument("--seed",          type=int, default=42)

    # ── audio ─────────────────────────────────────────────────────────────
    ap2 = sub.add_parser("audio", help="Stage 3: parallel MP3 download")
    ap2.add_argument("--fma-dir",    required=True)
    ap2.add_argument("--track-list", required=True,
                     help="CSV from the 'curate' stage")
    ap2.add_argument("--connections", type=int, default=32,
                     help="Parallel download threads")
    ap2.add_argument("--resume",      action="store_true",
                     help="Skip already-downloaded tracks")

    # ── all (run all three stages) ────────────────────────────────────────
    allp = sub.add_parser("all", help="Run all 3 stages end-to-end")
    allp.add_argument("--fma-dir",    required=True)
    allp.add_argument("--track-list", required=True)
    allp.add_argument("--connections", type=int, default=32)
    allp.add_argument("--seed",        type=int, default=42)
    allp.add_argument("--resume",      action="store_true")

    args = ap.parse_args()
    fma_dir = Path(args.fma_dir)

    if args.stage == "metadata":
        stage_metadata(fma_dir)

    elif args.stage == "curate":
        stage_metadata(fma_dir)   # ensure metadata exists
        stage_curate(fma_dir, Path(args.track_list), seed=args.seed)

    elif args.stage == "audio":
        stage_audio(fma_dir, Path(args.track_list),
                    args.connections, args.resume)

    elif args.stage == "all":
        stage_metadata(fma_dir)
        stage_curate(fma_dir, Path(args.track_list), seed=args.seed)
        stage_audio(fma_dir, Path(args.track_list),
                    args.connections, args.resume)


if __name__ == "__main__":
    main()
