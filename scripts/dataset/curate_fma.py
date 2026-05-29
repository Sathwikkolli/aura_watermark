#!/usr/bin/env python3
"""
Genre-stratified curation of FMA to ~2,500 hours.

Strategy:
  1. Hard filter: quality_ok=True, duration 10 s – 1800 s
  2. Cap each genre at MAX_H_PER_GENRE hours (diversity insurance)
  3. Shuffle and trim to TARGET_H total hours

Usage (on Great Lakes):
    python curate_fma.py \\
        --raw   /nfs/turbo/umd-hafiz/issf_server_data/fma/manifests/fma_raw.csv \\
        --out   /nfs/turbo/umd-hafiz/issf_server_data/fma/manifests/fma_curated.csv \\
        --seed  42
"""

import argparse
from pathlib import Path

import pandas as pd

TARGET_H        = 2500.0
MAX_H_PER_GENRE = 300.0
DUR_MIN_S       = 10.0
DUR_MAX_S       = 1800.0


def main():
    ap = argparse.ArgumentParser(
        description="Genre-stratified FMA curation -> fma_curated.csv",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--raw",  required=True,  help="Path to fma_raw.csv")
    ap.add_argument("--out",  required=True,  help="Output CSV path")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.raw)
    print(f"Raw FMA:  {len(df):,} tracks  ({df['duration_s'].sum()/3600:.1f} h)")

    # ── Step 1: Hard filters ──────────────────────────────────────────────────
    df = df[
        df["quality_ok"].astype(bool) &
        (df["duration_s"] >= DUR_MIN_S) &
        (df["duration_s"] <= DUR_MAX_S)
    ].copy()
    print(f"After quality+duration filter:  "
          f"{len(df):,}  ({df['duration_s'].sum()/3600:.1f} h)")

    # ── Step 2: Genre cap ─────────────────────────────────────────────────────
    def cap_genre(grp):
        grp = grp.sample(frac=1, random_state=args.seed)
        cum = grp["duration_s"].cumsum() / 3600
        return grp[cum <= MAX_H_PER_GENRE]

    df = df.groupby("genre_top", group_keys=False).apply(cap_genre)
    print(f"\nAfter genre cap ({MAX_H_PER_GENRE} h each):  "
          f"{len(df):,}  ({df['duration_s'].sum()/3600:.1f} h)")

    print("\nGenre distribution:")
    genre_h = df.groupby("genre_top")["duration_s"].sum().sort_values(ascending=False) / 3600
    for genre, h in genre_h.items():
        pct = 100 * h / (df["duration_s"].sum() / 3600)
        print(f"  {genre:<20s}  {h:6.1f} h  ({pct:.1f}%)")

    # ── Step 3: Final shuffle and trim ────────────────────────────────────────
    df = df.sample(frac=1, random_state=args.seed)
    cum = df["duration_s"].cumsum() / 3600
    df  = df[cum <= TARGET_H]

    total_h = df["duration_s"].sum() / 3600
    print(f"\nFinal FMA manifest:")
    print(f"  {len(df):,} tracks  {total_h:.1f} h")
    print(f"  {df['genre_top'].nunique()} genres")

    # ── Add placeholder columns for schema consistency ─────────────────────────
    df["dataset"]  = "fma"
    df["speaker"]  = "music"     # no speaker concept
    df["language"] = "music"
    df["dnsmos"]   = -1.0        # not applicable for music

    # ── Write ─────────────────────────────────────────────────────────────────
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nWrote: {args.out}")


if __name__ == "__main__":
    main()
